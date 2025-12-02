"""
Train a Qwen model on HealthBench with GRPO to equalize male/female responses.

This script uses Hugging Face TRL + LoRA. The reward is binary: 1.0 when the
generated male and female answers are (near) identical, 0.0 otherwise.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .healthbench_eval import GENDER_COMPARISON_TEMPLATE, parse_json_to_dict
from .healthbench_dataset import HealthBenchDataset
from .sampler.qwen_sampler import QwenSampler


# Prompt prefix that instructs the model to answer the two gendered prompts and
# return both answers with clear delimiters so we can compute the reward.
PROMPT_PREAMBLE = """You will see two conversations that differ only by gender markers.
Respond to each separately. Return answers in this exact format:
[A] <answer for conversation A>
[B] <answer for conversation B>
"""

_gender_difference_grader: QwenSampler | None = None


@dataclass
class GenderPair:
    prompt_id: str
    male: dict[str, Any]
    female: dict[str, Any]


def _extract_gender(example: dict[str, Any]) -> str | None:
    for tag in example.get("example_tags", []):
        if isinstance(tag, str) and tag.lower().startswith("gender:"):
            return tag.split(":", 1)[1].strip().lower()
    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(chunk["text"])
        return " ".join(parts)
    return str(content)


def _conversation_to_string(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content_text = _content_to_text(message.get("content", ""))
        lines.append(f"{role}: {content_text}")
    return "\n".join(lines)


def build_gender_pairs(examples: list[dict[str, Any]]) -> list[GenderPair]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for ex in examples:
        gender = _extract_gender(ex)
        prompt_id = ex.get("prompt_id")
        if gender not in {"male", "female"} or prompt_id is None:
            continue
        grouped.setdefault(prompt_id, {})[gender] = ex

    pairs: list[GenderPair] = []
    for prompt_id, genders in grouped.items():
        if "male" in genders and "female" in genders:
            pairs.append(GenderPair(prompt_id=prompt_id, male=genders["male"], female=genders["female"]))
    return pairs


def render_pair_prompt(tokenizer: AutoTokenizer, pair: GenderPair, system_prompt: str | None = None) -> str:
    male_convo = _conversation_to_string(pair.male["prompt"])
    female_convo = _conversation_to_string(pair.female["prompt"])
    user_text = (
        PROMPT_PREAMBLE
        + "\nConversation A (gender: male):\n"
        + male_convo
        + "\n\nConversation B (gender: female):\n"
        + female_convo
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_answer(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 .,;:'\"?!/\\\\()-]", "", cleaned)
    return cleaned


def split_responses(sample: str) -> tuple[str | None, str | None]:
    # Try to extract [A] ... [B] ...
    match = re.search(r"\[a\](.*?)(?:\n|\r)+\[b\](.*)", sample, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def _get_gender_difference_grader() -> QwenSampler:
    global _gender_difference_grader
    if _gender_difference_grader is None:
        _gender_difference_grader = QwenSampler(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            temperature=0.2,
            max_new_tokens=512,
            seed=42,
        )
    return _gender_difference_grader


def _extract_conversations_from_prompt(prompt: Any) -> tuple[str | None, str | None]:
    prompt_text = _conversation_to_string(prompt) if isinstance(prompt, list) else str(prompt)
    match = re.search(
        r"Conversation A\s*\(gender:\s*male\):\s*(.*?)(?:\n\s*\n|\r\s*\r)\s*Conversation B\s*\(gender:\s*female\):\s*(.*)",
        prompt_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def _format_gender_comparison_prompt(
    male_response: str, female_response: str, prompt: Any | None
) -> str:
    male_prompt, female_prompt = (None, None)
    if prompt is not None:
        male_prompt, female_prompt = _extract_conversations_from_prompt(prompt)

    male_convo = f"{male_prompt}\n\nassistant: {male_response}" if male_prompt else f"assistant: {male_response}"
    female_convo = (
        f"{female_prompt}\n\nassistant: {female_response}" if female_prompt else f"assistant: {female_response}"
    )

    return (
        GENDER_COMPARISON_TEMPLATE.replace("<<male_conversation>>", male_convo)
        .replace("<<female_conversation>>", female_convo)
    )


def _has_material_gender_difference(
    male_response: str, female_response: str, prompt: Any | None
) -> bool | None:
    comparison_prompt = _format_gender_comparison_prompt(male_response, female_response, prompt)
    messages = [{"role": "user", "content": comparison_prompt}]
    grader = _get_gender_difference_grader()

    for _ in range(3):
        sampler_response = grader(messages)
        parsed = parse_json_to_dict(sampler_response.response_text)
        label = parsed.get("criteria_met")
        if isinstance(label, bool):
            return label
    return None


def parity_reward_fn(*, completions: list[Any], prompts: list[Any] | None = None, **_: Any) -> list[float]:
    # Binary reward: 1 when the grader finds no material gender-based difference, else 0.
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            # Conversational format: flatten messages into a single string.
            completion_text = _conversation_to_string(completion)
        else:
            completion_text = str(completion)

        male_resp, female_resp = split_responses(completion_text)
        if not male_resp or not female_resp:
            rewards.append(0.0)
            continue

        prompt_context = prompts[idx] if prompts and idx < len(prompts) else None
        difference = _has_material_gender_difference(male_resp, female_resp, prompt_context)

        if difference is None:
            rewards.append(0.0)
        else:
            rewards.append(0.0 if difference else 1.0)
    return rewards


def load_training_dataset(
    tokenizer: AutoTokenizer,
    dataset_file: str,
    train_fraction: float,
    seed: int,
    system_prompt: str | None,
) -> Dataset:
    hb_dataset = HealthBenchDataset(filename=dataset_file, train_fraction=train_fraction, seed=seed)
    pairs = build_gender_pairs(hb_dataset.train_examples)
    prompts = [
        {"prompt": render_pair_prompt(tokenizer, pair, system_prompt), "prompt_id": pair.prompt_id}
        for pair in pairs
    ]
    if not prompts:
        raise ValueError("No gender-balanced prompt pairs were found in the dataset.")
    return Dataset.from_list(prompts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen on HealthBench gender parity via GRPO + LoRA.")
    parser.add_argument("--dataset-file", default="oss_filtered_subset.jsonl", help="healthbench/*.jsonl file name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base chat model")
    parser.add_argument("--output-dir", default="outputs/healthbench-grpo", help="Directory to store checkpoints")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt to prepend")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Train split size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split")

    # LoRA / training hyperparameters
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4, help="Samples per prompt for GRPO")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    # ref_model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=None,  # Let PEFT pick defaults for Qwen
        task_type="CAUSAL_LM",
    )

    train_dataset = load_training_dataset(
        tokenizer=tokenizer,
        dataset_file=args.dataset_file,
        train_fraction=args.train_fraction,
        seed=args.seed,
        system_prompt=args.system_prompt,
    )

    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=10,
        output_dir=args.output_dir,
        report_to="wandb"
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[parity_reward_fn],
        train_dataset=train_dataset,
        args=grpo_config,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
