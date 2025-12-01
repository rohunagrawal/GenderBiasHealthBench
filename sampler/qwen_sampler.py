import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types import MessageList, SamplerBase, SamplerResponse


class QwenSampler(SamplerBase):
    """
    Sample from Qwen models using HuggingFace transformers.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        system_instruction: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 32,
        max_new_tokens: int = 2048,
        device: str | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the Qwen sampler.
        
        Args:
            model: HuggingFace model name (default: Qwen2.5-0.5B-Instruct)
            system_instruction: Optional system instruction to prepend to conversations
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum number of tokens to generate
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            seed: Random seed for reproducibility
        """
        self.model_name = model
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Qwen model: {model} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set random seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _content_to_text(self, content: Any) -> str:
        """
        Convert content (which may be text, dict, or list) to plain text.
        For simplicity, this implementation only handles text content.
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        
        # Handle list of content items (e.g., text + images)
        if isinstance(content, list):
            text_parts = []
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                chunk_type = chunk.get("type")
                if chunk_type in {"text", "input_text"} and "text" in chunk:
                    text_parts.append(chunk["text"])
                elif chunk_type == "image_url":
                    # For now, we'll just add a placeholder for images
                    text_parts.append("[Image]")
                elif chunk_type == "input_image":
                    text_parts.append("[Image]")
            return " ".join(text_parts)
        
        # Fallback: stringify unknown content
        return str(content)

    def _message_list_to_chat(self, message_list: MessageList) -> list[dict[str, str]]:
        """
        Convert MessageList to the chat format expected by Qwen models.
        """
        chat_messages = []
        
        # Add system instruction if provided
        if self.system_instruction:
            chat_messages.append({
                "role": "system",
                "content": self.system_instruction
            })
        
        # Convert each message
        for message in message_list:
            role = message.get("role", "user")
            content = message.get("content", "")
            text_content = self._content_to_text(content)
            
            chat_messages.append({
                "role": role,
                "content": text_content
            })
        
        return chat_messages

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Generate a response for the given message list.
        """
        trial = 0
        max_retries = 3
        
        while trial < max_retries:
            try:
                # Convert messages to chat format
                chat_messages = self._message_list_to_chat(message_list)
                
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        do_sample=self.temperature > 0,
                    )
                
                # Decode only the generated tokens (exclude input)
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response_text = self.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
                
                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"model": self.model_name},
                    actual_queried_message_list=message_list,
                )
                
            except Exception as e:
                exception_backoff = 2 ** trial
                print(
                    f"Qwen generation error; retry {trial} after {exception_backoff} sec: {e}"
                )
                time.sleep(exception_backoff)
                trial += 1
        
        # If all retries failed, return empty response
        return SamplerResponse(
            response_text="",
            response_metadata={"model": self.model_name, "error": "Max retries exceeded"},
            actual_queried_message_list=message_list,
        )
