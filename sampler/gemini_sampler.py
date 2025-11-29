import base64
import os
import time
from typing import Any, Iterable

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from ..types import MessageList, SamplerBase, SamplerResponse


class GeminiSampler(SamplerBase):
    """
    Sample from Google's Gemini (and Gemma) models using the Vertex AI SDK.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash-lite",
        system_instruction: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 32,
        max_output_tokens: int = 2048,
        safety_settings: dict[str, Any] | None = None,
        project_id: str | None = None,
        location: str = "us-central1",
        seed: int | None = None,
    ):
        project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError(
                "GeminiSampler requires a Google Cloud project ID. Pass project_id=... or set GOOGLE_CLOUD_PROJECT."
            )
        
        vertexai.init(project=project_id, location=location)

        self.model_name = model
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.safety_settings = safety_settings
        self.seed = seed
        self.model = GenerativeModel(
            model_name=model, system_instruction=system_instruction
        )

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"text": text}

    def _handle_image(self, data_url: str) -> dict[str, Any]:
        """
        Convert an OpenAI-style data URL into the format expected by google-generativeai.
        """
        if data_url.startswith("data:"):
            header, encoded = data_url.split(",", 1)
            mime_type = "application/octet-stream"
            if ";" in header:
                mime_type = header.split(";")[0].removeprefix("data:") or mime_type
            image_bytes = base64.b64decode(encoded)
            return {"mime_type": mime_type, "data": image_bytes}
        # Remote HTTP(S) URIs aren't directly supported with API-key only access.
        return self._handle_text(f"[image: {data_url}]")

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _content_to_parts(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [self._handle_text(content)]

        parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                chunk_type = chunk.get("type")
                if chunk_type in {"text", "input_text"} and "text" in chunk:
                    parts.append(self._handle_text(chunk["text"]))
                elif chunk_type == "image_url":
                    image_info = chunk.get("image_url", {})
                    if isinstance(image_info, dict):
                        url = image_info.get("url")
                    else:
                        url = image_info
                    if url:
                        parts.append(self._handle_image(url))
                elif chunk_type == "input_image":
                    url = chunk.get("image_url")
                    if url:
                        parts.append(self._handle_image(url))
        elif isinstance(content, dict) and "text" in content:
            parts.append(self._handle_text(content["text"]))

        if not parts:
            # fallback: stringify unknown content
            parts.append(self._handle_text(str(content)))
        return parts

    def _message_list_to_contents(self, message_list: MessageList) -> Iterable[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for message in message_list:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts = self._content_to_parts(content)
            contents.append({"role": role, "parts": parts})
        return contents

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        contents = list(self._message_list_to_contents(message_list))
        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.seed is not None:
            generation_config["seed"] = self.seed
        trial = 0
        while True:
            try:
                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                )
                response_text = getattr(response, "text", None)
                if not response_text and getattr(response, "candidates", None):
                    candidate = response.candidates[0]
                    response_text = "".join(
                        part.text for part in candidate.content.parts if hasattr(part, "text")
                    )
                response_text = response_text or ""
                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"model": self.model_name},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial
                print(
                    f"Gemini API error; retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
