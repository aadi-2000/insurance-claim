"""
OpenAI GPT Client - Handles all OpenAI API interactions.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient

logger = logging.getLogger("insurance_claim_ai.llm")


class GPTClient(BaseLLMClient):
    """OpenAI GPT client with async support."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 max_tokens: int = 2048, temperature: float = 0.3, max_retries: int = 3):
        _api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not _api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key.")
        super().__init__(model, _api_key, max_tokens, temperature, max_retries)
        self.client = AsyncOpenAI(api_key=_api_key)

    async def complete(self, messages: List[Dict[str, Any]],
                       system_prompt: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """Send a chat completion request."""
        built_messages = self._build_messages(messages, system_prompt)
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=built_messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )
            result = {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }
            logger.info(f"GPT completion: {result['usage']['total_tokens']} tokens used")
            return result
        except Exception as e:
            logger.error(f"GPT completion failed: {e}")
            raise

    async def complete_with_vision(self, messages: List[Dict[str, Any]],
                                    images: List[str],
                                    system_prompt: Optional[str] = None,
                                    **kwargs) -> Dict[str, Any]:
        """Send a vision completion request with base64 images."""
        vision_model = kwargs.get("model", "gpt-4o-mini")  # Use gpt-4o-mini which has vision support
        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}
            })
        if messages:
            content.append({"type": "text", "text": messages[-1].get("content", "")})

        built_messages = []
        if system_prompt:
            built_messages.append({"role": "system", "content": system_prompt})
        built_messages.append({"role": "user", "content": content})

        try:
            response = await self.client.chat.completions.create(
                model=vision_model,
                messages=built_messages,
                max_tokens=kwargs.get("max_tokens", 4096),
            )
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            logger.error(f"GPT vision request failed: {e}")
            raise

    async def get_embeddings(self, texts: List[str],
                              model: str = "text-embedding-3-small") -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def complete_json(self, messages: List[Dict[str, Any]],
                             system_prompt: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
        """Get structured JSON response from GPT."""
        json_system = (system_prompt or "") + "\nRespond ONLY with valid JSON. No markdown, no backticks."
        result = await self.complete(messages, system_prompt=json_system, **kwargs)
        try:
            parsed = json.loads(result["content"].strip().strip("```json").strip("```"))
            result["parsed"] = parsed
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response from GPT")
            result["parsed"] = None
        return result
