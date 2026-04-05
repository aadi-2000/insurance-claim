"""
Base LLM Client - Abstract interface for all language model providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger("insurance_claim_ai.llm")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, api_key: str, max_tokens: int = 2048,
                 temperature: float = 0.3, max_retries: int = 3):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        logger.info(f"Initialized {self.__class__.__name__} with model={model}")

    @abstractmethod
    async def complete(self, messages: List[Dict[str, Any]],
                       system_prompt: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """Send a completion request to the LLM."""
        pass

    @abstractmethod
    async def complete_with_vision(self, messages: List[Dict[str, Any]],
                                    images: List[str],
                                    system_prompt: Optional[str] = None,
                                    **kwargs) -> Dict[str, Any]:
        """Send a completion request with image inputs."""
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    def _build_messages(self, messages: List[Dict], system_prompt: Optional[str] = None) -> List[Dict]:
        """Build message list with optional system prompt."""
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        result.extend(messages)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model})"
