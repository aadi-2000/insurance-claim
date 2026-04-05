"""
LLM Utilities - Token counting, image encoding, and helper functions.
"""

import base64
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger("insurance_claim_ai.llm")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Approximate token count for a text string."""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Rough approximation: ~4 chars per token
        return len(text) // 4


def encode_image_base64(image_path: Union[str, Path]) -> str:
    """Read an image file and return base64 encoded string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_bytes_base64(data: bytes) -> str:
    """Encode raw bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def get_mime_type(filename: str) -> str:
    """Determine MIME type from filename."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".pdf": "application/pdf",
        ".tiff": "image/tiff", ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "application/octet-stream")


def truncate_text(text: str, max_tokens: int = 4000, model: str = "gpt-4o-mini") -> str:
    """Truncate text to fit within token limit."""
    tokens = count_tokens(text, model)
    if tokens <= max_tokens:
        return text
    # Rough truncation by character ratio
    ratio = max_tokens / tokens
    return text[:int(len(text) * ratio)]
