"""Response Cache - In-memory LRU cache for LLM responses."""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Optional


class ResponseCache:
    """LRU cache for LLM API responses to avoid duplicate calls."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[Any]:
        key = self._make_key(prompt, model)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self._cache.move_to_end(key)
                self.hits += 1
                return entry["value"]
            else:
                del self._cache[key]
        self.misses += 1
        return None

    def set(self, prompt: str, model: str, value: Any):
        key = self._make_key(prompt, model)
        self._cache[key] = {"value": value, "timestamp": time.time()}
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"size": len(self._cache), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total > 0 else 0}
