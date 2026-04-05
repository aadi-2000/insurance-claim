"""
Rate Limiter - Controls API request rates to stay within provider limits.
"""

import asyncio
import time
from collections import deque
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 150000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.request_times: deque = deque()
        self.token_counts: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0):
        """Wait until a request can be made within rate limits."""
        async with self._lock:
            now = time.time()
            cutoff = now - 60

            # Clean old entries
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()
            while self.token_counts and self.token_counts[0][0] < cutoff:
                self.token_counts.popleft()

            # Check request rate
            if len(self.request_times) >= self.rpm:
                wait_time = self.request_times[0] - cutoff
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # Check token rate
            current_tokens = sum(t[1] for t in self.token_counts)
            if current_tokens + estimated_tokens > self.tpm:
                wait_time = self.token_counts[0][0] - cutoff if self.token_counts else 1
                if wait_time > 0:
                    await asyncio.sleep(max(wait_time, 0.1))

            self.request_times.append(time.time())
            if estimated_tokens > 0:
                self.token_counts.append((time.time(), estimated_tokens))

    def record_tokens(self, tokens_used: int):
        """Record actual tokens used after a request."""
        self.token_counts.append((time.time(), tokens_used))
