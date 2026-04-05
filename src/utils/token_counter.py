"""Token Counter - Tracks API token usage across requests."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TokenCounter:
    """Tracks cumulative token usage by model."""

    def __init__(self):
        self.usage: Dict[str, TokenUsage] = {}
        self.total_requests: int = 0

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        if model not in self.usage:
            self.usage[model] = TokenUsage()
        self.usage[model].prompt_tokens += prompt_tokens
        self.usage[model].completion_tokens += completion_tokens
        self.usage[model].total_tokens += prompt_tokens + completion_tokens
        self.total_requests += 1

    def get_summary(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "by_model": {m: {"prompt": u.prompt_tokens, "completion": u.completion_tokens,
                             "total": u.total_tokens} for m, u in self.usage.items()},
            "grand_total": sum(u.total_tokens for u in self.usage.values()),
        }

    def estimate_cost(self) -> Dict[str, float]:
        """Rough cost estimate based on typical pricing."""
        pricing = {
            "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
            "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
        }
        costs = {}
        for model, usage in self.usage.items():
            p = pricing.get(model, {"prompt": 0.001, "completion": 0.002})
            costs[model] = round(usage.prompt_tokens * p["prompt"] + usage.completion_tokens * p["completion"], 4)
        costs["total"] = round(sum(costs.values()), 4)
        return costs
