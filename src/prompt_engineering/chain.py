"""
Chain-of-Thought Builder - Constructs multi-step reasoning prompts for complex claim analysis.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("insurance_claim_ai.prompts")


class ChainOfThoughtBuilder:
    """Builds chain-of-thought prompts for multi-step reasoning tasks."""

    def __init__(self):
        self.steps = []

    def add_step(self, instruction: str, context: Optional[str] = None) -> "ChainOfThoughtBuilder":
        """Add a reasoning step."""
        self.steps.append({"instruction": instruction, "context": context})
        return self

    def build(self) -> str:
        """Build the complete chain-of-thought prompt."""
        prompt_parts = ["Let's analyze this step by step:\n"]
        for i, step in enumerate(self.steps, 1):
            prompt_parts.append(f"Step {i}: {step['instruction']}")
            if step["context"]:
                prompt_parts.append(f"  Context: {step['context']}")
        prompt_parts.append("\nProvide your reasoning for each step, then give a final conclusion.")
        return "\n".join(prompt_parts)

    def build_claim_analysis_chain(self, claim_data: Dict[str, Any]) -> str:
        """Build a chain-of-thought prompt for full claim analysis."""
        self.steps = []
        self.add_step(
            "Verify document completeness",
            f"Check if all required documents are present: discharge summary, policy, bills"
        )
        self.add_step(
            "Validate patient and policy information",
            f"Cross-reference patient details with policy holder information"
        )
        self.add_step(
            "Assess medical necessity",
            f"Evaluate if the diagnosis justifies the procedures performed"
        )
        self.add_step(
            "Check policy coverage",
            f"Verify procedures are covered, check sub-limits and exclusions"
        )
        self.add_step(
            "Analyze billing",
            f"Compare charges against standard rates for the region and procedure codes"
        )
        self.add_step(
            "Evaluate fraud risk",
            f"Check for anomalies, duplicate patterns, and suspicious indicators"
        )
        self.add_step(
            "Make final recommendation",
            "Based on all previous steps, recommend APPROVE, REJECT, REVIEW, or HOLD"
        )
        return self.build()

    def reset(self):
        """Clear all steps."""
        self.steps = []
        return self
