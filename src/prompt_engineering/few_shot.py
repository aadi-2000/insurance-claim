"""
Few-Shot Manager - Manages few-shot examples for in-context learning.
"""

import logging
from typing import List, Dict, Optional
from config import get_prompt_templates

logger = logging.getLogger("insurance_claim_ai.prompts")


class FewShotManager:
    """Manages few-shot examples for different tasks."""

    def __init__(self):
        templates = get_prompt_templates()
        self.examples = templates.get("few_shot", {})
        logger.info(f"Loaded few-shot examples for {len(self.examples)} tasks")

    def get_examples(self, task: str, max_examples: int = 3) -> List[Dict[str, str]]:
        """Get few-shot examples for a specific task."""
        task_examples = self.examples.get(task, [])
        return task_examples[:max_examples]

    def build_few_shot_messages(self, task: str, query: str,
                                 max_examples: int = 3) -> List[Dict[str, str]]:
        """Build message list with few-shot examples."""
        messages = []
        examples = self.get_examples(task, max_examples)
        for ex in examples:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
        messages.append({"role": "user", "content": query})
        return messages

    def add_example(self, task: str, input_text: str, output_text: str):
        """Add a new few-shot example at runtime."""
        if task not in self.examples:
            self.examples[task] = []
        self.examples[task].append({"input": input_text, "output": output_text})
        logger.info(f"Added few-shot example for task '{task}'")
