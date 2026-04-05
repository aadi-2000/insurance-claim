"""
Prompt Engineering Module.
Templates, few-shot examples, and chain-of-thought prompting for insurance claim processing.
"""

from .templates import PromptTemplateManager
from .few_shot import FewShotManager
from .chain import ChainOfThoughtBuilder

__all__ = ["PromptTemplateManager", "FewShotManager", "ChainOfThoughtBuilder"]
