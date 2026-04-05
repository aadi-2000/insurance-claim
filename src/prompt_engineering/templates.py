"""
Prompt Template Manager - Loads and renders prompt templates from YAML config.
"""

import logging
from typing import Dict, Any, Optional
from config import get_prompt_templates

logger = logging.getLogger("insurance_claim_ai.prompts")


class PromptTemplateManager:
    """Manages prompt templates for all agents."""

    def __init__(self):
        self.templates = get_prompt_templates()
        logger.info(f"Loaded {len(self.templates)} prompt template groups")

    def get_template(self, group: str, name: str) -> str:
        """Get a specific prompt template."""
        group_templates = self.templates.get(group, {})
        template = group_templates.get(name)
        if template is None:
            raise KeyError(f"Template not found: {group}.{name}")
        return template

    def render(self, group: str, name: str, **kwargs) -> str:
        """Render a template with variable substitution."""
        template = self.get_template(group, name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template

    def get_orchestrator_summary_prompt(self, claim_summary: Dict, decision: str,
                                         reasons: str, confidence: float) -> str:
        return self.render(
            "orchestrator", "summary",
            claim_summary=str(claim_summary),
            decision=decision,
            reasons=reasons,
            confidence=f"{confidence:.1f}"
        )

    def get_image_extraction_prompt(self) -> str:
        return self.get_template("image_processing", "extract_text")

    def get_pdf_extraction_prompt(self) -> str:
        return self.get_template("pdf_processing", "extract_structured")

    def get_fraud_analysis_prompt(self, claim_data: Dict) -> str:
        return self.render("fraud_detection", "analyze_patterns", claim_data=str(claim_data))

    def list_templates(self) -> Dict[str, list]:
        """List all available template groups and names."""
        return {group: list(templates.keys()) for group, templates in self.templates.items()}
