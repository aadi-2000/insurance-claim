"""
Error Handler - Standardized error handling for the agent pipeline.
"""

import logging
import traceback
from typing import Dict, Any

logger = logging.getLogger("insurance_claim_ai.handlers")


class ClaimProcessingError(Exception):
    """Base exception for claim processing errors."""
    def __init__(self, message: str, agent: str = "unknown", details: Dict = None):
        self.agent = agent
        self.details = details or {}
        super().__init__(message)


class AgentError(ClaimProcessingError):
    """Exception raised when an agent fails."""
    pass


def handle_agent_error(agent_name: str, error: Exception) -> Dict[str, Any]:
    """Create standardized error response for a failed agent."""
    logger.error(f"[{agent_name}] Error: {error}\n{traceback.format_exc()}")
    return {
        "agent": agent_name,
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
        "reasoning": [f"Agent encountered an error: {error}"],
        "output": {},
        "confidence": 0.0,
        "processing_time": "0s",
    }
