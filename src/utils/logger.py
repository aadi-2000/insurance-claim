"""Logging Setup - Configures application-wide logging from YAML config."""

import logging
import logging.config
import os
from pathlib import Path


def setup_logging(log_dir: str = "data/outputs"):
    """Configure logging from YAML config or defaults."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        from config import get_logging_config
        config = get_logging_config()
        # Ensure log directories exist
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(config)
    except Exception:
        # Fallback to basic config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, "app.log")),
            ],
        )
    
    logger = logging.getLogger("insurance_claim_ai")
    logger.info("Logging initialized")
    return logger
