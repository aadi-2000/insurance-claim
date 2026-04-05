"""
Configuration module for Insurance Claim AI.
Loads YAML configs and exposes them as Python objects.
"""

import os
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

def load_config(filename: str) -> dict:
    """Load a YAML configuration file."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

def get_model_config() -> dict:
    return load_config("model_config.yaml")

def get_prompt_templates() -> dict:
    return load_config("prompt_templates.yaml")

def get_logging_config() -> dict:
    return load_config("logging_config.yaml")
