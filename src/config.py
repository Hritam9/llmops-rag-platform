"""Shared utility to load configs/config.yaml from anywhere in the project."""
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config file into a plain dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_env(key: str, default: str | None = None) -> str | None:
    """Thin wrapper around os.getenv, kept here so tests can monkeypatch easily."""
    return os.getenv(key, default)
