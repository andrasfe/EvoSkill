"""Configuration loading for EvoSkill."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    """Return the OpenAI API key from environment."""
    key = os.getenv("EVOSKILL_API_SKILL", "")
    if not key:
        raise RuntimeError(
            "EVOSKILL_API_SKILL environment variable is not set. "
            "Please set it to your OpenAI API key."
        )
    return key


def get_storage_path() -> Path:
    """Return the storage directory path, creating it if needed."""
    raw = os.getenv("EVOSKILL_STORAGE_PATH", "./evoskill_data")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model() -> str:
    """Return the default model to use for skill synthesis."""
    return os.getenv("EVOSKILL_MODEL", "gpt-4o-mini")
