"""Locate and load the GB search configuration file."""

from __future__ import annotations

import json
import os
from importlib.resources import files
from typing import Any


def get_config_path() -> str:
    """Return the path to ``GB_search_config.json``.

    Resolution order:
    1. ``LDC_CONFIG`` environment variable
    2. ``./globalGB/GB_search_config.json`` in the current working directory
    """
    if env_path := os.environ.get("LDC_CONFIG"):
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"LDC_CONFIG points to missing file: {env_path}")
        return env_path

    cwd_path = os.path.join(os.getcwd(), "globalGB", "GB_search_config.json")
    if os.path.isfile(cwd_path):
        return cwd_path

    example = files("globalGB").joinpath("GB_search_config.json.example")
    raise FileNotFoundError(
        "GB search config not found. Either:\n"
        "  - set LDC_CONFIG to your config file path, or\n"
        f"  - create globalGB/GB_search_config.json in the working directory\n"
        f"    (see example shipped with the package: {example})"
    )


def load_config() -> dict[str, Any]:
    """Load and return the GB search configuration as a dictionary."""
    with open(get_config_path(), encoding="utf-8") as f:
        return json.load(f)
