"""Utilities for loading and working with YAML configs."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with Path(path).expanduser().open("r") as f:
        return yaml.safe_load(f)

