"""Configuration loading and structural validation."""

from pathlib import Path
from typing import Any

import yaml


_REQUIRED_SECTIONS = frozenset({"experiment", "universe", "data"}) #immutable set of required sections for the configuration


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate its top-level structure."""

    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    missing_sections = _REQUIRED_SECTIONS - config.keys()

    if missing_sections:
        missing_names = ", ".join(sorted(missing_sections))
        raise ValueError(
            f"Configuration is missing required section(s): {missing_names}"
        )

    return config