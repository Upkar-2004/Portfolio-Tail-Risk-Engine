"""Tests for configuration loading and structural validation."""

from pathlib import Path

import pytest

from tailrisk.config import load_config


def test_load_config_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "valid.yaml"
    config_path.write_text(
        """
experiment:
  name: "test"

universe:
  assets: []

data:
  interval: "1d"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, dict)
    assert config["experiment"]["name"] == "test"
    assert config["data"]["interval"] == "1d"


def test_load_config_rejects_list_at_root(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text(
        """
- GOOGL
- AMZN
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(config_path)


def test_load_config_rejects_missing_required_section(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "missing-section.yaml"
    config_path.write_text(
        """
experiment:
  name: "test"

data:
  interval: "1d"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="universe"):
        load_config(config_path)


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    nonexistent_path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(nonexistent_path)