"""Tests for configuration loading and structural validation."""

from pathlib import Path

import pytest

from tailrisk.config import (
    load_config,
    validate_portfolio_config,
)


def _valid_portfolio_config() -> dict[str, object]:
    """Return valid portfolio settings for controlled tests."""

    return {
        "weighting_method": "equal",
        "rebalancing_frequency": "monthly",
        "long_only": True,
        "cash_weight": 0.0,
        "initial_value": 1_000_000.0,
        "transaction_cost_bps": 0.0,
    }


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

portfolio:
  weighting_method: "equal"
  rebalancing_frequency: "monthly"
  long_only: true
  cash_weight: 0.0
  initial_value: 1000000.0
  transaction_cost_bps: 0.0
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, dict)
    assert config["experiment"]["name"] == "test"
    assert config["data"]["interval"] == "1d"
    assert config["portfolio"]["rebalancing_frequency"] == "monthly"


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


def test_validate_portfolio_config_accepts_valid_settings() -> None:
    portfolio = _valid_portfolio_config()

    validate_portfolio_config(portfolio)


def test_validate_portfolio_config_rejects_missing_field() -> None:
    portfolio = _valid_portfolio_config()
    portfolio.pop("initial_value")

    with pytest.raises(ValueError, match="initial_value"):
        validate_portfolio_config(portfolio)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weighting_method", "risk_parity"),
        ("rebalancing_frequency", "weekly"),
        ("long_only", "yes"),
        ("cash_weight", -0.10),
        ("cash_weight", 1.00),
        ("initial_value", 0.0),
        ("transaction_cost_bps", -1.0),
    ],
)
def test_validate_portfolio_config_rejects_invalid_settings(
    field: str,
    value: object,
) -> None:
    portfolio = _valid_portfolio_config()
    portfolio[field] = value

    with pytest.raises(ValueError, match=field):
        validate_portfolio_config(portfolio)
