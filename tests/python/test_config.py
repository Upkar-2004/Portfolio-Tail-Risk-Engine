"""Tests for configuration loading and structural validation."""

from pathlib import Path

import pytest

from tailrisk.config import (
    load_config,
    validate_portfolio_config,
    validate_forecasting_config,
    validate_models_config,
    validate_simulation_config,
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


def _valid_simulation_config() -> dict[str, object]:
    """Return valid Monte Carlo simulation settings."""

    return {
        "scenario_count": 100_000,
        "random_seed": 20261002,
        "reuse_standard_normal_shocks": True,
    }


def _valid_forecasting_config() -> dict[str, object]:
    """Return valid forecasting settings for controlled tests."""

    return {
        "horizon_sessions": 1,
        "estimation_window": 504,
        "confidence_levels": [
            0.95,
            0.975,
            0.99,
        ],
    }



def _valid_models_config() -> dict[str, object]:
    """Return valid model settings for controlled tests."""

    return {
        "ewma_gaussian": {
            "decay_factor": 0.94,
        },
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

forecasting:
  horizon_sessions: 1
  estimation_window: 504
  confidence_levels:
    - 0.95
    - 0.975
    - 0.99


models:
  ewma_gaussian:
    decay_factor: 0.94


simulation:
  scenario_count: 100000
  random_seed: 20261002
  reuse_standard_normal_shocks: true

""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, dict)
    assert config["experiment"]["name"] == "test"
    assert config["data"]["interval"] == "1d"
    assert config["portfolio"]["rebalancing_frequency"] == "monthly"
    assert config["forecasting"]["estimation_window"] == 504
    assert config["models"]["ewma_gaussian"]["decay_factor"] == 0.94
    assert config["simulation"]["scenario_count"] == 100_000


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


def test_validate_forecasting_config_accepts_valid_settings() -> None:
    forecasting = _valid_forecasting_config()

    validate_forecasting_config(forecasting)


def test_validate_forecasting_config_rejects_missing_field() -> None:
    forecasting = _valid_forecasting_config()
    forecasting.pop("estimation_window")

    with pytest.raises(
        ValueError,
        match="estimation_window",
    ):
        validate_forecasting_config(forecasting)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon_sessions", 0),
        ("horizon_sessions", 2),
        ("horizon_sessions", True),
        ("estimation_window", 0),
        ("estimation_window", -1),
        ("estimation_window", 504.0),
        ("confidence_levels", []),
        ("confidence_levels", [0.0, 0.95]),
        ("confidence_levels", [0.95, 1.0]),
        ("confidence_levels", [0.95, 0.95]),
        ("confidence_levels", [0.99, 0.95]),
    ],
)
def test_validate_forecasting_config_rejects_invalid_settings(
    field: str,
    value: object,
) -> None:
    forecasting = _valid_forecasting_config()
    forecasting[field] = value

    with pytest.raises(
        ValueError,
        match=field,
    ):
        validate_forecasting_config(forecasting)


def test_validate_simulation_config_accepts_valid_settings() -> None:
    simulation = _valid_simulation_config()

    validate_simulation_config(simulation)


@pytest.mark.parametrize(
    "missing_field",
    [
        "scenario_count",
        "random_seed",
        "reuse_standard_normal_shocks",
    ],
)
def test_validate_simulation_config_rejects_missing_field(
    missing_field: str,
) -> None:
    simulation = _valid_simulation_config()
    simulation.pop(missing_field)

    with pytest.raises(
        ValueError,
        match=missing_field,
    ):
        validate_simulation_config(simulation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_count", 0),
        ("scenario_count", -1),
        ("scenario_count", True),
        ("scenario_count", 100_000.0),
        ("random_seed", -1),
        ("random_seed", True),
        ("random_seed", 20261002.0),
        ("reuse_standard_normal_shocks", 1),
        ("reuse_standard_normal_shocks", "yes"),
        ("reuse_standard_normal_shocks", None),
    ],
)
def test_validate_simulation_config_rejects_invalid_settings(
    field: str,
    value: object,
) -> None:
    simulation = _valid_simulation_config()
    simulation[field] = value

    with pytest.raises(
        ValueError,
        match=field,
    ):
        validate_simulation_config(simulation)



def test_validate_models_config_accepts_valid_settings() -> None:
    """Verify that a valid EWMA decay factor is accepted."""

    models = _valid_models_config()

    validate_models_config(models)


def test_validate_models_config_rejects_missing_model() -> None:
    """Verify that the EWMA Gaussian model section is required."""

    with pytest.raises(
        ValueError,
        match="ewma_gaussian",
    ):
        validate_models_config({})


def test_validate_models_config_rejects_missing_decay_factor() -> None:
    """Verify that the EWMA decay factor is required."""

    models = _valid_models_config()
    models["ewma_gaussian"].pop("decay_factor")

    with pytest.raises(
        ValueError,
        match="decay_factor",
    ):
        validate_models_config(models)


@pytest.mark.parametrize(
    "decay_factor",
    [
        0.0,
        1.0,
        -0.01,
        1.01,
        True,
        False,
        "0.94",
        None,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_validate_models_config_rejects_invalid_decay_factor(
    decay_factor: object,
) -> None:
    """Verify that decay must be finite, numeric, and strictly bounded."""

    models = _valid_models_config()
    models["ewma_gaussian"]["decay_factor"] = (
        decay_factor
    )

    with pytest.raises(
        ValueError,
        match="decay_factor",
    ):
        validate_models_config(models)



@pytest.mark.parametrize(
    "invalid_models",
    [
        None,
        [],
        "models",
    ],
)
def test_validate_models_config_requires_mapping(
    invalid_models: object,
) -> None:
    """Verify that model settings must be supplied as a mapping."""

    with pytest.raises(
        ValueError,
        match="Models configuration must be a mapping",
    ):
        validate_models_config(invalid_models)


@pytest.mark.parametrize(
    "invalid_ewma_config",
    [
        None,
        [],
        0.94,
    ],
)
def test_validate_models_config_requires_ewma_mapping(
    invalid_ewma_config: object,
) -> None:
    """Verify that EWMA settings must be supplied as a mapping."""

    models = {
        "ewma_gaussian": invalid_ewma_config,
    }

    with pytest.raises(
        ValueError,
        match="EWMA Gaussian configuration must be a mapping",
    ):
        validate_models_config(models)
