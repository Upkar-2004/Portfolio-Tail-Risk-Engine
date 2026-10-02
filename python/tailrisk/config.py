"""Configuration loading and structural validation."""

from pathlib import Path
from typing import Any
from math import isfinite

import yaml

_REQUIRED_SECTIONS = frozenset(
    {
        "experiment",
        "universe",
        "data",
        "portfolio",
        "forecasting",
        "simulation",
    }
)

_REQUIRED_FORECASTING_FIELDS = frozenset(
    {
        "horizon_sessions",
        "estimation_window",
        "confidence_levels",
    }
)


_REQUIRED_SIMULATION_FIELDS = frozenset(
    {
        "scenario_count",
        "random_seed",
        "reuse_standard_normal_shocks",
    }
)


_REQUIRED_PORTFOLIO_FIELDS = frozenset(
    {
        "weighting_method",
        "rebalancing_frequency",
        "long_only",
        "cash_weight",
        "initial_value",
        "transaction_cost_bps",
    }
)

_SUPPORTED_WEIGHTING_METHODS = frozenset(
    {
        "equal",
    }
)

_SUPPORTED_REBALANCING_FREQUENCIES = frozenset(
    {
        "daily",
        "monthly",
    }
)


def _require_finite_number(
    value: Any,
    field: str,
) -> float:
    """Return a numeric configuration value after validating it."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(
            f"Portfolio field '{field}' must be a finite number."
        )

    return float(value)


def validate_portfolio_config(
    portfolio: dict[str, Any],
) -> None:
    """Validate portfolio construction settings."""

    if not isinstance(portfolio, dict):
        raise ValueError(
            "Portfolio configuration must be a mapping."
        )

    missing_fields = _REQUIRED_PORTFOLIO_FIELDS - portfolio.keys()

    if missing_fields:
        missing_names = ", ".join(sorted(missing_fields))
        raise ValueError(
            "Portfolio configuration is missing "
            f"required field(s): {missing_names}"
        )

    weighting_method = portfolio["weighting_method"]
    if weighting_method not in _SUPPORTED_WEIGHTING_METHODS:
        raise ValueError(
            "Unsupported portfolio weighting_method: "
            f"{weighting_method}"
        )

    frequency = portfolio["rebalancing_frequency"]
    if frequency not in _SUPPORTED_REBALANCING_FREQUENCIES:
        raise ValueError(
            "Unsupported rebalancing_frequency: "
            f"{frequency}"
        )

    if not isinstance(portfolio["long_only"], bool):
        raise ValueError(
            "Portfolio field 'long_only' must be Boolean."
        )

    cash_weight = _require_finite_number(
        portfolio["cash_weight"],
        "cash_weight",
    )
    if not 0.0 <= cash_weight < 1.0:
        raise ValueError(
            "Portfolio cash_weight must be between "
            "0 inclusive and 1 exclusive."
        )

    initial_value = _require_finite_number(
        portfolio["initial_value"],
        "initial_value",
    )
    if initial_value <= 0.0:
        raise ValueError(
            "Portfolio initial_value must be positive."
        )

    transaction_cost_bps = _require_finite_number(
        portfolio["transaction_cost_bps"],
        "transaction_cost_bps",
    )
    if transaction_cost_bps < 0.0:
        raise ValueError(
            "Portfolio transaction_cost_bps "
            "cannot be negative."
        )


def validate_forecasting_config(
    forecasting: dict[str, Any],
) -> None:
    """Validate the shared out-of-sample forecasting settings."""

    if not isinstance(forecasting, dict):
        raise ValueError(
            "Forecasting configuration must be a mapping."
        )

    missing_fields = (
        _REQUIRED_FORECASTING_FIELDS
        - forecasting.keys()
    )

    if missing_fields:
        missing_names = ", ".join(
            sorted(missing_fields)
        )
        raise ValueError(
            "Forecasting configuration is missing "
            f"required field(s): {missing_names}"
        )

    horizon_sessions = forecasting[
        "horizon_sessions"
    ]

    if (
        isinstance(horizon_sessions, bool)
        or not isinstance(horizon_sessions, int)
        or horizon_sessions != 1
    ):
        raise ValueError(
            "Forecasting field 'horizon_sessions' "
            "must equal 1."
        )

    estimation_window = forecasting[
        "estimation_window"
    ]

    if (
        isinstance(estimation_window, bool)
        or not isinstance(estimation_window, int)
        or estimation_window <= 0
    ):
        raise ValueError(
            "Forecasting field 'estimation_window' "
            "must be a positive integer."
        )

    confidence_levels = forecasting[
        "confidence_levels"
    ]

    if (
        not isinstance(confidence_levels, list)
        or not confidence_levels
    ):
        raise ValueError(
            "Forecasting field 'confidence_levels' "
            "must be a nonempty list."
        )

    for level in confidence_levels:
        if (
            isinstance(level, bool)
            or not isinstance(level, (int, float))
            or not isfinite(level)
            or not 0.0 < level < 1.0
        ):
            raise ValueError(
                "Forecasting field 'confidence_levels' "
                "must contain finite numbers strictly "
                "between 0 and 1."
            )

    if any(
        left >= right
        for left, right in zip(
            confidence_levels,
            confidence_levels[1:],
        )
    ):
        raise ValueError(
            "Forecasting field 'confidence_levels' "
            "must be unique and strictly increasing."
        )



def validate_simulation_config(
    simulation: dict[str, Any],
) -> None:
    """Validate reproducible Monte Carlo simulation settings."""

    if not isinstance(simulation, dict):
        raise ValueError(
            "Simulation configuration must be a mapping."
        )

    missing_fields = (
        _REQUIRED_SIMULATION_FIELDS
        - simulation.keys()
    )

    if missing_fields:
        missing_names = ", ".join(
            sorted(missing_fields)
        )
        raise ValueError(
            "Simulation configuration is missing "
            f"required field(s): {missing_names}"
        )

    scenario_count = simulation[
        "scenario_count"
    ]

    if (
        isinstance(scenario_count, bool)
        or not isinstance(scenario_count, int)
        or scenario_count < 2
    ):
        raise ValueError(
            "Simulation field 'scenario_count' "
            "must be an integer of at least two."
        )

    random_seed = simulation[
        "random_seed"
    ]

    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError(
            "Simulation field 'random_seed' "
            "must be a non-negative integer."
        )

    reuse_shocks = simulation[
        "reuse_standard_normal_shocks"
    ]

    if not isinstance(reuse_shocks, bool):
        raise ValueError(
            "Simulation field "
            "'reuse_standard_normal_shocks' "
            "must be Boolean."
        )



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

    validate_portfolio_config(config["portfolio"])
    validate_forecasting_config(config["forecasting"])
    validate_simulation_config(config["simulation"])

    return config
