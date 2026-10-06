"""Tests for Gaussian backtest command model selection."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_run_backtest_module() -> ModuleType:
    """Load the command script without making scripts a package."""

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_backtest.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "run_backtest",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load scripts/run_backtest.py."
        )

    module = importlib.util.module_from_spec(
        module_spec
    )
    module_spec.loader.exec_module(module)

    return module


run_backtest_module = _load_run_backtest_module()


def _placeholder_inputs() -> dict[str, object]:
    """Return small placeholders for testing model dispatch."""

    return {
        "asset_returns": pd.DataFrame(),
        "beginning_weights": pd.DataFrame(),
        "forecast_schedule": pd.DataFrame(),
        "confidence_levels": [0.95],
        "scenario_count": 500,
        "random_seed": 20261002,
        "decay_factor": 0.94,
    }


def test_model_forecasts_route_rolling_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rolling command selection calls the rolling model."""

    expected = pd.DataFrame({"result": [1.0]})
    received_arguments: dict[str, object] = {}

    def fake_rolling_forecasts(
        **arguments: object,
    ) -> pd.DataFrame:
        """Record rolling arguments and return a sentinel result."""

        received_arguments.update(arguments)
        return expected

    monkeypatch.setattr(
        run_backtest_module,
        "calculate_rolling_gaussian_monte_carlo_forecasts",
        fake_rolling_forecasts,
    )

    result = run_backtest_module.calculate_model_forecasts(
        model="rolling-gaussian",
        **_placeholder_inputs(),
    )

    assert result is expected
    assert "decay_factor" not in received_arguments
    assert received_arguments["scenario_count"] == 500
    assert received_arguments["random_seed"] == 20261002


def test_model_forecasts_route_ewma_decay_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EWMA command passes the configured decay factor."""

    expected = pd.DataFrame({"result": [2.0]})
    received_arguments: dict[str, object] = {}

    def fake_ewma_forecasts(
        **arguments: object,
    ) -> pd.DataFrame:
        """Record EWMA arguments and return a sentinel result."""

        received_arguments.update(arguments)
        return expected

    monkeypatch.setattr(
        run_backtest_module,
        "calculate_ewma_gaussian_monte_carlo_forecasts",
        fake_ewma_forecasts,
    )

    result = run_backtest_module.calculate_model_forecasts(
        model="ewma-gaussian",
        **_placeholder_inputs(),
    )

    assert result is expected
    assert received_arguments["decay_factor"] == 0.94
    assert received_arguments["scenario_count"] == 500
    assert received_arguments["random_seed"] == 20261002


def test_model_forecasts_reject_unknown_model() -> None:
    """Model dispatch rejects names outside the supported choices."""

    with pytest.raises(
        ValueError,
        match="Unsupported Gaussian model",
    ):
        run_backtest_module.calculate_model_forecasts(
            model="unknown-model",
            **_placeholder_inputs(),
        )
