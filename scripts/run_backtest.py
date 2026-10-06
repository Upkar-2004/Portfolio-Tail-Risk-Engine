"""Run and persist a Gaussian Monte Carlo backtest."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tailrisk.backtesting import (
    create_forecast_schedule,
    load_backtest_snapshot,
    save_backtest_forecasts,
    save_backtest_metadata,
    validate_backtest_forecasts,
)
from tailrisk.config import (
    load_config,
    validate_simulation_config,
)
from tailrisk.portfolio import load_portfolio_path
from tailrisk.returns import load_processed_returns
from tailrisk.simulation import (
    calculate_ewma_gaussian_monte_carlo_forecasts,
    calculate_rolling_gaussian_monte_carlo_forecasts,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_BACKTEST_DATA_ROOT = _PROJECT_ROOT / "data" / "backtests"
_ROLLING_MODEL = "rolling-gaussian"
_EWMA_MODEL = "ewma-gaussian"
_MODEL_NAMES = {
    _ROLLING_MODEL: "rolling_gaussian_monte_carlo",
    _EWMA_MODEL: "ewma_gaussian_monte_carlo",
}


def parse_args() -> argparse.Namespace:
    """Parse input snapshots and optional execution settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Run and save a Gaussian Monte Carlo backtest."
        )
    )
    parser.add_argument(
        "processed_snapshot",
        type=Path,
        help="Path to the processed asset-return snapshot.",
    )
    parser.add_argument(
        "portfolio_snapshot",
        type=Path,
        help="Path to the portfolio-path snapshot.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(_MODEL_NAMES),
        default=_ROLLING_MODEL,
        help=(
            "Gaussian model to run. Defaults to rolling-gaussian."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Configuration file. Defaults to configs/baseline.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_BACKTEST_DATA_ROOT,
        help="Directory under which the versioned snapshot is created.",
    )
    parser.add_argument(
        "--snapshot-id",
        help="Optional explicit output snapshot identifier.",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        help=(
            "Optional scenario-count override for controlled smoke runs. "
            "Omit it for the configured baseline."
        ),
    )

    return parser.parse_args()


def calculate_model_forecasts(
    model: str,
    asset_returns: pd.DataFrame,
    beginning_weights: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
    scenario_count: int,
    random_seed: int,
    decay_factor: float,
) -> pd.DataFrame:
    """Route validated inputs to the selected Gaussian model."""

    common_arguments = {
        "asset_returns": asset_returns,
        "beginning_weights": beginning_weights,
        "forecast_schedule": forecast_schedule,
        "confidence_levels": confidence_levels,
        "scenario_count": scenario_count,
        "random_seed": random_seed,
    }

    if model == _ROLLING_MODEL:
        return calculate_rolling_gaussian_monte_carlo_forecasts(
            **common_arguments,
        )

    if model == _EWMA_MODEL:
        return calculate_ewma_gaussian_monte_carlo_forecasts(
            **common_arguments,
            decay_factor=decay_factor,
        )

    raise ValueError(
        f"Unsupported Gaussian model: {model}."
    )


def main() -> None:
    """Run, save, and reload-verify the Gaussian backtest."""

    args = parse_args()
    config = deepcopy(load_config(args.config))

    if args.scenario_count is not None:
        config["simulation"]["scenario_count"] = (
            args.scenario_count
        )
        validate_simulation_config(
            config["simulation"]
        )

    if not config["simulation"][
        "reuse_standard_normal_shocks"
    ]:
        raise ValueError(
            "The Gaussian backtest requires shared "
            "standard-normal shocks."
        )

    generated_at = datetime.now(timezone.utc)
    asset_returns, returns_metadata = (
        load_processed_returns(
            args.processed_snapshot
        )
    )
    portfolio_path, portfolio_metadata = (
        load_portfolio_path(
            args.portfolio_snapshot
        )
    )

    expected_return_checksum = portfolio_metadata[
        "source"
    ]["asset_returns_sha256"]
    loaded_return_checksum = returns_metadata[
        "output"
    ]["file"]["sha256"]

    if expected_return_checksum != loaded_return_checksum:
        raise ValueError(
            "Portfolio and return snapshots do not share "
            "the same asset-return source."
        )

    if not asset_returns.index.equals(
        portfolio_path.daily.index
    ):
        raise ValueError(
            "Asset returns and portfolio data must use "
            "identical dates."
        )

    if set(asset_returns.columns) != set(
        portfolio_path.beginning_weights.columns
    ):
        raise ValueError(
            "Asset returns and portfolio weights must use "
            "the same tickers."
        )

    forecasting_config = config["forecasting"]
    simulation_config = config["simulation"]
    confidence_levels = forecasting_config[
        "confidence_levels"
    ]

    forecast_schedule = create_forecast_schedule(
        portfolio_returns=portfolio_path.daily[
            "portfolio_return"
        ],
        estimation_window=forecasting_config[
            "estimation_window"
        ],
    )
    forecasts = calculate_model_forecasts(
        model=args.model,
        asset_returns=asset_returns,
        beginning_weights=(
            portfolio_path.beginning_weights
        ),
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        scenario_count=simulation_config[
            "scenario_count"
        ],
        random_seed=simulation_config[
            "random_seed"
        ],
        decay_factor=config["models"][
            "ewma_gaussian"
        ]["decay_factor"],
    )

    model_name = _MODEL_NAMES[args.model]

    validate_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
    )

    snapshot_id = args.snapshot_id
    if snapshot_id is None:
        timestamp = generated_at.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        snapshot_id = (
            f"{timestamp}_"
            f"{config['experiment']['name']}_"
            f"{model_name}"
        )

    forecasts_path = save_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        output_root=args.output_root,
        snapshot_id=snapshot_id,
    )
    metadata_path = save_backtest_metadata(
        forecasts_path=forecasts_path,
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        config=config,
        model_name=model_name,
        source_returns_snapshot_id=(
            args.processed_snapshot.name
        ),
        source_returns_metadata=returns_metadata,
        source_portfolio_snapshot_id=(
            args.portfolio_snapshot.name
        ),
        source_portfolio_metadata=portfolio_metadata,
        generated_at=generated_at,
    )

    verified_forecasts, _ = load_backtest_snapshot(
        snapshot_directory=forecasts_path.parent,
        forecast_schedule=forecast_schedule,
    )

    print(f"{args.model} Monte Carlo backtest")
    print(
        "Forecast dates:",
        f"{forecast_schedule.index[0].date()} to "
        f"{forecast_schedule.index[-1].date()}",
    )
    print(
        "Forecast-date count:",
        f"{len(forecast_schedule):,}",
    )
    print(
        "Result rows:",
        f"{len(verified_forecasts):,}",
    )
    print(
        "Simulation scenarios per date:",
        f"{simulation_config['scenario_count']:,}",
    )
    print(
        "Random seed:",
        simulation_config["random_seed"],
    )
    if args.model == _EWMA_MODEL:
        print(
            "EWMA decay factor:",
            config["models"]["ewma_gaussian"][
                "decay_factor"
            ],
        )

    exceedance_counts = (
        verified_forecasts["var_exceedance"]
        .groupby(level="confidence_level")
        .sum()
    )
    print("VaR exceedances:")
    for level, count in exceedance_counts.items():
        print(
            f"  {float(level):.1%}: {int(count):,}"
        )

    print(f"Saved forecasts: {forecasts_path}")
    print(f"Saved metadata: {metadata_path}")
    print("Reload verification: passed")


if __name__ == "__main__":
    main()
