"""Compare analytic and Monte Carlo EWMA Gaussian forecasts."""

import argparse
from pathlib import Path

from tailrisk.backtesting import (
    create_forecast_schedule,
    load_backtest_snapshot,
)
from tailrisk.config import load_config
from tailrisk.evaluation import (
    summarize_gaussian_monte_carlo_errors,
)
from tailrisk.models import (
    calculate_ewma_gaussian_forecasts,
)
from tailrisk.portfolio import load_portfolio_path
from tailrisk.returns import load_processed_returns


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_EWMA_MODEL_NAME = "ewma_gaussian_monte_carlo"


def parse_args() -> argparse.Namespace:
    """Parse verified input and EWMA backtest snapshots."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare analytic and Monte Carlo EWMA "
            "Gaussian forecasts."
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
        "ewma_backtest_snapshot",
        type=Path,
        help="Path to the saved EWMA backtest snapshot.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Configuration file. Defaults to configs/baseline.yaml.",
    )

    return parser.parse_args()


def main() -> None:
    """Load verified data and report full-timeline EWMA errors."""

    args = parse_args()
    config = load_config(args.config)
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

    forecasting_config = config["forecasting"]
    forecast_schedule = create_forecast_schedule(
        portfolio_returns=portfolio_path.daily[
            "portfolio_return"
        ],
        estimation_window=forecasting_config[
            "estimation_window"
        ],
    )
    monte_carlo_forecasts, metadata = (
        load_backtest_snapshot(
            snapshot_directory=(
                args.ewma_backtest_snapshot
            ),
            forecast_schedule=forecast_schedule,
        )
    )

    if metadata["model"]["name"] != _EWMA_MODEL_NAME:
        raise ValueError(
            "Backtest snapshot must contain EWMA "
            "Gaussian Monte Carlo forecasts."
        )

    confidence_levels = forecasting_config[
        "confidence_levels"
    ]
    analytic_forecasts = calculate_ewma_gaussian_forecasts(
        asset_returns=asset_returns,
        beginning_weights=(
            portfolio_path.beginning_weights
        ),
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        decay_factor=config["models"][
            "ewma_gaussian"
        ]["decay_factor"],
    )
    summary = summarize_gaussian_monte_carlo_errors(
        analytic_forecasts=analytic_forecasts,
        monte_carlo_forecasts=monte_carlo_forecasts,
    )

    print("EWMA Gaussian analytic validation")
    print(
        "Forecast dates:",
        f"{len(forecast_schedule):,}",
    )
    print(
        "Compared rows:",
        f"{len(analytic_forecasts):,}",
    )
    print(
        "Monte Carlo scenarios per date:",
        f"{metadata['simulation']['scenario_count']:,}",
    )
    print(
        "EWMA decay factor:",
        config["models"]["ewma_gaussian"][
            "decay_factor"
        ],
    )
    print(
        "Error definition: Monte Carlo minus analytic; "
        "all errors below are in basis points."
    )
    print()
    print(
        f"{'Level':>8}"
        f"{'Metric':>20}"
        f"{'Bias':>10}"
        f"{'MAE':>10}"
        f"{'Median AE':>12}"
        f"{'95% AE':>10}"
        f"{'Max AE':>10}"
    )

    for (
        confidence_level,
        risk_metric,
    ), row in summary.iterrows():
        print(
            f"{confidence_level:>8.1%}"
            f"{risk_metric:>20}"
            f"{row['mean_error_bps']:>+10.4f}"
            f"{row['mean_absolute_error_bps']:>10.4f}"
            f"{row['median_absolute_error_bps']:>12.4f}"
            f"{row['p95_absolute_error_bps']:>10.4f}"
            f"{row['maximum_absolute_error_bps']:>10.4f}"
        )


if __name__ == "__main__":
    main()
