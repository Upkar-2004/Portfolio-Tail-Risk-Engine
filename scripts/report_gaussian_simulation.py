"""Report a Gaussian simulation for one real forecast date."""

import argparse
from pathlib import Path

import pandas as pd

from tailrisk.backtesting import create_forecast_schedule
from tailrisk.config import load_config
from tailrisk.portfolio import load_portfolio_path
from tailrisk.returns import load_processed_returns

from tailrisk.covariance import (
    calculate_sample_portfolio_moments,
)
from tailrisk.models import (
    calculate_gaussian_var_es,
)
from tailrisk.simulation import (
    calculate_empirical_var_es,
    generate_gaussian_portfolio_losses,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = (
    _PROJECT_ROOT
    / "configs"
    / "baseline.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse snapshot paths and an optional forecast date."""

    parser = argparse.ArgumentParser(
        description=(
            "Report a Gaussian simulation for one "
            "portfolio forecast date."
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
        "--forecast-date",
        help=(
            "Forecast date in YYYY-MM-DD format. "
            "Defaults to the final eligible date."
        ),
    )

    parser.add_argument(
        "--show-matrices",
        action="store_true",
        help=(
            "Display portfolio weights, asset means, "
            "covariance, and correlation matrices."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Load and align the data required for one forecast."""

    args = parse_args()
    config = load_config(_CONFIG_PATH)

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

    expected_return_checksum = (
        portfolio_metadata["source"][
            "asset_returns_sha256"
        ]
    )
    loaded_return_checksum = (
        returns_metadata["output"]["file"][
            "sha256"
        ]
    )

    if expected_return_checksum != loaded_return_checksum:
        raise ValueError(
            "Portfolio and return snapshots do not "
            "share the same asset-return source."
        )

    if not asset_returns.index.equals(
        portfolio_path.daily.index
    ):
        raise ValueError(
            "Asset returns and portfolio data must "
            "use identical dates."
        )

    if set(asset_returns.columns) != set(
        portfolio_path.beginning_weights.columns
    ):
        raise ValueError(
            "Asset returns and portfolio weights "
            "must use the same tickers."
        )

    estimation_window = config["forecasting"][
        "estimation_window"
    ]

    forecast_schedule = create_forecast_schedule(
        portfolio_returns=portfolio_path.daily[
            "portfolio_return"
        ],
        estimation_window=estimation_window,
    )

    if args.forecast_date is None:
        forecast_date = forecast_schedule.index[-1]
    else:
        forecast_date = pd.Timestamp(
            args.forecast_date
        )

    if forecast_date not in forecast_schedule.index:
        raise ValueError(
            "Requested date is not an eligible "
            "forecast date."
        )

    schedule_row = forecast_schedule.loc[
        forecast_date
    ]
    start_date = schedule_row[
        "estimation_start_date"
    ]
    end_date = schedule_row[
        "estimation_end_date"
    ]

    estimation_returns = asset_returns.loc[
        start_date:end_date
    ]
    forecast_weights = (
        portfolio_path.beginning_weights.loc[
            forecast_date
        ]
    )

    if len(estimation_returns) != estimation_window:
        raise ValueError(
            "Selected estimation window has an "
            "unexpected observation count."
        )

    moments = calculate_sample_portfolio_moments(
        asset_returns=estimation_returns,
        weights=forecast_weights,
    )

    simulation_config = config["simulation"]
    scenario_count = simulation_config[
        "scenario_count"
    ]
    random_seed = simulation_config[
        "random_seed"
    ]

    simulated_losses = (
        generate_gaussian_portfolio_losses(
            mean_vector=moments.mean_vector,
            covariance_matrix=(
                moments.covariance_matrix
            ),
            weights=forecast_weights,
            scenario_count=scenario_count,
            random_seed=random_seed,
        )
    )

    risk_results = []

    for confidence_level in config["forecasting"][
        "confidence_levels"
    ]:
        analytic_result = calculate_gaussian_var_es(
            mean_return=moments.portfolio_mean,
            volatility=moments.portfolio_volatility,
            confidence_level=confidence_level,
        )
        monte_carlo_result = calculate_empirical_var_es(
            losses=simulated_losses,
            confidence_level=confidence_level,
        )

        risk_results.append(
            (
                confidence_level,
                analytic_result,
                monte_carlo_result,
            )
        )

    simulated_mean_return = float(
        -simulated_losses.mean()
    )
    simulated_volatility = float(
        simulated_losses.std(ddof=1)
    )

    print("Gaussian simulation report")
    print(f"Forecast date: {forecast_date.date()}")
    print(
        "Estimation window:",
        f"{start_date.date()} to {end_date.date()}",
    )
    print(
        "Estimation observations:",
        f"{len(estimation_returns):,}",
    )
    print(
        "Assets:",
        len(estimation_returns.columns),
    )
    print(
        "Beginning-weight total:",
        f"{forecast_weights.sum():.6f}",
    )
    print(
        "Simulation scenarios:",
        f"{scenario_count:,}",
    )
    print(f"Random seed: {random_seed}")

    print("\nPortfolio moments")
    print(
        "Analytic mean return:",
        f"{moments.portfolio_mean:.6%}",
    )
    print(
        "Simulated mean return:",
        f"{simulated_mean_return:.6%}",
    )
    print(
        "Analytic volatility:",
        f"{moments.portfolio_volatility:.6%}",
    )
    print(
        "Simulated volatility:",
        f"{simulated_volatility:.6%}",
    )

    print("\nRisk comparison")
    print(
        f"{'Level':>8}"
        f"{'Analytic VaR':>15}"
        f"{'MC VaR':>12}"
        f"{'VaR diff':>12}"
        f"{'Analytic ES':>15}"
        f"{'MC ES':>12}"
        f"{'ES diff':>12}"
        f"{'Tail':>9}"
    )

    for (
        confidence_level,
        analytic_result,
        monte_carlo_result,
    ) in risk_results:
        var_difference_bps = (
            monte_carlo_result.value_at_risk
            - analytic_result.value_at_risk
        ) * 10_000
        es_difference_bps = (
            monte_carlo_result.expected_shortfall
            - analytic_result.expected_shortfall
        ) * 10_000

        print(
            f"{confidence_level:>8.1%}"
            f"{analytic_result.value_at_risk:>15.4%}"
            f"{monte_carlo_result.value_at_risk:>12.4%}"
            f"{var_difference_bps:>+11.2f}bp"
            f"{analytic_result.expected_shortfall:>15.4%}"
            f"{monte_carlo_result.expected_shortfall:>12.4%}"
            f"{es_difference_bps:>+11.2f}bp"
            f"{monte_carlo_result.tail_observation_count:>9,}"
        )

    if args.show_matrices:
        ticker_order = moments.mean_vector.index
        aligned_weights = forecast_weights.reindex(
            ticker_order
        )
        correlation_matrix = (
            estimation_returns
            .reindex(columns=ticker_order)
            .corr()
        )

        print("\nBeginning-of-day weights")
        print(
            aligned_weights.to_string(
                float_format=lambda value: (
                    f"{value:.4%}"
                )
            )
        )

        print("\nAsset mean vector")
        print(
            moments.mean_vector.to_string(
                float_format=lambda value: (
                    f"{value:.6%}"
                )
            )
        )

        print("\nSample covariance matrix")
        print(
            moments.covariance_matrix.to_string(
                float_format=lambda value: (
                    f"{value:.8f}"
                )
            )
        )

        print("\nSample correlation matrix")
        print(
            correlation_matrix.to_string(
                float_format=lambda value: (
                    f"{value:.3f}"
                )
            )
        )


if __name__ == "__main__":
    main()
