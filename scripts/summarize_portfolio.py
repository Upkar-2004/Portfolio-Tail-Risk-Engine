"""Command-line entry point for verified portfolio diagnostics."""

import argparse
from pathlib import Path

from tailrisk.diagnostics import calculate_return_diagnostics
from tailrisk.portfolio import load_portfolio_path


def parse_args() -> argparse.Namespace:
    """Parse the portfolio snapshot directory."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate descriptive diagnostics for a verified "
            "portfolio snapshot."
        )
    )
    parser.add_argument(
        "snapshot",
        type=Path,
        help="Path to a portfolio snapshot directory.",
    )

    return parser.parse_args()


def main() -> None:
    """Load a verified portfolio snapshot and report diagnostics."""

    args = parse_args()

    portfolio_path, metadata = load_portfolio_path(
        args.snapshot
    )

    diagnostics = calculate_return_diagnostics(
        portfolio_path.daily["portfolio_return"]
    )

    print(f"Snapshot: {args.snapshot.name}")
    print(
        "Source snapshot:",
        metadata["source"]["snapshot_id"],
    )
    print(f"Observations: {diagnostics.observation_count:,}")
    print(
        "Mean daily return:",
        f"{diagnostics.mean_daily_return:.6%}",
    )
    print(
        "Daily volatility:",
        f"{diagnostics.daily_volatility:.6%}",
    )
    print(
        "Annualized arithmetic mean:",
        f"{diagnostics.annualized_arithmetic_mean:.4%}",
    )
    print(
        "Annualized volatility:",
        f"{diagnostics.annualized_volatility:.4%}",
    )
    print(f"Skewness: {diagnostics.skewness:.4f}")
    print(
        "Excess kurtosis:",
        f"{diagnostics.excess_kurtosis:.4f}",
    )
    print(
        "95% empirical loss quantile:",
        f"{diagnostics.loss_quantile_95:.4%}",
    )
    print(
        "97.5% empirical loss quantile:",
        f"{diagnostics.loss_quantile_975:.4%}",
    )
    print(
        "99% empirical loss quantile:",
        f"{diagnostics.loss_quantile_99:.4%}",
    )
    print(
        "Worst daily return:",
        f"{diagnostics.worst_daily_return:.4%}",
        f"on {diagnostics.worst_return_date.date()}",
    )
    print(
        "Best daily return:",
        f"{diagnostics.best_daily_return:.4%}",
        f"on {diagnostics.best_return_date.date()}",
    )


if __name__ == "__main__":
    main()