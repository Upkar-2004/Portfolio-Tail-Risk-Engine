"""Command-line entry point for constructing the baseline portfolio path."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tailrisk.config import load_config
from tailrisk.portfolio import (
    create_equal_weights,
    run_portfolio_path,
    save_portfolio_metadata,
    save_portfolio_path,
)
from tailrisk.returns import load_processed_returns


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_PORTFOLIO_DATA_ROOT = _PROJECT_ROOT / "data" / "portfolio"


def parse_args() -> argparse.Namespace:
    """Parse the processed-return snapshot directory."""

    parser = argparse.ArgumentParser(
        description=(
            "Construct and save the baseline portfolio path."
        )
    )
    parser.add_argument(
        "snapshot",
        type=Path,
        help="Path to a processed-return snapshot directory.",
    )

    return parser.parse_args()


def main() -> None:
    """Construct and save the baseline portfolio path."""

    args = parse_args()
    config = load_config(_CONFIG_PATH)
    processed_at = datetime.now(timezone.utc)

    portfolio_config = config["portfolio"]

    asset_returns, source_metadata = (
        load_processed_returns(args.snapshot)
    )

    expected_tickers = [
        asset["ticker"]
        for asset in config["universe"]["assets"]
    ]

    if set(asset_returns.columns) != set(expected_tickers):
        raise ValueError(
            "Processed-return tickers do not match "
            "the configured portfolio universe."
        )

    if asset_returns.isna().to_numpy().any():
        raise ValueError(
            "Portfolio construction requires complete asset returns."
        )

    if float(
        portfolio_config["transaction_cost_bps"]
    ) != 0.0:
        raise ValueError(
            "Transaction costs are not yet supported "
            "by the portfolio-path engine."
        )

    target_weights = create_equal_weights(
        tickers=expected_tickers,
        cash_weight=float(
            portfolio_config["cash_weight"]
        ),
    )

    result = run_portfolio_path(
        asset_returns=asset_returns,
        initial_value=float(
            portfolio_config["initial_value"]
        ),
        target_weights=target_weights,
        rebalancing_frequency=portfolio_config[
            "rebalancing_frequency"
        ],
    )

    files = save_portfolio_path(
        result=result,
        output_root=_PORTFOLIO_DATA_ROOT,
        snapshot_id=args.snapshot.name,
    )

    metadata_path = save_portfolio_metadata(
        files=files,
        source_snapshot_id=args.snapshot.name,
        source_metadata=source_metadata,
        portfolio_config=portfolio_config,
        target_weights=target_weights,
        result=result,
        processed_at=processed_at,
    )

    daily = result.daily
    rebalance_count = int(
        daily["rebalanced_after_close"].sum()
    )
    total_turnover = float(
        daily["turnover"].sum()
    )

    print(f"Source snapshot: {args.snapshot.name}")
    print(f"Portfolio sessions: {len(daily)}")
    print(f"First date: {daily.index.min().date()}")
    print(f"Last date: {daily.index.max().date()}")
    print(
        "Initial value:",
        f"${portfolio_config['initial_value']:,.2f}",
    )
    print(
        "Ending value:",
        f"${daily['ending_value'].iloc[-1]:,.2f}",
    )
    print(f"Rebalances: {rebalance_count}")
    print(f"Total one-way turnover: {total_turnover:.4f}")
    print(f"Saved daily path: {files.daily}")
    print(
        "Saved beginning weights:",
        files.beginning_weights,
    )
    print(
        "Saved pre-rebalance weights:",
        files.pre_rebalance_weights,
    )
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
