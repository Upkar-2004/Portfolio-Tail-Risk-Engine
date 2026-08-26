"""Command-line entry point for processing asset returns."""

import argparse
from pathlib import Path

from tailrisk.config import load_config
from tailrisk.data import (
    extract_field,
    load_raw_snapshot,
    validate_market_data,
)
from tailrisk.returns import (
    calculate_simple_returns,
    flag_large_returns,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"


def parse_args() -> argparse.Namespace:
    """Parse the raw snapshot directory from the command line."""

    parser = argparse.ArgumentParser(
        description="Calculate and validate baseline asset returns."
    )
    parser.add_argument(
        "snapshot",
        type=Path,
        help="Path to a raw-data snapshot directory.",
    )

    return parser.parse_args()


def main() -> None:
    """Load a verified snapshot and summarize its asset returns."""

    args = parse_args()
    config = load_config(_CONFIG_PATH)

    expected_tickers = [
        asset["ticker"]
        for asset in config["universe"]["assets"]
    ]

    market_data, metadata = load_raw_snapshot(
        args.snapshot
    )
    validate_market_data(
        market_data,
        expected_tickers,
    )

    adjusted_close = extract_field(
        market_data,
        "Adj Close",
    )
    returns = calculate_simple_returns(adjusted_close)
    complete_returns = returns.dropna(how="any")

    threshold = config["data"]["validation"][
        "large_return_threshold"
    ]
    flags = flag_large_returns(
        returns,
        threshold=threshold,
    )
    flag_counts = flags.sum()

    print(f"Raw snapshot: {args.snapshot.name}")
    print(
        "Source SHA-256:",
        metadata["files"]["market_data"]["sha256"],
    )
    print(f"Return rows: {len(returns)}")
    print(f"Complete return rows: {len(complete_returns)}")
    print(f"First usable date: {complete_returns.index.min()}")
    print(f"Last usable date: {complete_returns.index.max()}")
    print(f"Large-return threshold: {threshold:.0%}")
    print("Flagged returns per ticker:")
    print(flag_counts.to_string())


if __name__ == "__main__":
    main()