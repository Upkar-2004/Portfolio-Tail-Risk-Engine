"""Command-line entry point for processing asset returns."""

import argparse
from datetime import datetime, timezone
from pathlib import Path


from tailrisk.config import load_config
from tailrisk.data import (
    calculate_file_sha256,
    extract_field,
    load_raw_snapshot,
    validate_market_data,
)
from tailrisk.returns import (
    calculate_simple_returns,
    flag_large_returns,
    save_processed_metadata,
    save_processed_returns,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_PROCESSED_DATA_ROOT = (_PROJECT_ROOT / "data" / "processed")



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
    processed_at = datetime.now(timezone.utc)

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
    processed_returns = returns.dropna(how="all")
    complete_returns = processed_returns.dropna(how="any")

    threshold = config["data"]["validation"][
        "large_return_threshold"
    ]
    flags = flag_large_returns(
        processed_returns,
        threshold=threshold,
    )
    flag_counts = flags.sum()

    data_path = save_processed_returns(
        returns=processed_returns,
        output_root=_PROCESSED_DATA_ROOT,
        snapshot_id=args.snapshot.name,
    )
    checksum = calculate_file_sha256(data_path)
    metadata_path = save_processed_metadata(
        data_path=data_path,
        source_snapshot_id=args.snapshot.name,
        source_metadata=metadata,
        returns=processed_returns,
        flags=flags,
        threshold=threshold,
        processed_at=processed_at,
        checksum=checksum,
        flagged_return_decision="retain",
    )

    print(f"Raw snapshot: {args.snapshot.name}")
    print(
        "Source SHA-256:",
        metadata["files"]["market_data"]["sha256"],
    )
    print(f"Return rows: {len(processed_returns)}")

    print(f"Complete return rows: {len(complete_returns)}")
    print(f"First usable date: {complete_returns.index.min()}")
    print(f"Last usable date: {complete_returns.index.max()}")
    print(f"Large-return threshold: {threshold:.0%}")
    print("Flagged returns per ticker:")
    print(flag_counts.to_string())
    print(f"Saved returns: {data_path}")
    print(f"SHA-256: {checksum}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
