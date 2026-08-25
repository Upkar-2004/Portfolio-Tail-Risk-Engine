"""Command-line entry point for reproducible data acquisition."""

from datetime import datetime, timezone
from pathlib import Path

from tailrisk.config import load_config
from tailrisk.data import (
    calculate_file_sha256,
    count_missing_prices,
    download_market_data,
    extract_field,
    save_raw_snapshot,
    validate_market_data,
    save_snapshot_metadata,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_RAW_DATA_ROOT = _PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Download, validate, save, and summarize the baseline market data."""

    config = load_config(_CONFIG_PATH)

    expected_tickers = [
        asset["ticker"]
        for asset in config["universe"]["assets"]
    ]

    retrieved_at = datetime.now(timezone.utc)

    market_data = download_market_data(config)

    validate_market_data(
        market_data,
        expected_tickers,
    )

    adjusted_close = extract_field(
        market_data,
        "Adj Close",
    )
    missing_counts = count_missing_prices(adjusted_close)

    snapshot_id = (
        retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + config["experiment"]["name"]
    )

    data_path = save_raw_snapshot(
        market_data,
        output_root=_RAW_DATA_ROOT,
        snapshot_id=snapshot_id,
    )
    checksum = calculate_file_sha256(data_path)

    metadata_path = save_snapshot_metadata(
    data_path=data_path,
    config=config,
    market_data=market_data,
    retrieved_at=retrieved_at,
    checksum=checksum,
    missing_counts=missing_counts,
    )

    print(f"Rows: {len(market_data)}")
    print(f"Columns: {len(market_data.columns)}")
    print(f"First date: {market_data.index.min()}")
    print(f"Last date: {market_data.index.max()}")
    print("Missing adjusted-close prices:")
    print(missing_counts.to_string())
    print(f"Saved snapshot: {data_path}")
    print(f"SHA-256: {checksum}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()