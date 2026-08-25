"""Command-line entry point for reproducible data acquisition."""

from pathlib import Path

from tailrisk.config import load_config
from tailrisk.data import (
    count_missing_prices,
    download_market_data,
    extract_field,
    validate_market_data,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"


def main() -> None:
    """Download, validate, and summarize the baseline market data."""

    config = load_config(_CONFIG_PATH)

    expected_tickers = [
        asset["ticker"]
        for asset in config["universe"]["assets"]
    ]

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

    print(f"Rows: {len(market_data)}")
    print(f"Columns: {len(market_data.columns)}")
    print(f"First date: {market_data.index.min()}")
    print(f"Last date: {market_data.index.max()}")
    print("Missing adjusted-close prices:")
    print(missing_counts.to_string())


if __name__ == "__main__":
    main()