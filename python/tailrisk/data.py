"""Data ingestion, validation, cleaning, and alignment."""

from typing import Any
from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import json


_REQUIRED_FIELDS = [
    "Adj Close",
    "Close",
    "Dividends",
    "Stock Splits",
]




def download_market_data(
    config: dict[str, Any],
) -> pd.DataFrame:
    """Download market data using the universe and settings in a configuration."""

    tickers = [
        asset["ticker"]
        for asset in config["universe"]["assets"]
    ]

    data_config = config["data"]
    download_options = data_config["download"].copy()

    return yf.download(
        tickers=tickers,
        start=data_config["start"],
        end=data_config["end"],
        interval=data_config["interval"],
        **download_options, #unpack the download options dictionary into keyword arguments for the yf.download function
    )



def extract_field(
    market_data: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    """Extract one price field into a ticker-column DataFrame."""

    if not isinstance(market_data.columns, pd.MultiIndex):
        raise ValueError(
            "Market data must use a MultiIndex"
        )

    required_levels = {"Price", "Ticker"}

    if not required_levels.issubset(market_data.columns.names):
        raise ValueError(
            f"Market data must have columns with levels: {required_levels}"
        )

    available_fields = market_data.columns.get_level_values("Price")

    if field not in available_fields:
        raise ValueError(
            f"Market data does not contain field: {field}"
        )

    return market_data.xs(
        field,
        axis="columns",
        level="Price",
    ).copy()


def validate_tickers(
    market_data: pd.DataFrame,
    expected_tickers: list[str],
) -> None:
    """Verify that all expected tickers are present in market data."""

    observed_tickers = set(
        market_data.columns.get_level_values("Ticker")
    )
    missing_tickers = set(expected_tickers) - observed_tickers

    if missing_tickers:
        missing_names = ", ".join(sorted(missing_tickers))
        raise ValueError(
            f"Market data is missing ticker(s): {missing_names}"
        )


def validate_fields(
    market_data: pd.DataFrame,
    required_fields: list[str],
) -> None:
    """Verify that all required fields are present in market data."""

    observed_fields = set(
        market_data.columns.get_level_values("Price")
    )
    missing_fields = set(required_fields) - observed_fields

    if missing_fields:
        missing_names = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Market data is missing field(s): {missing_names}"
        )



def validate_dates(
    market_data: pd.DataFrame,
) -> None:
    """Verify that market-data dates are unique and chronologically ordered."""

    dates = market_data.index

    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError(
            "Market data must use a DatetimeIndex."
        )

    if dates.has_duplicates:
        raise ValueError(
            "Market data contains duplicate dates."
        )

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Market data dates must be in chronological order."
        )


def validate_prices(
    prices: pd.DataFrame,
) -> None:
    """Verify that every observed price is positive and finite."""

    observed = prices.notna()

    invalid = observed & (
        (prices <= 0)
        | ~np.isfinite(prices)
    )

    if invalid.to_numpy().any():
        raise ValueError(
            "Observed prices must be positive and finite."
        )


def count_missing_prices(
    prices: pd.DataFrame,
) -> pd.Series:
    """Count missing price observations for each ticker."""

    return prices.isna().sum()


def validate_market_data(
    market_data: pd.DataFrame,
    expected_tickers: list[str],
) -> None:
    """Run all structural and numerical market-data validations."""

    validate_dates(market_data)
    validate_tickers(market_data, expected_tickers)
    validate_fields(market_data, _REQUIRED_FIELDS)

    for field in ("Adj Close", "Close"):
        prices = extract_field(market_data, field)
        validate_prices(prices)


"""SHA-256 is a hashing algorithm that converts any amount of data into a fixed-size digital fingerprint."""
"""The result always contains 256 bits, which is 64 hexadecimal characters."""

def calculate_file_sha256(
    path: str | Path,
) -> str:
    """Calculate the SHA-256 checksum of a file's contents."""

    digest = sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_raw_snapshot(
    market_data: pd.DataFrame,
    output_root: str | Path,
    snapshot_id: str,
) -> Path:
    """Save market data in a new snapshot directory without overwriting."""

    snapshot_directory = Path(output_root) / snapshot_id
    snapshot_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    data_path = snapshot_directory / "market_data.csv"
    market_data.to_csv(data_path)

    return data_path


def save_snapshot_metadata(
    data_path: str | Path,
    config: dict[str, Any],
    market_data: pd.DataFrame,
    retrieved_at: datetime,
    checksum: str,
    missing_counts: pd.Series,
) -> Path:
    """Save retrieval and validation metadata beside a raw snapshot."""

    if retrieved_at.utcoffset() is None:
        raise ValueError(
            "Retrieval timestamp must include a time zone."
        )

    metadata = {
        "schema_version": 1,
        "experiment": config["experiment"]["name"],
        "retrieved_at_utc": retrieved_at.astimezone(
            timezone.utc
        ).isoformat(),
        "provider": config["data"]["provider"],
        "access_library": config["data"]["access_library"],
        "access_library_version": yf.__version__,
        "request": {
            "tickers": [
                asset["ticker"]
                for asset in config["universe"]["assets"]
            ],
            "start": config["data"]["start"],
            "end_exclusive": config["data"]["end"],
            "interval": config["data"]["interval"],
            "download_options": config["data"]["download"],
        },
        "response": {
            "rows": len(market_data),
            "columns": len(market_data.columns),
            "first_date": market_data.index.min().isoformat(),
            "last_date": market_data.index.max().isoformat(),
            "fields": sorted(
                set(
                    market_data.columns.get_level_values(
                        "Price"
                    )
                )
            ),
            "tickers": sorted(
                set(
                    market_data.columns.get_level_values(
                        "Ticker"
                    )
                )
            ),
        },
        "validation": {
            "missing_adjusted_close": {
                str(ticker): int(count)
                for ticker, count in missing_counts.items()
            }
        },
        "files": {
            "market_data": {
                "name": Path(data_path).name,
                "sha256": checksum,
            }
        },
    }

    metadata_path = Path(data_path).parent / "metadata.json"

    with metadata_path.open(
        "x",
        encoding="utf-8",
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    return metadata_path



def load_raw_snapshot(
    snapshot_directory: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a raw snapshot after verifying its recorded checksum."""

    snapshot_path = Path(snapshot_directory)
    metadata_path = snapshot_path / "metadata.json"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    file_metadata = metadata["files"]["market_data"]
    data_path = snapshot_path / file_metadata["name"]

    expected_checksum = file_metadata["sha256"]
    actual_checksum = calculate_file_sha256(data_path)

    if actual_checksum != expected_checksum:
        raise ValueError(
            "Raw market-data snapshot checksum does not match."
        )

    market_data = pd.read_csv(
        data_path,
        header=[0, 1],
        index_col=0,
        parse_dates=[0],
    )

    return market_data, metadata
