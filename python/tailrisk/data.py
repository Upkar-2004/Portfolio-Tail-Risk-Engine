"""Data ingestion, validation, cleaning, and alignment."""

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


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
