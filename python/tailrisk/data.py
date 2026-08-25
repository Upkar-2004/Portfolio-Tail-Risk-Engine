"""Data ingestion, validation, cleaning, and alignment."""

from typing import Any

import pandas as pd
import yfinance as yf


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
