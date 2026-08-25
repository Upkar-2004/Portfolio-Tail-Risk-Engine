"""Tests for market-data processing."""

import pandas as pd
import pytest

from tailrisk.data import download_market_data, extract_field


def test_extract_field_returns_one_column_per_ticker() -> None:
    """Verify that extracting a field produces one column per ticker."""

    columns = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "GOOGL"),
            ("Adj Close", "AMZN"),
            ("Close", "GOOGL"),
            ("Close", "AMZN"),
        ],
        names=["Price", "Ticker"],
    )

    market_data = pd.DataFrame(
        [[100.0, 200.0, 101.0, 201.0]],
        index=pd.to_datetime(["2025-01-02"]),
        columns=columns,
    )

    adjusted_close = extract_field(market_data, "Adj Close")

    assert list(adjusted_close.columns) == ["GOOGL", "AMZN"]
    assert adjusted_close.loc["2025-01-02", "GOOGL"] == 100.0
    assert adjusted_close.loc["2025-01-02", "AMZN"] == 200.0


def test_extract_field_rejects_missing_field() -> None:
    """Verify that extraction rejects a field absent from the market data."""

    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "GOOGL"),
        ],
        names=["Price", "Ticker"],
    )

    market_data = pd.DataFrame(
        [[101.0]],
        columns=columns,
    )

    with pytest.raises(ValueError, match="Adj Close"):
        extract_field(market_data, "Adj Close")


def test_extract_field_rejects_flat_columns() -> None:
    """Verify that extraction rejects columns without the expected levels."""

    market_data = pd.DataFrame(
        {
            "Adj Close": [100.0],
            "Close": [101.0],
        }
    )

    with pytest.raises(ValueError, match="MultiIndex"):
        extract_field(market_data, "Adj Close")


def test_download_market_data_uses_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that configuration values are forwarded to yfinance."""

    received_arguments = {}
    expected_data = pd.DataFrame({"placeholder": [1.0]})

    def fake_download(**kwargs):
        """Record download arguments and return predictable test data."""

        received_arguments.update(kwargs)
        return expected_data

    monkeypatch.setattr(
        "tailrisk.data.yf.download",
        fake_download,
    )

    config = {
        "universe": {
            "assets": [
                {"ticker": "GOOGL"},
                {"ticker": "AMZN"},
            ]
        },
        "data": {
            "start": "2025-01-02",
            "end": "2025-01-10",
            "interval": "1d",
            "download": {
                "auto_adjust": False,
                "actions": True,
                "progress": False,
            },
        },
    }

    result = download_market_data(config)

    assert received_arguments["tickers"] == ["GOOGL", "AMZN"]
    assert received_arguments["start"] == "2025-01-02"
    assert received_arguments["end"] == "2025-01-10"
    assert received_arguments["interval"] == "1d"
    assert received_arguments["auto_adjust"] is False
    assert received_arguments["actions"] is True
    assert result is expected_data
