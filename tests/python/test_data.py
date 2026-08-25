"""Tests for market-data processing."""

import pandas as pd
import pytest


from tailrisk.data import (
    download_market_data,
    extract_field,
    validate_dates,
    validate_fields,
    validate_prices,
    validate_tickers,
    count_missing_prices,
    validate_market_data,
)


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


def test_validate_tickers_rejects_missing_ticker() -> None:
    """Verify that validation reports tickers missing from market data."""

    columns = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "GOOGL"),
            ("Close", "GOOGL"),
        ],
        names=["Price", "Ticker"],
    )

    market_data = pd.DataFrame(
        [[100.0, 101.0]],
        columns=columns,
    )

    expected_tickers = ["GOOGL", "AMZN"]

    with pytest.raises(ValueError, match="AMZN"):
        validate_tickers(market_data, expected_tickers)


def test_validate_fields_rejects_missing_fields() -> None:
    """Verify that validation reports required fields missing from market data."""

    columns = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "GOOGL"),
            ("Close", "GOOGL"),
        ],
        names=["Price", "Ticker"],
    )

    market_data = pd.DataFrame(
        [[100.0, 101.0]],
        columns=columns,
    )

    required_fields = [
        "Adj Close",
        "Close",
        "Dividends",
        "Stock Splits",
    ]

    with pytest.raises(ValueError, match="Dividends"):
        validate_fields(market_data, required_fields)



def test_validate_dates_rejects_duplicate_dates() -> None:
    """Verify that validation rejects duplicate market-data dates."""

    market_data = pd.DataFrame(
        {"GOOGL": [100.0, 101.0]},
        index=pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-02",
            ]
        ),
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_dates(market_data)


def test_validate_dates_rejects_unsorted_dates() -> None:
    """Verify that validation rejects dates outside chronological order."""

    market_data = pd.DataFrame(
        {"GOOGL": [101.0, 100.0]},
        index=pd.to_datetime(
            [
                "2025-01-03",
                "2025-01-02",
            ]
        ),
    )

    with pytest.raises(ValueError, match="chronological"):
        validate_dates(market_data)



@pytest.mark.parametrize(
    "invalid_price",
    [
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_prices_rejects_invalid_observations(
    invalid_price: float,
) -> None:
    """Verify that observed prices must be positive and finite."""

    prices = pd.DataFrame(
        {"GOOGL": [100.0, invalid_price]},
        index=pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
            ]
        ),
    )

    with pytest.raises(ValueError, match="positive and finite"):
        validate_prices(prices)


def test_validate_prices_allows_missing_observations() -> None:
    """Verify that missing prices remain available for separate handling."""

    prices = pd.DataFrame(
        {"GOOGL": [100.0, float("nan")]},
        index=pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
            ]
        ),
    )

    validate_prices(prices)


def test_count_missing_prices_returns_count_per_ticker() -> None:
    """Verify that missing price observations are counted for each ticker."""

    prices = pd.DataFrame(
        {
            "GOOGL": [100.0, float("nan"), 102.0],
            "AMZN": [float("nan"), 200.0, float("nan")],
        },
        index=pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
            ]
        ),
    )

    missing_counts = count_missing_prices(prices)

    assert missing_counts.to_dict() == {
        "GOOGL": 1,
        "AMZN": 2,
    }



def test_validate_market_data_accepts_valid_data() -> None:
    """Verify that complete and valid market data passes validation."""

    columns = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "GOOGL"),
            ("Close", "GOOGL"),
            ("Dividends", "GOOGL"),
            ("Stock Splits", "GOOGL"),
        ],
        names=["Price", "Ticker"],
    )

    market_data = pd.DataFrame(
        [[100.0, 101.0, 0.0, 0.0]],
        index=pd.to_datetime(["2025-01-02"]),
        columns=columns,
    )

    validate_market_data(
        market_data,
        expected_tickers=["GOOGL"],
    )