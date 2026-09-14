"""Tests for portfolio construction and accounting."""

import pandas as pd
import pytest

from tailrisk.portfolio import create_equal_weights


def test_create_equal_weights_allocates_investable_value() -> None:
    tickers = ["GOOGL", "AMZN", "WMT"]

    weights = create_equal_weights(
        tickers=tickers,
        cash_weight=0.10,
    )

    expected = pd.Series(
        [0.30, 0.30, 0.30],
        index=tickers,
        name="weight",
    )

    pd.testing.assert_series_equal(
        weights,
        expected,
    )
    assert weights.sum() == pytest.approx(0.90)


def test_create_equal_weights_rejects_empty_universe() -> None:
    with pytest.raises(
        ValueError,
        match="At least one ticker",
    ):
        create_equal_weights(
            tickers=[],
            cash_weight=0.0,
        )


def test_create_equal_weights_rejects_duplicate_tickers() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        create_equal_weights(
            tickers=["GOOGL", "GOOGL"],
            cash_weight=0.0,
        )


@pytest.mark.parametrize(
    "cash_weight",
    [
        -0.10,
        1.00,
        float("nan"),
    ],
)
def test_create_equal_weights_rejects_invalid_cash_weight(
    cash_weight: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Cash weight",
    ):
        create_equal_weights(
            tickers=["GOOGL", "AMZN"],
            cash_weight=cash_weight,
        )