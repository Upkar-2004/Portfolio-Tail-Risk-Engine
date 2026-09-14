"""Tests for portfolio construction and accounting."""

import pandas as pd
import pytest

import tailrisk.portfolio as portfolio
from tailrisk.portfolio import (
    calculate_portfolio_step,
    create_equal_weights,
)


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


def test_calculate_portfolio_step_matches_manual_example() -> None:
    beginning_weights = pd.Series(
        [0.50, 0.30, 0.20],
        index=["A", "B", "C"],
        name="weight",
    )
    asset_returns = pd.Series(
        [-0.04, 0.02, -0.01],
        index=["A", "B", "C"],
    )

    result = calculate_portfolio_step(
        beginning_value=1_000_000.0,
        beginning_weights=beginning_weights,
        asset_returns=asset_returns,
    )

    assert result.portfolio_return == pytest.approx(-0.016)
    assert result.pnl == pytest.approx(-16_000.0)
    assert result.loss == pytest.approx(16_000.0)
    assert result.ending_value == pytest.approx(984_000.0)

    expected_weights = pd.Series(
        [
            480_000.0 / 984_000.0,
            306_000.0 / 984_000.0,
            198_000.0 / 984_000.0,
        ],
        index=["A", "B", "C"],
        name="weight",
    )
    pd.testing.assert_series_equal(
        result.ending_weights,
        expected_weights,
    )
    assert result.ending_cash_weight == pytest.approx(0.0)


def test_calculate_portfolio_step_keeps_uninvested_weight_as_cash() -> None:
    beginning_weights = pd.Series(
        [0.45, 0.45],
        index=["A", "B"],
        name="weight",
    )
    asset_returns = pd.Series(
        [0.10, 0.00],
        index=["A", "B"],
    )

    result = calculate_portfolio_step(
        beginning_value=1_000.0,
        beginning_weights=beginning_weights,
        asset_returns=asset_returns,
    )

    assert result.ending_value == pytest.approx(1_045.0)
    assert result.portfolio_return == pytest.approx(0.045)
    assert result.ending_cash_weight == pytest.approx(100.0 / 1_045.0)
    assert (
        result.ending_weights.sum()
        + result.ending_cash_weight
    ) == pytest.approx(1.0)


def test_calculate_portfolio_step_aligns_assets_by_ticker() -> None:
    beginning_weights = pd.Series(
        [0.60, 0.40],
        index=["GOOGL", "AMZN"],
        name="weight",
    )
    asset_returns = pd.Series(
        [0.10, -0.10],
        index=["AMZN", "GOOGL"],
    )

    result = calculate_portfolio_step(
        beginning_value=1_000.0,
        beginning_weights=beginning_weights,
        asset_returns=asset_returns,
    )

    assert result.portfolio_return == pytest.approx(-0.02)


def test_run_portfolio_path_rebalances_after_month_end() -> None:
    """Drift during January, then restore targets before February."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.10, 0.00, 0.00],
            "B": [0.00, 0.00, 0.10],
        },
        index=pd.to_datetime(
            [
                "2025-01-30",
                "2025-01-31",
                "2025-02-03",
            ]
        ),
    )
    target_weights = pd.Series(
        [0.50, 0.50],
        index=["A", "B"],
        name="weight",
    )

    result = portfolio.run_portfolio_path(
        asset_returns=asset_returns,
        initial_value=1_000.0,
        target_weights=target_weights,
        rebalancing_frequency="monthly",
    )

    january_30 = pd.Timestamp("2025-01-30")
    january_31 = pd.Timestamp("2025-01-31")
    february_3 = pd.Timestamp("2025-02-03")

    # A's January gain makes it overweight before the rebalance.
    assert result.pre_rebalance_weights.loc[january_31, "A"] == pytest.approx(
        550.0 / 1_050.0
    )
    assert result.daily.loc[january_30, "portfolio_return"] == pytest.approx(
        0.05
    )
    assert not bool(
        result.daily.loc[january_30, "rebalanced_after_close"]
    )

    # The final January session is rebalanced after its return is applied.
    assert bool(
        result.daily.loc[january_31, "rebalanced_after_close"]
    )
    assert result.daily.loc[january_31, "turnover"] == pytest.approx(
        1.0 / 42.0
    )

    # February therefore starts at 50/50 and earns a 5% portfolio return.
    assert result.beginning_weights.loc[february_3, "A"] == pytest.approx(0.50)
    assert result.beginning_weights.loc[february_3, "B"] == pytest.approx(0.50)
    assert result.daily.loc[february_3, "beginning_value"] == pytest.approx(
        1_050.0
    )
    assert result.daily.loc[february_3, "portfolio_return"] == pytest.approx(
        0.05
    )
    assert result.daily.loc[february_3, "ending_value"] == pytest.approx(
        1_102.50
    )
