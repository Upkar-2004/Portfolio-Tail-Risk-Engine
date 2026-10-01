"""Tests for Gaussian portfolio-risk forecasts."""

import pytest

import pandas as pd

from tailrisk.backtesting import create_forecast_schedule
from tailrisk.models import (
    calculate_gaussian_var_es,
    calculate_rolling_gaussian_forecasts,
    calculate_rolling_multivariate_gaussian_forecasts,
)


def test_calculate_gaussian_var_es_matches_manual_example() -> None:
    """Verify Gaussian VaR and ES against a hand calculation."""

    result = calculate_gaussian_var_es(
        mean_return=0.001,
        volatility=0.02,
        confidence_level=0.95,
    )

    assert result.confidence_level == pytest.approx(0.95)
    assert result.value_at_risk == pytest.approx(
        0.03189707253902943
    )
    assert result.expected_shortfall == pytest.approx(
        0.040254256150148576
    )
    assert result.expected_shortfall > result.value_at_risk



@pytest.mark.parametrize(
    "mean_return",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_calculate_gaussian_var_es_rejects_invalid_mean(
    mean_return: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Mean return",
    ):
        calculate_gaussian_var_es(
            mean_return=mean_return,
            volatility=0.02,
            confidence_level=0.95,
        )


@pytest.mark.parametrize(
    "volatility",
    [
        0.0,
        -0.01,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_calculate_gaussian_var_es_rejects_invalid_volatility(
    volatility: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Volatility",
    ):
        calculate_gaussian_var_es(
            mean_return=0.001,
            volatility=volatility,
            confidence_level=0.95,
        )


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
        float("nan"),
        True,
    ],
)
def test_calculate_gaussian_var_es_rejects_invalid_confidence(
    confidence_level: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Confidence level",
    ):
        calculate_gaussian_var_es(
            mean_return=0.001,
            volatility=0.02,
            confidence_level=confidence_level,
        )


def test_rolling_gaussian_forecasts_use_prior_windows() -> None:
    """Verify rolling estimates and one-day forecast alignment."""

    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )
    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04, 0.05],
        index=dates,
        name="portfolio_return",
    )
    schedule = create_forecast_schedule(
        portfolio_returns,
        estimation_window=3,
    )

    result = calculate_rolling_gaussian_forecasts(
        portfolio_returns=portfolio_returns,
        forecast_schedule=schedule,
        confidence_levels=[
            0.95,
            0.99,
        ],
    )

    expected_index = pd.MultiIndex.from_tuples(
        [
            (dates[3], 0.95),
            (dates[3], 0.99),
            (dates[4], 0.95),
            (dates[4], 0.99),
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )
    pd.testing.assert_index_equal(
        result.index,
        expected_index,
    )

    first_forecast = result.loc[
        (dates[3], 0.95)
    ]
    assert first_forecast["mean_return"] == pytest.approx(
        0.006666666666666666
    )
    assert first_forecast["volatility"] == pytest.approx(
        0.025166114784235832
    )
    assert first_forecast["value_at_risk"] == pytest.approx(
        0.03472790851246069
    )
    assert first_forecast["expected_shortfall"] == pytest.approx(
        0.04524380061397863
    )

    second_forecast = result.loc[
        (dates[4], 0.95)
    ]
    assert second_forecast["mean_return"] == pytest.approx(
        -0.01
    )
    assert second_forecast["volatility"] == pytest.approx(
        0.03605551275463989
    )
    assert second_forecast["value_at_risk"] == pytest.approx(
        0.06930604092606446
    )
    assert second_forecast["expected_shortfall"] == pytest.approx(
        0.08437216794024315
    )

    assert (
        result.loc[
            (dates[3], 0.99),
            "value_at_risk",
        ]
        > result.loc[
            (dates[3], 0.95),
            "value_at_risk",
        ]
    )
    assert (
        result.loc[
            (dates[3], 0.99),
            "expected_shortfall",
        ]
        > result.loc[
            (dates[3], 0.95),
            "expected_shortfall",
        ]
    )


def test_rolling_gaussian_forecasts_requires_confidence_levels() -> None:
    dates = pd.date_range(
        "2025-01-02",
        periods=4,
        freq="B",
    )
    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=dates,
    )
    schedule = create_forecast_schedule(
        portfolio_returns,
        estimation_window=3,
    )

    with pytest.raises(
        ValueError,
        match="At least one confidence level is required",
    ):
        calculate_rolling_gaussian_forecasts(
            portfolio_returns=portfolio_returns,
            forecast_schedule=schedule,
            confidence_levels=[],
        )


def test_rolling_gaussian_forecasts_rejects_lookahead() -> None:
    dates = pd.date_range(
        "2025-01-02",
        periods=4,
        freq="B",
    )
    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=dates,
    )
    schedule = create_forecast_schedule(
        portfolio_returns,
        estimation_window=3,
    )

    forecast_date = dates[3]

    # Deliberately contaminate the estimation window with R_t.
    schedule.loc[
        forecast_date,
        "estimation_end_date",
    ] = forecast_date

    with pytest.raises(
        ValueError,
        match="before its forecast date",
    ):
        calculate_rolling_gaussian_forecasts(
            portfolio_returns=portfolio_returns,
            forecast_schedule=schedule,
            confidence_levels=[0.95],
        )


def test_rolling_multivariate_gaussian_forecasts_use_prior_asset_returns(
) -> None:
    """Verify rolling covariance estimates and forecast-date weights."""

    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )

    asset_returns = pd.DataFrame(
        {
            "A": [0.02, 0.00, -0.01, 0.03, 0.01],
            "B": [0.01, -0.01, 0.00, 0.02, -0.02],
        },
        index=dates,
    )

    beginning_weights = pd.DataFrame(
        {
            "A": [0.50, 0.50, 0.50, 0.60, 0.25],
            "B": [0.50, 0.50, 0.50, 0.40, 0.75],
        },
        index=dates,
    )

    portfolio_returns = (
        asset_returns * beginning_weights
    ).sum(axis=1)

    schedule = create_forecast_schedule(
        portfolio_returns,
        estimation_window=3,
    )

    result = calculate_rolling_multivariate_gaussian_forecasts(
        asset_returns=asset_returns,
        beginning_weights=beginning_weights,
        forecast_schedule=schedule,
        confidence_levels=[0.95, 0.99],
    )

    expected_index = pd.MultiIndex.from_tuples(
        [
            (dates[3], 0.95),
            (dates[3], 0.99),
            (dates[4], 0.95),
            (dates[4], 0.99),
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    pd.testing.assert_index_equal(
        result.index,
        expected_index,
    )

    first_forecast = result.loc[
        (dates[3], 0.95)
    ]

    assert first_forecast["mean_return"] == pytest.approx(
        0.002
    )
    assert first_forecast["volatility"] == pytest.approx(
        0.01216552506059644
    )
    assert first_forecast["value_at_risk"] == pytest.approx(
        0.018010508019691077
    )
    assert first_forecast["expected_shortfall"] == pytest.approx(
        0.023093984352544866
    )

    second_forecast = result.loc[
        (dates[4], 0.95)
    ]

    assert second_forecast["mean_return"] == pytest.approx(
        0.004166666666666667
    )
    assert second_forecast["volatility"] == pytest.approx(
        0.01607275126832159
    )
    assert second_forecast["value_at_risk"] == pytest.approx(
        0.022270656552120967
    )
    assert second_forecast["expected_shortfall"] == pytest.approx(
        0.028986803226381544
    )

    assert (
        result.loc[
            (dates[3], 0.99),
            "value_at_risk",
        ]
        > first_forecast["value_at_risk"]
    )
    assert (
        result.loc[
            (dates[3], 0.99),
            "expected_shortfall",
        ]
        > first_forecast["expected_shortfall"]
    )
