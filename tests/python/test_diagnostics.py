"""Tests for descriptive portfolio-return diagnostics."""

from math import sqrt

import pandas as pd
import pytest

from tailrisk.diagnostics import calculate_return_diagnostics


def test_calculate_return_diagnostics_matches_manual_example() -> None:
    """Verify basic return statistics against a hand calculation."""

    portfolio_returns = pd.Series(
        [0.02, -0.01, 0.01, -0.03, 0.01],
        index=pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
                "2025-01-07",
                "2025-01-08",
            ]
        ),
    )

    result = calculate_return_diagnostics(portfolio_returns)

    assert result.observation_count == 5
    assert result.mean_daily_return == pytest.approx(0.0)
    assert result.daily_variance == pytest.approx(0.0004)
    assert result.daily_volatility == pytest.approx(0.02)
    assert result.annualized_arithmetic_mean == pytest.approx(0.0)
    assert result.annualized_volatility == pytest.approx(
        sqrt(252) * 0.02
    )
    assert result.skewness == pytest.approx(-0.9375)
    assert result.excess_kurtosis == pytest.approx(-0.1875)

    assert result.loss_quantile_95 == pytest.approx(0.026)
    assert result.loss_quantile_975 == pytest.approx(0.028)
    assert result.loss_quantile_99 == pytest.approx(0.0292)

    assert result.worst_daily_return == pytest.approx(-0.03)
    assert result.worst_return_date == pd.Timestamp("2025-01-07")
    assert result.best_daily_return == pytest.approx(0.02)
    assert result.best_return_date == pd.Timestamp("2025-01-02")



def test_calculate_return_diagnostics_requires_four_returns() -> None:
    portfolio_returns = pd.Series(
        [0.01, -0.01, 0.02],
        index=pd.date_range(
            "2025-01-02",
            periods=3,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="At least four",
    ):
        calculate_return_diagnostics(portfolio_returns)


def test_calculate_return_diagnostics_requires_datetime_index() -> None:
    portfolio_returns = pd.Series(
        [0.01, -0.01, 0.02, -0.02]
    )

    with pytest.raises(
        ValueError,
        match="DatetimeIndex",
    ):
        calculate_return_diagnostics(portfolio_returns)


def test_calculate_return_diagnostics_rejects_constant_returns() -> None:
    portfolio_returns = pd.Series(
        [0.01, 0.01, 0.01, 0.01],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="positive variability",
    ):
        calculate_return_diagnostics(portfolio_returns)


@pytest.mark.parametrize(
    "nonfinite_return",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_calculate_return_diagnostics_rejects_nonfinite_returns(
    nonfinite_return: float,
) -> None:
    portfolio_returns = pd.Series(
        [0.01, -0.01, 0.02, nonfinite_return],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        calculate_return_diagnostics(portfolio_returns)


@pytest.mark.parametrize(
    "sessions_per_year",
    [0, -252],
)
def test_calculate_return_diagnostics_requires_positive_sessions(
    sessions_per_year: int,
) -> None:
    portfolio_returns = pd.Series(
        [0.01, -0.01, 0.02, -0.02],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Sessions per year must be positive",
    ):
        calculate_return_diagnostics(
            portfolio_returns,
            sessions_per_year=sessions_per_year,
        )