"""Tests for rolling backtests and forecast evaluation."""

import pandas as pd
import pytest
import numpy as np

from tailrisk.backtesting import (
    create_forecast_schedule,
    validate_backtest_forecasts,
)


def test_create_forecast_schedule_uses_only_prior_returns() -> None:
    """Verify that each forecast excludes its realized return."""

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

    result = create_forecast_schedule(
        portfolio_returns,
        estimation_window=3,
    )

    expected = pd.DataFrame(
        {
            "estimation_start_date": dates[:2],
            "estimation_end_date": dates[2:4],
            "realized_return": [-0.04, 0.05],
            "realized_loss": [0.04, -0.05],
        },
        index=pd.DatetimeIndex(
            dates[3:],
            name="forecast_date",
        ),
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )




def _create_valid_backtest_forecasts(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[float],
]:
    forecast_dates = pd.to_datetime(
        [
            "2025-01-07",
            "2025-01-08",
        ]
    )
    confidence_levels = [0.95, 0.99]

    forecast_schedule = pd.DataFrame(
        {
            "estimation_start_date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "estimation_end_date": pd.to_datetime(
                [
                    "2025-01-06",
                    "2025-01-07",
                ]
            ),
            "realized_return": [-0.03, 0.01],
            "realized_loss": [0.03, -0.01],
        },
        index=pd.DatetimeIndex(
            forecast_dates,
            name="forecast_date",
        ),
    )

    forecast_index = pd.MultiIndex.from_product(
        [
            forecast_dates,
            confidence_levels,
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    forecasts = pd.DataFrame(
        {
            "estimation_start_date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                ]
            ),
            "estimation_end_date": pd.to_datetime(
                [
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                ]
            ),
            "realized_return": [
                -0.03,
                -0.03,
                0.01,
                0.01,
            ],
            "realized_loss": [
                0.03,
                0.03,
                -0.01,
                -0.01,
            ],
            "value_at_risk": [
                0.02,
                0.03,
                0.018,
                0.028,
            ],
            "expected_shortfall": [
                0.025,
                0.035,
                0.023,
                0.033,
            ],
            "var_exceedance": [
                True,
                False,
                False,
                False,
            ],
        },
        index=forecast_index,
    )

    return (
        forecasts,
        forecast_schedule,
        confidence_levels,
    )





@pytest.mark.parametrize(
    "estimation_window",
    [
        0,
        -1,
        True,
        2.5,
    ],
)
def test_create_forecast_schedule_rejects_invalid_window(
    estimation_window: object,
) -> None:
    """Verify that estimation windows must be positive integers."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=estimation_window,
        )


def test_create_forecast_schedule_requires_out_of_sample_return() -> None:
    """Verify that the schedule requires a return after its first window."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="more observations",
    ):
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=4,
        )


@pytest.mark.parametrize(
    ("dates", "error_message"),
    [
        (
            [
                "2025-01-02",
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
            ],
            "unique",
        ),
        (
            [
                "2025-01-03",
                "2025-01-02",
                "2025-01-06",
                "2025-01-07",
            ],
            "increasing date",
        ),
        (
            [
                "2025-01-02",
                None,
                "2025-01-06",
                "2025-01-07",
            ],
            "must not be missing",
        ),
    ],
)
def test_create_forecast_schedule_rejects_invalid_dates(
    dates: list[str | None],
    error_message: str,
) -> None:
    """Verify that forecast inputs require valid ordered unique dates."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=pd.to_datetime(dates),
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=2,
        )


@pytest.mark.parametrize(
    "nonfinite_return",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_create_forecast_schedule_rejects_nonfinite_returns(
    nonfinite_return: float,
) -> None:
    """Verify that forecast scheduling rejects nonfinite returns."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, nonfinite_return],
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
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=2,
        )



def test_validate_backtest_forecasts_accepts_valid_results(
) -> None:
    """Verify that a complete and consistent forecast table is accepted."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    validate_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
    )


def test_validate_backtest_forecasts_requires_columns(
) -> None:
    """Verify that every shared backtest result column is required."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts = forecasts.drop(
        columns="expected_shortfall"
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        validate_backtest_forecasts(
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            confidence_levels=confidence_levels,
        )


def test_validate_backtest_forecasts_requires_complete_grid(
) -> None:
    """Verify that every forecast date has every confidence level."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts = forecasts.drop(
        index=(pd.Timestamp("2025-01-08"), 0.99)
    )

    with pytest.raises(
        ValueError,
        match="complete ordered grid",
    ):
        validate_backtest_forecasts(
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            confidence_levels=confidence_levels,
        )



def test_validate_backtest_forecasts_rejects_nonfinite_values(
) -> None:
    """Verify that numerical forecast results must be finite."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.95),
        "value_at_risk",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="finite numbers",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_requires_schedule_alignment(
) -> None:
    """Verify that realized outcomes agree with the shared schedule."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.99),
        "realized_loss",
    ] = 0.031

    with pytest.raises(
        ValueError,
        match="must match the forecast schedule",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_invalid_loss_sign(
) -> None:
    """Verify that realized loss is the negative of realized return."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecast_date = pd.Timestamp("2025-01-07")

    forecast_schedule.loc[
        forecast_date,
        "realized_loss",
    ] = 0.02
    forecasts.loc[
        (forecast_date, slice(None)),
        "realized_loss",
    ] = 0.02

    with pytest.raises(
        ValueError,
        match="negative of its realized return",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_es_below_var(
) -> None:
    """Verify that Expected Shortfall cannot be smaller than VaR."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.95),
        "expected_shortfall",
    ] = 0.019

    with pytest.raises(
        ValueError,
        match="Expected Shortfall",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_wrong_exceedance(
) -> None:
    """Verify that exceedance flags follow realized loss versus VaR."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.95),
        "var_exceedance",
    ] = False

    with pytest.raises(
        ValueError,
        match="exceedance indicators",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_lookahead(
) -> None:
    """Verify that each estimation window ends before its forecast date."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecast_date = pd.Timestamp("2025-01-07")

    forecast_schedule.loc[
        forecast_date,
        "estimation_end_date",
    ] = forecast_date
    forecasts.loc[
        (forecast_date, slice(None)),
        "estimation_end_date",
    ] = forecast_date

    with pytest.raises(
        ValueError,
        match="must end before",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )
