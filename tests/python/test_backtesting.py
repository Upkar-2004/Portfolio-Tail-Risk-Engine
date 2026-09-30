"""Tests for rolling backtests and forecast evaluation."""

import pandas as pd
import pytest

from tailrisk.backtesting import create_forecast_schedule


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