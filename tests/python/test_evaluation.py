"""Tests for forecast comparison diagnostics."""

import pandas as pd
import pytest

from tailrisk.evaluation import (
    summarize_gaussian_monte_carlo_errors,
)


def _create_comparison_forecasts(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create aligned analytic and Monte Carlo forecasts."""

    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
        ]
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [0.95, 0.99],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )
    analytic = pd.DataFrame(
        {
            "value_at_risk": [
                0.0200,
                0.0300,
                0.0210,
                0.0310,
            ],
            "expected_shortfall": [
                0.0250,
                0.0350,
                0.0260,
                0.0360,
            ],
        },
        index=index,
    )
    monte_carlo = analytic.copy()
    monte_carlo["value_at_risk"] += [
        0.0001,
        0.0004,
        -0.0002,
        -0.0002,
    ]
    monte_carlo["expected_shortfall"] += [
        0.0003,
        -0.0003,
        -0.0001,
        0.0001,
    ]

    return analytic, monte_carlo


def test_gaussian_error_summary_matches_manual_example() -> None:
    """Error summaries match controlled basis-point differences."""

    analytic, monte_carlo = _create_comparison_forecasts()

    result = summarize_gaussian_monte_carlo_errors(
        analytic_forecasts=analytic,
        monte_carlo_forecasts=monte_carlo,
    )

    var_95 = result.loc[
        (0.95, "value_at_risk")
    ]
    assert var_95["forecast_count"] == 2
    assert var_95["mean_error_bps"] == pytest.approx(
        -0.5
    )
    assert var_95[
        "mean_absolute_error_bps"
    ] == pytest.approx(1.5)
    assert var_95[
        "median_absolute_error_bps"
    ] == pytest.approx(1.5)
    assert var_95[
        "p95_absolute_error_bps"
    ] == pytest.approx(1.95)
    assert var_95[
        "maximum_absolute_error_bps"
    ] == pytest.approx(2.0)

    es_99 = result.loc[
        (0.99, "expected_shortfall")
    ]
    assert es_99["mean_error_bps"] == pytest.approx(
        -1.0
    )
    assert es_99[
        "mean_absolute_error_bps"
    ] == pytest.approx(2.0)
    assert es_99[
        "p95_absolute_error_bps"
    ] == pytest.approx(2.9)
    assert es_99[
        "maximum_absolute_error_bps"
    ] == pytest.approx(3.0)


def test_gaussian_error_summary_rejects_misaligned_index() -> None:
    """Analytic and Monte Carlo forecasts must align exactly."""

    analytic, monte_carlo = _create_comparison_forecasts()
    monte_carlo = monte_carlo.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="same index",
    ):
        summarize_gaussian_monte_carlo_errors(
            analytic_forecasts=analytic,
            monte_carlo_forecasts=monte_carlo,
        )
