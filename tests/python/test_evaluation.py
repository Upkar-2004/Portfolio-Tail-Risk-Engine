"""Tests for forecast comparison diagnostics."""

import pandas as pd
import pytest

from tailrisk.evaluation import (
    calculate_kupiec_unconditional_coverage,
    evaluate_var_unconditional_coverage,
    summarize_gaussian_monte_carlo_errors,
    summarize_var_exceedances,
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


def _create_var_exceedance_forecasts() -> pd.DataFrame:
    """Create controlled VaR forecasts with known exceedances."""

    dates = pd.date_range(
        "2025-01-02",
        periods=5,
        freq="B",
    )
    confidence_levels = [0.95, 0.99]
    index = pd.MultiIndex.from_product(
        [
            dates,
            confidence_levels,
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    realized_losses = [
        0.010,
        0.010,
        0.020,
        0.020,
        0.021,
        0.021,
        0.030,
        0.030,
        -0.010,
        -0.010,
    ]
    value_at_risk = [
        0.020,
        0.030,
        0.020,
        0.030,
        0.020,
        0.030,
        0.020,
        0.030,
        0.020,
        0.030,
    ]

    return pd.DataFrame(
        {
            "realized_loss": realized_losses,
            "value_at_risk": value_at_risk,
            "var_exceedance": [
                loss > var
                for loss, var in zip(
                    realized_losses,
                    value_at_risk,
                    strict=True,
                )
            ],
        },
        index=index,
    )


def test_var_exceedance_summary_matches_manual_example() -> None:
    """Exceedance summaries match manually calculated results."""

    forecasts = _create_var_exceedance_forecasts()

    result = summarize_var_exceedances(
        forecasts=forecasts,
    )

    summary_95 = result.loc[0.95]

    assert summary_95["observation_count"] == 5
    assert summary_95["exceedance_count"] == 2
    assert summary_95[
        "expected_exceedance_rate"
    ] == pytest.approx(0.05)
    assert summary_95[
        "observed_exceedance_rate"
    ] == pytest.approx(0.40)
    assert summary_95[
        "expected_exceedance_count"
    ] == pytest.approx(0.25)
    assert summary_95[
        "exceedance_count_difference"
    ] == pytest.approx(1.75)

    summary_99 = result.loc[0.99]

    assert summary_99["observation_count"] == 5
    assert summary_99["exceedance_count"] == 0
    assert summary_99[
        "expected_exceedance_rate"
    ] == pytest.approx(0.01)
    assert summary_99[
        "observed_exceedance_rate"
    ] == pytest.approx(0.0)
    assert summary_99[
        "expected_exceedance_count"
    ] == pytest.approx(0.05)
    assert summary_99[
        "exceedance_count_difference"
    ] == pytest.approx(-0.05)


def test_var_exceedance_requires_strict_loss_threshold() -> None:
    """A loss equal to VaR is not recorded as an exceedance."""

    forecasts = _create_var_exceedance_forecasts()

    equality_row = forecasts.loc[
        (
            pd.Timestamp("2025-01-03"),
            0.95,
        )
    ]

    assert equality_row["realized_loss"] == pytest.approx(
        equality_row["value_at_risk"]
    )
    assert not equality_row["var_exceedance"]


def test_var_exceedance_summary_rejects_incorrect_indicator() -> None:
    """Stored exceedances must agree with realized loss and VaR."""

    forecasts = _create_var_exceedance_forecasts()
    forecasts.loc[
        (
            pd.Timestamp("2025-01-02"),
            0.95,
        ),
        "var_exceedance",
    ] = True

    with pytest.raises(
        ValueError,
        match="must agree",
    ):
        summarize_var_exceedances(
            forecasts=forecasts,
        )


def test_kupiec_test_matches_manual_rejection_example() -> None:
    """Kupiec statistic matches a controlled excessive-breach case."""

    result = calculate_kupiec_unconditional_coverage(
        exceedance_count=10,
        observation_count=100,
        confidence_level=0.95,
        significance_level=0.05,
    )

    assert result.expected_exceedance_rate == pytest.approx(
        0.05
    )
    assert result.observed_exceedance_rate == pytest.approx(
        0.10
    )
    assert result.likelihood_ratio == pytest.approx(
        4.130843782549277
    )
    assert result.p_value == pytest.approx(
        0.042108350096184695
    )
    assert result.reject_null


def test_kupiec_test_does_not_reject_exact_expected_rate() -> None:
    """Exact nominal coverage produces zero LR and p-value one."""

    result = calculate_kupiec_unconditional_coverage(
        exceedance_count=5,
        observation_count=100,
        confidence_level=0.95,
    )

    assert result.likelihood_ratio == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.reject_null


@pytest.mark.parametrize(
    "exceedance_count",
    [
        0,
        10,
    ],
)
def test_kupiec_test_handles_boundary_exceedance_counts(
    exceedance_count: int,
) -> None:
    """Zero and all-exceedance samples have finite test results."""

    result = calculate_kupiec_unconditional_coverage(
        exceedance_count=exceedance_count,
        observation_count=10,
        confidence_level=0.99,
    )

    assert result.likelihood_ratio >= 0.0
    assert 0.0 <= result.p_value <= 1.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"observation_count": 0},
            "Observation count",
        ),
        (
            {"exceedance_count": 101},
            "Exceedance count",
        ),
        (
            {"confidence_level": 1.0},
            "Confidence level",
        ),
        (
            {"significance_level": True},
            "Significance level",
        ),
    ],
)
def test_kupiec_test_rejects_invalid_inputs(
    arguments: dict[str, float | int | bool],
    message: str,
) -> None:
    """Kupiec inputs must satisfy their statistical domains."""

    valid_arguments: dict[str, float | int | bool] = {
        "exceedance_count": 5,
        "observation_count": 100,
        "confidence_level": 0.95,
        "significance_level": 0.05,
    }
    valid_arguments.update(arguments)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        calculate_kupiec_unconditional_coverage(
            **valid_arguments,
        )


def test_var_coverage_evaluation_runs_kupiec_by_level() -> None:
    """Integrated coverage results use each level's breach count."""

    forecasts = _create_var_exceedance_forecasts()

    result = evaluate_var_unconditional_coverage(
        forecasts=forecasts,
        significance_level=0.05,
    )

    expected_95 = calculate_kupiec_unconditional_coverage(
        exceedance_count=2,
        observation_count=5,
        confidence_level=0.95,
    )
    expected_99 = calculate_kupiec_unconditional_coverage(
        exceedance_count=0,
        observation_count=5,
        confidence_level=0.99,
    )

    assert result.loc[
        0.95,
        "kupiec_likelihood_ratio",
    ] == pytest.approx(expected_95.likelihood_ratio)
    assert result.loc[
        0.95,
        "kupiec_p_value",
    ] == pytest.approx(expected_95.p_value)
    assert bool(
        result.loc[0.95, "kupiec_reject_null"]
    ) is expected_95.reject_null

    assert result.loc[
        0.99,
        "kupiec_likelihood_ratio",
    ] == pytest.approx(expected_99.likelihood_ratio)
    assert result.loc[
        0.99,
        "kupiec_p_value",
    ] == pytest.approx(expected_99.p_value)
    assert bool(
        result.loc[0.99, "kupiec_reject_null"]
    ) is expected_99.reject_null


@pytest.mark.parametrize(
    "missing_column",
    [
        "realized_loss",
        "value_at_risk",
        "var_exceedance",
    ],
)
def test_var_exceedance_summary_rejects_missing_columns(
    missing_column: str,
) -> None:
    """All inputs required for an exceedance must be present."""

    forecasts = (
        _create_var_exceedance_forecasts()
        .drop(columns=missing_column)
    )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        summarize_var_exceedances(
            forecasts=forecasts,
        )
