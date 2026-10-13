"""Tests for forecast comparison diagnostics."""

import pandas as pd
import pytest

from tailrisk.evaluation import (
    adjust_p_values_holm,
    calculate_christoffersen_conditional_coverage,
    calculate_christoffersen_independence,
    calculate_kupiec_unconditional_coverage,
    calculate_exact_binomial_interval,
    count_var_exceedance_transitions,
    evaluate_models_var_coverage,
    evaluate_var_coverage,
    evaluate_models_es_tail_severity,
    evaluate_var_unconditional_coverage,
    summarize_es_tail_severity,
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





def _create_es_tail_severity_forecasts() -> pd.DataFrame:
    """Create controlled ES forecasts with known tail severity."""

    dates = pd.date_range(
        "2025-01-02",
        periods=5,
        freq="B",
    )
    confidence_levels = [
        0.95,
        0.99,
    ]
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
        3.0,
        3.0,
        7.0,
        7.0,
        9.0,
        9.0,
        2.0,
        2.0,
        12.0,
        12.0,
    ]
    value_at_risk = [
        8.0,
        14.0,
        8.0,
        14.0,
        8.0,
        14.0,
        8.0,
        14.0,
        10.0,
        14.0,
    ]
    expected_shortfall = [
        10.0,
        17.0,
        10.0,
        17.0,
        10.0,
        17.0,
        10.0,
        17.0,
        13.0,
        17.0,
    ]

    return pd.DataFrame(
        {
            "realized_loss": realized_losses,
            "value_at_risk": value_at_risk,
            "expected_shortfall": (
                expected_shortfall
            ),
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




def test_es_tail_severity_matches_manual_example() -> None:
    """ES tail severity matches manually calculated values."""

    forecasts = _create_es_tail_severity_forecasts()

    result = summarize_es_tail_severity(
        forecasts=forecasts,
    )

    summary_95 = result.loc[0.95]

    assert summary_95["observation_count"] == 5
    assert summary_95["var_exceedance_count"] == 2

    assert summary_95[
        "mean_exceedance_loss"
    ] == pytest.approx(10.5)

    assert summary_95[
        "mean_var_on_exceedance_days"
    ] == pytest.approx(9.0)

    assert summary_95[
        "mean_es_on_exceedance_days"
    ] == pytest.approx(11.5)

    assert summary_95[
        "mean_var_excess"
    ] == pytest.approx(1.5)

    assert summary_95[
        "median_var_excess"
    ] == pytest.approx(1.5)

    assert summary_95[
        "maximum_var_excess"
    ] == pytest.approx(2.0)

    assert summary_95[
        "mean_es_gap"
    ] == pytest.approx(-1.0)


def test_es_tail_severity_uses_missing_values_without_breaches(
) -> None:
    """Tail statistics are undefined when no VaR breach occurs."""

    forecasts = _create_es_tail_severity_forecasts()

    result = summarize_es_tail_severity(
        forecasts=forecasts,
    )

    summary_99 = result.loc[0.99]

    assert summary_99["observation_count"] == 5
    assert summary_99["var_exceedance_count"] == 0

    undefined_columns = [
        "mean_exceedance_loss",
        "mean_var_on_exceedance_days",
        "mean_es_on_exceedance_days",
        "mean_var_excess",
        "median_var_excess",
        "maximum_var_excess",
        "mean_es_gap",
    ]

    assert summary_99[
        undefined_columns
    ].isna().all()


def test_es_tail_severity_rejects_missing_es_column(
) -> None:
    """Expected Shortfall forecasts are required."""

    forecasts = (
        _create_es_tail_severity_forecasts()
        .drop(columns="expected_shortfall")
    )

    with pytest.raises(
        ValueError,
        match="expected_shortfall",
    ):
        summarize_es_tail_severity(
            forecasts=forecasts,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        float("nan"),
        float("inf"),
        "not-a-number",
    ],
)
def test_es_tail_severity_rejects_invalid_es_values(
    bad_value: object,
) -> None:
    """Expected Shortfall forecasts must be finite numbers."""

    forecasts = _create_es_tail_severity_forecasts()
    forecasts["expected_shortfall"] = forecasts[
        "expected_shortfall"
    ].astype(object)
    forecasts.loc[
        (
            pd.Timestamp("2025-01-02"),
            0.95,
        ),
        "expected_shortfall",
    ] = bad_value

    with pytest.raises(
        ValueError,
        match="finite numbers",
    ):
        summarize_es_tail_severity(
            forecasts=forecasts,
        )


def test_es_tail_severity_rejects_es_below_var(
) -> None:
    """Expected Shortfall cannot be materially smaller than VaR."""

    forecasts = _create_es_tail_severity_forecasts()
    forecasts.loc[
        (
            pd.Timestamp("2025-01-02"),
            0.95,
        ),
        "expected_shortfall",
    ] = 7.99

    with pytest.raises(
        ValueError,
        match="at least as large",
    ):
        summarize_es_tail_severity(
            forecasts=forecasts,
        )


def test_es_tail_severity_allows_numerical_order_tolerance(
) -> None:
    """Tiny floating-point differences do not violate ES ordering."""

    forecasts = _create_es_tail_severity_forecasts()
    forecasts.loc[
        (
            pd.Timestamp("2025-01-02"),
            0.95,
        ),
        "expected_shortfall",
    ] = 8.0 - 0.5e-12

    result = summarize_es_tail_severity(
        forecasts=forecasts,
    )

    assert result.loc[
        0.95,
        "var_exceedance_count",
    ] == 2





def test_model_es_tail_severity_combines_aligned_models(
) -> None:
    """Aligned model summaries form one multi-indexed table."""

    rolling_forecasts = (
        _create_es_tail_severity_forecasts()
    )
    ewma_forecasts = rolling_forecasts.copy()

    ewma_forecasts["value_at_risk"] += 1.0
    ewma_forecasts["expected_shortfall"] += 1.0
    ewma_forecasts["var_exceedance"] = (
        ewma_forecasts["realized_loss"]
        > ewma_forecasts["value_at_risk"]
    )

    result = evaluate_models_es_tail_severity(
        forecasts_by_model={
            "rolling_gaussian": rolling_forecasts,
            "ewma_gaussian": ewma_forecasts,
        }
    )

    assert result.index.names == [
        "model",
        "confidence_level",
    ]
    assert list(
        result.index.get_level_values(
            "model"
        ).unique()
    ) == [
        "rolling_gaussian",
        "ewma_gaussian",
    ]
    assert len(result) == 4

    rolling_95 = result.loc[
        (
            "rolling_gaussian",
            0.95,
        )
    ]
    ewma_95 = result.loc[
        (
            "ewma_gaussian",
            0.95,
        )
    ]

    assert rolling_95[
        "var_exceedance_count"
    ] == 2
    assert rolling_95[
        "mean_var_excess"
    ] == pytest.approx(1.5)

    assert ewma_95[
        "var_exceedance_count"
    ] == 1
    assert ewma_95[
        "mean_exceedance_loss"
    ] == pytest.approx(12.0)
    assert ewma_95[
        "mean_var_excess"
    ] == pytest.approx(1.0)
    assert ewma_95[
        "mean_es_gap"
    ] == pytest.approx(-2.0)


def test_model_es_tail_severity_rejects_misaligned_index(
) -> None:
    """Compared models must use identical ordered forecast rows."""

    rolling_forecasts = (
        _create_es_tail_severity_forecasts()
    )
    ewma_forecasts = rolling_forecasts.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        evaluate_models_es_tail_severity(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_es_tail_severity_rejects_different_losses(
) -> None:
    """Compared models must use identical realized losses."""

    rolling_forecasts = (
        _create_es_tail_severity_forecasts()
    )
    ewma_forecasts = rolling_forecasts.copy()
    ewma_forecasts.loc[
        (
            pd.Timestamp("2025-01-02"),
            0.95,
        ),
        "realized_loss",
    ] = 3.1

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        evaluate_models_es_tail_severity(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_es_tail_severity_rejects_empty_mapping(
) -> None:
    """At least one model forecast table is required."""

    with pytest.raises(
        ValueError,
        match="At least one model",
    ):
        evaluate_models_es_tail_severity(
            forecasts_by_model={}
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
    assert summary_95[
        "interval_confidence_level"
    ] == pytest.approx(0.95)
    assert summary_95[
        "exceedance_rate_ci_lower"
    ] == pytest.approx(
        0.052744950526316926
    )
    assert summary_95[
        "exceedance_rate_ci_upper"
    ] == pytest.approx(
        0.853367200365327
    )

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
    assert summary_99[
        "interval_confidence_level"
    ] == pytest.approx(0.95)
    assert summary_99[
        "exceedance_rate_ci_lower"
    ] == pytest.approx(0.0)
    assert summary_99[
        "exceedance_rate_ci_upper"
    ] == pytest.approx(
        0.5218237501049814
    )


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



def test_exact_binomial_interval_matches_known_example(
) -> None:
    """Exact bounds match a published Clopper-Pearson example."""

    result = calculate_exact_binomial_interval(
        exceedance_count=7,
        observation_count=50,
        interval_confidence_level=0.95,
    )

    assert result.exceedance_count == 7
    assert result.observation_count == 50
    assert result.observed_exceedance_rate == pytest.approx(
        0.14
    )
    assert result.interval_confidence_level == pytest.approx(
        0.95
    )
    assert result.lower_bound == pytest.approx(
        0.05819170033997342
    )
    assert result.upper_bound == pytest.approx(
        0.26739600249700846
    )


@pytest.mark.parametrize(
    (
        "exceedance_count",
        "expected_lower",
        "expected_upper",
    ),
    [
        (
            0,
            0.0,
            0.30849710781876083,
        ),
        (
            10,
            0.6915028921812392,
            1.0,
        ),
    ],
)
def test_exact_binomial_interval_handles_boundary_counts(
    exceedance_count: int,
    expected_lower: float,
    expected_upper: float,
) -> None:
    """Zero and all-exceedance samples have valid exact bounds."""

    result = calculate_exact_binomial_interval(
        exceedance_count=exceedance_count,
        observation_count=10,
    )

    assert result.lower_bound == pytest.approx(
        expected_lower
    )
    assert result.upper_bound == pytest.approx(
        expected_upper
    )
    assert (
        result.lower_bound
        <= result.observed_exceedance_rate
        <= result.upper_bound
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"observation_count": 0},
            "Observation count",
        ),
        (
            {"observation_count": True},
            "Observation count",
        ),
        (
            {"exceedance_count": -1},
            "Exceedance count",
        ),
        (
            {"exceedance_count": 101},
            "Exceedance count",
        ),
        (
            {"interval_confidence_level": 1.0},
            "Interval confidence level",
        ),
        (
            {"interval_confidence_level": True},
            "Interval confidence level",
        ),
        (
            {
                "interval_confidence_level": (
                    float("nan")
                )
            },
            "Interval confidence level",
        ),
    ],
)



def test_exact_binomial_interval_rejects_invalid_inputs(
    arguments: dict[str, float | int | bool],
    message: str,
) -> None:
    """Exact interval inputs must satisfy their valid domains."""

    valid_arguments: dict[str, float | int | bool] = {
        "exceedance_count": 5,
        "observation_count": 100,
        "interval_confidence_level": 0.95,
    }
    valid_arguments.update(arguments)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        calculate_exact_binomial_interval(
            **valid_arguments,
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



def test_exceedance_transition_counts_match_manual_example() -> None:
    """Transition counts match a manually classified sequence."""

    exceedances = pd.Series(
        [
            False,
            False,
            True,
            True,
            False,
            True,
            False,
            False,
        ],
        index=pd.date_range(
            "2025-01-02",
            periods=8,
            freq="B",
        ),
        name="var_exceedance",
    )

    result = count_var_exceedance_transitions(
        exceedances=exceedances,
    )

    assert result.n00 == 2
    assert result.n01 == 2
    assert result.n10 == 2
    assert result.n11 == 1
    assert result.transition_count == 7


def test_exceedance_transitions_detect_different_ordering() -> None:
    """Equal breach totals can produce different transition counts."""

    dates = pd.date_range(
        "2025-01-02",
        periods=5,
        freq="B",
    )
    scattered = pd.Series(
        [
            False,
            True,
            False,
            True,
            False,
        ],
        index=dates,
        name="var_exceedance",
    )
    clustered = pd.Series(
        [
            False,
            False,
            False,
            True,
            True,
        ],
        index=dates,
        name="var_exceedance",
    )

    scattered_result = count_var_exceedance_transitions(
        exceedances=scattered,
    )
    clustered_result = count_var_exceedance_transitions(
        exceedances=clustered,
    )

    assert scattered.sum() == clustered.sum()

    assert scattered_result.n01 == 2
    assert scattered_result.n10 == 2
    assert scattered_result.n11 == 0

    assert clustered_result.n01 == 1
    assert clustered_result.n10 == 0
    assert clustered_result.n11 == 1


def test_exceedance_transitions_require_two_observations() -> None:
    """At least two observations are needed for one transition."""

    exceedances = pd.Series(
        [False],
        name="var_exceedance",
    )

    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        count_var_exceedance_transitions(
            exceedances=exceedances,
        )


def test_exceedance_transitions_require_boolean_values() -> None:
    """Transition indicators must contain Boolean values."""

    exceedances = pd.Series(
        [0, 1, 0, 1],
        name="var_exceedance",
    )

    with pytest.raises(
        ValueError,
        match="boolean",
    ):
        count_var_exceedance_transitions(
            exceedances=exceedances,
        )


def test_exceedance_transitions_reject_unordered_index() -> None:
    """Transition observations must remain chronologically ordered."""

    exceedances = pd.Series(
        [
            False,
            True,
            False,
        ],
        index=pd.to_datetime(
            [
                "2025-01-03",
                "2025-01-02",
                "2025-01-06",
            ]
        ),
        name="var_exceedance",
    )

    with pytest.raises(
        ValueError,
        match="ordered",
    ):
        count_var_exceedance_transitions(
            exceedances=exceedances,
        )




def test_christoffersen_independence_matches_manual_example(
) -> None:
    """Independence test matches a manually calculated example."""

    exceedances = pd.Series(
        [
            False,
            False,
            True,
            True,
            False,
            True,
            False,
            False,
        ],
        index=pd.date_range(
            "2025-01-02",
            periods=8,
            freq="B",
        ),
        name="var_exceedance",
    )

    result = calculate_christoffersen_independence(
        exceedances=exceedances,
        significance_level=0.05,
    )

    assert result.transitions.n00 == 2
    assert result.transitions.n01 == 2
    assert result.transitions.n10 == 2
    assert result.transitions.n11 == 1

    assert result.transition_breach_probability == pytest.approx(
        3.0 / 7.0
    )
    assert result.breach_probability_after_no_breach == (
        pytest.approx(0.5)
    )
    assert result.breach_probability_after_breach == pytest.approx(
        1.0 / 3.0
    )
    assert result.likelihood_ratio == pytest.approx(
        0.1964510115581639
    )
    assert result.p_value == pytest.approx(
        0.6576008676133729
    )
    assert not result.reject_null


def test_christoffersen_independence_rejects_clustered_breaches(
) -> None:
    """A concentrated breach block produces dependence evidence."""

    values = (
        [False] * 20
        + [True] * 5
        + [False]
    )
    exceedances = pd.Series(
        values,
        index=pd.date_range(
            "2025-01-02",
            periods=len(values),
            freq="B",
        ),
        name="var_exceedance",
    )

    result = calculate_christoffersen_independence(
        exceedances=exceedances,
    )

    assert result.transitions.n00 == 19
    assert result.transitions.n01 == 1
    assert result.transitions.n10 == 1
    assert result.transitions.n11 == 4

    assert (
        result.breach_probability_after_no_breach
        == pytest.approx(0.05)
    )
    assert (
        result.breach_probability_after_breach
        == pytest.approx(0.80)
    )
    assert result.likelihood_ratio == pytest.approx(
        12.075487207692609
    )
    assert result.p_value == pytest.approx(
        0.0005108911089523689
    )
    assert result.reject_null


def test_christoffersen_independence_handles_no_breaches(
) -> None:
    """An all-zero sequence remains mathematically defined."""

    exceedances = pd.Series(
        [False] * 10,
        index=pd.date_range(
            "2025-01-02",
            periods=10,
            freq="B",
        ),
        name="var_exceedance",
    )

    result = calculate_christoffersen_independence(
        exceedances=exceedances,
    )

    assert result.transition_breach_probability == pytest.approx(
        0.0
    )
    assert (
        result.breach_probability_after_no_breach
        == pytest.approx(0.0)
    )
    assert result.breach_probability_after_breach is None
    assert result.likelihood_ratio == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.reject_null


def test_christoffersen_independence_rejects_bad_significance(
) -> None:
    """Independence-test significance must be strictly between zero and one."""

    exceedances = pd.Series(
        [
            False,
            True,
            False,
        ],
        name="var_exceedance",
    )

    with pytest.raises(
        ValueError,
        match="Significance level",
    ):
        calculate_christoffersen_independence(
            exceedances=exceedances,
            significance_level=True,
        )




def test_christoffersen_conditional_coverage_combines_components(
) -> None:
    """Conditional coverage combines frequency and independence LR."""

    values = (
        [False] * 90
        + [True] * 10
    )
    exceedances = pd.Series(
        values,
        index=pd.date_range(
            "2025-01-02",
            periods=len(values),
            freq="B",
        ),
        name="var_exceedance",
    )

    result = calculate_christoffersen_conditional_coverage(
        exceedances=exceedances,
        confidence_level=0.95,
        significance_level=0.05,
    )

    assert (
        result.unconditional_coverage.likelihood_ratio
        == pytest.approx(4.130843782549277)
    )
    assert (
        result.independence.likelihood_ratio
        == pytest.approx(53.81628780732018)
    )
    assert result.likelihood_ratio == pytest.approx(
        57.94713158986946
    )
    assert result.likelihood_ratio == pytest.approx(
        result.unconditional_coverage.likelihood_ratio
        + result.independence.likelihood_ratio
    )
    assert result.p_value == pytest.approx(
        2.611802024625561e-13
    )
    assert result.reject_null


def test_christoffersen_conditional_coverage_handles_no_breaches(
) -> None:
    """Conditional coverage remains defined with no breaches."""

    exceedances = pd.Series(
        [False] * 100,
        index=pd.date_range(
            "2025-01-02",
            periods=100,
            freq="B",
        ),
        name="var_exceedance",
    )

    result = calculate_christoffersen_conditional_coverage(
        exceedances=exceedances,
        confidence_level=0.99,
    )

    assert (
        result.unconditional_coverage.exceedance_count
        == 0
    )
    assert (
        result.unconditional_coverage.likelihood_ratio
        == pytest.approx(2.01006717070029)
    )
    assert (
        result.independence.likelihood_ratio
        == pytest.approx(0.0)
    )
    assert result.likelihood_ratio == pytest.approx(
        2.01006717070029
    )
    assert result.p_value == pytest.approx(
        0.36603234127322914
    )
    assert not result.reject_null


def test_christoffersen_conditional_coverage_rejects_bad_level(
) -> None:
    """Conditional coverage requires a valid confidence level."""

    exceedances = pd.Series(
        [
            False,
            True,
            False,
        ],
        name="var_exceedance",
    )

    with pytest.raises(
        ValueError,
        match="Confidence level",
    ):
        calculate_christoffersen_conditional_coverage(
            exceedances=exceedances,
            confidence_level=True,
        )





def test_var_coverage_evaluation_combines_tests_by_level(
) -> None:
    """Coverage evaluation combines all tests by confidence level."""

    forecasts = _create_var_exceedance_forecasts()

    result = evaluate_var_coverage(
        forecasts=forecasts,
        significance_level=0.05,
    )

    assert list(result.index) == [
        0.95,
        0.99,
    ]

    result_95 = result.loc[0.95]

    assert result_95["observation_count"] == 5
    assert result_95["exceedance_count"] == 2

    assert result_95["transition_n00"] == 1
    assert result_95["transition_n01"] == 1
    assert result_95["transition_n10"] == 1
    assert result_95["transition_n11"] == 1

    assert result_95[
        "breach_probability_after_no_breach"
    ] == pytest.approx(0.5)
    assert result_95[
        "breach_probability_after_breach"
    ] == pytest.approx(0.5)

    assert result_95[
        "conditional_coverage_likelihood_ratio"
    ] == pytest.approx(
        result_95["kupiec_likelihood_ratio"]
        + result_95[
            "independence_likelihood_ratio"
        ]
    )


def test_var_coverage_evaluation_handles_missing_previous_state(
) -> None:
    """Undefined conditional probabilities become missing table values."""

    forecasts = _create_var_exceedance_forecasts()

    result = evaluate_var_coverage(
        forecasts=forecasts,
    )

    result_99 = result.loc[0.99]

    assert result_99["transition_n00"] == 4
    assert result_99["transition_n01"] == 0
    assert result_99["transition_n10"] == 0
    assert result_99["transition_n11"] == 0

    assert result_99[
        "breach_probability_after_no_breach"
    ] == pytest.approx(0.0)
    assert pd.isna(
        result_99[
            "breach_probability_after_breach"
        ]
    )

    assert result_99[
        "independence_likelihood_ratio"
    ] == pytest.approx(0.0)
    assert result_99[
        "conditional_coverage_likelihood_ratio"
    ] == pytest.approx(
        result_99["kupiec_likelihood_ratio"]
    )


def test_var_coverage_evaluation_propagates_significance(
) -> None:
    """All integrated tests use the requested significance level."""

    forecasts = _create_var_exceedance_forecasts()

    result = evaluate_var_coverage(
        forecasts=forecasts,
        significance_level=0.10,
    )

    assert (
        result["significance_level"]
        == 0.10
    ).all()


def test_holm_adjustment_matches_manual_example() -> None:
    """Holm-adjusted p-values match a controlled calculation."""

    p_values = pd.Series(
        [
            0.01,
            0.04,
            0.03,
            0.20,
        ],
        index=[
            "model_a",
            "model_b",
            "model_c",
            "model_d",
        ],
        name="p_value",
    )

    result = adjust_p_values_holm(
        p_values=p_values,
        significance_level=0.05,
    )

    assert list(result.index) == list(p_values.index)

    assert result[
        "raw_p_value"
    ].to_numpy() == pytest.approx(
        [
            0.01,
            0.04,
            0.03,
            0.20,
        ]
    )
    assert result[
        "holm_adjusted_p_value"
    ].to_numpy() == pytest.approx(
        [
            0.04,
            0.09,
            0.09,
            0.20,
        ]
    )

    assert bool(
        result.loc["model_a", "reject_null_holm"]
    )
    assert not bool(
        result.loc["model_b", "reject_null_holm"]
    )
    assert not bool(
        result.loc["model_c", "reject_null_holm"]
    )
    assert not bool(
        result.loc["model_d", "reject_null_holm"]
    )


def test_holm_adjustment_caps_p_values_at_one() -> None:
    """Holm-adjusted p-values cannot exceed one."""

    p_values = pd.Series(
        [
            0.40,
            0.50,
            0.90,
        ],
        name="p_value",
    )

    result = adjust_p_values_holm(
        p_values=p_values,
    )

    assert (
        result["holm_adjusted_p_value"] <= 1.0
    ).all()
    assert (
        result["holm_adjusted_p_value"] >= 0.0
    ).all()


@pytest.mark.parametrize(
    "values",
    [
        [-0.01, 0.20],
        [0.20, 1.01],
        [0.20, float("nan")],
    ],
)
def test_holm_adjustment_rejects_invalid_p_values(
    values: list[float],
) -> None:
    """Raw p-values must be finite numbers between zero and one."""

    with pytest.raises(
        ValueError,
        match="P-values",
    ):
        adjust_p_values_holm(
            p_values=pd.Series(values),
        )


def test_holm_adjustment_rejects_bad_significance() -> None:
    """Holm significance must be strictly between zero and one."""

    with pytest.raises(
        ValueError,
        match="Significance level",
    ):
        adjust_p_values_holm(
            p_values=pd.Series([0.01, 0.10]),
            significance_level=True,
        )



def test_model_coverage_evaluation_combines_model_tables(
) -> None:
    """Model coverage produces one aligned table with Holm results."""

    rolling_forecasts = (
        _create_var_exceedance_forecasts()
    )
    ewma_forecasts = rolling_forecasts.copy()
    ewma_forecasts["value_at_risk"] += 0.005
    ewma_forecasts["var_exceedance"] = (
        ewma_forecasts["realized_loss"]
        > ewma_forecasts["value_at_risk"]
    )

    result = evaluate_models_var_coverage(
        forecasts_by_model={
            "rolling_gaussian": rolling_forecasts,
            "ewma_gaussian": ewma_forecasts,
        },
        significance_level=0.05,
    )

    assert result.index.names == [
        "model",
        "confidence_level",
    ]
    assert list(
        result.index.get_level_values("model").unique()
    ) == [
        "rolling_gaussian",
        "ewma_gaussian",
    ]
    assert len(result) == 4

    expected_holm = adjust_p_values_holm(
        p_values=result[
            "conditional_coverage_p_value"
        ],
        significance_level=0.05,
    )

    assert result[
        "conditional_coverage_holm_p_value"
    ].to_numpy() == pytest.approx(
        expected_holm[
            "holm_adjusted_p_value"
        ].to_numpy()
    )
    assert (
        result[
            "conditional_coverage_reject_null_holm"
        ].to_numpy()
        == expected_holm[
            "reject_null_holm"
        ].to_numpy()
    ).all()


def test_model_coverage_propagates_interval_confidence_level(
) -> None:
    """Model coverage uses the requested interval confidence level."""

    forecasts = _create_var_exceedance_forecasts()

    result = evaluate_models_var_coverage(
        forecasts_by_model={
            "rolling_gaussian": forecasts,
        },
        significance_level=0.05,
        interval_confidence_level=0.90,
    )

    assert (
        result["interval_confidence_level"]
        == 0.90
    ).all()

    expected = calculate_exact_binomial_interval(
        exceedance_count=2,
        observation_count=5,
        interval_confidence_level=0.90,
    )

    result_95 = result.loc[
        (
            "rolling_gaussian",
            0.95,
        )
    ]

    assert result_95[
        "exceedance_rate_ci_lower"
    ] == pytest.approx(expected.lower_bound)
    assert result_95[
        "exceedance_rate_ci_upper"
    ] == pytest.approx(expected.upper_bound)


def test_model_coverage_evaluation_rejects_misaligned_index(
) -> None:
    """Compared model forecasts must have identical ordered rows."""

    rolling_forecasts = (
        _create_var_exceedance_forecasts()
    )
    ewma_forecasts = rolling_forecasts.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        evaluate_models_var_coverage(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_coverage_evaluation_rejects_different_losses(
) -> None:
    """Compared models must use identical realized losses."""

    rolling_forecasts = (
        _create_var_exceedance_forecasts()
    )
    ewma_forecasts = rolling_forecasts.copy()
    ewma_forecasts["realized_loss"] += 0.001
    ewma_forecasts["var_exceedance"] = (
        ewma_forecasts["realized_loss"]
        > ewma_forecasts["value_at_risk"]
    )

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        evaluate_models_var_coverage(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_coverage_evaluation_rejects_empty_mapping(
) -> None:
    """At least one model forecast table is required."""

    with pytest.raises(
        ValueError,
        match="At least one model",
    ):
        evaluate_models_var_coverage(
            forecasts_by_model={}
        )
