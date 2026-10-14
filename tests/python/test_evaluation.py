"""Tests for forecast comparison diagnostics."""

from math import log

import pandas as pd
import pytest

from tailrisk.evaluation import (
    adjust_p_values_holm,
    calculate_diebold_mariano_test,
    calculate_fz0_losses,
    calculate_christoffersen_conditional_coverage,
    calculate_christoffersen_independence,
    calculate_kupiec_unconditional_coverage,
    calculate_exact_binomial_interval,
    calculate_var_es_identification_residuals,
    calculate_newey_west_mean_test,
    calculate_basel_traffic_light_path,
    calculate_var_quantile_losses,
    classify_basel_traffic_light,
    count_var_exceedance_transitions,
    create_subperiod_definitions,
    evaluate_models_basel_traffic_light,
    evaluate_models_fz0_loss,
    evaluate_models_diebold_mariano,
    evaluate_models_subperiod_es_tail_severity,
    evaluate_models_subperiod_predictive_accuracy,
    evaluate_models_subperiod_var_es_identification,
    evaluate_models_var_quantile_loss,
    evaluate_models_subperiod_var_coverage,
    evaluate_models_var_es_identification,
    evaluate_models_var_coverage,
    evaluate_var_coverage,
    evaluate_models_es_tail_severity,
    evaluate_var_unconditional_coverage,
    summarize_es_tail_severity,
    summarize_fz0_loss,
    summarize_gaussian_monte_carlo_errors,
    summarize_models_basel_traffic_light,
    summarize_var_exceedances,
    summarize_var_es_identification,
    summarize_var_quantile_loss,
)


def _create_subperiod_forecasts(
) -> dict[str, pd.DataFrame]:
    """Create aligned forecasts spanning all predeclared years."""

    dates: list[pd.Timestamp] = []

    for year in range(2012, 2026):
        dates.extend(
            pd.to_datetime(
                [
                    f"{year}-01-03",
                    f"{year}-02-03",
                    f"{year}-03-02",
                    f"{year}-04-30",
                    f"{year}-06-15",
                    f"{year}-12-30",
                ]
            ).tolist()
        )
    dates = sorted(set(dates))
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
    date_losses = {
        date: position / 10_000.0
        for position, date in enumerate(dates)
    }
    realized_losses = [
        date_losses[forecast_date]
        for forecast_date, _ in index
    ]
    forecasts = pd.DataFrame(
        {
            "realized_loss": realized_losses,
        },
        index=index,
    )

    return {
        "model_a": forecasts.copy(),
        "model_b": forecasts.copy(),
        "model_c": forecasts.copy(),
    }


def test_create_subperiod_definitions_builds_calendar_and_nber_rows(
) -> None:
    """Definitions contain every calendar year and the NBER episode."""

    definitions = create_subperiod_definitions(
        forecasts_by_model=_create_subperiod_forecasts(),
    )

    assert list(definitions.index) == [
        *[
            f"calendar_{year}"
            for year in range(2012, 2026)
        ],
        "nber_recession_2020",
    ]
    assert definitions.loc[
        "calendar_2020",
        "period_type",
    ] == "calendar_year"
    assert definitions.loc[
        "nber_recession_2020",
        "period_type",
    ] == "nber_recession"


def test_create_subperiod_definitions_preserves_requested_and_realized_dates(
) -> None:
    """Requested boundaries remain distinct from observed sessions."""

    definitions = create_subperiod_definitions(
        forecasts_by_model=_create_subperiod_forecasts(),
    )
    calendar = definitions.loc["calendar_2019"]
    recession = definitions.loc[
        "nber_recession_2020"
    ]

    assert calendar["requested_start_date"] == pd.Timestamp(
        "2019-01-01"
    )
    assert calendar["first_forecast_date"] == pd.Timestamp(
        "2019-01-03"
    )
    assert calendar["requested_end_date"] == pd.Timestamp(
        "2019-12-31"
    )
    assert calendar["last_forecast_date"] == pd.Timestamp(
        "2019-12-30"
    )
    assert recession["requested_start_date"] == pd.Timestamp(
        "2020-02-01"
    )
    assert recession["requested_end_date"] == pd.Timestamp(
        "2020-04-30"
    )
    assert recession["first_forecast_date"] == pd.Timestamp(
        "2020-02-03"
    )
    assert recession["last_forecast_date"] == pd.Timestamp(
        "2020-04-30"
    )
    assert recession["observation_count"] == 3


def test_create_subperiod_definitions_flags_short_calendar_year(
) -> None:
    """A year below 90% of the maximum annual count is marked partial."""

    forecasts_by_model = _create_subperiod_forecasts()

    for model_name, forecasts in forecasts_by_model.items():
        retained_rows = (
            forecasts.index.get_level_values(
                "forecast_date"
            )
            != pd.Timestamp("2012-06-15")
        )
        forecasts_by_model[model_name] = forecasts.loc[
            retained_rows
        ]

    definitions = create_subperiod_definitions(
        forecasts_by_model=forecasts_by_model,
    )

    assert bool(
        definitions.loc["calendar_2012", "is_partial"]
    )
    assert not bool(
        definitions.loc["calendar_2019", "is_partial"]
    )


def test_create_subperiod_definitions_rejects_misaligned_models(
) -> None:
    """Models with different ordered forecast indexes are rejected."""

    forecasts_by_model = _create_subperiod_forecasts()
    model_c = forecasts_by_model["model_c"]
    last_date = model_c.index.get_level_values(
        "forecast_date"
    )[-1]
    forecasts_by_model["model_c"] = model_c.loc[
        model_c.index.get_level_values("forecast_date")
        != last_date
    ]

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        create_subperiod_definitions(
            forecasts_by_model=forecasts_by_model,
        )


def test_create_subperiod_definitions_rejects_different_losses(
) -> None:
    """Models with different realized losses are rejected."""

    forecasts_by_model = _create_subperiod_forecasts()
    forecasts_by_model["model_b"].iloc[
        0,
        forecasts_by_model["model_b"].columns.get_loc(
            "realized_loss"
        ),
    ] += 0.01
    forecasts_by_model["model_b"].iloc[
        1,
        forecasts_by_model["model_b"].columns.get_loc(
            "realized_loss"
        ),
    ] += 0.01

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        create_subperiod_definitions(
            forecasts_by_model=forecasts_by_model,
        )


def test_create_subperiod_definitions_rejects_inconsistent_date_loss(
) -> None:
    """Losses must agree across confidence levels on each date."""

    forecasts_by_model = _create_subperiod_forecasts()
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "realized_loss"
        ),
    ] += 0.01

    with pytest.raises(
        ValueError,
        match="identical across confidence levels",
    ):
        create_subperiod_definitions(
            forecasts_by_model=forecasts_by_model,
        )


def test_create_subperiod_definitions_rejects_missing_calendar_year(
) -> None:
    """Every predeclared calendar year must have common forecasts."""

    forecasts_by_model = _create_subperiod_forecasts()

    for model_name, forecasts in forecasts_by_model.items():
        year_mask = (
            forecasts.index.get_level_values(
                "forecast_date"
            ).year
            != 2017
        )
        forecasts_by_model[model_name] = forecasts.loc[
            year_mask
        ]

    with pytest.raises(
        ValueError,
        match="cover every calendar year",
    ):
        create_subperiod_definitions(
            forecasts_by_model=forecasts_by_model,
        )


def _create_subperiod_coverage_forecasts(
) -> dict[str, pd.DataFrame]:
    """Create aligned forecasts with controlled subperiod exceptions."""

    dates: list[pd.Timestamp] = []

    for year in range(2012, 2026):
        dates.extend(
            pd.bdate_range(
                start=f"{year}-01-02",
                periods=100,
            ).tolist()
        )

    dates = sorted(set(dates))
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
    base = pd.DataFrame(
        {
            "realized_loss": 1.0,
            "value_at_risk": 2.0,
            "expected_shortfall": 3.0,
        },
        index=index,
    )
    forecasts_by_model = {
        "model_a": base.copy(),
        "model_b": base.copy(),
        "model_c": base.copy(),
    }
    calendar_2012_dates = pd.DatetimeIndex(
        [
            date
            for date in dates
            if date.year == 2012
        ]
    )

    for model_name, exception_count in (
        ("model_a", 5),
        ("model_b", 10),
        ("model_c", 0),
    ):
        forecasts = forecasts_by_model[model_name]
        exception_dates = calendar_2012_dates[
            :exception_count
        ]
        forecasts.loc[
            pd.IndexSlice[exception_dates, 0.95],
            "value_at_risk",
        ] = 0.5
        forecasts.loc[
            (
                calendar_2012_dates[0],
                0.99,
            ),
            "value_at_risk",
        ] = 1.0

        if model_name == "model_a":
            forecasts.loc[
                (
                    calendar_2012_dates[1],
                    0.99,
                ),
                "value_at_risk",
            ] = 0.5

        forecasts["var_exceedance"] = (
            forecasts["realized_loss"]
            > forecasts["value_at_risk"]
        )

    return forecasts_by_model


def test_subperiod_var_coverage_matches_manual_calendar_example(
) -> None:
    """Calendar coverage counts, rates, gaps, and means are correct."""

    result = evaluate_models_subperiod_var_coverage(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.95,
        )
    ]

    assert row["observation_count"] == 100
    assert not bool(row["is_partial"])
    assert row["nominal_exceedance_rate"] == pytest.approx(
        0.05
    )
    assert row["expected_exceedance_count"] == pytest.approx(
        5.0
    )
    assert row["observed_exceedance_count"] == 5
    assert row["observed_exceedance_rate"] == pytest.approx(
        0.05
    )
    assert row[
        "observed_minus_expected_count"
    ] == pytest.approx(0.0)
    assert row[
        "observed_minus_expected_rate"
    ] == pytest.approx(0.0)
    assert row["mean_value_at_risk"] == pytest.approx(
        1.925
    )
    assert row["mean_expected_shortfall"] == pytest.approx(
        3.0
    )
    assert not bool(row["sparse_expected_tail"])
    assert not bool(row["sparse_observed_tail"])


def test_subperiod_var_coverage_uses_strict_exception_rule(
) -> None:
    """Loss equal to VaR remains outside the subperiod exception count."""

    result = evaluate_models_subperiod_var_coverage(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.99,
        )
    ]

    assert row["observed_exceedance_count"] == 1
    assert row["expected_exceedance_count"] == pytest.approx(
        1.0
    )
    assert bool(row["sparse_expected_tail"])
    assert bool(row["sparse_observed_tail"])


def test_subperiod_var_coverage_reuses_exact_binomial_interval(
) -> None:
    """Subperiod bounds equal the established exact-binomial result."""

    result = evaluate_models_subperiod_var_coverage(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
        interval_confidence_level=0.90,
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.95,
        )
    ]
    expected = calculate_exact_binomial_interval(
        exceedance_count=5,
        observation_count=100,
        interval_confidence_level=0.90,
    )

    assert row[
        "interval_confidence_level"
    ] == pytest.approx(0.90)
    assert row[
        "exact_interval_lower_bound"
    ] == pytest.approx(expected.lower_bound)
    assert row[
        "exact_interval_upper_bound"
    ] == pytest.approx(expected.upper_bound)


def test_subperiod_var_coverage_contains_only_descriptive_columns(
) -> None:
    """The initial table contains no test p-values or decisions."""

    result = evaluate_models_subperiod_var_coverage(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )

    assert len(result) == 15 * 3 * 2
    assert list(result.index.names) == [
        "period_id",
        "model",
        "confidence_level",
    ]
    assert not any(
        "p_value" in column
        or "reject" in column
        or "test_statistic" in column
        for column in result.columns
    )


def test_subperiod_var_coverage_retains_overlapping_nber_period(
) -> None:
    """The NBER episode is reported alongside its containing year."""

    result = evaluate_models_subperiod_var_coverage(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    calendar = result.loc[
        (
            "calendar_2020",
            "model_a",
            0.95,
        )
    ]
    recession = result.loc[
        (
            "nber_recession_2020",
            "model_a",
            0.95,
        )
    ]

    assert calendar["observation_count"] == 100
    assert 0 < recession["observation_count"] < 100
    assert recession["period_type"] == "nber_recession"
    assert recession["first_forecast_date"] >= pd.Timestamp(
        "2020-02-01"
    )
    assert recession["last_forecast_date"] <= pd.Timestamp(
        "2020-04-30"
    )


def test_subperiod_var_coverage_rejects_incorrect_indicator(
) -> None:
    """Stored period exceptions must match strict loss-versus-VaR values."""

    forecasts_by_model = (
        _create_subperiod_coverage_forecasts()
    )
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "var_exceedance"
        ),
    ] = False

    with pytest.raises(
        ValueError,
        match="must agree",
    ):
        evaluate_models_subperiod_var_coverage(
            forecasts_by_model=forecasts_by_model,
        )


def test_subperiod_var_coverage_rejects_es_below_var(
) -> None:
    """Subperiod Expected Shortfall must not be smaller than VaR."""

    forecasts_by_model = (
        _create_subperiod_coverage_forecasts()
    )
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "expected_shortfall"
        ),
    ] = 0.25

    with pytest.raises(
        ValueError,
        match="at least as large as VaR",
    ):
        evaluate_models_subperiod_var_coverage(
            forecasts_by_model=forecasts_by_model,
        )


def test_subperiod_tail_severity_matches_manual_example(
) -> None:
    """Conditional severity statistics match controlled exceptions."""

    result = evaluate_models_subperiod_es_tail_severity(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.95,
        )
    ]

    assert row["observation_count"] == 100
    assert row["expected_exceedance_count"] == pytest.approx(
        5.0
    )
    assert row["var_exceedance_count"] == 5
    assert row["mean_exceedance_loss"] == pytest.approx(
        1.0
    )
    assert row[
        "mean_var_on_exceedance_days"
    ] == pytest.approx(0.5)
    assert row[
        "mean_es_on_exceedance_days"
    ] == pytest.approx(3.0)
    assert row["mean_var_excess"] == pytest.approx(0.5)
    assert row["median_var_excess"] == pytest.approx(0.5)
    assert row["maximum_var_excess"] == pytest.approx(0.5)
    assert row["mean_es_gap"] == pytest.approx(-2.0)
    assert not bool(row["sparse_expected_tail"])
    assert not bool(row["sparse_observed_tail"])


def test_subperiod_tail_severity_preserves_sparse_finite_result(
) -> None:
    """One observed tail loss is retained and marked as sparse."""

    result = evaluate_models_subperiod_es_tail_severity(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.99,
        )
    ]

    assert row["var_exceedance_count"] == 1
    assert row["mean_exceedance_loss"] == pytest.approx(
        1.0
    )
    assert row["mean_var_excess"] == pytest.approx(0.5)
    assert row["mean_es_gap"] == pytest.approx(-2.0)
    assert bool(row["sparse_expected_tail"])
    assert bool(row["sparse_observed_tail"])


def test_subperiod_tail_severity_uses_missing_values_for_empty_tail(
) -> None:
    """A period with no exceptions has missing conditional severity."""

    result = evaluate_models_subperiod_es_tail_severity(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_c",
            0.95,
        )
    ]
    conditional_columns = [
        "mean_exceedance_loss",
        "mean_var_on_exceedance_days",
        "mean_es_on_exceedance_days",
        "mean_var_excess",
        "median_var_excess",
        "maximum_var_excess",
        "mean_es_gap",
    ]

    assert row["var_exceedance_count"] == 0
    assert row[conditional_columns].isna().all()
    assert bool(row["sparse_observed_tail"])


def test_subperiod_tail_severity_contains_only_descriptive_columns(
) -> None:
    """The tail table contains every period and no inferential output."""

    result = evaluate_models_subperiod_es_tail_severity(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )

    assert len(result) == 15 * 3 * 2
    assert list(result.index.names) == [
        "period_id",
        "model",
        "confidence_level",
    ]
    assert not any(
        "p_value" in column
        or "reject" in column
        or "test_statistic" in column
        for column in result.columns
    )


def test_subperiod_tail_severity_retains_nber_case_study(
) -> None:
    """The NBER episode receives its own descriptive severity rows."""

    result = evaluate_models_subperiod_es_tail_severity(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    recession = result.loc[
        (
            "nber_recession_2020",
            "model_a",
            0.99,
        )
    ]

    assert recession["period_type"] == "nber_recession"
    assert 0 < recession["observation_count"] < 100
    assert bool(recession["sparse_expected_tail"])


def test_subperiod_tail_severity_rejects_incorrect_indicator(
) -> None:
    """Tail rows reject exception indicators inconsistent with VaR."""

    forecasts_by_model = (
        _create_subperiod_coverage_forecasts()
    )
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "var_exceedance"
        ),
    ] = False

    with pytest.raises(
        ValueError,
        match="must agree",
    ):
        evaluate_models_subperiod_es_tail_severity(
            forecasts_by_model=forecasts_by_model,
        )


def test_subperiod_tail_severity_rejects_es_below_var(
) -> None:
    """Tail rows preserve the established ES-at-least-VaR contract."""

    forecasts_by_model = (
        _create_subperiod_coverage_forecasts()
    )
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "expected_shortfall"
        ),
    ] = 0.25

    with pytest.raises(
        ValueError,
        match="at least as large as VaR",
    ):
        evaluate_models_subperiod_es_tail_severity(
            forecasts_by_model=forecasts_by_model,
        )


def test_subperiod_joint_calibration_matches_manual_example(
) -> None:
    """Mean tail contribution and residual match a controlled period."""

    result = evaluate_models_subperiod_var_es_identification(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.95,
        )
    ]

    assert row["observation_count"] == 100
    assert row["expected_exceedance_count"] == pytest.approx(
        5.0
    )
    assert row["var_exceedance_count"] == 5
    assert row["mean_tail_contribution"] == pytest.approx(
        2.425
    )
    assert row["mean_expected_shortfall"] == pytest.approx(
        3.0
    )
    assert row[
        "mean_identification_residual"
    ] == pytest.approx(-0.575)
    assert not bool(row["sparse_expected_tail"])
    assert not bool(row["sparse_observed_tail"])


def test_subperiod_joint_calibration_uses_every_session(
) -> None:
    """Joint residuals remain defined when no VaR exception occurs."""

    result = evaluate_models_subperiod_var_es_identification(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_c",
            0.95,
        )
    ]

    assert row["var_exceedance_count"] == 0
    assert row["mean_tail_contribution"] == pytest.approx(
        2.0
    )
    assert row["mean_expected_shortfall"] == pytest.approx(
        3.0
    )
    assert row[
        "mean_identification_residual"
    ] == pytest.approx(-1.0)
    assert bool(row["sparse_observed_tail"])


def test_subperiod_joint_calibration_flags_sparse_tail(
) -> None:
    """Rare expected and observed tails retain both warning flags."""

    result = evaluate_models_subperiod_var_es_identification(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.99,
        )
    ]

    assert row["expected_exceedance_count"] == pytest.approx(
        1.0
    )
    assert row["var_exceedance_count"] == 1
    assert row["mean_tail_contribution"] == pytest.approx(
        2.475
    )
    assert row[
        "mean_identification_residual"
    ] == pytest.approx(-0.525)
    assert bool(row["sparse_expected_tail"])
    assert bool(row["sparse_observed_tail"])


def test_subperiod_joint_calibration_is_descriptive_only(
) -> None:
    """The joint table contains all periods and no inferential fields."""

    result = evaluate_models_subperiod_var_es_identification(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )

    assert len(result) == 15 * 3 * 2
    assert list(result.index.names) == [
        "period_id",
        "model",
        "confidence_level",
    ]
    assert not any(
        "hac" in column
        or "p_value" in column
        or "reject" in column
        or "test_statistic" in column
        for column in result.columns
    )


def test_subperiod_joint_calibration_retains_nber_case_study(
) -> None:
    """The NBER episode receives descriptive joint-calibration rows."""

    result = evaluate_models_subperiod_var_es_identification(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    recession = result.loc[
        (
            "nber_recession_2020",
            "model_a",
            0.99,
        )
    ]

    assert recession["period_type"] == "nber_recession"
    assert 0 < recession["observation_count"] < 100
    assert bool(recession["sparse_expected_tail"])


def test_subperiod_joint_calibration_rejects_incorrect_indicator(
) -> None:
    """Joint rows require indicators consistent with strict VaR breaches."""

    forecasts_by_model = (
        _create_subperiod_coverage_forecasts()
    )
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "var_exceedance"
        ),
    ] = False

    with pytest.raises(
        ValueError,
        match="must agree",
    ):
        evaluate_models_subperiod_var_es_identification(
            forecasts_by_model=forecasts_by_model,
        )


def test_subperiod_joint_calibration_rejects_es_below_var(
) -> None:
    """Joint rows preserve the established ES-at-least-VaR contract."""

    forecasts_by_model = (
        _create_subperiod_coverage_forecasts()
    )
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "expected_shortfall"
        ),
    ] = 0.25

    with pytest.raises(
        ValueError,
        match="at least as large as VaR",
    ):
        evaluate_models_subperiod_var_es_identification(
            forecasts_by_model=forecasts_by_model,
        )


def test_subperiod_predictive_accuracy_matches_manual_scores(
) -> None:
    """Mean quantile and FZ0 scores match a controlled calendar year."""

    result = evaluate_models_subperiod_predictive_accuracy(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    row = result.loc[
        (
            "calendar_2012",
            "model_a",
            0.95,
        )
    ]
    expected_quantile_loss = (
        5 * 0.95 * 0.5
        + 95 * 0.05 * 1.0
    ) / 100
    non_exception_fz0 = (
        2.0 / 3.0
        + log(3.0)
        - 1.0
    )
    exception_fz0 = (
        0.5 / (0.05 * 3.0)
        + 0.5 / 3.0
        + log(3.0)
        - 1.0
    )
    expected_fz0_loss = (
        5 * exception_fz0
        + 95 * non_exception_fz0
    ) / 100

    assert row["observation_count"] == 100
    assert row["mean_var_quantile_loss"] == pytest.approx(
        expected_quantile_loss
    )
    assert row["mean_fz0_loss"] == pytest.approx(
        expected_fz0_loss
    )


def test_subperiod_predictive_accuracy_ranks_models_and_ties(
) -> None:
    """Lower period scores rank first and exact ties retain rank one."""

    result = evaluate_models_subperiod_predictive_accuracy(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )

    for score_rank in (
        "var_quantile_loss_rank",
        "fz0_loss_rank",
    ):
        assert result.loc[
            ("calendar_2012", "model_c", 0.95),
            score_rank,
        ] == 1
        assert result.loc[
            ("calendar_2012", "model_a", 0.95),
            score_rank,
        ] == 2
        assert result.loc[
            ("calendar_2012", "model_b", 0.95),
            score_rank,
        ] == 3
        assert result.xs(
            ("calendar_2013", 0.95),
            level=("period_id", "confidence_level"),
        )[score_rank].eq(1).all()


def test_subperiod_predictive_accuracy_is_descriptive_only(
) -> None:
    """The table contains every period and no inferential output."""

    result = evaluate_models_subperiod_predictive_accuracy(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )

    assert len(result) == 15 * 3 * 2
    assert list(result.index.names) == [
        "period_id",
        "model",
        "confidence_level",
    ]
    assert not any(
        "hac" in column
        or "p_value" in column
        or "reject" in column
        or "test_statistic" in column
        for column in result.columns
    )


def test_subperiod_predictive_accuracy_retains_sparse_context(
) -> None:
    """The NBER case study and rare-tail warnings accompany scores."""

    result = evaluate_models_subperiod_predictive_accuracy(
        forecasts_by_model=(
            _create_subperiod_coverage_forecasts()
        ),
    )
    recession = result.loc[
        (
            "nber_recession_2020",
            "model_a",
            0.99,
        )
    ]

    assert recession["period_type"] == "nber_recession"
    assert 0 < recession["observation_count"] < 100
    assert bool(recession["sparse_expected_tail"])
    assert bool(recession["sparse_observed_tail"])
    assert pd.notna(recession["mean_var_quantile_loss"])
    assert pd.notna(recession["mean_fz0_loss"])


def test_subperiod_predictive_accuracy_rejects_bad_indicator(
) -> None:
    """Period scores require indicators consistent with strict breaches."""

    forecasts_by_model = (
        _create_subperiod_coverage_forecasts()
    )
    forecasts_by_model["model_a"].iloc[
        0,
        forecasts_by_model["model_a"].columns.get_loc(
            "var_exceedance"
        ),
    ] = False

    with pytest.raises(
        ValueError,
        match="must agree",
    ):
        evaluate_models_subperiod_predictive_accuracy(
            forecasts_by_model=forecasts_by_model,
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





def test_var_quantile_losses_match_manual_example(
) -> None:
    """VaR quantile losses match controlled forecast errors."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=3,
        freq="D",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [0.95],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    forecasts = pd.DataFrame(
        {
            "realized_loss": [
                0.010,
                0.020,
                0.030,
            ],
            "value_at_risk": [
                0.020,
                0.020,
                0.020,
            ],
            "var_exceedance": [
                False,
                False,
                True,
            ],
        },
        index=index,
    )

    result = calculate_var_quantile_losses(
        forecasts=forecasts,
    )

    assert result.index.equals(forecasts.index)

    assert result[
        "var_quantile_loss"
    ].to_numpy() == pytest.approx(
        [
            0.0005,
            0.0000,
            0.0095,
        ]
    )

    assert (
        result["var_quantile_loss"] >= 0.0
    ).all()

    assert (
        result.iloc[2]["var_quantile_loss"]
        / result.iloc[0]["var_quantile_loss"]
    ) == pytest.approx(19.0)





def test_var_es_identification_residuals_match_manual_example(
) -> None:
    """Joint VaR-ES residuals match a calibrated manual example."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=4,
        freq="D",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [0.75],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    forecasts = pd.DataFrame(
        {
            "realized_loss": [
                0.000,
                0.010,
                0.019,
                0.030,
            ],
            "value_at_risk": [
                0.020,
                0.020,
                0.020,
                0.020,
            ],
            "expected_shortfall": [
                0.030,
                0.030,
                0.030,
                0.030,
            ],
            "var_exceedance": [
                False,
                False,
                False,
                True,
            ],
        },
        index=index,
    )

    result = (
        calculate_var_es_identification_residuals(
            forecasts=forecasts,
        )
    )

    assert result.index.equals(forecasts.index)

    assert result[
        "tail_contribution"
    ].to_numpy() == pytest.approx(
        [
            0.020,
            0.020,
            0.020,
            0.060,
        ]
    )

    assert result[
        "var_es_identification_residual"
    ].to_numpy() == pytest.approx(
        [
            -0.010,
            -0.010,
            -0.010,
            0.030,
        ]
    )

    assert result[
        "var_es_identification_residual"
    ].mean() == pytest.approx(0.0)




def test_var_es_identification_summary_matches_manual_example(
) -> None:
    """Joint VaR-ES summary matches a calibrated manual example."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=4,
        freq="D",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [0.75],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    forecasts = pd.DataFrame(
        {
            "realized_loss": [
                0.000,
                0.010,
                0.019,
                0.030,
            ],
            "value_at_risk": [
                0.020,
                0.020,
                0.020,
                0.020,
            ],
            "expected_shortfall": [
                0.030,
                0.030,
                0.030,
                0.030,
            ],
            "var_exceedance": [
                False,
                False,
                False,
                True,
            ],
        },
        index=index,
    )

    result = summarize_var_es_identification(
        forecasts=forecasts,
        bandwidth=1,
        significance_level=0.05,
    )

    summary = result.loc[0.75]

    assert summary[
        "observation_count"
    ] == 4

    assert summary[
        "mean_tail_contribution"
    ] == pytest.approx(0.030)

    assert summary[
        "mean_expected_shortfall"
    ] == pytest.approx(0.030)

    assert summary[
        "mean_identification_residual"
    ] == pytest.approx(0.0)

    assert summary["hac_bandwidth"] == 1
    assert summary[
        "hac_long_run_variance"
    ] == pytest.approx(0.000275)
    assert summary[
        "hac_standard_error"
    ] == pytest.approx(
        0.0082915619758885
    )
    assert summary[
        "hac_test_statistic"
    ] == pytest.approx(0.0)
    assert summary["hac_p_value"] == pytest.approx(
        1.0
    )
    assert summary[
        "significance_level"
    ] == pytest.approx(0.05)
    assert not summary["hac_reject_null"]


def test_var_es_identification_applies_hac_by_level(
) -> None:
    """Each confidence level receives its own HAC calibration test."""

    forecasts = _create_es_tail_severity_forecasts()

    result = summarize_var_es_identification(
        forecasts=forecasts,
        bandwidth=1,
        significance_level=0.10,
    )
    residuals = (
        calculate_var_es_identification_residuals(
            forecasts=forecasts,
        )
    )

    assert list(result.index) == [0.95, 0.99]

    for confidence_level in result.index:
        level_residuals = residuals.xs(
            confidence_level,
            level="confidence_level",
        )[
            "var_es_identification_residual"
        ]
        expected = calculate_newey_west_mean_test(
            observations=level_residuals,
            bandwidth=1,
            significance_level=0.10,
        )
        summary = result.loc[confidence_level]

        assert summary[
            "mean_identification_residual"
        ] == pytest.approx(expected.sample_mean)
        assert summary[
            "hac_bandwidth"
        ] == expected.bandwidth
        assert summary[
            "hac_long_run_variance"
        ] == pytest.approx(
            expected.long_run_variance
        )
        assert summary[
            "hac_standard_error"
        ] == pytest.approx(
            expected.standard_error
        )
        assert summary[
            "hac_test_statistic"
        ] == pytest.approx(
            expected.test_statistic
        )
        assert summary[
            "hac_p_value"
        ] == pytest.approx(expected.p_value)
        assert summary[
            "significance_level"
        ] == pytest.approx(0.10)
        assert bool(
            summary["hac_reject_null"]
        ) is expected.reject_null







def test_model_var_es_identification_combines_and_adjusts(
) -> None:
    """Aligned model tests form one Holm-adjusted calibration table."""

    rolling_forecasts = (
        _create_es_tail_severity_forecasts()
    )
    ewma_forecasts = rolling_forecasts.copy()
    ewma_forecasts["expected_shortfall"] += 1.0
    forecasts_by_model = {
        "rolling_gaussian": rolling_forecasts,
        "ewma_gaussian": ewma_forecasts,
    }

    result = evaluate_models_var_es_identification(
        forecasts_by_model=forecasts_by_model,
        bandwidth=1,
        significance_level=0.10,
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

    for model_name, forecasts in (
        forecasts_by_model.items()
    ):
        expected = summarize_var_es_identification(
            forecasts=forecasts,
            bandwidth=1,
            significance_level=0.10,
        )
        pd.testing.assert_frame_equal(
            result.loc[model_name, expected.columns],
            expected,
        )

    expected_holm = adjust_p_values_holm(
        p_values=result["hac_p_value"],
        significance_level=0.10,
    )
    assert result[
        "hac_holm_p_value"
    ].to_numpy() == pytest.approx(
        expected_holm[
            "holm_adjusted_p_value"
        ].to_numpy()
    )
    assert result[
        "hac_holm_reject_null"
    ].equals(
        expected_holm["reject_null_holm"]
    )


def test_model_var_es_identification_rejects_misaligned_index(
) -> None:
    """Compared calibration tests require identical ordered rows."""

    rolling_forecasts = (
        _create_es_tail_severity_forecasts()
    )
    ewma_forecasts = rolling_forecasts.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        evaluate_models_var_es_identification(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            },
            bandwidth=1,
        )


def test_model_var_es_identification_rejects_different_losses(
) -> None:
    """Compared calibration tests require identical realized losses."""

    rolling_forecasts = (
        _create_es_tail_severity_forecasts()
    )
    ewma_forecasts = rolling_forecasts.copy()
    ewma_forecasts.iloc[
        0,
        ewma_forecasts.columns.get_loc(
            "realized_loss"
        ),
    ] += 0.25
    ewma_forecasts["var_exceedance"] = (
        ewma_forecasts["realized_loss"]
        > ewma_forecasts["value_at_risk"]
    )

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        evaluate_models_var_es_identification(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            },
            bandwidth=1,
        )


def test_model_var_es_identification_rejects_empty_mapping(
) -> None:
    """At least one model is required for joint calibration."""

    with pytest.raises(
        ValueError,
        match="At least one model",
    ):
        evaluate_models_var_es_identification(
            forecasts_by_model={},
        )


def test_newey_west_mean_test_matches_manual_example(
) -> None:
    """Newey-West calculations match a bandwidth-one example."""

    observations = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range(
            start="2025-01-02",
            periods=4,
            freq="D",
        ),
        name="identification_residual",
    )

    result = calculate_newey_west_mean_test(
        observations=observations,
        bandwidth=1,
        significance_level=0.05,
    )

    assert result.observation_count == 4
    assert result.bandwidth == 1
    assert result.sample_mean == pytest.approx(2.5)
    assert result.long_run_variance == pytest.approx(
        1.5625
    )
    assert result.mean_variance == pytest.approx(
        0.390625
    )
    assert result.standard_error == pytest.approx(
        0.625
    )
    assert result.test_statistic == pytest.approx(
        4.0
    )
    assert result.p_value == pytest.approx(
        0.00006334248366623984
    )
    assert result.significance_level == pytest.approx(
        0.05
    )
    assert result.reject_null


@pytest.mark.parametrize(
    "bad_bandwidth",
    [
        True,
        -1,
        4,
        1.5,
    ],
)
def test_newey_west_mean_test_rejects_invalid_bandwidth(
    bad_bandwidth: object,
) -> None:
    """HAC bandwidth must be a valid lag count."""

    observations = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range(
            start="2025-01-02",
            periods=4,
            freq="D",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Bandwidth must be an integer",
    ):
        calculate_newey_west_mean_test(
            observations=observations,
            bandwidth=bad_bandwidth,
        )


def test_diebold_mariano_test_matches_hac_manual_example(
) -> None:
    """DM results equal the HAC test of A-minus-B scores."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=4,
        freq="D",
    )
    model_a_scores = pd.Series(
        [2.0, 4.0, 6.0, 8.0],
        index=dates,
        name="model_a_score",
    )
    model_b_scores = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=dates,
        name="model_b_score",
    )

    result = calculate_diebold_mariano_test(
        model_a_scores=model_a_scores,
        model_b_scores=model_b_scores,
        bandwidth=1,
        significance_level=0.05,
    )

    assert result.observation_count == 4
    assert result.bandwidth == 1
    assert result.mean_score_difference == pytest.approx(
        2.5
    )
    assert result.long_run_variance == pytest.approx(
        1.5625
    )
    assert result.mean_variance == pytest.approx(
        0.390625
    )
    assert result.standard_error == pytest.approx(
        0.625
    )
    assert result.test_statistic == pytest.approx(4.0)
    assert result.p_value == pytest.approx(
        0.00006334248366623984
    )
    assert result.significance_level == pytest.approx(
        0.05
    )
    assert result.reject_null


def test_diebold_mariano_sign_identifies_better_model(
) -> None:
    """A negative A-minus-B mean identifies model A as better."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=4,
        freq="D",
    )
    model_a_scores = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=dates,
    )
    model_b_scores = pd.Series(
        [2.0, 4.0, 6.0, 8.0],
        index=dates,
    )

    result = calculate_diebold_mariano_test(
        model_a_scores=model_a_scores,
        model_b_scores=model_b_scores,
        bandwidth=1,
    )

    assert result.mean_score_difference == pytest.approx(
        -2.5
    )
    assert result.test_statistic == pytest.approx(-4.0)
    assert result.p_value == pytest.approx(
        0.00006334248366623984
    )


@pytest.mark.parametrize(
    (
        "model_a_values",
        "model_b_values",
        "expected_mean",
        "expected_statistic",
        "expected_p_value",
    ),
    [
        (
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            0.0,
            0.0,
            1.0,
        ),
        (
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            1.0,
            float("inf"),
            0.0,
        ),
    ],
)
def test_diebold_mariano_handles_zero_hac_variance(
    model_a_values: list[float],
    model_b_values: list[float],
    expected_mean: float,
    expected_statistic: float,
    expected_p_value: float,
) -> None:
    """Constant differentials have defined DM boundary results."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=3,
        freq="D",
    )

    result = calculate_diebold_mariano_test(
        model_a_scores=pd.Series(
            model_a_values,
            index=dates,
        ),
        model_b_scores=pd.Series(
            model_b_values,
            index=dates,
        ),
        bandwidth=0,
    )

    assert result.mean_score_difference == pytest.approx(
        expected_mean
    )
    assert result.standard_error == pytest.approx(0.0)
    assert result.test_statistic == expected_statistic
    assert result.p_value == pytest.approx(
        expected_p_value
    )


def test_diebold_mariano_rejects_misaligned_score_index(
) -> None:
    """Paired DM scores must use the same ordered observations."""

    model_a_scores = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range(
            "2025-01-02",
            periods=3,
            freq="D",
        ),
    )
    model_b_scores = model_a_scores.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="same ordered index",
    ):
        calculate_diebold_mariano_test(
            model_a_scores=model_a_scores,
            model_b_scores=model_b_scores,
        )


@pytest.mark.parametrize(
    "bad_values",
    [
        [1.0, float("nan"), 3.0],
        [1.0, float("inf"), 3.0],
        [True, False, True],
    ],
)
def test_diebold_mariano_rejects_invalid_scores(
    bad_values: list[object],
) -> None:
    """Each DM score series must contain finite non-boolean numbers."""

    dates = pd.date_range(
        "2025-01-02",
        periods=3,
        freq="D",
    )

    with pytest.raises(
        ValueError,
        match="Model A scores must contain finite numbers",
    ):
        calculate_diebold_mariano_test(
            model_a_scores=pd.Series(
                bad_values,
                index=dates,
            ),
            model_b_scores=pd.Series(
                [1.0, 2.0, 3.0],
                index=dates,
            ),
        )


def _create_dm_integration_forecasts(
    value_at_risk: float,
    expected_shortfall: float,
) -> pd.DataFrame:
    """Create three-level forecasts for integrated DM tests."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=4,
        freq="D",
    )
    confidence_levels = [
        0.95,
        0.975,
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
        loss
        for loss in [
            0.010,
            0.020,
            0.030,
            0.040,
        ]
        for _ in confidence_levels
    ]
    value_at_risk_values = [
        value_at_risk
    ] * len(index)

    return pd.DataFrame(
        {
            "realized_loss": realized_losses,
            "value_at_risk": value_at_risk_values,
            "expected_shortfall": [
                expected_shortfall
            ] * len(index),
            "var_exceedance": [
                loss > forecast
                for loss, forecast in zip(
                    realized_losses,
                    value_at_risk_values,
                    strict=True,
                )
            ],
        },
        index=index,
    )


def test_model_diebold_mariano_runs_every_pair_and_level(
) -> None:
    """Three models and levels produce both nine-test families."""

    forecasts_by_model = {
        "rolling_gaussian": (
            _create_dm_integration_forecasts(
                value_at_risk=0.020,
                expected_shortfall=0.030,
            )
        ),
        "ewma_gaussian": (
            _create_dm_integration_forecasts(
                value_at_risk=0.025,
                expected_shortfall=0.035,
            )
        ),
        "filtered_historical": (
            _create_dm_integration_forecasts(
                value_at_risk=0.022,
                expected_shortfall=0.032,
            )
        ),
    }

    result = evaluate_models_diebold_mariano(
        forecasts_by_model=forecasts_by_model,
        bandwidth=1,
        significance_level=0.10,
    )

    assert result.index.names == [
        "score_name",
        "confidence_level",
        "model_a",
        "model_b",
    ]
    assert len(result) == 18
    assert list(
        result.index
        .get_level_values("score_name")
        .unique()
    ) == [
        "var_quantile_loss",
        "fz0_loss",
    ]

    for score_name in (
        "var_quantile_loss",
        "fz0_loss",
    ):
        assert len(result.xs(
            score_name,
            level="score_name",
        )) == 9

    expected_pairs = [
        (
            "rolling_gaussian",
            "ewma_gaussian",
        ),
        (
            "rolling_gaussian",
            "filtered_historical",
        ),
        (
            "ewma_gaussian",
            "filtered_historical",
        ),
    ]
    actual_pairs = list(
        result.xs(
            (
                "var_quantile_loss",
                0.95,
            ),
            level=(
                "score_name",
                "confidence_level",
            ),
        ).index
    )
    assert actual_pairs == expected_pairs


def test_model_diebold_mariano_matches_daily_score_pipeline(
) -> None:
    """Integrated DM rows match separately calculated daily scores."""

    rolling_forecasts = _create_dm_integration_forecasts(
        value_at_risk=0.020,
        expected_shortfall=0.030,
    )
    ewma_forecasts = _create_dm_integration_forecasts(
        value_at_risk=0.025,
        expected_shortfall=0.035,
    )
    filtered_forecasts = (
        _create_dm_integration_forecasts(
            value_at_risk=0.022,
            expected_shortfall=0.032,
        )
    )

    result = evaluate_models_diebold_mariano(
        forecasts_by_model={
            "rolling_gaussian": rolling_forecasts,
            "ewma_gaussian": ewma_forecasts,
            "filtered_historical": (
                filtered_forecasts
            ),
        },
        bandwidth=1,
        significance_level=0.10,
    )

    rolling_scores = calculate_fz0_losses(
        forecasts=rolling_forecasts,
    ).xs(
        0.975,
        level="confidence_level",
    )["fz0_loss"]
    filtered_scores = calculate_fz0_losses(
        forecasts=filtered_forecasts,
    ).xs(
        0.975,
        level="confidence_level",
    )["fz0_loss"]
    expected = calculate_diebold_mariano_test(
        model_a_scores=rolling_scores,
        model_b_scores=filtered_scores,
        bandwidth=1,
        significance_level=0.10,
    )
    row = result.loc[
        (
            "fz0_loss",
            0.975,
            "rolling_gaussian",
            "filtered_historical",
        )
    ]

    assert row["observation_count"] == 4
    assert row["mean_score_difference"] == (
        pytest.approx(
            expected.mean_score_difference
        )
    )
    assert row["hac_long_run_variance"] == (
        pytest.approx(
            expected.long_run_variance
        )
    )
    assert row["hac_standard_error"] == (
        pytest.approx(expected.standard_error)
    )
    assert row["hac_bandwidth"] == 1
    assert row["dm_statistic"] == pytest.approx(
        expected.test_statistic
    )
    assert row["dm_p_value"] == pytest.approx(
        expected.p_value
    )
    assert bool(
        row["dm_reject_equal_accuracy"]
    ) is expected.reject_null


def test_model_diebold_mariano_adjusts_score_families_separately(
) -> None:
    """Quantile and FZ0 p-values receive separate Holm adjustments."""

    result = evaluate_models_diebold_mariano(
        forecasts_by_model={
            "rolling_gaussian": (
                _create_dm_integration_forecasts(
                    value_at_risk=0.020,
                    expected_shortfall=0.030,
                )
            ),
            "ewma_gaussian": (
                _create_dm_integration_forecasts(
                    value_at_risk=0.025,
                    expected_shortfall=0.035,
                )
            ),
            "filtered_historical": (
                _create_dm_integration_forecasts(
                    value_at_risk=0.022,
                    expected_shortfall=0.032,
                )
            ),
        },
        bandwidth=1,
        significance_level=0.10,
    )

    for score_name in (
        "var_quantile_loss",
        "fz0_loss",
    ):
        family = result.xs(
            score_name,
            level="score_name",
        )
        expected = adjust_p_values_holm(
            p_values=family["dm_p_value"],
            significance_level=0.10,
        )

        assert family[
            "holm_adjusted_p_value"
        ].to_numpy() == pytest.approx(
            expected[
                "holm_adjusted_p_value"
            ].to_numpy()
        )
        assert family[
            "holm_reject_equal_accuracy"
        ].to_numpy(dtype=bool).tolist() == (
            expected[
                "reject_null_holm"
            ].to_numpy(dtype=bool).tolist()
        )


def test_model_diebold_mariano_requires_two_models(
) -> None:
    """Pairwise predictive-accuracy tests need two models."""

    forecasts = _create_dm_integration_forecasts(
        value_at_risk=0.020,
        expected_shortfall=0.030,
    )

    with pytest.raises(
        ValueError,
        match="At least two model",
    ):
        evaluate_models_diebold_mariano(
            forecasts_by_model={
                "rolling_gaussian": forecasts,
            },
        )


def test_model_diebold_mariano_rejects_misaligned_index(
) -> None:
    """Integrated DM tests require identical ordered forecast rows."""

    rolling_forecasts = _create_dm_integration_forecasts(
        value_at_risk=0.020,
        expected_shortfall=0.030,
    )
    ewma_forecasts = _create_dm_integration_forecasts(
        value_at_risk=0.025,
        expected_shortfall=0.035,
    ).iloc[::-1]

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        evaluate_models_diebold_mariano(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            },
        )


def test_model_diebold_mariano_rejects_different_losses(
) -> None:
    """Integrated DM tests require identical realized outcomes."""

    rolling_forecasts = _create_dm_integration_forecasts(
        value_at_risk=0.020,
        expected_shortfall=0.030,
    )
    ewma_forecasts = _create_dm_integration_forecasts(
        value_at_risk=0.025,
        expected_shortfall=0.035,
    )
    ewma_forecasts.iloc[
        0,
        ewma_forecasts.columns.get_loc(
            "realized_loss"
        ),
    ] += 0.001
    ewma_forecasts["var_exceedance"] = (
        ewma_forecasts["realized_loss"]
        > ewma_forecasts["value_at_risk"]
    )

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        evaluate_models_diebold_mariano(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            },
            bandwidth=1,
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




def _create_var_quantile_ranking_forecasts(
    value_at_risk: float,
) -> pd.DataFrame:
    """Create aligned forecasts for quantile-loss ranking tests."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=3,
        freq="D",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [0.95],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )
    realized_losses = [
        0.010,
        0.020,
        0.030,
    ]
    value_at_risk_values = [
        value_at_risk,
        value_at_risk,
        value_at_risk,
    ]

    return pd.DataFrame(
        {
            "realized_loss": realized_losses,
            "value_at_risk": (
                value_at_risk_values
            ),
            "var_exceedance": [
                loss > forecast
                for loss, forecast in zip(
                    realized_losses,
                    value_at_risk_values,
                    strict=True,
                )
            ],
        },
        index=index,
    )


def test_var_quantile_loss_summary_matches_manual_example(
) -> None:
    """Mean quantile loss is calculated separately by level."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=3,
        freq="D",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [
                0.95,
                0.99,
            ],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    forecasts = pd.DataFrame(
        {
            "realized_loss": [
                0.010,
                0.010,
                0.020,
                0.020,
                0.030,
                0.030,
            ],
            "value_at_risk": [
                0.020,
                0.025,
                0.020,
                0.025,
                0.020,
                0.025,
            ],
            "var_exceedance": [
                False,
                False,
                False,
                False,
                True,
                True,
            ],
        },
        index=index,
    )

    result = summarize_var_quantile_loss(
        forecasts=forecasts,
    )

    assert list(result.index) == [
        0.95,
        0.99,
    ]

    assert result.loc[
        0.95,
        "observation_count",
    ] == 3
    assert result.loc[
        0.99,
        "observation_count",
    ] == 3

    assert result.loc[
        0.95,
        "mean_var_quantile_loss",
    ] == pytest.approx(
        (
            0.0005
            + 0.0000
            + 0.0095
        )
        / 3.0
    )

    assert result.loc[
        0.99,
        "mean_var_quantile_loss",
    ] == pytest.approx(
        (
            0.00015
            + 0.00005
            + 0.00495
        )
        / 3.0
    )


def test_model_var_quantile_loss_ranks_aligned_models(
) -> None:
    """Aligned models are ranked by their mean quantile loss."""

    rolling_forecasts = (
        _create_var_quantile_ranking_forecasts(
            value_at_risk=0.020,
        )
    )
    ewma_forecasts = (
        _create_var_quantile_ranking_forecasts(
            value_at_risk=0.025,
        )
    )
    filtered_forecasts = (
        _create_var_quantile_ranking_forecasts(
            value_at_risk=0.021,
        )
    )

    result = evaluate_models_var_quantile_loss(
        forecasts_by_model={
            "rolling_gaussian": rolling_forecasts,
            "ewma_gaussian": ewma_forecasts,
            "filtered_historical": (
                filtered_forecasts
            ),
        }
    )

    assert result.index.names == [
        "model",
        "confidence_level",
    ]

    assert result.loc[
        (
            "rolling_gaussian",
            0.95,
        ),
        "mean_var_quantile_loss",
    ] == pytest.approx(
        0.0100 / 3.0
    )

    assert result.loc[
        (
            "ewma_gaussian",
            0.95,
        ),
        "mean_var_quantile_loss",
    ] == pytest.approx(
        0.00575 / 3.0
    )

    assert result.loc[
        (
            "filtered_historical",
            0.95,
        ),
        "mean_var_quantile_loss",
    ] == pytest.approx(
        0.00915 / 3.0
    )

    assert result.loc[
        (
            "ewma_gaussian",
            0.95,
        ),
        "var_quantile_loss_rank",
    ] == 1

    assert result.loc[
        (
            "filtered_historical",
            0.95,
        ),
        "var_quantile_loss_rank",
    ] == 2

    assert result.loc[
        (
            "rolling_gaussian",
            0.95,
        ),
        "var_quantile_loss_rank",
    ] == 3


def test_model_var_quantile_loss_rejects_misaligned_index(
) -> None:
    """Quantile-loss comparisons require identical ordered rows."""

    rolling_forecasts = (
        _create_var_quantile_ranking_forecasts(
            value_at_risk=0.020,
        )
    )
    ewma_forecasts = (
        _create_var_quantile_ranking_forecasts(
            value_at_risk=0.025,
        )
        .iloc[::-1]
    )

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        evaluate_models_var_quantile_loss(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_var_quantile_loss_rejects_different_losses(
) -> None:
    """Quantile-loss comparisons require identical realized losses."""

    rolling_forecasts = (
        _create_var_quantile_ranking_forecasts(
            value_at_risk=0.020,
        )
    )
    ewma_forecasts = (
        _create_var_quantile_ranking_forecasts(
            value_at_risk=0.025,
        )
    )
    ewma_forecasts.iloc[
        0,
        ewma_forecasts.columns.get_loc(
            "realized_loss"
        ),
    ] += 0.001
    ewma_forecasts["var_exceedance"] = (
        ewma_forecasts["realized_loss"]
        > ewma_forecasts["value_at_risk"]
    )

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        evaluate_models_var_quantile_loss(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_var_quantile_loss_rejects_empty_mapping(
) -> None:
    """At least one model is required for quantile-loss ranking."""

    with pytest.raises(
        ValueError,
        match="At least one model",
    ):
        evaluate_models_var_quantile_loss(
            forecasts_by_model={},
        )


def _create_fz0_forecasts(
    value_at_risk: float = 0.020,
    expected_shortfall: float = 0.030,
) -> pd.DataFrame:
    """Create controlled forecasts for daily FZ0 tests."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=3,
        freq="D",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [0.95],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    return pd.DataFrame(
        {
            "realized_loss": [
                0.010,
                0.020,
                0.030,
            ],
            "value_at_risk": [
                value_at_risk,
                value_at_risk,
                value_at_risk,
            ],
            "expected_shortfall": [
                expected_shortfall,
                expected_shortfall,
                expected_shortfall,
            ],
            "var_exceedance": [
                loss > value_at_risk
                for loss in [
                    0.010,
                    0.020,
                    0.030,
                ]
            ],
        },
        index=index,
    )


def test_fz0_losses_match_positive_loss_manual_example(
) -> None:
    """Positive-loss FZ0 scores match a manual calculation."""

    forecasts = _create_fz0_forecasts()

    result = calculate_fz0_losses(
        forecasts=forecasts,
    )

    non_breach_loss = (
        0.020 / 0.030
        + log(0.030)
        - 1.0
    )
    breach_tail_penalty = (
        (0.030 - 0.020)
        / (
            (1.0 - 0.95)
            * 0.030
        )
    )
    breach_loss = (
        non_breach_loss
        + breach_tail_penalty
    )

    assert result.index.equals(forecasts.index)

    assert result[
        "fz0_loss"
    ].to_numpy() == pytest.approx(
        [
            non_breach_loss,
            non_breach_loss,
            breach_loss,
        ]
    )

    assert non_breach_loss == pytest.approx(
        -3.839891743
    )
    assert breach_tail_penalty == pytest.approx(
        6.666666667
    )
    assert breach_loss == pytest.approx(
        2.826774924
    )


def test_fz0_losses_require_strictly_positive_es(
) -> None:
    """FZ0 scoring requires a positive Expected Shortfall scale."""

    forecasts = _create_fz0_forecasts()
    forecasts["value_at_risk"] = 0.0
    forecasts["expected_shortfall"] = 0.0
    forecasts["var_exceedance"] = (
        forecasts["realized_loss"]
        > forecasts["value_at_risk"]
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        calculate_fz0_losses(
            forecasts=forecasts,
        )


def test_fz0_loss_summary_matches_manual_example(
) -> None:
    """Mean FZ0 loss matches a controlled manual calculation."""

    forecasts = _create_fz0_forecasts()

    result = summarize_fz0_loss(
        forecasts=forecasts,
    )

    non_breach_loss = (
        0.020 / 0.030
        + log(0.030)
        - 1.0
    )
    breach_tail_penalty = (
        (0.030 - 0.020)
        / (
            (1.0 - 0.95)
            * 0.030
        )
    )
    expected_mean = (
        non_breach_loss
        + breach_tail_penalty / 3.0
    )

    assert result.loc[
        0.95,
        "observation_count",
    ] == 3
    assert result.loc[
        0.95,
        "mean_fz0_loss",
    ] == pytest.approx(expected_mean)


def test_model_fz0_loss_ranks_aligned_models(
) -> None:
    """Aligned models are ranked by their mean joint FZ0 loss."""

    rolling_forecasts = _create_fz0_forecasts(
        value_at_risk=0.020,
        expected_shortfall=0.030,
    )
    ewma_forecasts = _create_fz0_forecasts(
        value_at_risk=0.025,
        expected_shortfall=0.035,
    )
    filtered_forecasts = _create_fz0_forecasts(
        value_at_risk=0.021,
        expected_shortfall=0.031,
    )

    result = evaluate_models_fz0_loss(
        forecasts_by_model={
            "rolling_gaussian": rolling_forecasts,
            "ewma_gaussian": ewma_forecasts,
            "filtered_historical": (
                filtered_forecasts
            ),
        }
    )

    def expected_mean_fz0(
        value_at_risk: float,
        expected_shortfall: float,
    ) -> float:
        """Calculate the controlled three-session mean FZ0 loss."""

        non_breach_loss = (
            value_at_risk / expected_shortfall
            + log(expected_shortfall)
            - 1.0
        )
        tail_penalty = (
            (0.030 - value_at_risk)
            / (
                (1.0 - 0.95)
                * expected_shortfall
            )
        )
        return (
            non_breach_loss
            + tail_penalty / 3.0
        )

    assert result.index.names == [
        "model",
        "confidence_level",
    ]
    assert result.loc[
        ("rolling_gaussian", 0.95),
        "mean_fz0_loss",
    ] == pytest.approx(
        expected_mean_fz0(0.020, 0.030)
    )
    assert result.loc[
        ("ewma_gaussian", 0.95),
        "mean_fz0_loss",
    ] == pytest.approx(
        expected_mean_fz0(0.025, 0.035)
    )
    assert result.loc[
        ("filtered_historical", 0.95),
        "mean_fz0_loss",
    ] == pytest.approx(
        expected_mean_fz0(0.021, 0.031)
    )

    assert result.loc[
        ("ewma_gaussian", 0.95),
        "fz0_loss_rank",
    ] == 1
    assert result.loc[
        ("filtered_historical", 0.95),
        "fz0_loss_rank",
    ] == 2
    assert result.loc[
        ("rolling_gaussian", 0.95),
        "fz0_loss_rank",
    ] == 3


def test_model_fz0_loss_rejects_misaligned_index(
) -> None:
    """FZ0 comparisons require identical ordered forecast rows."""

    rolling_forecasts = _create_fz0_forecasts()
    ewma_forecasts = _create_fz0_forecasts(
        value_at_risk=0.025,
        expected_shortfall=0.035,
    ).iloc[::-1]

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        evaluate_models_fz0_loss(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_fz0_loss_rejects_different_losses(
) -> None:
    """FZ0 comparisons require identical realized losses."""

    rolling_forecasts = _create_fz0_forecasts()
    ewma_forecasts = _create_fz0_forecasts(
        value_at_risk=0.025,
        expected_shortfall=0.035,
    )
    ewma_forecasts.iloc[
        0,
        ewma_forecasts.columns.get_loc(
            "realized_loss"
        ),
    ] += 0.001
    ewma_forecasts["var_exceedance"] = (
        ewma_forecasts["realized_loss"]
        > ewma_forecasts["value_at_risk"]
    )

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        evaluate_models_fz0_loss(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_fz0_loss_rejects_empty_mapping(
) -> None:
    """At least one model is required for joint FZ0 ranking."""

    with pytest.raises(
        ValueError,
        match="At least one model",
    ):
        evaluate_models_fz0_loss(
            forecasts_by_model={},
        )


@pytest.mark.parametrize(
    ("exception_count", "expected_zone"),
    [
        (0, "green"),
        (4, "green"),
        (5, "amber"),
        (9, "amber"),
        (10, "red"),
        (250, "red"),
    ],
)
def test_basel_traffic_light_classifies_boundaries(
    exception_count: int,
    expected_zone: str,
) -> None:
    """Basel zones change at five and ten exceptions."""

    assert classify_basel_traffic_light(
        exception_count=exception_count,
    ) == expected_zone


@pytest.mark.parametrize(
    "bad_exception_count",
    [
        True,
        -1,
        251,
        4.0,
        "5",
    ],
)
def test_basel_traffic_light_rejects_invalid_counts(
    bad_exception_count: object,
) -> None:
    """A 250-session exception count must be an integer in range."""

    with pytest.raises(
        ValueError,
        match="Exception count must be an integer",
    ):
        classify_basel_traffic_light(
            exception_count=bad_exception_count,
        )


def _create_basel_traffic_light_forecasts(
    exception_positions: set[int],
    period_count: int = 250,
    confidence_level: float = 0.99,
) -> pd.DataFrame:
    """Create controlled forecasts for rolling traffic-light tests."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=period_count,
        freq="B",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [confidence_level],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )
    realized_losses = [
        2.0 if position in exception_positions else 0.5
        for position in range(period_count)
    ]
    value_at_risk = [1.0] * period_count

    return pd.DataFrame(
        {
            "realized_loss": realized_losses,
            "value_at_risk": value_at_risk,
            "var_exceedance": [
                loss > forecast
                for loss, forecast in zip(
                    realized_losses,
                    value_at_risk,
                    strict=True,
                )
            ],
        },
        index=index,
    )


@pytest.mark.parametrize(
    ("exception_count", "expected_zone"),
    [
        (4, "green"),
        (5, "amber"),
        (9, "amber"),
        (10, "red"),
    ],
)
def test_basel_traffic_light_path_uses_complete_window(
    exception_count: int,
    expected_zone: str,
) -> None:
    """A complete 250-session window receives the expected zone."""

    forecasts = _create_basel_traffic_light_forecasts(
        exception_positions=set(
            range(exception_count)
        ),
    )

    result = calculate_basel_traffic_light_path(
        forecasts=forecasts,
    )

    assert len(result) == 1
    assert result.index.name == "forecast_date"
    assert result.index[0] == forecasts.index[
        -1
    ][0]
    assert result.iloc[0][
        "window_start_date"
    ] == forecasts.index[0][0]
    assert result.iloc[0][
        "window_observation_count"
    ] == 250
    assert result.iloc[0][
        "confidence_level"
    ] == pytest.approx(0.99)
    assert result.iloc[0][
        "exception_count"
    ] == exception_count
    assert result.iloc[0][
        "traffic_light_zone"
    ] == expected_zone


def test_basel_traffic_light_path_drops_expired_exception(
) -> None:
    """An exception leaves the count after rolling beyond 250 dates."""

    forecasts = _create_basel_traffic_light_forecasts(
        exception_positions={0, 1, 2, 3, 4},
        period_count=251,
    )

    result = calculate_basel_traffic_light_path(
        forecasts=forecasts,
    )

    assert len(result) == 2
    assert result["exception_count"].tolist() == [
        5,
        4,
    ]
    assert result["traffic_light_zone"].tolist() == [
        "amber",
        "green",
    ]
    assert result["window_start_date"].tolist() == [
        forecasts.index[0][0],
        forecasts.index[1][0],
    ]


def test_basel_traffic_light_path_rejects_partial_window(
) -> None:
    """Fewer than 250 forecast sessions cannot form a Basel window."""

    forecasts = _create_basel_traffic_light_forecasts(
        exception_positions=set(),
        period_count=249,
    )

    with pytest.raises(
        ValueError,
        match="At least 250 forecast sessions",
    ):
        calculate_basel_traffic_light_path(
            forecasts=forecasts,
        )


def test_basel_traffic_light_path_requires_99_percent_var(
) -> None:
    """Basel traffic-light thresholds apply only to 99% VaR rows."""

    forecasts = _create_basel_traffic_light_forecasts(
        exception_positions=set(),
        confidence_level=0.95,
    )

    with pytest.raises(
        ValueError,
        match="contain 99% VaR rows",
    ):
        calculate_basel_traffic_light_path(
            forecasts=forecasts,
        )


def test_basel_traffic_light_path_rejects_reverse_dates(
) -> None:
    """Rolling windows require increasing forecast dates."""

    forecasts = _create_basel_traffic_light_forecasts(
        exception_positions=set(),
    ).iloc[::-1]

    with pytest.raises(
        ValueError,
        match="ordered by increasing forecast date",
    ):
        calculate_basel_traffic_light_path(
            forecasts=forecasts,
        )


def _create_aligned_basel_model_forecasts(
    exception_positions: set[int],
    period_count: int = 251,
) -> pd.DataFrame:
    """Create model forecasts with common realized losses."""

    dates = pd.date_range(
        start="2025-01-02",
        periods=period_count,
        freq="B",
    )
    index = pd.MultiIndex.from_product(
        [
            dates,
            [0.99],
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )
    realized_losses = [1.0] * period_count
    value_at_risk = [
        0.5 if position in exception_positions else 1.5
        for position in range(period_count)
    ]

    return pd.DataFrame(
        {
            "realized_loss": realized_losses,
            "value_at_risk": value_at_risk,
            "var_exceedance": [
                loss > forecast
                for loss, forecast in zip(
                    realized_losses,
                    value_at_risk,
                    strict=True,
                )
            ],
        },
        index=index,
    )


def test_model_basel_traffic_light_combines_aligned_paths(
) -> None:
    """Aligned model forecasts produce one combined rolling table."""

    forecasts_by_model = {
        "rolling_gaussian": (
            _create_aligned_basel_model_forecasts(
                exception_positions=set(range(10)),
            )
        ),
        "ewma_gaussian": (
            _create_aligned_basel_model_forecasts(
                exception_positions=set(range(5)),
            )
        ),
        "filtered_historical": (
            _create_aligned_basel_model_forecasts(
                exception_positions={1, 2, 3, 4},
            )
        ),
    }

    result = evaluate_models_basel_traffic_light(
        forecasts_by_model=forecasts_by_model,
    )

    assert result.index.names == [
        "model",
        "forecast_date",
    ]
    assert len(result) == 6
    assert result.loc[
        "rolling_gaussian",
        "exception_count",
    ].tolist() == [10, 9]
    assert result.loc[
        "rolling_gaussian",
        "traffic_light_zone",
    ].tolist() == ["red", "amber"]
    assert result.loc[
        "ewma_gaussian",
        "traffic_light_zone",
    ].tolist() == ["amber", "green"]
    assert result.loc[
        "filtered_historical",
        "traffic_light_zone",
    ].tolist() == ["green", "green"]


def test_model_basel_traffic_light_summary_matches_paths(
) -> None:
    """The compact summary reports latest, worst, and zone durations."""

    paths = evaluate_models_basel_traffic_light(
        forecasts_by_model={
            "rolling_gaussian": (
                _create_aligned_basel_model_forecasts(
                    exception_positions=set(range(10)),
                )
            ),
            "ewma_gaussian": (
                _create_aligned_basel_model_forecasts(
                    exception_positions=set(range(5)),
                )
            ),
            "filtered_historical": (
                _create_aligned_basel_model_forecasts(
                    exception_positions={1, 2, 3, 4},
                )
            ),
        }
    )

    result = summarize_models_basel_traffic_light(
        traffic_light_paths=paths,
    )

    assert list(result.index) == [
        "rolling_gaussian",
        "ewma_gaussian",
        "filtered_historical",
    ]

    rolling = result.loc["rolling_gaussian"]
    assert rolling["window_count"] == 2
    assert rolling["latest_exception_count"] == 9
    assert rolling["latest_zone"] == "amber"
    assert rolling["maximum_exception_count"] == 10
    assert rolling["worst_zone"] == "red"
    assert rolling["green_window_count"] == 0
    assert rolling["amber_window_count"] == 1
    assert rolling["red_window_count"] == 1
    assert rolling["green_window_rate"] == pytest.approx(
        0.0
    )
    assert rolling["amber_window_rate"] == pytest.approx(
        0.5
    )
    assert rolling["red_window_rate"] == pytest.approx(
        0.5
    )
    assert rolling["maximum_exception_date"] == (
        paths.loc["rolling_gaussian"].index[0]
    )
    assert rolling["first_amber_date"] == (
        paths.loc["rolling_gaussian"].index[1]
    )
    assert rolling["first_red_date"] == (
        paths.loc["rolling_gaussian"].index[0]
    )

    ewma = result.loc["ewma_gaussian"]
    assert ewma["latest_zone"] == "green"
    assert ewma["maximum_exception_count"] == 5
    assert ewma["worst_zone"] == "amber"
    assert ewma["green_window_count"] == 1
    assert ewma["amber_window_count"] == 1
    assert ewma["red_window_count"] == 0
    assert pd.isna(ewma["first_red_date"])

    filtered = result.loc["filtered_historical"]
    assert filtered["latest_zone"] == "green"
    assert filtered["maximum_exception_count"] == 4
    assert filtered["worst_zone"] == "green"
    assert filtered["green_window_count"] == 2
    assert pd.isna(filtered["first_amber_date"])
    assert pd.isna(filtered["first_red_date"])


def test_model_basel_traffic_light_rejects_misaligned_index(
) -> None:
    """Combined traffic lights require identical ordered rows."""

    rolling_forecasts = (
        _create_aligned_basel_model_forecasts(
            exception_positions=set(),
        )
    )
    ewma_forecasts = (
        _create_aligned_basel_model_forecasts(
            exception_positions=set(),
        ).iloc[::-1]
    )

    with pytest.raises(
        ValueError,
        match="same ordered forecast index",
    ):
        evaluate_models_basel_traffic_light(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_model_basel_traffic_light_rejects_different_losses(
) -> None:
    """Combined traffic lights require identical realized losses."""

    rolling_forecasts = (
        _create_aligned_basel_model_forecasts(
            exception_positions=set(),
        )
    )
    ewma_forecasts = (
        _create_aligned_basel_model_forecasts(
            exception_positions=set(),
        )
    )
    ewma_forecasts.iloc[
        0,
        ewma_forecasts.columns.get_loc(
            "realized_loss"
        ),
    ] += 0.1

    with pytest.raises(
        ValueError,
        match="identical realized losses",
    ):
        evaluate_models_basel_traffic_light(
            forecasts_by_model={
                "rolling_gaussian": rolling_forecasts,
                "ewma_gaussian": ewma_forecasts,
            }
        )


def test_basel_traffic_light_summary_rejects_wrong_zone(
) -> None:
    """Stored traffic-light zones must agree with exception counts."""

    paths = evaluate_models_basel_traffic_light(
        forecasts_by_model={
            "rolling_gaussian": (
                _create_aligned_basel_model_forecasts(
                    exception_positions=set(),
                )
            ),
        }
    )
    paths.iloc[
        0,
        paths.columns.get_loc(
            "traffic_light_zone"
        ),
    ] = "red"

    with pytest.raises(
        ValueError,
        match="must agree with exception counts",
    ):
        summarize_models_basel_traffic_light(
            traffic_light_paths=paths,
        )
