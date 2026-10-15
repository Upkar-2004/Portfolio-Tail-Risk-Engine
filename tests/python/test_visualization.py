"""Tests for research-result figures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailrisk.models import calculate_gaussian_var_es
from tailrisk.simulation import calculate_empirical_var_es
from tailrisk.visualization import (
    create_basel_traffic_light_figure,
    create_subperiod_exception_interval_figure,
    create_subperiod_exception_rate_figure,
    prepare_subperiod_exception_interval_tables,
    prepare_subperiod_exception_rate_matrices,
    save_basel_traffic_light_figure,
    save_gaussian_tail_risk_figure,
    save_subperiod_exception_interval_figure,
    save_subperiod_exception_rate_figure,
)


def _create_basel_figure_tables(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create controlled traffic-light paths and summaries."""

    models = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
        "filtered_historical_simulation",
    ]
    dates = pd.date_range(
        "2025-01-02",
        periods=5,
        freq="B",
    )
    count_paths = {
        models[0]: [3, 5, 9, 10, 12],
        models[1]: [2, 4, 5, 7, 6],
        models[2]: [1, 2, 3, 4, 3],
    }
    rows = []

    for model_name in models:
        for forecast_date, count in zip(
            dates,
            count_paths[model_name],
            strict=True,
        ):
            rows.append(
                {
                    "model": model_name,
                    "forecast_date": forecast_date,
                    "exception_count": count,
                    "traffic_light_zone": (
                        "green"
                        if count <= 4
                        else "amber"
                        if count <= 9
                        else "red"
                    ),
                }
            )

    paths = pd.DataFrame(rows).set_index(
        ["model", "forecast_date"]
    )
    summary = pd.DataFrame(
        {
            "latest_exception_count": [12, 6, 3],
            "latest_zone": ["red", "amber", "green"],
            "maximum_exception_count": [12, 7, 4],
            "maximum_exception_date": [
                dates[4],
                dates[3],
                dates[3],
            ],
        },
        index=pd.Index(models, name="model"),
    )

    return paths, summary


def _create_subperiod_coverage_table() -> pd.DataFrame:
    """Create a complete controlled subperiod coverage table."""

    period_ids = [
        *(f"calendar_{year}" for year in range(2012, 2026)),
        "nber_recession_2020",
    ]
    models = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
        "filtered_historical_simulation",
    ]
    confidence_levels = [0.95, 0.975, 0.99]
    index = pd.MultiIndex.from_product(
        [period_ids, models, confidence_levels],
        names=[
            "period_id",
            "model",
            "confidence_level",
        ],
    )
    row_numbers = np.arange(len(index), dtype=int)
    observation_counts = np.full(len(index), 250)
    exception_counts = (row_numbers % 15) + 1
    observed_rates = (
        exception_counts / observation_counts
    )
    confidence_values = index.get_level_values(
        "confidence_level"
    ).to_numpy(dtype=float)
    nominal_rates = 1.0 - confidence_values
    deviations = observed_rates - nominal_rates
    period_types = [
        (
            "nber_recession"
            if period_id == "nber_recession_2020"
            else "calendar_year"
        )
        for period_id in index.get_level_values("period_id")
    ]

    return pd.DataFrame(
        {
            "period_type": period_types,
            "observation_count": observation_counts,
            "nominal_exceedance_rate": nominal_rates,
            "observed_exceedance_count": exception_counts,
            "observed_exceedance_rate": observed_rates,
            "observed_minus_expected_rate": deviations,
            "interval_confidence_level": 0.95,
            "exact_interval_lower_bound": np.maximum(
                0.0,
                observed_rates - 0.01,
            ),
            "exact_interval_upper_bound": np.minimum(
                1.0,
                observed_rates + 0.02,
            ),
        },
        index=index,
    )


def test_save_gaussian_tail_risk_figure_creates_png(
    tmp_path: Path,
) -> None:
    """The Gaussian comparison figure is saved as a valid PNG."""

    rng = np.random.default_rng(12345)

    analytic_mean_return = 0.001
    analytic_volatility = 0.02

    simulated_returns = (
        analytic_mean_return
        + analytic_volatility * rng.standard_normal(20_000)
    )
    simulated_losses = -simulated_returns

    confidence_levels = (0.95, 0.975, 0.99)

    analytic_forecasts = tuple(
        calculate_gaussian_var_es(
            mean_return=analytic_mean_return,
            volatility=analytic_volatility,
            confidence_level=confidence_level,
        )
        for confidence_level in confidence_levels
    )

    empirical_estimates = tuple(
        calculate_empirical_var_es(
            simulated_losses,
            confidence_level=confidence_level,
        )
        for confidence_level in confidence_levels
    )

    output_path = tmp_path / "gaussian_tail_risk.png"

    saved_path = save_gaussian_tail_risk_figure(
        losses=simulated_losses,
        analytic_mean_return=analytic_mean_return,
        analytic_volatility=analytic_volatility,
        analytic_forecasts=analytic_forecasts,
        empirical_estimates=empirical_estimates,
        realized_loss=0.025,
        forecast_date=pd.Timestamp("2025-12-31"),
        estimation_start_date=pd.Timestamp("2024-01-24"),
        estimation_end_date=pd.Timestamp("2025-12-30"),
        estimation_observation_count=504,
        asset_count=11,
        scenario_count=20_000,
        random_seed=12345,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_create_basel_traffic_light_figure_uses_aligned_panels(
) -> None:
    """Each model receives one panel on common axes and thresholds."""

    paths, summary = _create_basel_figure_tables()

    figure = create_basel_traffic_light_figure(
        traffic_light_paths=paths,
        summary=summary,
    )

    assert len(figure.axes) == 3
    assert figure.axes[0].get_shared_x_axes().joined(
        figure.axes[0],
        figure.axes[1],
    )
    assert figure.axes[0].get_shared_y_axes().joined(
        figure.axes[0],
        figure.axes[2],
    )
    assert [
        axis.get_title(loc="left")
        for axis in figure.axes
    ] == [
        "Rolling Gaussian",
        "EWMA Gaussian",
        "Filtered Historical Simulation",
    ]

    for axis, model_name in zip(
        figure.axes,
        summary.index,
        strict=True,
    ):
        count_lines = [
            line
            for line in axis.lines
            if line.get_label() == "Rolling exception count"
        ]
        expected_counts = paths.xs(
            model_name,
            level="model",
        )["exception_count"].to_numpy()

        assert len(count_lines) == 1
        np.testing.assert_array_equal(
            count_lines[0].get_ydata(),
            expected_counts,
        )
        assert len(axis.patches) == 3
        assert axis.get_ylim() == figure.axes[0].get_ylim()


def test_save_basel_traffic_light_figure_creates_png_and_svg(
    tmp_path: Path,
) -> None:
    """The publication figure is saved in raster and vector formats."""

    paths, summary = _create_basel_figure_tables()
    png_path = tmp_path / "basel_traffic_light.png"
    svg_path = tmp_path / "basel_traffic_light.svg"

    saved_png, saved_svg = save_basel_traffic_light_figure(
        traffic_light_paths=paths,
        summary=summary,
        png_output_path=png_path,
        svg_output_path=svg_path,
    )

    assert saved_png == png_path
    assert saved_svg == svg_path
    assert png_path.stat().st_size > 10_000
    assert svg_path.stat().st_size > 10_000
    assert png_path.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert "<svg" in svg_path.read_text(
        encoding="utf-8"
    )


def test_prepare_subperiod_exception_rate_matrices_maps_values(
) -> None:
    """Coverage rows map to fixed model, year, and level positions."""

    coverage = _create_subperiod_coverage_table()
    original = coverage.copy(deep=True)
    calendar, recession = (
        prepare_subperiod_exception_rate_matrices(
            coverage
        )
    )

    assert list(calendar) == [0.95, 0.975, 0.99]
    assert calendar[0.95].shape == (3, 14)
    assert recession[0.99].shape == (3, 1)
    assert calendar[0.975].loc[
        "ewma_gaussian_monte_carlo",
        "calendar_2018",
    ] == coverage.loc[
        (
            "calendar_2018",
            "ewma_gaussian_monte_carlo",
            0.975,
        ),
        "observed_minus_expected_rate",
    ]
    assert recession[0.99].loc[
        "filtered_historical_simulation",
        "nber_recession_2020",
    ] == coverage.loc[
        (
            "nber_recession_2020",
            "filtered_historical_simulation",
            0.99,
        ),
        "observed_minus_expected_rate",
    ]
    pd.testing.assert_frame_equal(
        coverage,
        original,
    )


def test_create_subperiod_exception_rate_figure_uses_six_panels(
) -> None:
    """Three confidence levels each receive calendar and NBER panels."""

    coverage = _create_subperiod_coverage_table()
    figure = create_subperiod_exception_rate_figure(
        coverage
    )

    assert len(figure.axes) == 7
    assert figure.axes[0].images[0].get_array().shape == (
        3,
        14,
    )
    assert figure.axes[1].images[0].get_array().shape == (
        3,
        1,
    )
    assert figure.axes[0].get_ylabel() == "95.0% VaR"
    assert figure.axes[2].get_ylabel() == "97.5% VaR"
    assert figure.axes[4].get_ylabel() == "99.0% VaR"
    assert figure.axes[0].images[0].norm.vmin == (
        -figure.axes[0].images[0].norm.vmax
    )
    assert figure.axes[1].images[0].norm is (
        figure.axes[0].images[0].norm
    )


def test_prepare_subperiod_exception_rate_matrices_rejects_gap(
) -> None:
    """A missing predeclared cell cannot silently become blank."""

    coverage = _create_subperiod_coverage_table().drop(
        index=(
            "calendar_2019",
            "ewma_gaussian_monte_carlo",
            0.99,
        )
    )

    with pytest.raises(
        ValueError,
        match="must contain exactly",
    ):
        prepare_subperiod_exception_rate_matrices(
            coverage
        )


def test_save_subperiod_exception_rate_figure_creates_png_and_svg(
    tmp_path: Path,
) -> None:
    """The subperiod figure is saved in publication formats."""

    coverage = _create_subperiod_coverage_table()
    png_path = tmp_path / "subperiod_exception_rates.png"
    svg_path = tmp_path / "subperiod_exception_rates.svg"

    saved_png, saved_svg = (
        save_subperiod_exception_rate_figure(
            var_coverage=coverage,
            png_output_path=png_path,
            svg_output_path=svg_path,
        )
    )

    assert saved_png == png_path
    assert saved_svg == svg_path
    assert png_path.stat().st_size > 25_000
    assert svg_path.stat().st_size > 25_000
    assert png_path.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "NBER recession" in svg_text


def test_prepare_subperiod_exception_interval_tables_maps_rows(
) -> None:
    """Interval inputs retain fixed period and model ordering."""

    coverage = _create_subperiod_coverage_table()
    original = coverage.copy(deep=True)
    calendar, recession = (
        prepare_subperiod_exception_interval_tables(
            coverage
        )
    )

    assert list(calendar) == [0.95, 0.975, 0.99]
    assert calendar[0.95].shape == (42, 6)
    assert recession[0.99].shape == (3, 6)
    assert calendar[0.975].loc[
        (
            "calendar_2018",
            "ewma_gaussian_monte_carlo",
        ),
        "observed_exceedance_rate",
    ] == coverage.loc[
        (
            "calendar_2018",
            "ewma_gaussian_monte_carlo",
            0.975,
        ),
        "observed_exceedance_rate",
    ]
    pd.testing.assert_frame_equal(coverage, original)


def test_create_subperiod_exception_interval_figure_uses_panels(
) -> None:
    """Calendar and recession intervals use three level panels."""

    coverage = _create_subperiod_coverage_table()
    figure = create_subperiod_exception_interval_figure(
        coverage
    )

    assert len(figure.axes) == 6
    calendar_axes = figure.axes[:3]
    recession_axes = figure.axes[3:]
    assert [
        axis.get_title()
        for axis in calendar_axes
    ] == [
        "95% VaR",
        "97.5% VaR",
        "99% VaR",
    ]
    assert calendar_axes[0].get_yticklabels()[0].get_text() == (
        "2012"
    )
    assert all(
        tick.get_text() == ""
        for tick in calendar_axes[1].get_yticklabels()
    )
    assert [axis.get_xlim() for axis in calendar_axes] == [
        (-0.004, 0.20),
        (-0.004, 0.16),
        (-0.004, 0.125),
    ]
    assert all(
        np.allclose(axis.get_xlim(), (-0.004, 0.40))
        for axis in recession_axes
    )
    assert sum(
        bool(axis.get_xlabel()) for axis in calendar_axes
    ) == 1
    assert sum(
        bool(axis.get_xlabel()) for axis in recession_axes
    ) == 1
    assert all(axis.patches for axis in recession_axes)
    assert all(
        any(
            text.get_text().startswith("nominal ")
            for text in axis.texts
        )
        for axis in calendar_axes
    )
    assert all(
        "Nominal" not in text.get_text()
        for axis in figure.axes
        for text in axis.texts
    )

    for calendar_axis, expected_target in zip(
        calendar_axes,
        [0.05, 0.025, 0.01],
        strict=True,
    ):
        target_lines = [
            line
            for line in calendar_axis.lines
            if len(line.get_xdata()) == 2
            and np.allclose(
                line.get_xdata(),
                expected_target,
            )
        ]
        assert len(target_lines) == 1


def test_prepare_subperiod_exception_intervals_rejects_bad_rate(
) -> None:
    """Observed rates must agree with their exception counts."""

    coverage = _create_subperiod_coverage_table()
    coverage.loc[
        (
            "calendar_2019",
            "ewma_gaussian_monte_carlo",
            0.99,
        ),
        "observed_exceedance_rate",
    ] = 0.50

    with pytest.raises(
        ValueError,
        match="must equal exception counts",
    ):
        prepare_subperiod_exception_interval_tables(
            coverage
        )


def test_save_subperiod_exception_interval_figure_creates_formats(
    tmp_path: Path,
) -> None:
    """The exact interval figure is saved as PNG and SVG."""

    coverage = _create_subperiod_coverage_table()
    png_path = tmp_path / "subperiod_exception_intervals.png"
    svg_path = tmp_path / "subperiod_exception_intervals.svg"

    saved_png, saved_svg = (
        save_subperiod_exception_interval_figure(
            var_coverage=coverage,
            png_output_path=png_path,
            svg_output_path=svg_path,
        )
    )

    assert saved_png == png_path
    assert saved_svg == svg_path
    assert png_path.stat().st_size > 25_000
    assert svg_path.stat().st_size > 25_000
    assert png_path.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Clopper-Pearson" in svg_text
