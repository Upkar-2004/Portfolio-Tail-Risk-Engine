"""Tests for research-result figures."""

from pathlib import Path

import numpy as np
import pandas as pd

from tailrisk.models import calculate_gaussian_var_es
from tailrisk.simulation import calculate_empirical_var_es
from tailrisk.visualization import (
    create_basel_traffic_light_figure,
    save_basel_traffic_light_figure,
    save_gaussian_tail_risk_figure,
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
