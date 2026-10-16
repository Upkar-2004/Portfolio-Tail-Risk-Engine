"""Tests for the descriptive predictive-score rank figure."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailrisk.visualization import (
    create_subperiod_predictive_first_place_figure,
    create_subperiod_predictive_rank_figure,
    prepare_subperiod_predictive_first_place_counts,
    prepare_subperiod_predictive_rank_matrices,
    save_subperiod_predictive_first_place_figure,
    save_subperiod_predictive_rank_figure,
)


def create_predictive_accuracy_table() -> pd.DataFrame:
    """Create complete controlled predictive-score results."""

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
    records = []

    for period_index, period_id in enumerate(period_ids):
        for level_index, confidence_level in enumerate(
            confidence_levels
        ):
            ql_order = np.roll(
                np.arange(1, 4),
                period_index + level_index,
            )
            fz0_order = np.roll(
                np.arange(1, 4),
                period_index + (2 * level_index) + 1,
            )

            for model_index, model in enumerate(models):
                ql_rank = int(ql_order[model_index])
                fz0_rank = int(fz0_order[model_index])
                records.append(
                    {
                        "period_id": period_id,
                        "model": model,
                        "confidence_level": confidence_level,
                        "period_type": (
                            "nber_recession"
                            if period_id == "nber_recession_2020"
                            else "calendar_year"
                        ),
                        "mean_var_quantile_loss": (
                            0.001 * ql_rank
                        ),
                        "var_quantile_loss_rank": ql_rank,
                        "mean_fz0_loss": float(fz0_rank - 5),
                        "fz0_loss_rank": fz0_rank,
                    }
                )

    return pd.DataFrame.from_records(records).set_index(
        [
            "period_id",
            "model",
            "confidence_level",
        ]
    )


def test_prepare_predictive_rank_matrices_maps_fixed_layout() -> None:
    """Ranks map to score blocks, years, models, and levels."""

    predictive_accuracy = create_predictive_accuracy_table()
    original = predictive_accuracy.copy(deep=True)
    calendar, recession = (
        prepare_subperiod_predictive_rank_matrices(
            predictive_accuracy
        )
    )

    assert list(calendar) == [0.95, 0.975, 0.99]
    assert calendar[0.95].shape == (14, 6)
    assert recession[0.99].shape == (1, 6)
    assert list(calendar[0.95].columns.names) == [
        "score_family",
        "model",
    ]
    assert calendar[0.975].loc[
        "calendar_2018",
        ("QL", "ewma_gaussian_monte_carlo"),
    ] == predictive_accuracy.loc[
        (
            "calendar_2018",
            "ewma_gaussian_monte_carlo",
            0.975,
        ),
        "var_quantile_loss_rank",
    ]
    assert recession[0.99].loc[
        "nber_recession_2020",
        ("FZ0", "filtered_historical_simulation"),
    ] == predictive_accuracy.loc[
        (
            "nber_recession_2020",
            "filtered_historical_simulation",
            0.99,
        ),
        "fz0_loss_rank",
    ]
    pd.testing.assert_frame_equal(
        predictive_accuracy,
        original,
    )


def test_prepare_predictive_rank_matrices_rejects_bad_rank() -> None:
    """A rank inconsistent with the saved means is rejected."""

    predictive_accuracy = create_predictive_accuracy_table()
    predictive_accuracy.loc[
        (
            "calendar_2019",
            "ewma_gaussian_monte_carlo",
            0.99,
        ),
        "var_quantile_loss_rank",
    ] = 1

    with pytest.raises(ValueError, match="must match"):
        prepare_subperiod_predictive_rank_matrices(
            predictive_accuracy
        )


def test_prepare_first_place_counts_excludes_stress_period() -> None:
    """First-place counts use the fourteen calendar years only."""

    counts = prepare_subperiod_predictive_first_place_counts(
        create_predictive_accuracy_table()
    )

    assert counts.shape == (3, 6)
    assert list(counts.index) == [0.95, 0.975, 0.99]

    for score_family in ("QL", "FZ0"):
        assert (
            counts.loc[:, score_family].sum(axis=1) == 14
        ).all()


def test_create_predictive_summary_uses_two_score_panels() -> None:
    """The main figure compares first-place counts by score."""

    figure = create_subperiod_predictive_first_place_figure(
        create_predictive_accuracy_table()
    )

    assert len(figure.axes) == 2
    assert [axis.get_title(loc="left") for axis in figure.axes] == [
        "(a)  Quantile loss (VaR)",
        "(b)  FZ0 loss (joint VaR–ES)",
    ]
    assert all(len(axis.lines) == 4 for axis in figure.axes)
    assert len(figure.legends) == 0
    assert figure.get_size_inches().tolist() == [7.5, 3.35]


def test_create_predictive_rank_figure_groups_score_first() -> None:
    """The supporting grid groups confidence levels within scores."""

    figure = create_subperiod_predictive_rank_figure(
        create_predictive_accuracy_table()
    )

    assert len(figure.axes) == 4
    calendar_axes = figure.axes[0::2]
    recession_axes = figure.axes[1::2]
    assert [axis.get_title(loc="left") for axis in calendar_axes] == [
        "(a)  Quantile loss (VaR)",
        "(b)  FZ0 loss (joint VaR–ES)",
    ]
    assert all(
        axis.images[0].get_array().shape == (14, 9, 4)
        for axis in calendar_axes
    )
    assert all(
        axis.images[0].get_array().shape == (1, 9, 4)
        for axis in recession_axes
    )
    assert len(figure.legends) == 0
    assert figure.get_size_inches().tolist() == [7.5, 5.65]


def test_save_predictive_rank_figure_creates_png_and_svg(
    tmp_path: Path,
) -> None:
    """The supporting rank grid is saved in publication formats."""

    png_path = tmp_path / "predictive_rank.png"
    svg_path = tmp_path / "predictive_rank.svg"
    saved_png, saved_svg = save_subperiod_predictive_rank_figure(
        predictive_accuracy=create_predictive_accuracy_table(),
        png_output_path=png_path,
        svg_output_path=svg_path,
    )

    assert saved_png == png_path
    assert saved_svg == svg_path
    assert png_path.stat().st_size > 25_000
    assert svg_path.stat().st_size > 25_000
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Annual Predictive-Score Ranks" in svg_text
    assert all(
        line == line.rstrip()
        for line in svg_text.splitlines()
    )


def test_save_predictive_summary_creates_png_and_svg(
    tmp_path: Path,
) -> None:
    """The main summary is saved in publication formats."""

    png_path = tmp_path / "predictive_summary.png"
    svg_path = tmp_path / "predictive_summary.svg"
    saved_png, saved_svg = (
        save_subperiod_predictive_first_place_figure(
            predictive_accuracy=create_predictive_accuracy_table(),
            png_output_path=png_path,
            svg_output_path=svg_path,
        )
    )

    assert saved_png == png_path
    assert saved_svg == svg_path
    assert png_path.stat().st_size > 20_000
    assert svg_path.stat().st_size > 20_000
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "Predictive-Score First-Place Counts" in svg_text
    assert all(
        line == line.rstrip()
        for line in svg_text.splitlines()
    )
