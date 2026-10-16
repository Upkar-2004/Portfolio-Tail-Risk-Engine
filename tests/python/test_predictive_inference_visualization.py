"""Tests for the full-sample predictive-accuracy inference figure."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tailrisk.figures.inference import (
    create_predictive_inference_figure,
    prepare_predictive_inference_table,
    save_predictive_inference_figure,
)


MODELS = [
    "rolling_gaussian_monte_carlo",
    "ewma_gaussian_monte_carlo",
    "filtered_historical_simulation",
]
LEVELS = [0.95, 0.975, 0.99]
SCORES = ["var_quantile_loss", "fz0_loss"]
PAIRS = [
    (MODELS[0], MODELS[1]),
    (MODELS[0], MODELS[2]),
    (MODELS[1], MODELS[2]),
]


def create_model_means() -> pd.DataFrame:
    """Create positive QL means and negative FZ0 means."""

    records = []

    for level_index, confidence_level in enumerate(LEVELS):
        for model_index, model in enumerate(MODELS):
            records.append(
                {
                    "model": model,
                    "confidence_level": confidence_level,
                    "observation_count": 100,
                    "mean_var_quantile_loss": (
                        0.001
                        + (0.0001 * model_index)
                        + (0.00001 * level_index)
                    ),
                    "mean_fz0_loss": (
                        -3.0
                        - (0.2 * model_index)
                        - (0.1 * level_index)
                    ),
                }
            )

    return pd.DataFrame.from_records(records).set_index(
        ["model", "confidence_level"]
    )


def create_comparisons(
    bandwidth: int = 8,
    flip_one_decision: bool = False,
) -> pd.DataFrame:
    """Create complete internally consistent pairwise results."""

    means = create_model_means()
    records = []

    for score_name in SCORES:
        mean_column = (
            "mean_var_quantile_loss"
            if score_name == "var_quantile_loss"
            else "mean_fz0_loss"
        )

        for confidence_level in LEVELS:
            for pair_index, (model_a, model_b) in enumerate(PAIRS):
                difference = float(
                    means.loc[(model_a, confidence_level), mean_column]
                    - means.loc[(model_b, confidence_level), mean_column]
                )
                standard_error = abs(difference) / (2.5 + pair_index)
                statistic = difference / standard_error
                adjusted_p_value = 0.02 if pair_index == 0 else 0.20
                reject = adjusted_p_value <= 0.05

                if (
                    flip_one_decision
                    and score_name == "var_quantile_loss"
                    and confidence_level == 0.99
                    and pair_index == 1
                ):
                    adjusted_p_value = 0.03
                    reject = True

                records.append(
                    {
                        "score_name": score_name,
                        "confidence_level": confidence_level,
                        "model_a": model_a,
                        "model_b": model_b,
                        "observation_count": 100,
                        "mean_score_difference": difference,
                        "hac_long_run_variance": 0.01,
                        "hac_mean_variance": standard_error**2,
                        "hac_standard_error": standard_error,
                        "hac_bandwidth": bandwidth,
                        "dm_statistic": statistic,
                        "dm_p_value": min(adjusted_p_value, 0.10),
                        "significance_level": 0.05,
                        "dm_reject_equal_accuracy": (
                            min(adjusted_p_value, 0.10) <= 0.05
                        ),
                        "holm_adjusted_p_value": adjusted_p_value,
                        "holm_reject_equal_accuracy": reject,
                    }
                )

    return pd.DataFrame.from_records(records).set_index(
        [
            "score_name",
            "confidence_level",
            "model_a",
            "model_b",
        ]
    )


def test_prepare_inference_table_applies_challenger_orientation() -> None:
    """All displayed rows use challenger-minus-comparator ordering."""

    baseline = create_comparisons()
    sensitivity = create_comparisons(
        bandwidth=0,
        flip_one_decision=True,
    )
    prepared = prepare_predictive_inference_table(
        comparisons=baseline,
        sensitivity_comparisons=sensitivity,
        model_mean_scores=create_model_means(),
    )

    assert len(prepared) == 18
    assert list(
        prepared.xs(
            ("var_quantile_loss", 0.95),
            level=("score_name", "confidence_level"),
        ).index
    ) == ["EWMA - RG", "FHS - RG", "FHS - EWMA"]
    ewma_vs_rg = prepared.loc[
        ("var_quantile_loss", 0.95, "EWMA - RG")
    ]
    source = baseline.loc[
        (
            "var_quantile_loss",
            0.95,
            MODELS[0],
            MODELS[1],
        )
    ]
    assert ewma_vs_rg["mean_score_difference"] == pytest.approx(
        -source["mean_score_difference"]
    )
    assert ewma_vs_rg["dm_statistic"] == pytest.approx(
        -source["dm_statistic"]
    )
    assert ewma_vs_rg["holm_adjusted_p_value"] == (
        source["holm_adjusted_p_value"]
    )
    assert prepared["sensitivity_decision_changed"].sum() == 1
    assert prepared.loc[
        ("var_quantile_loss", 0.99, "FHS - RG"),
        "sensitivity_decision_changed",
    ]


def test_prepare_inference_table_rejects_inconsistent_means() -> None:
    """Magnitude inputs must reproduce every verified differential."""

    means = create_model_means()
    means.loc[(MODELS[1], 0.95), "mean_var_quantile_loss"] += 0.01

    with pytest.raises(ValueError, match="do not reproduce"):
        prepare_predictive_inference_table(
            comparisons=create_comparisons(),
            model_mean_scores=means,
        )


def test_create_inference_figure_uses_two_score_panels() -> None:
    """The figure has fixed orientation and no unadjusted cutoffs."""

    prepared = prepare_predictive_inference_table(
        comparisons=create_comparisons(),
        sensitivity_comparisons=create_comparisons(
            bandwidth=0,
            flip_one_decision=True,
        ),
        model_mean_scores=create_model_means(),
    )
    figure = create_predictive_inference_figure(prepared)

    assert len(figure.axes) == 2
    assert [axis.get_title(loc="left") for axis in figure.axes] == [
        "(a)  Quantile loss (VaR)",
        "(b)  FZ0 loss (joint VaR–ES)",
    ]
    assert figure.get_size_inches().tolist() == [7.5, 4.7]
    assert figure.texts[0].get_text() == (
        "Adaptive models beat Rolling Gaussian in 3 of 12 "
        "comparisons;\nEWMA and FHS are never distinguishable"
    )
    assert "† significant at 5% with HAC lag 0" in (
        figure.texts[1].get_text()
    )
    assert "Source:" not in " ".join(
        text.get_text() for text in figure.texts
    )
    assert [text.get_text() for text in figure.legends[0].texts] == [
        "EWMA lower loss",
        "FHS lower loss",
        r"filled: $p_{\mathrm{Holm}} < 0.05$",
        "open: not significant",
    ]

    for axis in figure.axes:
        vertical_references = [
            line
            for line in axis.lines
            if np.array_equal(
                np.asarray(line.get_xdata(), dtype=float),
                np.array([0.0, 0.0]),
            )
        ]
        assert len(vertical_references) == 1
        assert all(
            not np.allclose(
                np.asarray(line.get_xdata(), dtype=float),
                [-1.96, -1.96],
            )
            for line in axis.lines
            if len(line.get_xdata()) == 2
        )
        assert sum(
            text.get_text() == r"$p_{\mathrm{Holm}}$"
            for text in axis.texts
        ) == 1

    assert [
        label.get_text() for label in figure.axes[0].get_yticklabels()
    ] == [
        "EWMA − RG",
        "FHS − RG",
        "FHS − EWMA",
    ] * 3
    assert [
        text.get_text()
        for text in figure.axes[0].texts
        if text.get_text() in {"95%", "97.5%", "99%"}
    ] == ["95%", "97.5%", "99%"]


def test_save_inference_figure_creates_png_and_svg(
    tmp_path: Path,
) -> None:
    """The formal comparison figure saves in publication formats."""

    prepared = prepare_predictive_inference_table(
        comparisons=create_comparisons(),
        sensitivity_comparisons=create_comparisons(
            bandwidth=0,
            flip_one_decision=True,
        ),
        model_mean_scores=create_model_means(),
    )
    png_path = tmp_path / "inference.png"
    svg_path = tmp_path / "inference.svg"
    saved_png, saved_svg = save_predictive_inference_figure(
        inference_table=prepared,
        png_output_path=png_path,
        svg_output_path=svg_path,
    )

    assert saved_png == png_path
    assert saved_svg == svg_path
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 25_000
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "Full-Sample Predictive-Accuracy Comparisons" in svg_text
    assert all(
        line == line.rstrip() for line in svg_text.splitlines()
    )

    second_png_path = tmp_path / "inference-second.png"
    second_svg_path = tmp_path / "inference-second.svg"
    save_predictive_inference_figure(
        inference_table=prepared,
        png_output_path=second_png_path,
        svg_output_path=second_svg_path,
    )

    assert png_path.read_bytes() == second_png_path.read_bytes()
    assert svg_path.read_bytes() == second_svg_path.read_bytes()
