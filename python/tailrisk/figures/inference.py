"""Full-sample predictive-accuracy inference figures."""

from math import ceil
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from tailrisk import style
from tailrisk.style import (
    normalize_svg_whitespace as _normalize_svg_whitespace,
)


_SCORE_NAMES = (
    "var_quantile_loss",
    "fz0_loss",
)
_SCORE_LABELS = {
    "var_quantile_loss": "Quantile loss (VaR)",
    "fz0_loss": "FZ0 loss (joint VaR–ES)",
}
_CONFIDENCE_LEVELS = (
    0.95,
    0.975,
    0.99,
)
_DISPLAY_PAIRS = (
    (
        "ewma_gaussian_monte_carlo",
        "rolling_gaussian_monte_carlo",
    ),
    (
        "filtered_historical_simulation",
        "rolling_gaussian_monte_carlo",
    ),
    (
        "filtered_historical_simulation",
        "ewma_gaussian_monte_carlo",
    ),
)
_COMPARISON_COLUMNS = {
    "observation_count",
    "mean_score_difference",
    "hac_long_run_variance",
    "hac_mean_variance",
    "hac_standard_error",
    "hac_bandwidth",
    "dm_statistic",
    "dm_p_value",
    "significance_level",
    "dm_reject_equal_accuracy",
    "holm_adjusted_p_value",
    "holm_reject_equal_accuracy",
}
_MODEL_MEAN_COLUMNS = {
    "observation_count",
    "mean_var_quantile_loss",
    "mean_fz0_loss",
}


def _validate_comparison_results(
    comparisons: pd.DataFrame,
    table_name: str,
) -> None:
    """Validate one complete pairwise predictive-accuracy table."""

    if not isinstance(comparisons, pd.DataFrame):
        raise TypeError(f"{table_name} must be a pandas DataFrame.")

    if comparisons.empty:
        raise ValueError(f"{table_name} must not be empty.")

    expected_index_names = [
        "score_name",
        "confidence_level",
        "model_a",
        "model_b",
    ]

    if (
        not isinstance(comparisons.index, pd.MultiIndex)
        or list(comparisons.index.names) != expected_index_names
    ):
        raise ValueError(
            f"{table_name} must use score_name, confidence_level, "
            "model_a, and model_b index levels."
        )

    if not comparisons.index.is_unique:
        raise ValueError(f"{table_name} index entries must be unique.")

    missing_columns = _COMPARISON_COLUMNS - set(comparisons.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{table_name} is missing required column(s): "
            f"{missing_names}."
        )

    numeric_columns = sorted(
        _COMPARISON_COLUMNS
        - {
            "dm_reject_equal_accuracy",
            "holm_reject_equal_accuracy",
        }
    )

    try:
        numeric_table = comparisons[numeric_columns].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{table_name} numeric fields must contain numbers."
        ) from error

    if not np.isfinite(numeric_table.to_numpy()).all():
        raise ValueError(f"{table_name} numeric fields must be finite.")

    if (
        (numeric_table["observation_count"] <= 0.0).any()
        or (
            numeric_table["observation_count"]
            != np.floor(numeric_table["observation_count"])
        ).any()
    ):
        raise ValueError(
            f"{table_name} observation counts must be positive integers."
        )

    if (numeric_table["hac_standard_error"] < 0.0).any():
        raise ValueError(
            f"{table_name} HAC standard errors must be nonnegative."
        )

    for p_value_column in (
        "dm_p_value",
        "holm_adjusted_p_value",
        "significance_level",
    ):
        values = numeric_table[p_value_column]

        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(
                f"{table_name} {p_value_column} values must be "
                "between zero and one."
            )

    significance_levels = numeric_table[
        "significance_level"
    ].unique()

    if len(significance_levels) != 1:
        raise ValueError(
            f"{table_name} must use one significance level."
        )

    for decision_column in (
        "dm_reject_equal_accuracy",
        "holm_reject_equal_accuracy",
    ):
        if not comparisons[decision_column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(
                f"{table_name} {decision_column} must be Boolean."
            )

    significance_level = float(significance_levels[0])
    adjusted_decisions = comparisons[
        "holm_adjusted_p_value"
    ].astype(float) <= significance_level

    if not np.array_equal(
        adjusted_decisions.to_numpy(dtype=bool),
        comparisons[
            "holm_reject_equal_accuracy"
        ].to_numpy(dtype=bool),
    ):
        raise ValueError(
            f"{table_name} Holm decisions must match the adjusted "
            "p-values and significance level."
        )

    model_ids = set(style.MODEL_IDS.values())
    expected_pair_sets = {
        frozenset(pair) for pair in _DISPLAY_PAIRS
    }

    for score_name in _SCORE_NAMES:
        for confidence_level in _CONFIDENCE_LEVELS:
            try:
                block = comparisons.xs(
                    (score_name, confidence_level),
                    level=("score_name", "confidence_level"),
                )
            except KeyError as error:
                raise ValueError(
                    f"{table_name} is missing {score_name} at "
                    f"{confidence_level:g}."
                ) from error

            observed_pairs = []

            for model_a, model_b in block.index:
                if model_a not in model_ids or model_b not in model_ids:
                    raise ValueError(
                        f"{table_name} contains an unknown model."
                    )

                if model_a == model_b:
                    raise ValueError(
                        f"{table_name} cannot compare a model with itself."
                    )

                observed_pairs.append(frozenset((model_a, model_b)))

            if (
                len(observed_pairs) != len(expected_pair_sets)
                or set(observed_pairs) != expected_pair_sets
            ):
                raise ValueError(
                    f"{table_name} must contain exactly the three "
                    "predeclared model pairs for every score and level."
                )

    if len(comparisons) != 18:
        raise ValueError(f"{table_name} must contain exactly 18 rows.")


def _oriented_row(
    comparisons: pd.DataFrame,
    score_name: str,
    confidence_level: float,
    model_a: str,
    model_b: str,
) -> tuple[pd.Series, float]:
    """Return one row and the sign needed for the display order."""

    direct_key = (
        score_name,
        confidence_level,
        model_a,
        model_b,
    )
    reverse_key = (
        score_name,
        confidence_level,
        model_b,
        model_a,
    )

    if direct_key in comparisons.index:
        return comparisons.loc[direct_key], 1.0

    if reverse_key in comparisons.index:
        return comparisons.loc[reverse_key], -1.0

    raise ValueError(
        "Predictive-accuracy results are missing a required model pair."
    )


def _validate_model_mean_scores(
    model_mean_scores: pd.DataFrame,
) -> None:
    """Validate model-level full-sample score means."""

    if not isinstance(model_mean_scores, pd.DataFrame):
        raise TypeError(
            "Full-sample model means must be a pandas DataFrame."
        )

    if (
        not isinstance(model_mean_scores.index, pd.MultiIndex)
        or list(model_mean_scores.index.names)
        != ["model", "confidence_level"]
        or not model_mean_scores.index.is_unique
    ):
        raise ValueError(
            "Full-sample model means must use unique model and "
            "confidence_level index levels."
        )

    missing_columns = _MODEL_MEAN_COLUMNS - set(
        model_mean_scores.columns
    )

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Full-sample model means are missing required "
            f"column(s): {missing_names}."
        )

    expected_index = pd.MultiIndex.from_product(
        [
            list(style.MODEL_IDS.values()),
            _CONFIDENCE_LEVELS,
        ],
        names=["model", "confidence_level"],
    )

    if set(model_mean_scores.index) != set(expected_index):
        raise ValueError(
            "Full-sample model means must contain all model-level "
            "and confidence-level combinations."
        )

    numeric = model_mean_scores[
        sorted(_MODEL_MEAN_COLUMNS)
    ].astype(float)

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Full-sample model means must be finite.")

    if (numeric["mean_var_quantile_loss"] <= 0.0).any():
        raise ValueError(
            "Full-sample mean quantile losses must be positive."
        )


def prepare_predictive_inference_table(
    comparisons: pd.DataFrame,
    sensitivity_comparisons: pd.DataFrame | None = None,
    model_mean_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Orient pairwise results and attach magnitude and sensitivity fields."""

    _validate_comparison_results(
        comparisons,
        "Predictive-accuracy results",
    )

    if sensitivity_comparisons is not None:
        _validate_comparison_results(
            sensitivity_comparisons,
            "Sensitivity predictive-accuracy results",
        )

    if model_mean_scores is not None:
        _validate_model_mean_scores(model_mean_scores)

    records: list[dict[str, object]] = []

    for score_name in _SCORE_NAMES:
        mean_column = (
            "mean_var_quantile_loss"
            if score_name == "var_quantile_loss"
            else "mean_fz0_loss"
        )

        for confidence_level in _CONFIDENCE_LEVELS:
            for model_a, model_b in _DISPLAY_PAIRS:
                row, direction = _oriented_row(
                    comparisons,
                    score_name,
                    confidence_level,
                    model_a,
                    model_b,
                )
                mean_difference = direction * float(
                    row["mean_score_difference"]
                )
                dm_statistic = direction * float(
                    row["dm_statistic"]
                )
                model_a_mean = np.nan
                model_b_mean = np.nan
                ql_relative_difference = np.nan

                if model_mean_scores is not None:
                    model_a_mean = float(
                        model_mean_scores.loc[
                            (model_a, confidence_level),
                            mean_column,
                        ]
                    )
                    model_b_mean = float(
                        model_mean_scores.loc[
                            (model_b, confidence_level),
                            mean_column,
                        ]
                    )

                    if not np.isclose(
                        model_a_mean - model_b_mean,
                        mean_difference,
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            "Full-sample model means do not reproduce "
                            "the verified mean score differential."
                        )

                    if score_name == "var_quantile_loss":
                        ql_relative_difference = (
                            100.0 * mean_difference / model_b_mean
                        )

                winning_model = (
                    model_a
                    if mean_difference < 0.0
                    else model_b
                    if mean_difference > 0.0
                    else ""
                )
                result: dict[str, object] = {
                    "score_name": score_name,
                    "score_label": _SCORE_LABELS[score_name],
                    "confidence_level": confidence_level,
                    "comparison": (
                        f"{style.MODEL_ABBREVIATIONS[model_a]} - "
                        f"{style.MODEL_ABBREVIATIONS[model_b]}"
                    ),
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_mean_score": model_a_mean,
                    "model_b_mean_score": model_b_mean,
                    "mean_score_difference": mean_difference,
                    "ql_relative_difference_percent": (
                        ql_relative_difference
                    ),
                    "winning_model": winning_model,
                    "observation_count": int(
                        row["observation_count"]
                    ),
                    "hac_bandwidth": int(row["hac_bandwidth"]),
                    "hac_standard_error": float(
                        row["hac_standard_error"]
                    ),
                    "dm_statistic": dm_statistic,
                    "dm_p_value": float(row["dm_p_value"]),
                    "holm_adjusted_p_value": float(
                        row["holm_adjusted_p_value"]
                    ),
                    "holm_reject_equal_accuracy": bool(
                        row["holm_reject_equal_accuracy"]
                    ),
                }

                if sensitivity_comparisons is not None:
                    sensitivity_row, sensitivity_direction = (
                        _oriented_row(
                            sensitivity_comparisons,
                            score_name,
                            confidence_level,
                            model_a,
                            model_b,
                        )
                    )
                    sensitivity_mean = sensitivity_direction * float(
                        sensitivity_row["mean_score_difference"]
                    )

                    if not np.isclose(
                        sensitivity_mean,
                        mean_difference,
                        rtol=1e-12,
                        atol=1e-15,
                    ):
                        raise ValueError(
                            "Sensitivity results must use the same "
                            "daily score differentials as the baseline."
                        )

                    sensitivity_reject = bool(
                        sensitivity_row[
                            "holm_reject_equal_accuracy"
                        ]
                    )
                    result.update(
                        {
                            "sensitivity_hac_bandwidth": int(
                                sensitivity_row["hac_bandwidth"]
                            ),
                            "sensitivity_dm_statistic": (
                                sensitivity_direction
                                * float(
                                    sensitivity_row["dm_statistic"]
                                )
                            ),
                            "sensitivity_holm_adjusted_p_value": (
                                float(
                                    sensitivity_row[
                                        "holm_adjusted_p_value"
                                    ]
                                )
                            ),
                            "sensitivity_holm_reject_equal_accuracy": (
                                sensitivity_reject
                            ),
                            "sensitivity_decision_changed": (
                                sensitivity_reject
                                != bool(
                                    row[
                                        "holm_reject_equal_accuracy"
                                    ]
                                )
                            ),
                        }
                    )

                records.append(result)

    inference_table = pd.DataFrame.from_records(records).set_index(
        [
            "score_name",
            "confidence_level",
            "comparison",
        ]
    )

    return inference_table


def _format_adjusted_p_value(value: float) -> str:
    """Format an adjusted p-value for the figure's numeric column."""

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


def create_predictive_inference_figure(
    inference_table: pd.DataFrame,
) -> Figure:
    """Create the full-sample DM-statistic comparison figure."""

    required_columns = {
        "model_a",
        "model_b",
        "winning_model",
        "mean_score_difference",
        "dm_statistic",
        "holm_adjusted_p_value",
        "holm_reject_equal_accuracy",
    }
    missing_columns = required_columns - set(inference_table.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Prepared predictive-inference results are missing "
            f"required column(s): {missing_names}."
        )

    expected_index_names = [
        "score_name",
        "confidence_level",
        "comparison",
    ]

    if (
        not isinstance(inference_table.index, pd.MultiIndex)
        or list(inference_table.index.names) != expected_index_names
        or not inference_table.index.is_unique
    ):
        raise ValueError(
            "Prepared predictive-inference results must use score, "
            "confidence-level, and comparison indexes."
        )

    has_sensitivity = "sensitivity_decision_changed" in (
        inference_table.columns
    )

    with style.context("report"):
        figure, axes = style.figure(
            1,
            2,
            height=4.7,
            sharex=True,
            sharey=True,
        )
        figure.subplots_adjust(
            left=0.22,
            right=0.925,
            top=0.74,
            bottom=0.15,
            wspace=0.34,
        )
        flat_table = inference_table.reset_index()
        adaptive_vs_rg = flat_table[
            flat_table["comparison"].isin(
                ["EWMA - RG", "FHS - RG"]
            )
        ]
        significant_adaptive_wins = int(
            (
                adaptive_vs_rg[
                    "holm_reject_equal_accuracy"
                ].astype(bool)
                & (
                    adaptive_vs_rg["mean_score_difference"].astype(
                        float
                    )
                    < 0.0
                )
            ).sum()
        )
        direct_comparisons = flat_table[
            flat_table["comparison"] == "FHS - EWMA"
        ]
        direct_rejections = int(
            direct_comparisons[
                "holm_reject_equal_accuracy"
            ].astype(bool).sum()
        )
        direct_clause = (
            "EWMA and FHS are never distinguishable"
            if direct_rejections == 0
            else (
                "EWMA and FHS differ in "
                f"{direct_rejections} of {len(direct_comparisons)} "
                "comparisons"
            )
        )
        title = (
            "Adaptive models beat Rolling Gaussian in "
            f"{significant_adaptive_wins} of {len(adaptive_vs_rg)} "
            f"comparisons;\n{direct_clause}"
        )
        subtitle = (
            "Full-sample DM statistics"
        )

        if has_sensitivity:
            changed_rows = inference_table.loc[
                inference_table[
                    "sensitivity_decision_changed"
                ].astype(bool)
            ]
            lag_zero_only_rejections = (
                not changed_rows.empty
                and not changed_rows[
                    "holm_reject_equal_accuracy"
                ].astype(bool).any()
                and changed_rows[
                    "sensitivity_holm_reject_equal_accuracy"
                ].astype(bool).all()
            )

            if lag_zero_only_rejections:
                subtitle += (
                    " · † significant at 5% with HAC lag 0; the "
                    "reported Newey–West specification is more "
                    "conservative"
                )
            elif not changed_rows.empty:
                subtitle += " · † denotes a decision change at HAC lag 0"

        figure_height_points = figure.get_figheight() * 72.0
        figure.text(
            0.012,
            1.0 - (8.0 / figure_height_points),
            title,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=style.INK,
        )
        figure.text(
            0.012,
            1.0 - (38.0 / figure_height_points),
            subtitle,
            ha="left",
            va="top",
            fontsize=8.2,
            color=style.INK_4,
        )
        y_positions = np.array(
            [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0]
        )
        row_labels = []

        for confidence_level in _CONFIDENCE_LEVELS:
            for model_a, model_b in _DISPLAY_PAIRS:
                row_labels.append(
                    style.MODEL_ABBREVIATIONS[model_a]
                    + " − "
                    + style.MODEL_ABBREVIATIONS[model_b]
                )

        maximum_statistic = float(
            np.abs(
                inference_table["dm_statistic"].to_numpy(dtype=float)
            ).max()
        )
        axis_limit = float(max(2, ceil(maximum_statistic * 1.08)))

        for axis, score_name in zip(
            axes,
            _SCORE_NAMES,
            strict=True,
        ):
            score_table = inference_table.xs(
                score_name,
                level="score_name",
            )
            ordered_rows = []

            for confidence_level in _CONFIDENCE_LEVELS:
                for model_a, model_b in _DISPLAY_PAIRS:
                    comparison = (
                        f"{style.MODEL_ABBREVIATIONS[model_a]} - "
                        f"{style.MODEL_ABBREVIATIONS[model_b]}"
                    )
                    ordered_rows.append(
                        score_table.loc[
                            (confidence_level, comparison)
                        ]
                    )

            for y_position, row in zip(
                y_positions,
                ordered_rows,
                strict=True,
            ):
                winning_model = str(row["winning_model"])
                model_style = style.model_style(winning_model)
                rejected = bool(
                    row["holm_reject_equal_accuracy"]
                )
                axis.plot(
                    float(row["dm_statistic"]),
                    y_position,
                    marker=model_style["marker"],
                    markersize=6.2,
                    markerfacecolor=(
                        model_style["color"]
                        if rejected
                        else style.PAPER
                    ),
                    markeredgecolor=model_style["color"],
                    markeredgewidth=1.15,
                    linestyle="none",
                    clip_on=False,
                    zorder=4,
                )
                changed = (
                    has_sensitivity
                    and bool(row["sensitivity_decision_changed"])
                )
                p_label = _format_adjusted_p_value(
                    float(row["holm_adjusted_p_value"])
                )

                if changed:
                    p_label += "†"

                axis.text(
                    1.035,
                    y_position,
                    p_label,
                    transform=axis.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=7.0,
                    color=style.INK_2,
                    clip_on=False,
                )

            axis.axvline(
                0.0,
                color=style.INK_3,
                linewidth=0.8,
                linestyle="-",
                zorder=2,
            )

            for separator in (3.0, 7.0):
                axis.axhline(
                    separator,
                    color=style.RULE,
                    linewidth=0.6,
                    zorder=1,
                )

            axis.set_xlim(-axis_limit, axis_limit)
            axis.set_ylim(10.7, -0.7)
            axis.xaxis.set_major_locator(
                MaxNLocator(nbins=5, symmetric=True)
            )
            axis.grid(
                axis="x",
                color=style.RULE,
                linewidth=0.4,
                zorder=0,
            )
            axis.set_xlabel("DM statistic")
            axis.set_yticks(y_positions)
            axis.tick_params(axis="y", length=0)
            axis.text(
                0.0,
                -0.14,
                "← first model better",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color=style.INK_3,
            )
            axis.text(
                1.0,
                -0.14,
                "second model better →",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.0,
                color=style.INK_3,
            )
            axis.text(
                1.035,
                1.025,
                r"$p_{\mathrm{Holm}}$",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=style.INK_2,
                clip_on=False,
            )

        axes[0].set_yticklabels(row_labels, fontsize=7.2)

        for confidence_level, group_center in zip(
            _CONFIDENCE_LEVELS,
            (1.0, 5.0, 9.0),
            strict=True,
        ):
            axes[0].text(
                -0.52,
                group_center,
                style.CONF_LABELS[confidence_level],
                transform=axes[0].get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color=style.INK_2,
                clip_on=False,
            )

        axes[1].tick_params(labelleft=False)
        style.panel_labels(
            axes,
            [
                _SCORE_LABELS[score_name]
                for score_name in _SCORE_NAMES
            ],
        )
        encoding_handles = [
            Line2D(
                [0],
                [0],
                marker=style.MODEL_MARKERS[model],
                markersize=5.5,
                markerfacecolor=style.MODEL_COLORS[model],
                markeredgecolor=style.MODEL_COLORS[model],
                linestyle="none",
                label=f"{model} lower loss",
            )
            for model in ("EWMA", "FHS")
        ]
        encoding_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markersize=5.0,
                    markerfacecolor=style.INK_2,
                    markeredgecolor=style.INK_2,
                    linestyle="none",
                    label=r"filled: $p_{\mathrm{Holm}} < 0.05$",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markersize=5.0,
                    markerfacecolor=style.PAPER,
                    markeredgecolor=style.INK_2,
                    linestyle="none",
                    label="open: not significant",
                ),
            ]
        )
        figure.legend(
            handles=encoding_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.84),
            ncols=4,
            frameon=False,
            handletextpad=0.45,
            columnspacing=1.2,
        )

        return figure


def save_predictive_inference_figure(
    inference_table: pd.DataFrame,
    png_output_path: str | Path,
    svg_output_path: str | Path,
) -> tuple[Path, Path]:
    """Save the predictive-inference figure in raster and vector formats."""

    png_path = Path(png_output_path)
    svg_path = Path(svg_output_path)

    if png_path.suffix.lower() != ".png":
        raise ValueError("The raster figure must use a .png path.")

    if svg_path.suffix.lower() != ".svg":
        raise ValueError("The vector figure must use a .svg path.")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure = create_predictive_inference_figure(inference_table)
    metadata = {
        "Title": "Full-Sample Predictive-Accuracy Comparisons",
        "Description": (
            "Diebold-Mariano statistics and Holm-adjusted p-values "
            "for full-sample VaR and joint VaR-ES comparisons."
        ),
    }
    figure.savefig(
        png_path,
        dpi=300,
        format="png",
        metadata=metadata,
    )
    svg_metadata = {**metadata, "Date": None}

    with mpl.rc_context(
        {"svg.hashsalt": "tailrisk-predictive-inference"}
    ):
        figure.savefig(
            svg_path,
            format="svg",
            metadata=svg_metadata,
        )
    _normalize_svg_whitespace(svg_path)
    figure.clear()

    return png_path, svg_path
