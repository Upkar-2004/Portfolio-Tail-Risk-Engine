"""Predictive-score figures for subperiod tail-risk results."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from tailrisk import style
from tailrisk.style import (
    MODEL_STYLES_BY_ID,
    PREDICTIVE_RANK_PALETTE,
    normalize_svg_whitespace as _normalize_svg_whitespace,
)


SUBPERIOD_PREDICTIVE_RANK_PALETTE = dict(
    PREDICTIVE_RANK_PALETTE
)
TAIL_RISK_MODEL_STYLES = {
    model_id: {
        "label": model_style["label"],
        "color": model_style["color"],
        "marker": model_style["marker"],
    }
    for model_id, model_style in MODEL_STYLES_BY_ID.items()
}

_SUBPERIOD_MODEL_ORDER = (
    "rolling_gaussian_monte_carlo",
    "ewma_gaussian_monte_carlo",
    "filtered_historical_simulation",
)
_SUBPERIOD_CONFIDENCE_LEVELS = (
    0.95,
    0.975,
    0.99,
)
_SUBPERIOD_CALENDAR_IDS = tuple(
    f"calendar_{year}"
    for year in range(2012, 2026)
)
_SUBPERIOD_RECESSION_ID = "nber_recession_2020"

_PREDICTIVE_SCORE_SPECS = (
    (
        "QL",
        "mean_var_quantile_loss",
        "var_quantile_loss_rank",
    ),
    (
        "FZ0",
        "mean_fz0_loss",
        "fz0_loss_rank",
    ),
)


def prepare_subperiod_predictive_rank_matrices(
    predictive_accuracy: pd.DataFrame,
) -> tuple[
    dict[float, pd.DataFrame],
    dict[float, pd.DataFrame],
]:
    """Validate predictive scores and arrange rank matrices."""

    if not isinstance(predictive_accuracy, pd.DataFrame):
        raise TypeError(
            "Subperiod predictive accuracy must be a pandas "
            "DataFrame."
        )

    if predictive_accuracy.empty:
        raise ValueError(
            "Subperiod predictive accuracy must not be empty."
        )

    if (
        not isinstance(predictive_accuracy.index, pd.MultiIndex)
        or list(predictive_accuracy.index.names)
        != [
            "period_id",
            "model",
            "confidence_level",
        ]
    ):
        raise ValueError(
            "Subperiod predictive accuracy must use period_id, "
            "model, and confidence_level index levels."
        )

    if not predictive_accuracy.index.is_unique:
        raise ValueError(
            "Subperiod predictive-accuracy index entries must be "
            "unique."
        )

    required_columns = {
        "period_type",
        *(
            column_name
            for _, mean_column, rank_column
            in _PREDICTIVE_SCORE_SPECS
            for column_name in (mean_column, rank_column)
        ),
    }
    missing_columns = (
        required_columns - set(predictive_accuracy.columns)
    )

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Subperiod predictive accuracy is missing required "
            f"column(s): {missing_names}."
        )

    expected_index = pd.MultiIndex.from_product(
        [
            (
                *_SUBPERIOD_CALENDAR_IDS,
                _SUBPERIOD_RECESSION_ID,
            ),
            _SUBPERIOD_MODEL_ORDER,
            _SUBPERIOD_CONFIDENCE_LEVELS,
        ],
        names=[
            "period_id",
            "model",
            "confidence_level",
        ],
    )

    if set(predictive_accuracy.index) != set(expected_index):
        raise ValueError(
            "Subperiod predictive accuracy must contain exactly "
            "the predeclared 2012-2025 calendar and 2020 NBER "
            "period-model-confidence combinations."
        )

    expected_period_types = {
        **{
            period_id: "calendar_year"
            for period_id in _SUBPERIOD_CALENDAR_IDS
        },
        _SUBPERIOD_RECESSION_ID: "nber_recession",
    }

    for period_id, expected_type in (
        expected_period_types.items()
    ):
        observed_types = set(
            predictive_accuracy.xs(
                period_id,
                level="period_id",
            )["period_type"].astype(str)
        )

        if observed_types != {expected_type}:
            raise ValueError(
                f"Period {period_id} must use period_type "
                f"{expected_type}."
            )

    numeric_columns = [
        column_name
        for _, mean_column, rank_column
        in _PREDICTIVE_SCORE_SPECS
        for column_name in (mean_column, rank_column)
    ]

    try:
        numeric_table = predictive_accuracy[
            numeric_columns
        ].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Subperiod predictive scores and ranks must be "
            "numeric."
        ) from error

    if not np.isfinite(numeric_table.to_numpy()).all():
        raise ValueError(
            "Subperiod predictive scores and ranks must be finite."
        )

    if (
        numeric_table["mean_var_quantile_loss"] < 0.0
    ).any():
        raise ValueError(
            "Mean VaR quantile losses must be nonnegative."
        )

    grouping_levels = ["period_id", "confidence_level"]

    for _, mean_column, rank_column in _PREDICTIVE_SCORE_SPECS:
        rank_values = numeric_table[rank_column].to_numpy()

        if (
            (rank_values < 1.0).any()
            or (rank_values > 3.0).any()
            or not np.equal(
                rank_values,
                np.floor(rank_values),
            ).all()
        ):
            raise ValueError(
                "Subperiod predictive ranks must be integers from "
                "one through three."
            )

        expected_ranks = (
            numeric_table.groupby(
                level=grouping_levels,
                sort=False,
            )[mean_column]
            .rank(method="min", ascending=True)
            .astype(int)
        )

        if not np.array_equal(
            numeric_table[rank_column].to_numpy(dtype=int),
            expected_ranks.to_numpy(dtype=int),
        ):
            raise ValueError(
                f"{rank_column} must match the minimum ranks of "
                f"{mean_column}."
            )

    matrix_columns = pd.MultiIndex.from_product(
        [
            [spec[0] for spec in _PREDICTIVE_SCORE_SPECS],
            _SUBPERIOD_MODEL_ORDER,
        ],
        names=["score_family", "model"],
    )
    calendar_matrices: dict[float, pd.DataFrame] = {}
    recession_matrices: dict[float, pd.DataFrame] = {}

    for confidence_level in _SUBPERIOD_CONFIDENCE_LEVELS:
        level_table = predictive_accuracy.xs(
            confidence_level,
            level="confidence_level",
        )
        rank_matrix = pd.DataFrame(
            index=(
                *_SUBPERIOD_CALENDAR_IDS,
                _SUBPERIOD_RECESSION_ID,
            ),
            columns=matrix_columns,
            dtype=int,
        )
        rank_matrix.index.name = "period_id"

        for score_label, _, rank_column in (
            _PREDICTIVE_SCORE_SPECS
        ):
            score_ranks = (
                level_table[rank_column]
                .unstack("model")
                .reindex(
                    index=rank_matrix.index,
                    columns=_SUBPERIOD_MODEL_ORDER,
                )
            )
            rank_matrix.loc[:, score_label] = (
                score_ranks.to_numpy(dtype=int)
            )

        calendar_matrices[confidence_level] = (
            rank_matrix.loc[list(_SUBPERIOD_CALENDAR_IDS)]
            .astype(int)
            .copy()
        )
        recession_matrices[confidence_level] = (
            rank_matrix.loc[[_SUBPERIOD_RECESSION_ID]]
            .astype(int)
            .copy()
        )

    return calendar_matrices, recession_matrices


def prepare_subperiod_predictive_first_place_counts(
    predictive_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    """Count calendar-year rank-one finishes by score and level."""

    calendar_matrices, _ = prepare_subperiod_predictive_rank_matrices(
        predictive_accuracy
    )
    columns = pd.MultiIndex.from_product(
        [
            [score_spec[0] for score_spec in _PREDICTIVE_SCORE_SPECS],
            _SUBPERIOD_MODEL_ORDER,
        ],
        names=["score_family", "model"],
    )
    counts = pd.DataFrame(
        index=pd.Index(
            _SUBPERIOD_CONFIDENCE_LEVELS,
            name="confidence_level",
        ),
        columns=columns,
        dtype=int,
    )

    for confidence_level, rank_matrix in calendar_matrices.items():
        counts.loc[confidence_level] = (
            (rank_matrix == 1).sum(axis=0).to_numpy(dtype=int)
        )

    return counts.astype(int)


def create_subperiod_predictive_first_place_figure(
    predictive_accuracy: pd.DataFrame,
) -> Figure:
    """Create the main-text first-place-count summary figure."""

    with style.context("report"):
        return _create_subperiod_predictive_first_place_figure(
            predictive_accuracy
        )


def _create_subperiod_predictive_first_place_figure(
    predictive_accuracy: pd.DataFrame,
) -> Figure:
    """Build the first-place summary inside its local style."""

    counts = prepare_subperiod_predictive_first_place_counts(
        predictive_accuracy
    )
    figure, axes = style.figure(
        1,
        2,
        height=3.35,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.94,
        top=0.70,
        bottom=0.21,
        wspace=0.16,
    )
    style.title_block(
        figure,
        (
            "EWMA leads VaR accuracy; FHS leads joint "
            "VaR–ES accuracy"
        ),
        (
            "Years with the lowest period-average score, 2012–2025, "
            "equal-weight 11-stock portfolio, one-day-ahead forecasts.\n"
            "Feb–Apr 2020: FHS ranked first for both scores at "
            "every confidence level"
        ),
    )
    base_x = np.arange(len(_SUBPERIOD_CONFIDENCE_LEVELS))
    x_offsets = (-0.045, 0.0, 0.045)
    model_labels = ("RG", "EWMA", "FHS")
    even_split = len(_SUBPERIOD_CALENDAR_IDS) / 3.0
    score_titles = (
        "Quantile loss (VaR)",
        "FZ0 loss (joint VaR–ES)",
    )

    for score_index, (axis, score_spec, score_title) in enumerate(
        zip(axes, _PREDICTIVE_SCORE_SPECS, score_titles)
    ):
        score_label = score_spec[0]
        style.reference_line(
            axis,
            even_split,
            label=(
                "Even split (14/3)"
                if score_index == 1
                else None
            ),
        )

        for model_index, (model, model_label, x_offset) in enumerate(
            zip(
                _SUBPERIOD_MODEL_ORDER,
                model_labels,
                x_offsets,
            )
        ):
            values = counts[(score_label, model)].to_numpy(
                dtype=int
            )
            x_values = base_x + x_offset
            model_style = TAIL_RISK_MODEL_STYLES[model]
            label_offset = 0

            if score_index == 1 and model_index == 0:
                label_offset = -7
            elif score_index == 1 and model_index == 1:
                label_offset = 7

            style.model_line(
                axis,
                x_values,
                values,
                model_label,
                dy_pts=label_offset,
            )

            for level_index, (x_value, value) in enumerate(
                zip(x_values, values)
            ):
                vertical_offset = 0.42

                if (
                    score_index == 1
                    and level_index == 2
                    and model_index == 0
                ):
                    vertical_offset = -0.72

                axis.text(
                    x_value,
                    value + vertical_offset,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=7.1,
                    color=model_style["color"],
                    fontweight="bold",
                )
        axis.set_xticks(
            base_x,
            ["95%", "97.5%", "99%"],
        )
        axis.set_xlabel("VaR confidence level")
        axis.set_xlim(-0.22, 2.48)
        axis.set_ylim(0.0, 14.0)
        axis.set_yticks([0, 7, 14])
        axis.tick_params(axis="x", length=0)
        style.value_grid(axis, "y")

    axes[0].set_ylabel(
        "Calendar years ranked first (of 14)",
    )
    style.panel_labels(
        axes,
        [score_title for score_title in score_titles],
    )
    style.source_line(
        figure,
        (
            "Source: Yahoo Finance adjusted closes; author's "
            "calculations. Ranks are descriptive; full-sample "
            "Holm-adjusted DM tests provide inference."
        ),
    )

    return figure


def _predictive_rank_cell_colors(
    rank_values: np.ndarray,
) -> np.ndarray:
    """Color rank-one cells by the winning model."""

    row_count, column_count = rank_values.shape
    colors = np.empty((row_count, column_count, 4), dtype=float)
    colors[:] = mpl.colors.to_rgba("#FFFFFF")

    for column_index in range(column_count):
        model = _SUBPERIOD_MODEL_ORDER[column_index % 3]
        winning_rows = rank_values[:, column_index] == 1
        colors[winning_rows, column_index] = mpl.colors.to_rgba(
            TAIL_RISK_MODEL_STYLES[model]["color"]
        )

    return colors


def _style_predictive_rank_axis(
    axis: object,
    row_count: int,
    show_model_labels: bool,
) -> None:
    """Apply cell boundaries to a nine-column rank axis."""

    column_count = 9
    axis.set_xticks(range(column_count))
    axis.set_xlim(-0.5, column_count - 0.5)
    axis.set_ylim(row_count - 0.5, -0.5)
    axis.set_xticks(
        np.arange(-0.5, column_count, 1.0),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, row_count, 1.0),
        minor=True,
    )
    axis.grid(
        which="minor",
        color=SUBPERIOD_PREDICTIVE_RANK_PALETTE["grid"],
        linewidth=0.5,
    )

    for divider in (2.5, 5.5):
        axis.axvline(
            divider,
            color=SUBPERIOD_PREDICTIVE_RANK_PALETTE["divider"],
            linewidth=0.8,
            zorder=4,
        )

    axis.tick_params(axis="both", which="both", length=0)

    for spine in axis.spines.values():
        spine.set_visible(False)

    if show_model_labels:
        axis.set_xticklabels(
            ["RG", "EWMA", "FHS"] * 3,
            fontsize=6.8,
        )
        axis.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            pad=3,
        )
    else:
        axis.set_xticklabels([])


def _annotate_predictive_ranks(
    axis: object,
    rank_values: np.ndarray,
) -> None:
    """Write rank values with contrast-safe text."""

    for row_index in range(rank_values.shape[0]):
        for column_index in range(rank_values.shape[1]):
            rank = int(rank_values[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(rank),
                ha="center",
                va="center",
                fontsize=7.2,
                color=(
                    "white"
                    if rank == 1
                    else SUBPERIOD_PREDICTIVE_RANK_PALETTE[
                        (
                            "secondary_text"
                            if rank == 2
                            else "tertiary_text"
                        )
                    ]
                ),
                fontweight=("bold" if rank == 1 else "normal"),
                zorder=5,
            )


def create_subperiod_predictive_rank_figure(
    predictive_accuracy: pd.DataFrame,
) -> Figure:
    """Create the supporting annual predictive-rank grid."""

    with style.context("report"):
        return _create_subperiod_predictive_rank_figure(
            predictive_accuracy
        )


def _create_subperiod_predictive_rank_figure(
    predictive_accuracy: pd.DataFrame,
) -> Figure:
    """Build the score-first annual rank grid."""

    calendar_matrices, recession_matrices = (
        prepare_subperiod_predictive_rank_matrices(
            predictive_accuracy
        )
    )
    figure = Figure(
        figsize=(style.figure_widths("report")[0], 5.65)
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=(14.0, 1.15),
        left=0.095,
        right=0.995,
        top=0.72,
        bottom=0.14,
        hspace=0.17,
        wspace=0.12,
    )
    style.title_block(
        figure,
        "Annual ranks underlying the predictive-score summary",
        (
            "Equal-weight 11-stock portfolio, 2012–2025. Winner colors "
            "identify rank one; cell text reports ranks 1–3."
        ),
    )
    calendar_year_labels = [
        str(year) for year in range(2012, 2026)
    ]

    for score_index, score_spec in enumerate(
        _PREDICTIVE_SCORE_SPECS
    ):
        score_label = score_spec[0]
        calendar_axis = figure.add_subplot(grid[0, score_index])
        recession_axis = figure.add_subplot(grid[1, score_index])
        calendar_values = np.hstack(
            [
                calendar_matrices[level]
                .loc[:, score_label]
                .to_numpy(dtype=int)
                for level in _SUBPERIOD_CONFIDENCE_LEVELS
            ]
        )
        recession_values = np.hstack(
            [
                recession_matrices[level]
                .loc[:, score_label]
                .to_numpy(dtype=int)
                for level in _SUBPERIOD_CONFIDENCE_LEVELS
            ]
        )

        calendar_axis.imshow(
            _predictive_rank_cell_colors(calendar_values),
            aspect="auto",
            interpolation="none",
        )
        recession_axis.imshow(
            _predictive_rank_cell_colors(recession_values),
            aspect="auto",
            interpolation="none",
        )
        _style_predictive_rank_axis(
            calendar_axis,
            row_count=14,
            show_model_labels=True,
        )
        _style_predictive_rank_axis(
            recession_axis,
            row_count=1,
            show_model_labels=False,
        )
        _annotate_predictive_ranks(
            calendar_axis,
            calendar_values,
        )
        _annotate_predictive_ranks(
            recession_axis,
            recession_values,
        )
        calendar_axis.set_title(
            (
                "(a)  Quantile loss (VaR)"
                if score_label == "QL"
                else "(b)  FZ0 loss (joint VaR–ES)"
            ),
            loc="left",
            fontsize=8.8,
            fontweight="bold",
            pad=38,
        )

        for center, confidence_level in zip(
            (1.0, 4.0, 7.0),
            _SUBPERIOD_CONFIDENCE_LEVELS,
        ):
            calendar_axis.text(
                center,
                1.105,
                f"{confidence_level * 100:g}%",
                transform=calendar_axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=SUBPERIOD_PREDICTIVE_RANK_PALETTE["text"],
            )

        calendar_axis.set_yticks(range(14))
        recession_axis.set_yticks([0])

        if score_index == 0:
            calendar_axis.set_yticklabels(
                calendar_year_labels,
                fontsize=7.2,
            )
            recession_axis.set_yticklabels(
                ["Stress period\nFeb–Apr 2020\nn = 62"],
                fontsize=7.0,
            )
        else:
            calendar_axis.set_yticklabels([])
            recession_axis.set_yticklabels([])

    style.source_line(
        figure,
        (
            "Source: Yahoo Finance adjusted closes; author's "
            "calculations. RG = Rolling Gaussian; EWMA = EWMA "
            "Gaussian; FHS = Filtered Historical Simulation."
        ),
    )

    return figure


def _save_subperiod_predictive_figure(
    figure: Figure,
    png_output_path: str | Path,
    svg_output_path: str | Path,
    metadata: dict[str, str],
) -> tuple[Path, Path]:
    """Save one predictive figure in raster and vector formats."""

    png_path = Path(png_output_path)
    svg_path = Path(svg_output_path)

    if png_path.suffix.lower() != ".png":
        raise ValueError("The raster figure must use a .png path.")

    if svg_path.suffix.lower() != ".svg":
        raise ValueError("The vector figure must use a .svg path.")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path,
        dpi=300,
        format="png",
        metadata=metadata,
    )
    figure.savefig(
        svg_path,
        format="svg",
        metadata=metadata,
    )
    _normalize_svg_whitespace(svg_path)
    figure.clear()

    return png_path, svg_path


def save_subperiod_predictive_first_place_figure(
    predictive_accuracy: pd.DataFrame,
    png_output_path: str | Path,
    svg_output_path: str | Path,
) -> tuple[Path, Path]:
    """Save PNG and SVG copies of the main summary figure."""

    return _save_subperiod_predictive_figure(
        figure=create_subperiod_predictive_first_place_figure(
            predictive_accuracy
        ),
        png_output_path=png_output_path,
        svg_output_path=svg_output_path,
        metadata={
            "Title": "Predictive-Score First-Place Counts",
            "Description": (
                "Calendar-year first-place counts by model, score, "
                "and confidence level."
            ),
        },
    )


def save_subperiod_predictive_rank_figure(
    predictive_accuracy: pd.DataFrame,
    png_output_path: str | Path,
    svg_output_path: str | Path,
) -> tuple[Path, Path]:
    """Save PNG and SVG copies of the supporting annual grid."""

    return _save_subperiod_predictive_figure(
        figure=create_subperiod_predictive_rank_figure(
            predictive_accuracy
        ),
        png_output_path=png_output_path,
        svg_output_path=svg_output_path,
        metadata={
            "Title": "Annual Predictive-Score Ranks",
            "Description": (
                "Calendar-year and February-April 2020 descriptive "
                "ranks grouped by score family and confidence level."
            ),
        },
    )
