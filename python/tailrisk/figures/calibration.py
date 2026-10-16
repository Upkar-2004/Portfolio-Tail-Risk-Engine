"""VaR calibration figures for subperiod tail-risk results."""

from math import ceil, isfinite
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import (
    LinearSegmentedColormap,
    TwoSlopeNorm,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import binom

from tailrisk.style import (
    EXCEPTION_RATE_PALETTE,
    MODEL_STYLES_BY_ID,
    REPORT_FONT_STACK,
    normalize_svg_whitespace as _normalize_svg_whitespace,
)


SUBPERIOD_EXCEPTION_RATE_PALETTE = dict(
    EXCEPTION_RATE_PALETTE
)
TAIL_RISK_MODEL_STYLES = {
    model_id: {
        "label": model_style["label"],
        "color": model_style["color"],
        "marker": model_style["marker"],
    }
    for model_id, model_style in MODEL_STYLES_BY_ID.items()
}

_TAIL_RISK_PUBLICATION_STYLE = {
    "font.family": REPORT_FONT_STACK,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

_EXCEPTION_INTERVAL_FIGURE_STYLE = {
    **_TAIL_RISK_PUBLICATION_STYLE,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
}

_CALENDAR_REFERENCE_OBSERVATIONS = 252
_RECESSION_REFERENCE_OBSERVATIONS = 62
_AXIS_LEFT_PADDING = -0.004
_CALENDAR_AXIS_LIMITS = {
    0.95: 0.20,
    0.975: 0.16,
    0.99: 0.125,
}
_CALENDAR_AXIS_TICKS = {
    0.95: np.arange(0.00, 0.201, 0.04),
    0.975: np.arange(0.00, 0.151, 0.03),
    0.99: np.arange(0.00, 0.121, 0.03),
}
_RECESSION_AXIS_LIMIT = 0.40
_RECESSION_AXIS_TICKS = np.arange(0.00, 0.401, 0.08)

_SUBPERIOD_MODEL_ORDER = (
    "rolling_gaussian_monte_carlo",
    "ewma_gaussian_monte_carlo",
    "filtered_historical_simulation",
)
_SUBPERIOD_MODEL_LABELS = (
    "Rolling Gaussian",
    "EWMA Gaussian",
    "Filtered Historical",
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


def prepare_subperiod_exception_rate_matrices(
    var_coverage: pd.DataFrame,
) -> tuple[
    dict[float, pd.DataFrame],
    dict[float, pd.DataFrame],
]:
    """Validate coverage results and arrange plotting matrices."""

    if not isinstance(var_coverage, pd.DataFrame):
        raise TypeError(
            "Subperiod VaR coverage must be a pandas DataFrame."
        )

    if var_coverage.empty:
        raise ValueError(
            "Subperiod VaR coverage must not be empty."
        )

    if (
        not isinstance(var_coverage.index, pd.MultiIndex)
        or list(var_coverage.index.names)
        != [
            "period_id",
            "model",
            "confidence_level",
        ]
    ):
        raise ValueError(
            "Subperiod VaR coverage must use period_id, model, "
            "and confidence_level index levels."
        )

    if not var_coverage.index.is_unique:
        raise ValueError(
            "Subperiod VaR coverage index entries must be unique."
        )

    required_columns = {
        "period_type",
        "observed_minus_expected_rate",
    }
    missing_columns = (
        required_columns - set(var_coverage.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Subperiod VaR coverage is missing required "
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

    if set(var_coverage.index) != set(expected_index):
        raise ValueError(
            "Subperiod VaR coverage must contain exactly the "
            "predeclared 2012-2025 calendar and 2020 NBER "
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
            var_coverage.xs(
                period_id,
                level="period_id",
            )["period_type"].astype(str)
        )

        if observed_types != {expected_type}:
            raise ValueError(
                f"Period {period_id} must use period_type "
                f"{expected_type}."
            )

    try:
        deviations = var_coverage[
            "observed_minus_expected_rate"
        ].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Observed-minus-expected rates must be numeric."
        ) from error

    deviation_values = deviations.to_numpy()

    if (
        not np.isfinite(deviation_values).all()
        or (np.abs(deviation_values) > 1.0).any()
    ):
        raise ValueError(
            "Observed-minus-expected rates must be finite and "
            "between -1 and 1."
        )

    plotting_table = deviations.rename("deviation").reset_index()
    calendar_matrices: dict[float, pd.DataFrame] = {}
    recession_matrices: dict[float, pd.DataFrame] = {}

    for confidence_level in _SUBPERIOD_CONFIDENCE_LEVELS:
        level_rows = plotting_table.loc[
            np.isclose(
                plotting_table["confidence_level"],
                confidence_level,
            )
        ]
        level_matrix = level_rows.pivot(
            index="model",
            columns="period_id",
            values="deviation",
        )
        calendar_matrices[confidence_level] = (
            level_matrix.reindex(
                index=_SUBPERIOD_MODEL_ORDER,
                columns=_SUBPERIOD_CALENDAR_IDS,
            )
        )
        recession_matrices[confidence_level] = (
            level_matrix.reindex(
                index=_SUBPERIOD_MODEL_ORDER,
                columns=[_SUBPERIOD_RECESSION_ID],
            )
        )

    return calendar_matrices, recession_matrices


def prepare_subperiod_exception_interval_tables(
    var_coverage: pd.DataFrame,
) -> tuple[
    dict[float, pd.DataFrame],
    dict[float, pd.DataFrame],
]:
    """Validate and arrange exact exception-rate interval results."""

    prepare_subperiod_exception_rate_matrices(
        var_coverage
    )
    required_columns = {
        "observation_count",
        "nominal_exceedance_rate",
        "observed_exceedance_count",
        "observed_exceedance_rate",
        "interval_confidence_level",
        "exact_interval_lower_bound",
        "exact_interval_upper_bound",
    }
    missing_columns = (
        required_columns - set(var_coverage.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Subperiod VaR coverage is missing required interval "
            f"column(s): {missing_names}."
        )

    numeric_columns = list(required_columns)

    try:
        numeric_table = var_coverage[
            numeric_columns
        ].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Subperiod exception-rate interval inputs must be "
            "numeric."
        ) from error

    if not np.isfinite(numeric_table.to_numpy()).all():
        raise ValueError(
            "Subperiod exception-rate interval inputs must be "
            "finite."
        )

    observation_counts = numeric_table[
        "observation_count"
    ].to_numpy()
    exception_counts = numeric_table[
        "observed_exceedance_count"
    ].to_numpy()
    observed_rates = numeric_table[
        "observed_exceedance_rate"
    ].to_numpy()
    lower_bounds = numeric_table[
        "exact_interval_lower_bound"
    ].to_numpy()
    upper_bounds = numeric_table[
        "exact_interval_upper_bound"
    ].to_numpy()
    interval_levels = numeric_table[
        "interval_confidence_level"
    ].to_numpy()

    if (
        (observation_counts <= 0.0).any()
        or not np.equal(
            observation_counts,
            np.floor(observation_counts),
        ).all()
    ):
        raise ValueError(
            "Subperiod observation counts must be positive "
            "integers."
        )

    if (
        (exception_counts < 0.0).any()
        or (exception_counts > observation_counts).any()
        or not np.equal(
            exception_counts,
            np.floor(exception_counts),
        ).all()
    ):
        raise ValueError(
            "Subperiod exception counts must be integers between "
            "zero and the observation count."
        )

    calculated_rates = exception_counts / observation_counts

    if not np.allclose(
        observed_rates,
        calculated_rates,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Observed exception rates must equal exception counts "
            "divided by observation counts."
        )

    if (
        (lower_bounds < 0.0).any()
        or (upper_bounds > 1.0).any()
        or (lower_bounds > observed_rates).any()
        or (observed_rates > upper_bounds).any()
    ):
        raise ValueError(
            "Exact interval bounds must contain the observed rate "
            "and remain between zero and one."
        )

    if not np.allclose(
        interval_levels,
        0.95,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Subperiod exception-rate intervals must use the "
            "predeclared 95% confidence level."
        )

    confidence_values = var_coverage.index.get_level_values(
        "confidence_level"
    ).to_numpy(dtype=float)
    nominal_rates = numeric_table[
        "nominal_exceedance_rate"
    ].to_numpy()

    if not np.allclose(
        nominal_rates,
        1.0 - confidence_values,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Nominal exception rates must equal one minus the "
            "VaR confidence level."
        )

    plotting_columns = [
        "observation_count",
        "nominal_exceedance_rate",
        "observed_exceedance_count",
        "observed_exceedance_rate",
        "exact_interval_lower_bound",
        "exact_interval_upper_bound",
    ]
    calendar_tables: dict[float, pd.DataFrame] = {}
    recession_tables: dict[float, pd.DataFrame] = {}

    for confidence_level in _SUBPERIOD_CONFIDENCE_LEVELS:
        level_table = var_coverage.xs(
            confidence_level,
            level="confidence_level",
        )[plotting_columns]
        calendar_index = pd.MultiIndex.from_product(
            [
                _SUBPERIOD_CALENDAR_IDS,
                _SUBPERIOD_MODEL_ORDER,
            ],
            names=["period_id", "model"],
        )
        recession_index = pd.MultiIndex.from_product(
            [
                [_SUBPERIOD_RECESSION_ID],
                _SUBPERIOD_MODEL_ORDER,
            ],
            names=["period_id", "model"],
        )
        calendar_tables[confidence_level] = (
            level_table.reindex(calendar_index).copy()
        )
        recession_tables[confidence_level] = (
            level_table.reindex(recession_index).copy()
        )

    return calendar_tables, recession_tables


def _plot_exception_rate_intervals(
    axis: object,
    interval_table: pd.DataFrame,
    period_ids: Sequence[str],
    show_counts: bool,
) -> None:
    """Plot model-dodged rates and exact horizontal intervals."""

    base_positions = np.arange(len(period_ids), dtype=float)
    model_offsets = (-0.22, 0.0, 0.22)

    for model_name, offset in zip(
        _SUBPERIOD_MODEL_ORDER,
        model_offsets,
        strict=True,
    ):
        style = TAIL_RISK_MODEL_STYLES[model_name]
        model_rows = interval_table.xs(
            model_name,
            level="model",
        ).reindex(period_ids)
        observed_rates = model_rows[
            "observed_exceedance_rate"
        ].to_numpy(dtype=float)
        lower_bounds = model_rows[
            "exact_interval_lower_bound"
        ].to_numpy(dtype=float)
        upper_bounds = model_rows[
            "exact_interval_upper_bound"
        ].to_numpy(dtype=float)
        y_positions = base_positions + offset
        axis.errorbar(
            observed_rates,
            y_positions,
            xerr=np.vstack(
                [
                    observed_rates - lower_bounds,
                    upper_bounds - observed_rates,
                ]
            ),
            fmt=str(style["marker"]),
            markersize=5.0,
            markerfacecolor=str(style["color"]),
            markeredgecolor="white",
            markeredgewidth=0.65,
            ecolor=str(style["color"]),
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.15,
            linestyle="none",
            label=str(style["label"]),
            zorder=3,
        )

        if show_counts:
            exception_counts = model_rows[
                "observed_exceedance_count"
            ].to_numpy(dtype=int)
            observation_counts = model_rows[
                "observation_count"
            ].to_numpy(dtype=int)

            for upper, y_position, count, total in zip(
                upper_bounds,
                y_positions,
                exception_counts,
                observation_counts,
                strict=True,
            ):
                axis.annotate(
                    f"{count}/{total}",
                    xy=(upper, y_position),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8.0,
                    color=str(style["color"]),
                )


def create_subperiod_exception_interval_figure(
    var_coverage: pd.DataFrame,
) -> Figure:
    """Create pointwise exact intervals for subperiod VaR rates."""

    with mpl.rc_context(_EXCEPTION_INTERVAL_FIGURE_STYLE):
        return _create_subperiod_exception_interval_figure(
            var_coverage
        )


def _create_subperiod_exception_interval_figure(
    var_coverage: pd.DataFrame,
) -> Figure:
    """Build the exact-interval figure inside its local style."""

    calendar_tables, recession_tables = (
        prepare_subperiod_exception_interval_tables(
            var_coverage
        )
    )
    figure = Figure(
        figsize=(14.0, 9.0),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    header, panels, footer = figure.subfigures(
        nrows=3,
        ncols=1,
        height_ratios=(0.85, 6.7, 0.52),
    )
    panel_axes = panels.subplots(
        nrows=2,
        ncols=3,
        sharex=False,
        squeeze=False,
        gridspec_kw={
            "height_ratios": (6.0, 1.0),
            "wspace": 0.20,
        },
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=str(
                TAIL_RISK_MODEL_STYLES[model]["color"]
            ),
            marker=str(
                TAIL_RISK_MODEL_STYLES[model]["marker"]
            ),
            linestyle="none",
            markersize=6.5,
            label=str(
                TAIL_RISK_MODEL_STYLES[model]["label"]
            ),
        )
        for model in _SUBPERIOD_MODEL_ORDER
    ]
    header.text(
        0.5,
        0.78,
        "Observed VaR exception rates and exact uncertainty intervals",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    header.text(
        0.5,
        0.47,
        "Calendar years and the Feb-Apr 2020 NBER recession",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#666666",
    )
    header.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
        fontsize=9.5,
        frameon=False,
    )
    calendar_year_labels = [
        str(year) for year in range(2012, 2026)
    ]

    for column_index, confidence_level in enumerate(
        _SUBPERIOD_CONFIDENCE_LEVELS
    ):
        calendar_axis = panel_axes[0, column_index]
        recession_axis = panel_axes[1, column_index]
        calendar_table = calendar_tables[
            confidence_level
        ]
        recession_table = recession_tables[
            confidence_level
        ]
        nominal_rate = 1.0 - confidence_level
        reference_counts = binom.ppf(
            [0.025, 0.975],
            _CALENDAR_REFERENCE_OBSERVATIONS,
            nominal_rate,
        )
        reference_rates = (
            reference_counts
            / _CALENDAR_REFERENCE_OBSERVATIONS
        )
        recession_reference_counts = binom.ppf(
            [0.025, 0.975],
            _RECESSION_REFERENCE_OBSERVATIONS,
            nominal_rate,
        )
        recession_reference_rates = (
            recession_reference_counts
            / _RECESSION_REFERENCE_OBSERVATIONS
        )

        for row_index in range(len(calendar_year_labels)):
            if row_index % 2 == 0:
                calendar_axis.axhspan(
                    row_index - 0.5,
                    row_index + 0.5,
                    color="#FAFAFA",
                    zorder=0,
                )

        calendar_axis.axvspan(
            float(reference_rates[0]),
            float(reference_rates[1]),
            color="#E3F1EA",
            zorder=0.2,
        )
        recession_axis.set_facecolor("#F2F4F7")
        recession_axis.axvspan(
            float(recession_reference_rates[0]),
            float(recession_reference_rates[1]),
            color="#E3F1EA",
            zorder=0.2,
        )
        _plot_exception_rate_intervals(
            calendar_axis,
            calendar_table,
            _SUBPERIOD_CALENDAR_IDS,
            show_counts=False,
        )
        _plot_exception_rate_intervals(
            recession_axis,
            recession_table,
            [_SUBPERIOD_RECESSION_ID],
            show_counts=True,
        )

        for axis in (calendar_axis, recession_axis):
            axis.axvline(
                nominal_rate,
                color="#444444",
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=2,
            )
            axis.xaxis.set_major_formatter(
                PercentFormatter(
                    xmax=1.0,
                    decimals=0,
                )
            )
            axis.grid(
                axis="x",
                color="#E5E5E5",
                linewidth=0.6,
                alpha=1.0,
                zorder=0,
            )
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_visible(False)
            axis.tick_params(
                axis="y",
                length=0,
            )
            axis.tick_params(
                axis="x",
                labelsize=8.5,
            )
            axis.spines["bottom"].set_color("#9AA0A6")
            axis.spines["bottom"].set_linewidth(0.7)

        calendar_axis.set_xlim(
            _AXIS_LEFT_PADDING,
            _CALENDAR_AXIS_LIMITS[confidence_level],
        )
        calendar_axis.set_xticks(
            _CALENDAR_AXIS_TICKS[confidence_level]
        )
        recession_axis.set_xlim(
            _AXIS_LEFT_PADDING,
            _RECESSION_AXIS_LIMIT,
        )
        recession_axis.set_xticks(
            _RECESSION_AXIS_TICKS
        )
        calendar_axis.set_ylim(-0.65, 13.65)
        calendar_axis.invert_yaxis()
        calendar_axis.set_yticks(range(14))
        recession_axis.set_ylim(-0.58, 0.58)
        recession_axis.invert_yaxis()
        recession_axis.set_yticks([0])
        calendar_axis.set_title(
            f"{confidence_level * 100:g}% VaR",
            fontsize=12,
            fontweight="bold",
            pad=18,
        )
        calendar_axis.text(
            0.5,
            1.005,
            f"nominal {nominal_rate * 100:g}%",
            transform=calendar_axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#777777",
        )
        if column_index == 0:
            calendar_axis.set_yticklabels(
                calendar_year_labels,
                fontsize=8.5,
            )
            recession_axis.set_yticklabels(
                ["STRESS PERIOD\nFeb-Apr 2020\nn = 62"],
                fontsize=8.5,
            )
        else:
            calendar_axis.set_yticklabels([])
            recession_axis.set_yticklabels([])

    panel_axes[0, 1].set_xlabel(
        "Observed annual exception rate",
        fontsize=9.5,
    )
    panel_axes[1, 1].set_xlabel(
        "Observed stress-period exception rate",
        fontsize=9.5,
    )

    footer.text(
        0.5,
        0.55,
        (
            "Bars: pointwise 95% exact Clopper-Pearson intervals; "
            "dashes: nominal rates; pale-green bands: central 95% "
            "binomial reference ranges (252 annual sessions; 62 stress "
            "sessions).\n"
            "Intervals are not simultaneous or multiplicity-adjusted. "
            "The stress period overlaps calendar 2020; labels show "
            "exceptions/observations."
        ),
        ha="center",
        va="center",
        fontsize=8.0,
        color="#666666",
    )

    return figure


def save_subperiod_exception_interval_figure(
    var_coverage: pd.DataFrame,
    png_output_path: str | Path,
    svg_output_path: str | Path,
) -> tuple[Path, Path]:
    """Save PNG and SVG copies of the exact interval figure."""

    png_path = Path(png_output_path)
    svg_path = Path(svg_output_path)

    if png_path.suffix.lower() != ".png":
        raise ValueError(
            "The raster interval figure must use a .png path."
        )

    if svg_path.suffix.lower() != ".svg":
        raise ValueError(
            "The vector interval figure must use a .svg path."
        )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure = create_subperiod_exception_interval_figure(
        var_coverage
    )
    metadata = {
        "Title": (
            "Observed VaR Exception Rates and Exact "
            "Uncertainty Intervals"
        ),
        "Description": (
            "Calendar-year and February-April 2020 observed "
            "VaR exception rates with pointwise 95% exact "
            "Clopper-Pearson intervals."
        ),
    }
    figure.savefig(
        png_path,
        dpi=300,
        format="png",
        metadata=metadata,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    figure.savefig(
        svg_path,
        format="svg",
        metadata=metadata,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    _normalize_svg_whitespace(svg_path)
    figure.clear()

    return png_path, svg_path


def _annotate_exception_rate_heatmap(
    axis: object,
    values: np.ndarray,
    color_limit: float,
) -> None:
    """Write signed percentage-point values inside heatmap cells."""

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = float(values[row_index, column_index])
            text_color = (
                "white"
                if abs(value) >= 0.52 * color_limit
                else SUBPERIOD_EXCEPTION_RATE_PALETTE["text"]
            )
            axis.text(
                column_index,
                row_index,
                f"{100.0 * value:+.1f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color=text_color,
                fontweight=(
                    "bold"
                    if abs(value) >= 0.52 * color_limit
                    else "normal"
                ),
            )


def create_subperiod_exception_rate_figure(
    var_coverage: pd.DataFrame,
) -> Figure:
    """Create the annual and recession VaR exception-rate heatmap."""

    with mpl.rc_context(_TAIL_RISK_PUBLICATION_STYLE):
        return _create_subperiod_exception_rate_figure(
            var_coverage
        )


def _create_subperiod_exception_rate_figure(
    var_coverage: pd.DataFrame,
) -> Figure:
    """Build the exception-rate heatmap inside its local style."""

    calendar_matrices, recession_matrices = (
        prepare_subperiod_exception_rate_matrices(
            var_coverage
        )
    )
    all_values = np.concatenate(
        [
            matrix.to_numpy().ravel()
            for matrix in (
                *calendar_matrices.values(),
                *recession_matrices.values(),
            )
        ]
    )
    maximum_absolute_deviation = float(
        np.max(np.abs(all_values))
    )
    color_limit = max(
        0.01,
        ceil(maximum_absolute_deviation * 100.0) / 100.0,
    )
    color_map = LinearSegmentedColormap.from_list(
        "tailrisk_exception_rate",
        [
            SUBPERIOD_EXCEPTION_RATE_PALETTE["negative"],
            SUBPERIOD_EXCEPTION_RATE_PALETTE["neutral"],
            SUBPERIOD_EXCEPTION_RATE_PALETTE["positive"],
        ],
    )
    figure = Figure(
        figsize=(16.0, 9.4),
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=(14.0, 1.15),
        left=0.10,
        right=0.97,
        top=0.84,
        bottom=0.18,
        hspace=0.42,
        wspace=0.08,
    )
    figure.suptitle(
        "Annual VaR exception-rate deviations from nominal coverage\n"
        "Observed minus nominal exception rate (percentage points)",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    calendar_image = None

    for row_index, confidence_level in enumerate(
        _SUBPERIOD_CONFIDENCE_LEVELS
    ):
        calendar_axis = figure.add_subplot(
            grid[row_index, 0]
        )
        recession_axis = figure.add_subplot(
            grid[row_index, 1]
        )
        calendar_values = 100.0 * calendar_matrices[
            confidence_level
        ].to_numpy()
        recession_values = 100.0 * recession_matrices[
            confidence_level
        ].to_numpy()
        percentage_point_limit = 100.0 * color_limit
        calendar_image = calendar_axis.imshow(
            calendar_values,
            cmap=color_map,
            norm=TwoSlopeNorm(
                vmin=-percentage_point_limit,
                vcenter=0.0,
                vmax=percentage_point_limit,
            ),
            aspect="auto",
        )
        recession_axis.imshow(
            recession_values,
            cmap=color_map,
            norm=calendar_image.norm,
            aspect="auto",
        )
        _annotate_exception_rate_heatmap(
            calendar_axis,
            calendar_values / 100.0,
            color_limit,
        )
        _annotate_exception_rate_heatmap(
            recession_axis,
            recession_values / 100.0,
            color_limit,
        )

        for axis in (calendar_axis, recession_axis):
            axis.set_yticks(range(3))
            axis.set_yticklabels([])
            axis.tick_params(
                axis="both",
                which="both",
                length=0,
            )
            for spine in axis.spines.values():
                spine.set_visible(False)

        calendar_axis.set_yticklabels(
            _SUBPERIOD_MODEL_LABELS,
            fontsize=9,
        )
        calendar_axis.set_xticks(range(14))
        calendar_axis.set_xticklabels(
            [str(year) for year in range(2012, 2026)],
            fontsize=8,
        )
        recession_axis.set_xticks([0])
        recession_axis.set_xticklabels(
            ["Feb-Apr\n2020"],
            fontsize=8,
        )
        calendar_axis.set_ylabel(
            f"{confidence_level:.1%} VaR",
            fontsize=11,
            fontweight="bold",
            labelpad=12,
        )

        if row_index == 0:
            calendar_axis.set_title(
                "Calendar-year diagnostics",
                fontsize=10,
                fontweight="bold",
                pad=8,
            )
            recession_axis.set_title(
                "NBER recession*",
                fontsize=9,
                fontweight="bold",
                pad=8,
            )

    if calendar_image is None:
        raise RuntimeError(
            "Could not construct the exception-rate heatmap."
        )

    color_bar_axis = figure.add_axes(
        [0.10, 0.095, 0.87, 0.025]
    )
    color_bar = figure.colorbar(
        calendar_image,
        cax=color_bar_axis,
        orientation="horizontal",
    )
    color_bar.set_label(
        "Observed minus nominal exception rate (percentage points)",
        fontsize=10,
    )
    color_bar.ax.tick_params(labelsize=8)
    figure.text(
        0.5,
        0.017,
        (
            "Red = more exceptions than nominal (tail-risk "
            "underestimation); blue = fewer.\n"
            "*The 62-session recession case study overlaps calendar "
            "2020. This figure is descriptive; sparse tail counts can "
            "produce large deviations, especially at 99% VaR."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=SUBPERIOD_EXCEPTION_RATE_PALETTE["text"],
        wrap=True,
    )

    return figure


def save_subperiod_exception_rate_figure(
    var_coverage: pd.DataFrame,
    png_output_path: str | Path,
    svg_output_path: str | Path,
) -> tuple[Path, Path]:
    """Save PNG and SVG copies of the subperiod coverage figure."""

    png_path = Path(png_output_path)
    svg_path = Path(svg_output_path)

    if png_path.suffix.lower() != ".png":
        raise ValueError(
            "The raster subperiod figure must use a .png path."
        )

    if svg_path.suffix.lower() != ".svg":
        raise ValueError(
            "The vector subperiod figure must use a .svg path."
        )

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    svg_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure = create_subperiod_exception_rate_figure(
        var_coverage
    )
    metadata = {
        "Title": (
            "Annual VaR Exception-Rate Deviations from "
            "Nominal Coverage"
        ),
        "Description": (
            "Calendar-year and February-April 2020 NBER "
            "recession exception-rate deviations for three VaR "
            "models and three confidence levels."
        ),
    }
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
