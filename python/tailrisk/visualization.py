"""Figures for portfolio tail-risk results."""

from math import ceil, isfinite, pi
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    TwoSlopeNorm,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import binom

from tailrisk import style
from tailrisk.models import GaussianRiskForecast
from tailrisk.simulation import EmpiricalRiskEstimate
from tailrisk.style import (
    EXCEPTION_RATE_PALETTE,
    MODEL_STYLES_BY_ID,
    PREDICTIVE_RANK_PALETTE,
    REPORT_FONT_STACK,
    normalize_svg_whitespace as _normalize_svg_whitespace,
)


BASEL_TRAFFIC_LIGHT_PALETTE = {
    "green": "#DCEBDD",
    "amber": "#F3E2AD",
    "red": "#EFCFCF",
    "line": "#17324D",
    "threshold": "#59636E",
    "grid": "#AAB2BB",
}

_BASEL_MODEL_LABELS = {
    "rolling_gaussian_monte_carlo": "Rolling Gaussian",
    "ewma_gaussian_monte_carlo": "EWMA Gaussian",
    "filtered_historical_simulation": (
        "Filtered Historical Simulation"
    ),
}

SUBPERIOD_EXCEPTION_RATE_PALETTE = dict(
    EXCEPTION_RATE_PALETTE
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


def create_basel_traffic_light_figure(
    traffic_light_paths: pd.DataFrame,
    summary: pd.DataFrame,
) -> Figure:
    """Create aligned rolling-exception panels for VaR models."""

    if not isinstance(traffic_light_paths, pd.DataFrame):
        raise TypeError(
            "Traffic-light paths must be a pandas DataFrame."
        )

    if traffic_light_paths.empty:
        raise ValueError(
            "Traffic-light paths must not be empty."
        )

    if (
        not isinstance(
            traffic_light_paths.index,
            pd.MultiIndex,
        )
        or list(traffic_light_paths.index.names)
        != ["model", "forecast_date"]
    ):
        raise ValueError(
            "Traffic-light paths must use model and forecast_date "
            "index levels."
        )

    required_path_columns = {
        "exception_count",
        "traffic_light_zone",
    }
    missing_path_columns = (
        required_path_columns
        - set(traffic_light_paths.columns)
    )

    if missing_path_columns:
        missing_names = ", ".join(
            sorted(missing_path_columns)
        )
        raise ValueError(
            "Traffic-light paths are missing required "
            f"column(s): {missing_names}."
        )

    if not isinstance(summary, pd.DataFrame):
        raise TypeError(
            "Traffic-light summary must be a pandas DataFrame."
        )

    required_summary_columns = {
        "latest_exception_count",
        "latest_zone",
        "maximum_exception_count",
        "maximum_exception_date",
    }
    missing_summary_columns = (
        required_summary_columns - set(summary.columns)
    )

    if missing_summary_columns:
        missing_names = ", ".join(
            sorted(missing_summary_columns)
        )
        raise ValueError(
            "Traffic-light summary is missing required "
            f"column(s): {missing_names}."
        )

    model_names = list(
        traffic_light_paths.index
        .get_level_values("model")
        .unique()
    )

    if model_names != list(summary.index):
        raise ValueError(
            "Traffic-light paths and summary must use the same "
            "ordered models."
        )

    try:
        exception_counts = traffic_light_paths[
            "exception_count"
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Traffic-light exception counts must be numeric."
        ) from error

    if (
        not np.isfinite(exception_counts).all()
        or (exception_counts < 0.0).any()
    ):
        raise ValueError(
            "Traffic-light exception counts must be finite "
            "and nonnegative."
        )

    figure = Figure(
        figsize=(13.0, 9.5),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axes_array = figure.subplots(
        nrows=len(model_names),
        ncols=1,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes_array[:, 0]
    maximum_count = float(exception_counts.max())
    y_upper = float(
        max(12, ceil(maximum_count + 1.5))
    )

    figure.suptitle(
        "Basel-style 99% VaR traffic-light monitoring\n"
        "Rolling exceptions over complete 250-session windows",
        fontsize=15,
        fontweight="bold",
    )
    figure.supylabel(
        "Exceptions in the prior 250 sessions",
        fontsize=11,
    )
    figure.supxlabel(
        "Rolling-window end date",
        fontsize=11,
    )

    for axis, model_name in zip(
        axes,
        model_names,
        strict=True,
    ):
        model_path = traffic_light_paths.xs(
            model_name,
            level="model",
        )
        dates = pd.to_datetime(model_path.index)
        counts = model_path[
            "exception_count"
        ].to_numpy(dtype=float)
        model_summary = summary.loc[model_name]
        display_name = _BASEL_MODEL_LABELS.get(
            model_name,
            model_name.replace("_", " ").title(),
        )

        axis.axhspan(
            -0.5,
            5.0,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["green"],
            alpha=0.72,
            zorder=0,
        )
        axis.axhspan(
            5.0,
            10.0,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["amber"],
            alpha=0.72,
            zorder=0,
        )
        axis.axhspan(
            10.0,
            y_upper,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["red"],
            alpha=0.72,
            zorder=0,
        )
        axis.axhline(
            5.0,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["threshold"],
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )
        axis.axhline(
            10.0,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["threshold"],
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )
        axis.plot(
            dates,
            counts,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["line"],
            linewidth=1.55,
            drawstyle="steps-post",
            label="Rolling exception count",
            zorder=3,
        )
        axis.set_title(
            display_name,
            loc="left",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        axis.text(
            1.0,
            1.025,
            (
                "Maximum "
                f"{int(model_summary['maximum_exception_count'])} "
                "on "
                f"{pd.Timestamp(model_summary['maximum_exception_date']).date()}"
                "  |  Latest "
                f"{str(model_summary['latest_zone']).title()} "
                f"({int(model_summary['latest_exception_count'])})"
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["line"],
        )
        axis.text(
            0.99,
            2.25,
            "GREEN  0–4",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["threshold"],
        )
        axis.text(
            0.99,
            7.5,
            "AMBER  5–9",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["threshold"],
        )
        axis.text(
            0.99,
            (10.0 + y_upper) / 2.0,
            "RED  10+",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color=BASEL_TRAFFIC_LIGHT_PALETTE["threshold"],
        )
        axis.set_ylim(-0.5, y_upper)
        axis.yaxis.set_major_locator(
            MaxNLocator(
                nbins=7,
                integer=True,
            )
        )
        axis.grid(
            axis="y",
            color=BASEL_TRAFFIC_LIGHT_PALETTE["grid"],
            linewidth=0.6,
            alpha=0.45,
            zorder=1,
        )
        axis.tick_params(
            axis="both",
            labelsize=9,
        )

    axes[-1].xaxis.set_major_locator(
        mdates.YearLocator(2)
    )
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    return figure


def save_basel_traffic_light_figure(
    traffic_light_paths: pd.DataFrame,
    summary: pd.DataFrame,
    png_output_path: str | Path,
    svg_output_path: str | Path,
) -> tuple[Path, Path]:
    """Save PNG and SVG traffic-light figures from tested tables."""

    png_path = Path(png_output_path)
    svg_path = Path(svg_output_path)

    if png_path.suffix.lower() != ".png":
        raise ValueError(
            "The raster traffic-light figure must use a .png path."
        )

    if svg_path.suffix.lower() != ".svg":
        raise ValueError(
            "The vector traffic-light figure must use a .svg path."
        )

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    svg_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure = create_basel_traffic_light_figure(
        traffic_light_paths=traffic_light_paths,
        summary=summary,
    )
    metadata = {
        "Title": "Basel-style 99% VaR Traffic-Light Monitoring",
        "Description": (
            "Rolling 250-session VaR exception counts for the "
            "rolling Gaussian, EWMA Gaussian, and filtered "
            "historical simulation models."
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
    figure.clear()

    return png_path, svg_path


def save_gaussian_tail_risk_figure(
    losses: np.ndarray,
    analytic_mean_return: float,
    analytic_volatility: float,
    analytic_forecasts: Sequence[GaussianRiskForecast],
    empirical_estimates: Sequence[EmpiricalRiskEstimate],
    realized_loss: float,
    forecast_date: pd.Timestamp,
    estimation_start_date: pd.Timestamp,
    estimation_end_date: pd.Timestamp,
    estimation_observation_count: int,
    asset_count: int,
    scenario_count: int,
    random_seed: int,
    output_path: str | Path,
) -> Path:
    """Save a Gaussian loss distribution with VaR and ES comparisons."""

    if not isinstance(losses, np.ndarray):
        raise TypeError(
            "Losses must be a NumPy array."
        )

    if losses.ndim != 1 or losses.size == 0:
        raise ValueError(
            "Losses must be a nonempty one-dimensional array."
        )

    numeric_losses = losses.astype(
        float,
        copy=False,
    )

    if not np.isfinite(numeric_losses).all():
        raise ValueError(
            "Losses must contain finite numbers."
        )

    if (
        not isfinite(analytic_mean_return)
        or not isfinite(analytic_volatility)
        or analytic_volatility <= 0.0
    ):
        raise ValueError(
            "Analytic moments must be finite and volatility positive."
        )

    if not analytic_forecasts or not empirical_estimates:
        raise ValueError(
            "At least one analytic and empirical estimate is required."
        )

    if len(analytic_forecasts) != len(empirical_estimates):
        raise ValueError(
            "Analytic and empirical estimates must have equal lengths."
        )

    for analytic, empirical in zip(
        analytic_forecasts,
        empirical_estimates,
        strict=True,
    ):
        if not np.isclose(
            analytic.confidence_level,
            empirical.confidence_level,
        ):
            raise ValueError(
                "Analytic and empirical confidence levels must match."
            )

    figure_path = Path(output_path)
    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = Figure(
        figsize=(12, 8),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(4.0, 1.35),
    )
    distribution_axis = figure.add_subplot(
        grid[0]
    )
    table_axis = figure.add_subplot(
        grid[1]
    )

    distribution_axis.hist(
        numeric_losses,
        bins=90,
        density=True,
        alpha=0.55,
        color="tab:blue",
        label="Monte Carlo losses",
    )

    lower_bound, upper_bound = np.quantile(
        numeric_losses,
        [0.001, 0.9995],
    )
    density_x = np.linspace(
        lower_bound,
        upper_bound,
        700,
    )
    analytic_loss_mean = -float(
        analytic_mean_return
    )
    standardized_x = (
        density_x - analytic_loss_mean
    ) / analytic_volatility
    analytic_density = (
        np.exp(-0.5 * standardized_x**2)
        / (analytic_volatility * np.sqrt(2.0 * pi))
    )

    distribution_axis.plot(
        density_x,
        analytic_density,
        color="black",
        linewidth=1.7,
        label="Analytic Gaussian density",
    )

    colors = (
        "tab:green",
        "tab:orange",
        "tab:red",
        "tab:purple",
    )

    for color, analytic, empirical in zip(
        colors,
        analytic_forecasts,
        empirical_estimates,
        strict=False,
    ):
        level_label = (
            f"{analytic.confidence_level:.1%}"
        )
        distribution_axis.axvline(
            empirical.value_at_risk,
            color=color,
            linewidth=1.7,
            label=f"MC VaR {level_label}",
        )
        distribution_axis.axvline(
            analytic.value_at_risk,
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=f"Analytic VaR {level_label}",
        )

    most_extreme_estimate = max(
        empirical_estimates,
        key=lambda estimate: estimate.confidence_level,
    )
    distribution_axis.axvspan(
        most_extreme_estimate.value_at_risk,
        float(numeric_losses.max()),
        color="tab:red",
        alpha=0.12,
        label=(
            f"Worst "
            f"{1.0 - most_extreme_estimate.confidence_level:.1%} tail"
        ),
    )
    distribution_axis.axvline(
        realized_loss,
        color="tab:purple",
        linewidth=1.6,
        linestyle=":",
        label="Realized loss",
    )

    distribution_axis.set_title(
        "Gaussian Portfolio Loss Distribution\n"
        f"Forecast date {pd.Timestamp(forecast_date).date()}"
    )
    distribution_axis.set_xlabel(
        "One-session portfolio loss"
    )
    distribution_axis.set_ylabel("Density")
    distribution_axis.xaxis.set_major_formatter(
        PercentFormatter(xmax=1.0)
    )
    distribution_axis.grid(
        axis="y",
        alpha=0.22,
    )
    distribution_axis.legend(
        loc="upper left",
        fontsize=8,
        ncols=2,
    )

    table_axis.axis("off")
    table_rows = []

    for analytic, empirical in zip(
        analytic_forecasts,
        empirical_estimates,
        strict=True,
    ):
        table_rows.append(
            [
                f"{analytic.confidence_level:.1%}",
                f"{analytic.value_at_risk:.4%}",
                f"{empirical.value_at_risk:.4%}",
                f"{analytic.expected_shortfall:.4%}",
                f"{empirical.expected_shortfall:.4%}",
                f"{empirical.tail_observation_count:,}",
            ]
        )

    result_table = table_axis.table(
        cellText=table_rows,
        colLabels=[
            "Level",
            "Analytic VaR",
            "MC VaR",
            "Analytic ES",
            "MC ES",
            "Tail scenarios",
        ],
        cellLoc="center",
        loc="upper center",
    )
    result_table.auto_set_font_size(False)
    result_table.set_fontsize(9)
    result_table.scale(1.0, 1.3)

    table_axis.text(
        0.5,
        0.02,
        (
            f"Window: {pd.Timestamp(estimation_start_date).date()} to "
            f"{pd.Timestamp(estimation_end_date).date()} | "
            f"{estimation_observation_count:,} observations | "
            f"{asset_count} assets | "
            f"{scenario_count:,} scenarios | seed {random_seed} | "
            f"realized loss {realized_loss:.4%}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    figure.savefig(
        figure_path,
        dpi=180,
        format="png",
        metadata={
            "Title": "Gaussian Portfolio Tail-Risk Report",
            "Description": (
                "Analytic and Monte Carlo Gaussian VaR and "
                "Expected Shortfall comparison."
            ),
        },
    )
    figure.clear()

    return figure_path
