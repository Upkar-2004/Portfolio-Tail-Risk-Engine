"""Backtesting-monitoring figures for portfolio tail-risk results."""

from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


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
