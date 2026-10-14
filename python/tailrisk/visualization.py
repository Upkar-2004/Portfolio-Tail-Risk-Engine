"""Figures for portfolio tail-risk results."""

from math import ceil, isfinite, pi
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from tailrisk.models import GaussianRiskForecast
from tailrisk.simulation import EmpiricalRiskEstimate


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
        fontweight="semibold",
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
            fontweight="semibold",
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
