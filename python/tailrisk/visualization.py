"""Figures for portfolio tail-risk results."""

from math import isfinite, pi
from pathlib import Path
from typing import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from tailrisk.models import GaussianRiskForecast
from tailrisk.simulation import EmpiricalRiskEstimate


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
