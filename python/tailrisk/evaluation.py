"""Forecast diagnostics and comparative model evaluation."""

import numpy as np
import pandas as pd


def summarize_gaussian_monte_carlo_errors(
    analytic_forecasts: pd.DataFrame,
    monte_carlo_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize Monte Carlo errors relative to analytic forecasts.

    Errors are defined as Monte Carlo minus analytic and expressed in
    basis points. Results are grouped by confidence level and risk metric.
    """

    if not isinstance(analytic_forecasts, pd.DataFrame):
        raise TypeError(
            "Analytic forecasts must be a pandas DataFrame."
        )

    if not isinstance(monte_carlo_forecasts, pd.DataFrame):
        raise TypeError(
            "Monte Carlo forecasts must be a pandas DataFrame."
        )

    if analytic_forecasts.empty:
        raise ValueError(
            "Analytic forecasts must not be empty."
        )

    if monte_carlo_forecasts.empty:
        raise ValueError(
            "Monte Carlo forecasts must not be empty."
        )

    if not analytic_forecasts.index.equals(
        monte_carlo_forecasts.index
    ):
        raise ValueError(
            "Analytic and Monte Carlo forecasts must use "
            "the same index."
        )

    if not isinstance(
        analytic_forecasts.index,
        pd.MultiIndex,
    ) or "confidence_level" not in (
        analytic_forecasts.index.names
    ):
        raise ValueError(
            "Forecasts must use a MultiIndex containing "
            "confidence_level."
        )

    required_columns = {
        "value_at_risk",
        "expected_shortfall",
    }

    for forecasts, label in (
        (analytic_forecasts, "Analytic forecasts"),
        (monte_carlo_forecasts, "Monte Carlo forecasts"),
    ):
        if not required_columns.issubset(
            forecasts.columns
        ):
            raise ValueError(
                f"{label} must contain VaR and Expected "
                "Shortfall columns."
            )

        try:
            values = forecasts[
                sorted(required_columns)
            ].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} risk estimates must be numeric."
            ) from error

        if not np.isfinite(values).all():
            raise ValueError(
                f"{label} risk estimates must be finite."
            )

    records: list[dict[str, float | int | str]] = []
    summary_keys: list[tuple[float, str]] = []
    confidence_levels = (
        analytic_forecasts.index
        .get_level_values("confidence_level")
        .unique()
    )

    for confidence_level in confidence_levels:
        analytic_level = analytic_forecasts.xs(
            confidence_level,
            level="confidence_level",
        )
        monte_carlo_level = monte_carlo_forecasts.xs(
            confidence_level,
            level="confidence_level",
        )

        for metric in (
            "value_at_risk",
            "expected_shortfall",
        ):
            errors_bps = (
                monte_carlo_level[metric]
                - analytic_level[metric]
            ) * 10_000.0
            absolute_errors_bps = errors_bps.abs()

            summary_keys.append(
                (
                    float(confidence_level),
                    metric,
                )
            )
            records.append(
                {
                    "forecast_count": len(errors_bps),
                    "mean_error_bps": float(
                        errors_bps.mean()
                    ),
                    "mean_absolute_error_bps": float(
                        absolute_errors_bps.mean()
                    ),
                    "median_absolute_error_bps": float(
                        absolute_errors_bps.median()
                    ),
                    "p95_absolute_error_bps": float(
                        absolute_errors_bps.quantile(
                            0.95,
                            interpolation="linear",
                        )
                    ),
                    "maximum_absolute_error_bps": float(
                        absolute_errors_bps.max()
                    ),
                }
            )

    summary_index = pd.MultiIndex.from_tuples(
        summary_keys,
        names=[
            "confidence_level",
            "risk_metric",
        ],
    )

    return pd.DataFrame(
        records,
        index=summary_index,
    )
