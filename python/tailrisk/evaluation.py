"""Forecast diagnostics and comparative model evaluation."""

from dataclasses import dataclass
from math import erfc, isfinite, log, sqrt
from numbers import Integral, Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KupiecTestResult:
    """Result of a Kupiec unconditional-coverage test."""

    confidence_level: float
    observation_count: int
    exceedance_count: int
    expected_exceedance_rate: float
    observed_exceedance_rate: float
    likelihood_ratio: float
    p_value: float
    significance_level: float
    reject_null: bool


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


def summarize_var_exceedances(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize observed and expected VaR exceedances.

    An exceedance occurs when the realized loss is strictly greater
    than the forecast VaR. Results are grouped by confidence level.
    """

    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError(
            "Forecasts must be a pandas DataFrame."
        )

    if forecasts.empty:
        raise ValueError(
            "Forecasts must not be empty."
        )

    if (
        not isinstance(forecasts.index, pd.MultiIndex)
        or "confidence_level"
        not in forecasts.index.names
    ):
        raise ValueError(
            "Forecasts must use a MultiIndex containing "
            "confidence_level."
        )

    if forecasts.index.has_duplicates:
        raise ValueError(
            "Forecast index entries must be unique."
        )

    required_columns = {
        "realized_loss",
        "value_at_risk",
        "var_exceedance",
    }
    missing_columns = (
        required_columns - set(forecasts.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Forecasts are missing required column(s): "
            f"{missing_names}."
        )

    confidence_index = forecasts.index.get_level_values(
        "confidence_level"
    )

    if any(
        isinstance(value, (bool, np.bool_))
        for value in confidence_index
    ):
        raise ValueError(
            "Confidence levels must be finite numbers "
            "strictly between 0 and 1."
        )

    try:
        confidence_values = confidence_index.to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Confidence levels must be finite numbers "
            "strictly between 0 and 1."
        ) from error

    if (
        not np.isfinite(confidence_values).all()
        or (confidence_values <= 0.0).any()
        or (confidence_values >= 1.0).any()
    ):
        raise ValueError(
            "Confidence levels must be finite numbers "
            "strictly between 0 and 1."
        )

    try:
        risk_values = forecasts[
            [
                "realized_loss",
                "value_at_risk",
            ]
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Realized losses and VaR forecasts must "
            "contain numbers."
        ) from error

    if not np.isfinite(risk_values).all():
        raise ValueError(
            "Realized losses and VaR forecasts must "
            "contain finite numbers."
        )

    if (
        not pd.api.types.is_bool_dtype(
            forecasts["var_exceedance"]
        )
        or forecasts["var_exceedance"].isna().any()
    ):
        raise ValueError(
            "VaR exceedance indicators must be boolean."
        )

    calculated_exceedances = (
        forecasts["realized_loss"].to_numpy(dtype=float)
        > forecasts["value_at_risk"].to_numpy(dtype=float)
    )
    stored_exceedances = forecasts[
        "var_exceedance"
    ].to_numpy(dtype=bool)

    if not np.array_equal(
        stored_exceedances,
        calculated_exceedances,
    ):
        raise ValueError(
            "Stored VaR exceedance indicators must agree "
            "with realized losses and VaR forecasts."
        )

    records: list[dict[str, float | int]] = []

    for confidence_level, level_forecasts in (
        forecasts.groupby(
            level="confidence_level",
            sort=False,
        )
    ):
        observation_count = len(level_forecasts)
        exceedance_count = int(
            level_forecasts["var_exceedance"].sum()
        )
        expected_rate = 1.0 - float(
            confidence_level
        )
        observed_rate = (
            exceedance_count / observation_count
        )
        expected_count = (
            observation_count * expected_rate
        )

        records.append(
            {
                "confidence_level": float(
                    confidence_level
                ),
                "observation_count": observation_count,
                "exceedance_count": exceedance_count,
                "expected_exceedance_rate": (
                    expected_rate
                ),
                "observed_exceedance_rate": (
                    observed_rate
                ),
                "expected_exceedance_count": (
                    expected_count
                ),
                "exceedance_count_difference": (
                    exceedance_count - expected_count
                ),
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("confidence_level")
    )


def calculate_kupiec_unconditional_coverage(
    exceedance_count: int,
    observation_count: int,
    confidence_level: float,
    significance_level: float = 0.05,
) -> KupiecTestResult:
    """Test whether the unconditional VaR breach rate is correct.

    The null hypothesis is that the true exceedance probability equals
    one minus the VaR confidence level. The likelihood-ratio statistic
    has an asymptotic chi-square distribution with one degree of freedom.
    """

    if (
        isinstance(observation_count, bool)
        or not isinstance(observation_count, Integral)
        or observation_count <= 0
    ):
        raise ValueError(
            "Observation count must be a positive integer."
        )

    if (
        isinstance(exceedance_count, bool)
        or not isinstance(exceedance_count, Integral)
        or exceedance_count < 0
        or exceedance_count > observation_count
    ):
        raise ValueError(
            "Exceedance count must be an integer between "
            "zero and the observation count."
        )

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError(
            "Confidence level must be a finite real number "
            "strictly between 0 and 1."
        )

    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, Real)
        or not isfinite(significance_level)
        or not 0.0 < significance_level < 1.0
    ):
        raise ValueError(
            "Significance level must be a finite real number "
            "strictly between 0 and 1."
        )

    observation_count = int(observation_count)
    exceedance_count = int(exceedance_count)
    confidence_level = float(confidence_level)
    significance_level = float(significance_level)

    expected_rate = 1.0 - confidence_level
    observed_rate = exceedance_count / observation_count
    non_exceedance_count = (
        observation_count - exceedance_count
    )

    null_log_likelihood = (
        exceedance_count * log(expected_rate)
        + non_exceedance_count
        * log(1.0 - expected_rate)
    )

    alternative_log_likelihood = 0.0

    if exceedance_count > 0:
        alternative_log_likelihood += (
            exceedance_count * log(observed_rate)
        )

    if non_exceedance_count > 0:
        alternative_log_likelihood += (
            non_exceedance_count
            * log(1.0 - observed_rate)
        )

    likelihood_ratio = max(
        0.0,
        -2.0
        * (
            null_log_likelihood
            - alternative_log_likelihood
        ),
    )

    # For one degree of freedom, the chi-square survival function is
    # erfc(sqrt(x / 2)). This avoids adding a statistics dependency for
    # this single distribution while retaining the exact tail formula.
    p_value = erfc(
        sqrt(likelihood_ratio / 2.0)
    )

    return KupiecTestResult(
        confidence_level=confidence_level,
        observation_count=observation_count,
        exceedance_count=exceedance_count,
        expected_exceedance_rate=expected_rate,
        observed_exceedance_rate=observed_rate,
        likelihood_ratio=likelihood_ratio,
        p_value=p_value,
        significance_level=significance_level,
        reject_null=p_value < significance_level,
    )


def evaluate_var_unconditional_coverage(
    forecasts: pd.DataFrame,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Summarize exceedances and run Kupiec tests by VaR level."""

    summary = summarize_var_exceedances(
        forecasts=forecasts,
    )
    test_records: list[dict[str, float | bool]] = []

    for confidence_level, row in summary.iterrows():
        test_result = (
            calculate_kupiec_unconditional_coverage(
                exceedance_count=int(
                    row["exceedance_count"]
                ),
                observation_count=int(
                    row["observation_count"]
                ),
                confidence_level=float(confidence_level),
                significance_level=significance_level,
            )
        )
        test_records.append(
            {
                "kupiec_likelihood_ratio": (
                    test_result.likelihood_ratio
                ),
                "kupiec_p_value": test_result.p_value,
                "significance_level": (
                    test_result.significance_level
                ),
                "kupiec_reject_null": (
                    test_result.reject_null
                ),
            }
        )

    test_results = pd.DataFrame(
        test_records,
        index=summary.index,
    )

    return summary.join(test_results)
