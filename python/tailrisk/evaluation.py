"""Forecast diagnostics and comparative model evaluation."""

from dataclasses import dataclass
from math import erfc, exp, isfinite, log, sqrt
from numbers import Integral, Real
from scipy.stats import binomtest

import numpy as np
import pandas as pd



_ES_VAR_ORDER_TOLERANCE = 1e-12


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




@dataclass(frozen=True)
class ExactBinomialIntervalResult:
    """Exact confidence interval for a binomial proportion."""

    exceedance_count: int
    observation_count: int
    observed_exceedance_rate: float
    interval_confidence_level: float
    lower_bound: float
    upper_bound: float






@dataclass(frozen=True)
class VaRExceedanceTransitions:
    """Counts of consecutive VaR exceedance states."""

    n00: int
    n01: int
    n10: int
    n11: int

    @property
    def transition_count(self) -> int:
        """Return the total number of observed transitions."""

        return (
            self.n00
            + self.n01
            + self.n10
            + self.n11
        )




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




def calculate_exact_binomial_interval(
    exceedance_count: int,
    observation_count: int,
    interval_confidence_level: float = 0.95,
) -> ExactBinomialIntervalResult:
    """Calculate an exact interval for an exceedance rate.

    The interval uses the two-sided Clopper-Pearson method for a
    binomial proportion.
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
        isinstance(interval_confidence_level, bool)
        or not isinstance(
            interval_confidence_level,
            Real,
        )
        or not isfinite(interval_confidence_level)
        or not 0.0 < interval_confidence_level < 1.0
    ):
        raise ValueError(
            "Interval confidence level must be a finite "
            "real number strictly between 0 and 1."
        )

    observation_count = int(observation_count)
    exceedance_count = int(exceedance_count)
    interval_confidence_level = float(
        interval_confidence_level
    )

    binomial_result = binomtest(
        k=exceedance_count,
        n=observation_count,
    )
    interval = binomial_result.proportion_ci(
        confidence_level=interval_confidence_level,
        method="exact",
    )

    return ExactBinomialIntervalResult(
        exceedance_count=exceedance_count,
        observation_count=observation_count,
        observed_exceedance_rate=(
            exceedance_count / observation_count
        ),
        interval_confidence_level=(
            interval_confidence_level
        ),
        lower_bound=float(interval.low),
        upper_bound=float(interval.high),
    )






def summarize_var_exceedances(
    forecasts: pd.DataFrame,
    interval_confidence_level: float = 0.95,
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
        interval = calculate_exact_binomial_interval(
            exceedance_count=exceedance_count,
            observation_count=observation_count,
            interval_confidence_level=(
                interval_confidence_level
            ),
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
                "interval_confidence_level": (
                    interval.interval_confidence_level
                ),
                "exceedance_rate_ci_lower": (
                    interval.lower_bound
                ),
                "exceedance_rate_ci_upper": (
                    interval.upper_bound
                ),
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("confidence_level")
    )




def summarize_es_tail_severity(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize loss severity on VaR-exceedance sessions.

    The returned table contains one row per confidence level.
    Statistics conditional on a VaR exceedance are missing when
    no exceedance is observed.
    """

    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError(
            "Forecasts must be a pandas DataFrame."
        )

    if forecasts.empty:
        raise ValueError(
            "Forecasts must not be empty."
        )

    if "expected_shortfall" not in forecasts.columns:
        raise ValueError(
            "Forecasts are missing required column: "
            "expected_shortfall."
        )

    # Reuse the established VaR validation contract. This checks the
    # index, confidence levels, loss and VaR values, boolean indicators,
    # and agreement with the strict realized_loss > value_at_risk rule.
    summarize_var_exceedances(
        forecasts=forecasts,
    )

    expected_shortfall = forecasts[
        "expected_shortfall"
    ]

    if any(
        isinstance(value, (bool, np.bool_))
        for value in expected_shortfall
    ):
        raise ValueError(
            "Expected Shortfall forecasts must contain "
            "finite numbers."
        )

    try:
        expected_shortfall_values = (
            expected_shortfall.to_numpy(dtype=float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Expected Shortfall forecasts must contain "
            "finite numbers."
        ) from error

    if not np.isfinite(
        expected_shortfall_values
    ).all():
        raise ValueError(
            "Expected Shortfall forecasts must contain "
            "finite numbers."
        )

    value_at_risk_values = forecasts[
        "value_at_risk"
    ].to_numpy(dtype=float)

    if (
        expected_shortfall_values
        < (
            value_at_risk_values
            - _ES_VAR_ORDER_TOLERANCE
        )
    ).any():
        raise ValueError(
            "Expected Shortfall forecasts must be at "
            "least as large as VaR forecasts."
        )

    records: list[dict[str, float | int]] = []

    for confidence_level, level_forecasts in (
        forecasts.groupby(
            level="confidence_level",
            sort=False,
        )
    ):
        observation_count = len(level_forecasts)
        exceedance_forecasts = level_forecasts.loc[
            level_forecasts["var_exceedance"]
        ]
        exceedance_count = len(
            exceedance_forecasts
        )

        if exceedance_count == 0:
            mean_exceedance_loss = float("nan")
            mean_var_on_exceedance_days = float("nan")
            mean_es_on_exceedance_days = float("nan")
            mean_var_excess = float("nan")
            median_var_excess = float("nan")
            maximum_var_excess = float("nan")
            mean_es_gap = float("nan")
        else:
            exceedance_losses = exceedance_forecasts[
                "realized_loss"
            ].to_numpy(dtype=float)
            exceedance_var = exceedance_forecasts[
                "value_at_risk"
            ].to_numpy(dtype=float)
            exceedance_es = exceedance_forecasts[
                "expected_shortfall"
            ].to_numpy(dtype=float)

            var_excess = (
                exceedance_losses - exceedance_var
            )
            es_gap = (
                exceedance_losses - exceedance_es
            )

            mean_exceedance_loss = float(
                exceedance_losses.mean()
            )
            mean_var_on_exceedance_days = float(
                exceedance_var.mean()
            )
            mean_es_on_exceedance_days = float(
                exceedance_es.mean()
            )
            mean_var_excess = float(
                var_excess.mean()
            )
            median_var_excess = float(
                np.median(var_excess)
            )
            maximum_var_excess = float(
                var_excess.max()
            )
            mean_es_gap = float(
                es_gap.mean()
            )

        records.append(
            {
                "confidence_level": float(
                    confidence_level
                ),
                "observation_count": (
                    observation_count
                ),
                "var_exceedance_count": (
                    exceedance_count
                ),
                "mean_exceedance_loss": (
                    mean_exceedance_loss
                ),
                "mean_var_on_exceedance_days": (
                    mean_var_on_exceedance_days
                ),
                "mean_es_on_exceedance_days": (
                    mean_es_on_exceedance_days
                ),
                "mean_var_excess": (
                    mean_var_excess
                ),
                "median_var_excess": (
                    median_var_excess
                ),
                "maximum_var_excess": (
                    maximum_var_excess
                ),
                "mean_es_gap": mean_es_gap,
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("confidence_level")
    )





def evaluate_models_es_tail_severity(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize aligned ES tail severity across multiple models.

    Each model contributes one row per confidence level. Models must
    use identical ordered forecast indexes and realized portfolio losses.
    """

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if not forecasts_by_model:
        raise ValueError(
            "At least one model forecast table is required."
        )

    severity_tables: dict[str, pd.DataFrame] = {}
    reference_index: pd.Index | None = None
    reference_losses: np.ndarray | None = None
    reference_model: str | None = None

    for model_name, forecasts in (
        forecasts_by_model.items()
    ):
        if (
            not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise ValueError(
                "Each model name must be a non-empty string."
            )

        if not isinstance(forecasts, pd.DataFrame):
            raise TypeError(
                "Each model's forecasts must be a pandas DataFrame."
            )

        if (
            reference_index is not None
            and not forecasts.index.equals(
                reference_index
            )
        ):
            raise ValueError(
                "All models must use the same ordered "
                "forecast index."
            )

        severity_table = summarize_es_tail_severity(
            forecasts=forecasts,
        )

        if reference_index is None:
            reference_index = forecasts.index
            reference_losses = forecasts[
                "realized_loss"
            ].to_numpy(dtype=float)
            reference_model = model_name
        else:
            model_losses = forecasts[
                "realized_loss"
            ].to_numpy(dtype=float)

            if not np.allclose(
                model_losses,
                reference_losses,
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError(
                    "All models must use identical realized "
                    f"losses; {model_name} differs from "
                    f"{reference_model}."
                )

        severity_tables[model_name] = (
            severity_table
        )

    return pd.concat(
        severity_tables,
        names=["model"],
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
    interval_confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Summarize exceedances and run Kupiec tests by VaR level."""

    summary = summarize_var_exceedances(
        forecasts=forecasts,
        interval_confidence_level=(
            interval_confidence_level
        ),
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


def adjust_p_values_holm(
    p_values: pd.Series,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Adjust a family of p-values using Holm's procedure.

    The returned table preserves the original index and includes the
    raw p-value, adjusted p-value, and adjusted rejection decision.
    """

    if not isinstance(p_values, pd.Series):
        raise TypeError(
            "P-values must be a pandas Series."
        )

    if p_values.empty:
        raise ValueError(
            "P-values must not be empty."
        )

    if p_values.index.has_duplicates:
        raise ValueError(
            "P-value index entries must be unique."
        )

    if pd.api.types.is_bool_dtype(p_values):
        raise ValueError(
            "P-values must be finite numbers between zero and one."
        )

    try:
        values = p_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "P-values must be finite numbers between zero and one."
        ) from error

    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError(
            "P-values must be finite numbers between zero and one."
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

    test_count = len(values)

    sorted_positions = np.argsort(
        values,
        kind="stable",
    )
    sorted_p_values = values[
        sorted_positions
    ]
    multipliers = np.arange(
        test_count,
        0,
        -1,
        dtype=float,
    )

    adjusted_sorted = (
        sorted_p_values * multipliers
    )
    adjusted_sorted = np.maximum.accumulate(
        adjusted_sorted
    )
    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0,
    )

    adjusted_values = np.empty(
        test_count,
        dtype=float,
    )
    adjusted_values[
        sorted_positions
    ] = adjusted_sorted

    return pd.DataFrame(
        {
            "raw_p_value": values,
            "holm_adjusted_p_value": adjusted_values,
            "reject_null_holm": (
                adjusted_values
                < float(significance_level)
            ),
        },
        index=p_values.index.copy(),
    )




def count_var_exceedance_transitions(
    exceedances: pd.Series,
) -> VaRExceedanceTransitions:
    """Count consecutive states in an ordered VaR breach sequence.

    The first digit represents the preceding session and the second
    digit represents the current session. False is state zero and
    True is state one.
    """

    if not isinstance(exceedances, pd.Series):
        raise TypeError(
            "Exceedances must be a pandas Series."
        )

    if len(exceedances) < 2:
        raise ValueError(
            "Exceedances must contain at least two observations."
        )

    if (
        not pd.api.types.is_bool_dtype(exceedances)
        or exceedances.isna().any()
    ):
        raise ValueError(
            "Exceedance indicators must be boolean values."
        )

    if exceedances.index.has_duplicates:
        raise ValueError(
            "Exceedance index entries must be unique."
        )

    if not exceedances.index.is_monotonic_increasing:
        raise ValueError(
            "Exceedance observations must be ordered by "
            "increasing index."
        )

    values = exceedances.to_numpy(dtype=bool)
    previous_states = values[:-1]
    current_states = values[1:]

    n00 = int(
        np.count_nonzero(
            ~previous_states & ~current_states
        )
    )
    n01 = int(
        np.count_nonzero(
            ~previous_states & current_states
        )
    )
    n10 = int(
        np.count_nonzero(
            previous_states & ~current_states
        )
    )
    n11 = int(
        np.count_nonzero(
            previous_states & current_states
        )
    )

    return VaRExceedanceTransitions(
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
    )



@dataclass(frozen=True)
class ChristoffersenIndependenceTestResult:
    """Result of a Christoffersen exceedance-independence test."""

    transitions: VaRExceedanceTransitions
    transition_breach_probability: float
    breach_probability_after_no_breach: float | None
    breach_probability_after_breach: float | None
    likelihood_ratio: float
    p_value: float
    significance_level: float
    reject_null: bool




@dataclass(frozen=True)
class ChristoffersenConditionalCoverageTestResult:
    """Result of a Christoffersen conditional-coverage test."""

    unconditional_coverage: KupiecTestResult
    independence: ChristoffersenIndependenceTestResult
    likelihood_ratio: float
    p_value: float
    significance_level: float
    reject_null: bool





def _calculate_bernoulli_log_likelihood(
    success_count: int,
    failure_count: int,
    probability: float,
) -> float:
    """Calculate a boundary-safe Bernoulli log likelihood."""

    log_likelihood = 0.0

    if success_count > 0:
        log_likelihood += (
            success_count * log(probability)
        )

    if failure_count > 0:
        log_likelihood += (
            failure_count * log(1.0 - probability)
        )

    return log_likelihood



def calculate_christoffersen_independence(
    exceedances: pd.Series,
    significance_level: float = 0.05,
) -> ChristoffersenIndependenceTestResult:
    """Test whether VaR exceedances are independent through time."""

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

    significance_level = float(significance_level)

    transitions = count_var_exceedance_transitions(
        exceedances=exceedances,
    )

    breach_transition_count = (
        transitions.n01 + transitions.n11
    )
    non_breach_transition_count = (
        transitions.n00 + transitions.n10
    )
    transition_count = transitions.transition_count

    transition_breach_probability = (
        breach_transition_count / transition_count
    )

    after_no_breach_count = (
        transitions.n00 + transitions.n01
    )
    after_breach_count = (
        transitions.n10 + transitions.n11
    )

    probability_after_no_breach: float | None = None
    probability_after_breach: float | None = None

    null_log_likelihood = (
        _calculate_bernoulli_log_likelihood(
            success_count=breach_transition_count,
            failure_count=non_breach_transition_count,
            probability=transition_breach_probability,
        )
    )

    alternative_log_likelihood = 0.0

    if after_no_breach_count > 0:
        probability_after_no_breach = (
            transitions.n01 / after_no_breach_count
        )
        alternative_log_likelihood += (
            _calculate_bernoulli_log_likelihood(
                success_count=transitions.n01,
                failure_count=transitions.n00,
                probability=probability_after_no_breach,
            )
        )

    if after_breach_count > 0:
        probability_after_breach = (
            transitions.n11 / after_breach_count
        )
        alternative_log_likelihood += (
            _calculate_bernoulli_log_likelihood(
                success_count=transitions.n11,
                failure_count=transitions.n10,
                probability=probability_after_breach,
            )
        )

    likelihood_ratio = max(
        0.0,
        -2.0
        * (
            null_log_likelihood
            - alternative_log_likelihood
        ),
    )

    p_value = erfc(
        sqrt(likelihood_ratio / 2.0)
    )

    return ChristoffersenIndependenceTestResult(
        transitions=transitions,
        transition_breach_probability=(
            transition_breach_probability
        ),
        breach_probability_after_no_breach=(
            probability_after_no_breach
        ),
        breach_probability_after_breach=(
            probability_after_breach
        ),
        likelihood_ratio=likelihood_ratio,
        p_value=p_value,
        significance_level=significance_level,
        reject_null=p_value < significance_level,
    )




def calculate_christoffersen_conditional_coverage(
    exceedances: pd.Series,
    confidence_level: float,
    significance_level: float = 0.05,
) -> ChristoffersenConditionalCoverageTestResult:
    """Test correct VaR coverage and exceedance independence jointly.

    The conditional-coverage likelihood ratio is the sum of the
    Kupiec unconditional-coverage statistic and the Christoffersen
    independence statistic. Under the joint null hypothesis, it has
    an asymptotic chi-square distribution with two degrees of freedom.
    """

    independence_result = (
        calculate_christoffersen_independence(
            exceedances=exceedances,
            significance_level=significance_level,
        )
    )

    unconditional_result = (
        calculate_kupiec_unconditional_coverage(
            exceedance_count=int(exceedances.sum()),
            observation_count=len(exceedances),
            confidence_level=confidence_level,
            significance_level=significance_level,
        )
    )

    likelihood_ratio = (
        unconditional_result.likelihood_ratio
        + independence_result.likelihood_ratio
    )

    # The chi-square survival function with two degrees of freedom
    # simplifies to exp(-x / 2).
    p_value = exp(
        -likelihood_ratio / 2.0
    )

    return ChristoffersenConditionalCoverageTestResult(
        unconditional_coverage=unconditional_result,
        independence=independence_result,
        likelihood_ratio=likelihood_ratio,
        p_value=p_value,
        significance_level=float(significance_level),
        reject_null=p_value < significance_level,
    )




def evaluate_var_coverage(
    forecasts: pd.DataFrame,
    significance_level: float = 0.05,
    interval_confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Evaluate VaR coverage and independence by confidence level.

    The returned table combines descriptive exceedance summaries,
    Kupiec unconditional coverage, Christoffersen independence, and
    Christoffersen conditional coverage.
    """

    summary = summarize_var_exceedances(
        forecasts=forecasts,
        interval_confidence_level=(
            interval_confidence_level
        ),
    )
    test_records: list[
        dict[str, float | int | bool | None]
    ] = []

    for confidence_level in summary.index:
        level_forecasts = forecasts.xs(
            confidence_level,
            level="confidence_level",
        )
        exceedances = level_forecasts[
            "var_exceedance"
        ]

        conditional_result = (
            calculate_christoffersen_conditional_coverage(
                exceedances=exceedances,
                confidence_level=float(confidence_level),
                significance_level=significance_level,
            )
        )
        unconditional_result = (
            conditional_result.unconditional_coverage
        )
        independence_result = (
            conditional_result.independence
        )
        transitions = independence_result.transitions

        test_records.append(
            {
                "transition_n00": transitions.n00,
                "transition_n01": transitions.n01,
                "transition_n10": transitions.n10,
                "transition_n11": transitions.n11,
                "transition_breach_probability": (
                    independence_result
                    .transition_breach_probability
                ),
                "breach_probability_after_no_breach": (
                    independence_result
                    .breach_probability_after_no_breach
                ),
                "breach_probability_after_breach": (
                    independence_result
                    .breach_probability_after_breach
                ),
                "kupiec_likelihood_ratio": (
                    unconditional_result.likelihood_ratio
                ),
                "kupiec_p_value": (
                    unconditional_result.p_value
                ),
                "kupiec_reject_null": (
                    unconditional_result.reject_null
                ),
                "independence_likelihood_ratio": (
                    independence_result.likelihood_ratio
                ),
                "independence_p_value": (
                    independence_result.p_value
                ),
                "independence_reject_null": (
                    independence_result.reject_null
                ),
                "conditional_coverage_likelihood_ratio": (
                    conditional_result.likelihood_ratio
                ),
                "conditional_coverage_p_value": (
                    conditional_result.p_value
                ),
                "conditional_coverage_reject_null": (
                    conditional_result.reject_null
                ),
                "significance_level": (
                    conditional_result.significance_level
                ),
            }
        )

    test_results = pd.DataFrame(
        test_records,
        index=summary.index,
    )

    return summary.join(test_results)


def evaluate_models_var_coverage(
    forecasts_by_model: dict[str, pd.DataFrame],
    significance_level: float = 0.05,
    interval_confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Evaluate aligned VaR forecasts across multiple models.

    Each model receives one row per confidence level. Holm adjustment
    is applied to the complete family of conditional-coverage tests.
    """

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if not forecasts_by_model:
        raise ValueError(
            "At least one model forecast table is required."
        )

    coverage_tables: dict[str, pd.DataFrame] = {}
    reference_index: pd.Index | None = None
    reference_losses: np.ndarray | None = None
    reference_model: str | None = None

    for model_name, forecasts in (
        forecasts_by_model.items()
    ):
        if (
            not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise ValueError(
                "Each model name must be a non-empty string."
            )

        if not isinstance(forecasts, pd.DataFrame):
            raise TypeError(
                "Each model's forecasts must be a pandas DataFrame."
            )

        if (
            reference_index is not None
            and not forecasts.index.equals(reference_index)
        ):
            raise ValueError(
                "All models must use the same ordered "
                "forecast index."
            )

        coverage_table = evaluate_var_coverage(
            forecasts=forecasts,
            significance_level=significance_level,
            interval_confidence_level=(
                interval_confidence_level
            ),
        )

        if reference_index is None:
            reference_index = forecasts.index
            reference_losses = forecasts[
                "realized_loss"
            ].to_numpy(dtype=float)
            reference_model = model_name
        else:
            model_losses = forecasts[
                "realized_loss"
            ].to_numpy(dtype=float)

            if not np.allclose(
                model_losses,
                reference_losses,
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError(
                    "All models must use identical realized "
                    f"losses; {model_name} differs from "
                    f"{reference_model}."
                )

        coverage_tables[model_name] = coverage_table

    combined = pd.concat(
        coverage_tables,
        names=["model"],
    )

    holm_results = adjust_p_values_holm(
        p_values=combined[
            "conditional_coverage_p_value"
        ],
        significance_level=significance_level,
    )

    combined[
        "conditional_coverage_holm_p_value"
    ] = holm_results[
        "holm_adjusted_p_value"
    ]
    combined[
        "conditional_coverage_reject_null_holm"
    ] = holm_results[
        "reject_null_holm"
    ]

    return combined
