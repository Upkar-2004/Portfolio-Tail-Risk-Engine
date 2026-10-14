"""Forecast diagnostics and comparative model evaluation."""

from dataclasses import dataclass
from math import erfc, exp, isfinite, log, sqrt
from numbers import Integral, Real
from scipy.stats import binomtest

import numpy as np
import pandas as pd



_ES_VAR_ORDER_TOLERANCE = 1e-12
_BASEL_TRAFFIC_LIGHT_CONFIDENCE_LEVEL = 0.99
_BASEL_TRAFFIC_LIGHT_WINDOW = 250
_SUBPERIOD_FIRST_CALENDAR_YEAR = 2012
_SUBPERIOD_LAST_CALENDAR_YEAR = 2025
_SUBPERIOD_PARTIAL_YEAR_FRACTION = 0.90
_NBER_BUSINESS_CYCLE_SOURCE = (
    "https://www.nber.org/research/business-cycle-dating"
)
_NBER_RECESSION_PERIODS = (
    {
        "period_id": "nber_recession_2020",
        "label": "NBER recession: February-April 2020",
        "requested_start_date": pd.Timestamp("2020-02-01"),
        "requested_end_date": pd.Timestamp("2020-04-30"),
    },
)


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





@dataclass(frozen=True)
class NeweyWestMeanTestResult:
    """Result of a Newey-West HAC test of a population mean."""

    observation_count: int
    bandwidth: int
    sample_mean: float
    long_run_variance: float
    mean_variance: float
    standard_error: float
    test_statistic: float
    p_value: float
    significance_level: float
    reject_null: bool






@dataclass(frozen=True)
class DieboldMarianoTestResult:
    """Result of a pairwise Diebold-Mariano score comparison."""

    observation_count: int
    bandwidth: int
    mean_score_difference: float
    long_run_variance: float
    mean_variance: float
    standard_error: float
    test_statistic: float
    p_value: float
    significance_level: float
    reject_null: bool





def _extract_common_forecast_dates(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """Validate aligned model panels and return common forecast dates."""

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if not forecasts_by_model:
        raise ValueError(
            "At least one model forecast table is required."
        )

    reference_index: pd.MultiIndex | None = None
    reference_losses: np.ndarray | None = None
    reference_model: str | None = None
    common_dates: pd.DatetimeIndex | None = None

    for model_name, forecasts in forecasts_by_model.items():
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

        if forecasts.empty:
            raise ValueError(
                "Each model's forecasts must not be empty."
            )

        if (
            not isinstance(forecasts.index, pd.MultiIndex)
            or list(forecasts.index.names)
            != ["forecast_date", "confidence_level"]
        ):
            raise ValueError(
                "Forecasts must use forecast_date and "
                "confidence_level as their ordered index levels."
            )

        if forecasts.index.has_duplicates:
            raise ValueError(
                "Forecast index entries must be unique."
            )

        if not forecasts.index.is_monotonic_increasing:
            raise ValueError(
                "Forecast rows must be ordered by forecast date and "
                "confidence level."
            )

        if "realized_loss" not in forecasts.columns:
            raise ValueError(
                "Forecasts are missing required column: "
                "realized_loss."
            )

        date_level = forecasts.index.get_level_values(
            "forecast_date"
        )

        if not isinstance(date_level, pd.DatetimeIndex):
            raise ValueError(
                "Forecast dates must be pandas datetime values."
            )

        if date_level.hasnans:
            raise ValueError(
                "Forecast dates must not contain missing values."
            )

        if date_level.tz is not None:
            raise ValueError(
                "Forecast dates must be timezone-naive."
            )

        try:
            realized_losses = forecasts[
                "realized_loss"
            ].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Realized losses must contain numbers."
            ) from error

        if not np.isfinite(realized_losses).all():
            raise ValueError(
                "Realized losses must contain finite numbers."
            )

        loss_table = forecasts[
            "realized_loss"
        ].unstack("confidence_level")

        if loss_table.isna().any().any():
            raise ValueError(
                "Every forecast date must contain every confidence "
                "level."
            )

        reference_date_losses = loss_table.iloc[
            :, 0
        ].to_numpy(dtype=float)

        if not np.allclose(
            loss_table.to_numpy(dtype=float),
            reference_date_losses[:, np.newaxis],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "Realized losses must be identical across confidence "
                "levels on each forecast date."
            )

        if (
            reference_index is not None
            and not forecasts.index.equals(reference_index)
        ):
            raise ValueError(
                "All models must use the same ordered forecast index."
            )

        if reference_index is None:
            reference_index = forecasts.index
            reference_losses = realized_losses
            reference_model = model_name
            common_dates = pd.DatetimeIndex(
                loss_table.index.copy(),
                name="forecast_date",
            )
        elif not np.allclose(
            realized_losses,
            reference_losses,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "All models must use identical realized losses; "
                f"{model_name} differs from {reference_model}."
            )

    if common_dates is None:
        raise RuntimeError(
            "Common forecast dates could not be constructed."
        )

    return common_dates





def create_subperiod_definitions(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create predeclared calendar and NBER subperiod definitions.

    All model panels must contain the same ordered forecast rows and
    realized losses. Calendar periods cover 2012 through 2025. The
    external event period uses the documented NBER monthly chronology.
    """

    common_dates = _extract_common_forecast_dates(
        forecasts_by_model=forecasts_by_model,
    )
    calendar_records: list[dict[str, object]] = []

    for year in range(
        _SUBPERIOD_FIRST_CALENDAR_YEAR,
        _SUBPERIOD_LAST_CALENDAR_YEAR + 1,
    ):
        requested_start = pd.Timestamp(
            year=year,
            month=1,
            day=1,
        )
        requested_end = pd.Timestamp(
            year=year,
            month=12,
            day=31,
        )
        period_dates = common_dates[
            (common_dates >= requested_start)
            & (common_dates <= requested_end)
        ]

        if period_dates.empty:
            raise ValueError(
                "Common forecast dates must cover every calendar year "
                "from 2012 through 2025."
            )

        calendar_records.append(
            {
                "period_id": f"calendar_{year}",
                "period_type": "calendar_year",
                "label": f"Calendar year {year}",
                "requested_start_date": requested_start,
                "requested_end_date": requested_end,
                "first_forecast_date": period_dates[0],
                "last_forecast_date": period_dates[-1],
                "observation_count": len(period_dates),
                "is_partial": False,
                "source": "predeclared_calendar_year_rule",
            }
        )

    maximum_annual_count = max(
        int(record["observation_count"])
        for record in calendar_records
    )

    for record in calendar_records:
        record["is_partial"] = bool(
            int(record["observation_count"])
            < (
                _SUBPERIOD_PARTIAL_YEAR_FRACTION
                * maximum_annual_count
            )
        )

    stress_records: list[dict[str, object]] = []

    for recession in _NBER_RECESSION_PERIODS:
        requested_start = recession[
            "requested_start_date"
        ]
        requested_end = recession[
            "requested_end_date"
        ]
        period_dates = common_dates[
            (common_dates >= requested_start)
            & (common_dates <= requested_end)
        ]

        if period_dates.empty:
            raise ValueError(
                "The predeclared NBER recession must intersect the "
                "common forecast dates."
            )

        stress_records.append(
            {
                "period_id": recession["period_id"],
                "period_type": "nber_recession",
                "label": recession["label"],
                "requested_start_date": requested_start,
                "requested_end_date": requested_end,
                "first_forecast_date": period_dates[0],
                "last_forecast_date": period_dates[-1],
                "observation_count": len(period_dates),
                "is_partial": False,
                "source": _NBER_BUSINESS_CYCLE_SOURCE,
            }
        )

    definitions = pd.DataFrame(
        calendar_records + stress_records
    ).set_index("period_id")
    definitions.index.name = "period_id"

    return definitions





def classify_basel_traffic_light(
    exception_count: int,
) -> str:
    """Classify 99% VaR exceptions in a 250-session Basel window.

    Zero through four exceptions are green, five through nine are
    amber, and ten or more are red.
    """

    if (
        isinstance(exception_count, bool)
        or not isinstance(exception_count, Integral)
        or exception_count < 0
        or exception_count > _BASEL_TRAFFIC_LIGHT_WINDOW
    ):
        raise ValueError(
            "Exception count must be an integer between "
            "zero and 250."
        )

    exception_count = int(exception_count)

    if exception_count <= 4:
        return "green"

    if exception_count <= 9:
        return "amber"

    return "red"





def calculate_basel_traffic_light_path(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate rolling 250-session zones for 99% VaR forecasts.

    The returned index contains each eligible window's ending forecast
    date. Partial windows are not reported.
    """

    # Reuse the established validation for the forecast index, finite
    # losses and VaR values, and strict exceedance indicators.
    summarize_var_exceedances(
        forecasts=forecasts,
    )

    if list(forecasts.index.names) != [
        "forecast_date",
        "confidence_level",
    ]:
        raise ValueError(
            "Forecasts must use forecast_date and confidence_level "
            "as their ordered index levels."
        )

    confidence_levels = (
        forecasts.index.get_level_values(
            "confidence_level"
        )
    )
    level_mask = (
        confidence_levels
        == _BASEL_TRAFFIC_LIGHT_CONFIDENCE_LEVEL
    )

    if not level_mask.any():
        raise ValueError(
            "Forecasts must contain 99% VaR rows for the Basel "
            "traffic-light diagnostic."
        )

    level_forecasts = forecasts.loc[level_mask].copy()
    level_forecasts.index = (
        level_forecasts.index.droplevel(
            "confidence_level"
        )
    )

    if not level_forecasts.index.is_monotonic_increasing:
        raise ValueError(
            "The 99% VaR forecasts must be ordered by increasing "
            "forecast date."
        )

    if len(level_forecasts) < _BASEL_TRAFFIC_LIGHT_WINDOW:
        raise ValueError(
            "At least 250 forecast sessions are required for the "
            "Basel traffic-light diagnostic."
        )

    rolling_counts = (
        level_forecasts["var_exceedance"]
        .astype(int)
        .rolling(
            window=_BASEL_TRAFFIC_LIGHT_WINDOW,
            min_periods=_BASEL_TRAFFIC_LIGHT_WINDOW,
        )
        .sum()
        .iloc[_BASEL_TRAFFIC_LIGHT_WINDOW - 1:]
        .astype(int)
    )
    window_count = len(rolling_counts)
    window_start_dates = level_forecasts.index[
        :window_count
    ]
    zones = [
        classify_basel_traffic_light(
            exception_count=int(exception_count),
        )
        for exception_count in rolling_counts
    ]

    return pd.DataFrame(
        {
            "window_start_date": window_start_dates,
            "window_observation_count": (
                _BASEL_TRAFFIC_LIGHT_WINDOW
            ),
            "confidence_level": (
                _BASEL_TRAFFIC_LIGHT_CONFIDENCE_LEVEL
            ),
            "exception_count": rolling_counts.to_numpy(
                dtype=int
            ),
            "traffic_light_zone": zones,
        },
        index=rolling_counts.index.copy(),
    )





def evaluate_models_basel_traffic_light(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine aligned model traffic-light paths.

    Models must use identical ordered forecast rows and realized losses.
    The returned index contains model and rolling-window end date.
    """

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if not forecasts_by_model:
        raise ValueError(
            "At least one model forecast table is required."
        )

    paths_by_model: dict[str, pd.DataFrame] = {}
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

        model_path = calculate_basel_traffic_light_path(
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

        paths_by_model[model_name] = model_path

    return pd.concat(
        paths_by_model,
        names=["model"],
    )


def summarize_models_basel_traffic_light(
    traffic_light_paths: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize rolling Basel-style zones for each model."""

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

    if traffic_light_paths.index.has_duplicates:
        raise ValueError(
            "Traffic-light path index entries must be unique."
        )

    required_columns = {
        "window_start_date",
        "window_observation_count",
        "confidence_level",
        "exception_count",
        "traffic_light_zone",
    }
    missing_columns = (
        required_columns - set(traffic_light_paths.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Traffic-light paths are missing required "
            f"column(s): {missing_names}."
        )

    if not (
        traffic_light_paths["window_observation_count"]
        == _BASEL_TRAFFIC_LIGHT_WINDOW
    ).all():
        raise ValueError(
            "Traffic-light paths must use 250-session windows."
        )

    if not np.allclose(
        traffic_light_paths["confidence_level"].to_numpy(
            dtype=float
        ),
        _BASEL_TRAFFIC_LIGHT_CONFIDENCE_LEVEL,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Traffic-light paths must use 99% VaR forecasts."
        )

    exception_counts = traffic_light_paths[
        "exception_count"
    ]

    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        for value in exception_counts
    ):
        raise ValueError(
            "Traffic-light exception counts must be integers."
        )

    expected_zones = pd.Series(
        [
            classify_basel_traffic_light(
                exception_count=int(value),
            )
            for value in exception_counts
        ],
        index=traffic_light_paths.index,
    )

    if not expected_zones.equals(
        traffic_light_paths[
            "traffic_light_zone"
        ].astype(str)
    ):
        raise ValueError(
            "Traffic-light zones must agree with exception counts."
        )

    records: list[dict[str, object]] = []

    for model_name, model_path in (
        traffic_light_paths.groupby(
            level="model",
            sort=False,
        )
    ):
        model_path = model_path.droplevel("model")

        if not model_path.index.is_monotonic_increasing:
            raise ValueError(
                "Each model's traffic-light path must be ordered "
                "by increasing forecast date."
            )

        window_count = len(model_path)
        zone_counts = (
            model_path["traffic_light_zone"]
            .value_counts()
        )
        green_count = int(
            zone_counts.get("green", 0)
        )
        amber_count = int(
            zone_counts.get("amber", 0)
        )
        red_count = int(
            zone_counts.get("red", 0)
        )
        maximum_exception_count = int(
            model_path["exception_count"].max()
        )
        maximum_mask = (
            model_path["exception_count"]
            == maximum_exception_count
        )
        maximum_exception_date = (
            model_path.index[maximum_mask][0]
        )
        latest = model_path.iloc[-1]
        amber_dates = model_path.index[
            model_path["traffic_light_zone"]
            == "amber"
        ]
        red_dates = model_path.index[
            model_path["traffic_light_zone"]
            == "red"
        ]

        records.append(
            {
                "model": model_name,
                "window_count": window_count,
                "latest_forecast_date": (
                    model_path.index[-1]
                ),
                "latest_exception_count": int(
                    latest["exception_count"]
                ),
                "latest_zone": latest[
                    "traffic_light_zone"
                ],
                "maximum_exception_count": (
                    maximum_exception_count
                ),
                "maximum_exception_date": (
                    maximum_exception_date
                ),
                "worst_zone": (
                    classify_basel_traffic_light(
                        exception_count=(
                            maximum_exception_count
                        ),
                    )
                ),
                "green_window_count": green_count,
                "amber_window_count": amber_count,
                "red_window_count": red_count,
                "green_window_rate": (
                    green_count / window_count
                ),
                "amber_window_rate": (
                    amber_count / window_count
                ),
                "red_window_rate": (
                    red_count / window_count
                ),
                "first_amber_date": (
                    amber_dates[0]
                    if len(amber_dates) > 0
                    else pd.NaT
                ),
                "first_red_date": (
                    red_dates[0]
                    if len(red_dates) > 0
                    else pd.NaT
                ),
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("model")
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





def evaluate_models_subperiod_var_coverage(
    forecasts_by_model: dict[str, pd.DataFrame],
    interval_confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Summarize descriptive VaR coverage in predeclared periods.

    The result contains calendar-year and NBER rows for every aligned
    model-confidence pair. Exact intervals are pointwise diagnostics;
    this function performs no hypothesis tests or rejection decisions.
    """

    definitions = create_subperiod_definitions(
        forecasts_by_model=forecasts_by_model,
    )
    records: list[
        dict[str, bool | float | int | str | pd.Timestamp]
    ] = []

    for period_id, definition in definitions.iterrows():
        requested_start = pd.Timestamp(
            definition["requested_start_date"]
        )
        requested_end = pd.Timestamp(
            definition["requested_end_date"]
        )

        for model_name, forecasts in (
            forecasts_by_model.items()
        ):
            forecast_dates = (
                forecasts.index.get_level_values(
                    "forecast_date"
                )
            )
            period_mask = (
                (forecast_dates >= requested_start)
                & (forecast_dates <= requested_end)
            )
            period_forecasts = forecasts.loc[
                period_mask
            ]

            # This reuses the established positive-loss VaR/ES
            # validation, including the strict exception rule and
            # Expected Shortfall greater-than-or-equal-to-VaR check.
            summarize_es_tail_severity(
                forecasts=period_forecasts,
            )
            coverage = summarize_var_exceedances(
                forecasts=period_forecasts,
                interval_confidence_level=(
                    interval_confidence_level
                ),
            )
            mean_var = period_forecasts[
                "value_at_risk"
            ].groupby(
                level="confidence_level",
                sort=False,
            ).mean()
            mean_es = period_forecasts[
                "expected_shortfall"
            ].groupby(
                level="confidence_level",
                sort=False,
            ).mean()

            for confidence_level, summary in (
                coverage.iterrows()
            ):
                observation_count = int(
                    summary["observation_count"]
                )
                expected_count = float(
                    summary[
                        "expected_exceedance_count"
                    ]
                )
                observed_count = int(
                    summary["exceedance_count"]
                )
                nominal_rate = float(
                    summary[
                        "expected_exceedance_rate"
                    ]
                )
                observed_rate = float(
                    summary[
                        "observed_exceedance_rate"
                    ]
                )

                if observation_count != int(
                    definition["observation_count"]
                ):
                    raise ValueError(
                        "Subperiod observation counts must agree "
                        "with the period definitions."
                    )

                records.append(
                    {
                        "period_id": str(period_id),
                        "model": model_name,
                        "confidence_level": float(
                            confidence_level
                        ),
                        "period_type": str(
                            definition["period_type"]
                        ),
                        "first_forecast_date": pd.Timestamp(
                            definition[
                                "first_forecast_date"
                            ]
                        ),
                        "last_forecast_date": pd.Timestamp(
                            definition[
                                "last_forecast_date"
                            ]
                        ),
                        "observation_count": observation_count,
                        "is_partial": bool(
                            definition["is_partial"]
                        ),
                        "nominal_exceedance_rate": nominal_rate,
                        "expected_exceedance_count": (
                            expected_count
                        ),
                        "observed_exceedance_count": (
                            observed_count
                        ),
                        "observed_exceedance_rate": (
                            observed_rate
                        ),
                        "observed_minus_expected_count": (
                            observed_count - expected_count
                        ),
                        "observed_minus_expected_rate": (
                            observed_rate - nominal_rate
                        ),
                        "interval_confidence_level": float(
                            summary[
                                "interval_confidence_level"
                            ]
                        ),
                        "exact_interval_lower_bound": float(
                            summary[
                                "exceedance_rate_ci_lower"
                            ]
                        ),
                        "exact_interval_upper_bound": float(
                            summary[
                                "exceedance_rate_ci_upper"
                            ]
                        ),
                        "mean_value_at_risk": float(
                            mean_var.loc[confidence_level]
                        ),
                        "mean_expected_shortfall": float(
                            mean_es.loc[confidence_level]
                        ),
                        "sparse_expected_tail": bool(
                            expected_count < 5.0
                        ),
                        "sparse_observed_tail": bool(
                            observed_count < 5
                        ),
                    }
                )

    result = pd.DataFrame.from_records(records).set_index(
        [
            "period_id",
            "model",
            "confidence_level",
        ]
    )

    return result





def evaluate_models_subperiod_es_tail_severity(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize descriptive ES tail severity in predeclared periods.

    Severity statistics condition on the project's strict VaR
    exception set. Rows with no exception retain missing conditional
    statistics, while sparse expected and observed tails are flagged.
    """

    definitions = create_subperiod_definitions(
        forecasts_by_model=forecasts_by_model,
    )
    records: list[
        dict[str, bool | float | int | str | pd.Timestamp]
    ] = []

    for period_id, definition in definitions.iterrows():
        requested_start = pd.Timestamp(
            definition["requested_start_date"]
        )
        requested_end = pd.Timestamp(
            definition["requested_end_date"]
        )

        for model_name, forecasts in (
            forecasts_by_model.items()
        ):
            forecast_dates = (
                forecasts.index.get_level_values(
                    "forecast_date"
                )
            )
            period_mask = (
                (forecast_dates >= requested_start)
                & (forecast_dates <= requested_end)
            )
            period_forecasts = forecasts.loc[
                period_mask
            ]
            severity = summarize_es_tail_severity(
                forecasts=period_forecasts,
            )

            for confidence_level, summary in (
                severity.iterrows()
            ):
                observation_count = int(
                    summary["observation_count"]
                )
                confidence_level = float(
                    confidence_level
                )
                nominal_rate = 1.0 - confidence_level
                expected_count = (
                    observation_count * nominal_rate
                )
                observed_count = int(
                    summary["var_exceedance_count"]
                )

                if observation_count != int(
                    definition["observation_count"]
                ):
                    raise ValueError(
                        "Subperiod observation counts must agree "
                        "with the period definitions."
                    )

                records.append(
                    {
                        "period_id": str(period_id),
                        "model": model_name,
                        "confidence_level": confidence_level,
                        "period_type": str(
                            definition["period_type"]
                        ),
                        "first_forecast_date": pd.Timestamp(
                            definition[
                                "first_forecast_date"
                            ]
                        ),
                        "last_forecast_date": pd.Timestamp(
                            definition[
                                "last_forecast_date"
                            ]
                        ),
                        "observation_count": observation_count,
                        "is_partial": bool(
                            definition["is_partial"]
                        ),
                        "nominal_exceedance_rate": (
                            nominal_rate
                        ),
                        "expected_exceedance_count": (
                            expected_count
                        ),
                        "var_exceedance_count": observed_count,
                        "mean_exceedance_loss": float(
                            summary["mean_exceedance_loss"]
                        ),
                        "mean_var_on_exceedance_days": float(
                            summary[
                                "mean_var_on_exceedance_days"
                            ]
                        ),
                        "mean_es_on_exceedance_days": float(
                            summary[
                                "mean_es_on_exceedance_days"
                            ]
                        ),
                        "mean_var_excess": float(
                            summary["mean_var_excess"]
                        ),
                        "median_var_excess": float(
                            summary["median_var_excess"]
                        ),
                        "maximum_var_excess": float(
                            summary["maximum_var_excess"]
                        ),
                        "mean_es_gap": float(
                            summary["mean_es_gap"]
                        ),
                        "sparse_expected_tail": bool(
                            expected_count < 5.0
                        ),
                        "sparse_observed_tail": bool(
                            observed_count < 5
                        ),
                    }
                )

    result = pd.DataFrame.from_records(records).set_index(
        [
            "period_id",
            "model",
            "confidence_level",
        ]
    )

    return result





def evaluate_models_subperiod_var_es_identification(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize descriptive joint VaR-ES residuals by subperiod.

    The mean identification residual uses every forecast session. A
    positive value indicates average tail-risk underestimation and a
    negative value indicates overestimation. No inference is performed.
    """

    definitions = create_subperiod_definitions(
        forecasts_by_model=forecasts_by_model,
    )
    records: list[
        dict[str, bool | float | int | str | pd.Timestamp]
    ] = []

    for period_id, definition in definitions.iterrows():
        requested_start = pd.Timestamp(
            definition["requested_start_date"]
        )
        requested_end = pd.Timestamp(
            definition["requested_end_date"]
        )

        for model_name, forecasts in (
            forecasts_by_model.items()
        ):
            forecast_dates = (
                forecasts.index.get_level_values(
                    "forecast_date"
                )
            )
            period_mask = (
                (forecast_dates >= requested_start)
                & (forecast_dates <= requested_end)
            )
            period_forecasts = forecasts.loc[
                period_mask
            ]
            identification = (
                calculate_var_es_identification_residuals(
                    forecasts=period_forecasts,
                )
            )
            analysis = identification.copy()
            analysis["expected_shortfall"] = (
                period_forecasts[
                    "expected_shortfall"
                ].to_numpy(dtype=float)
            )
            analysis["var_exceedance"] = (
                period_forecasts[
                    "var_exceedance"
                ].to_numpy(dtype=bool)
            )

            for confidence_level, level_analysis in (
                analysis.groupby(
                    level="confidence_level",
                    sort=False,
                )
            ):
                observation_count = len(level_analysis)
                confidence_level = float(
                    confidence_level
                )
                expected_count = (
                    observation_count
                    * (1.0 - confidence_level)
                )
                observed_count = int(
                    level_analysis[
                        "var_exceedance"
                    ].sum()
                )

                if observation_count != int(
                    definition["observation_count"]
                ):
                    raise ValueError(
                        "Subperiod observation counts must agree "
                        "with the period definitions."
                    )

                records.append(
                    {
                        "period_id": str(period_id),
                        "model": model_name,
                        "confidence_level": confidence_level,
                        "period_type": str(
                            definition["period_type"]
                        ),
                        "first_forecast_date": pd.Timestamp(
                            definition[
                                "first_forecast_date"
                            ]
                        ),
                        "last_forecast_date": pd.Timestamp(
                            definition[
                                "last_forecast_date"
                            ]
                        ),
                        "observation_count": observation_count,
                        "is_partial": bool(
                            definition["is_partial"]
                        ),
                        "expected_exceedance_count": (
                            expected_count
                        ),
                        "var_exceedance_count": observed_count,
                        "mean_tail_contribution": float(
                            level_analysis[
                                "tail_contribution"
                            ].mean()
                        ),
                        "mean_expected_shortfall": float(
                            level_analysis[
                                "expected_shortfall"
                            ].mean()
                        ),
                        "mean_identification_residual": float(
                            level_analysis[
                                "var_es_identification_residual"
                            ].mean()
                        ),
                        "sparse_expected_tail": bool(
                            expected_count < 5.0
                        ),
                        "sparse_observed_tail": bool(
                            observed_count < 5
                        ),
                    }
                )

    result = pd.DataFrame.from_records(records).set_index(
        [
            "period_id",
            "model",
            "confidence_level",
        ]
    )

    return result




def calculate_var_quantile_losses(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the quantile loss of every VaR forecast.

    Quantile loss penalizes VaR underprediction by the confidence
    level and overprediction by one minus the confidence level.
    Smaller values indicate more accurate VaR forecasts.
    """

    # Reuse the established validation for the forecast index,
    # confidence levels, finite losses and VaR values, and strict
    # realized_loss > value_at_risk exceedance indicators.
    summarize_var_exceedances(
        forecasts=forecasts,
    )

    confidence_levels = (
        forecasts.index
        .get_level_values("confidence_level")
        .to_numpy(dtype=float)
    )
    realized_losses = forecasts[
        "realized_loss"
    ].to_numpy(dtype=float)
    value_at_risk = forecasts[
        "value_at_risk"
    ].to_numpy(dtype=float)
    exceedances = forecasts[
        "var_exceedance"
    ].to_numpy(dtype=bool)

    absolute_forecast_errors = np.abs(
        realized_losses - value_at_risk
    )
    error_weights = np.where(
        exceedances,
        confidence_levels,
        1.0 - confidence_levels,
    )
    quantile_losses = (
        error_weights * absolute_forecast_errors
    )

    return pd.DataFrame(
        {
            "var_quantile_loss": quantile_losses,
        },
        index=forecasts.index.copy(),
    )






def summarize_var_quantile_loss(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize average VaR quantile loss by confidence level.

    Lower mean quantile loss indicates more accurate VaR forecasts.
    Scores should only be compared across models using the same
    confidence level and forecast observations.
    """

    quantile_losses = calculate_var_quantile_losses(
        forecasts=forecasts,
    )

    records: list[dict[str, float | int]] = []

    for confidence_level, level_losses in (
        quantile_losses.groupby(
            level="confidence_level",
            sort=False,
        )
    ):
        records.append(
            {
                "confidence_level": float(
                    confidence_level
                ),
                "observation_count": len(
                    level_losses
                ),
                "mean_var_quantile_loss": float(
                    level_losses[
                        "var_quantile_loss"
                    ].mean()
                ),
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("confidence_level")
    )






def evaluate_models_var_quantile_loss(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare average VaR quantile loss across aligned models.

    Models must use identical ordered forecast indexes and realized
    losses. Lower mean quantile loss receives a better descriptive
    rank within each confidence level.
    """

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if not forecasts_by_model:
        raise ValueError(
            "At least one model forecast table is required."
        )

    loss_tables: dict[str, pd.DataFrame] = {}
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

        loss_table = summarize_var_quantile_loss(
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

        loss_tables[model_name] = loss_table

    combined = pd.concat(
        loss_tables,
        names=["model"],
    )

    combined["var_quantile_loss_rank"] = (
        combined.groupby(
            level="confidence_level",
            sort=False,
        )["mean_var_quantile_loss"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    return combined


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





def calculate_fz0_losses(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the joint FZ0 loss of every VaR-ES forecast.

    This is the positive-loss equivalent of the canonical return-domain
    FZ0 score. Lower values indicate better joint VaR-ES forecasts.
    """

    # Reuse the established validation for forecast structure,
    # confidence levels, realized losses, VaR exceedances, finite
    # Expected Shortfall values, and the ES >= VaR requirement.
    summarize_es_tail_severity(
        forecasts=forecasts,
    )

    confidence_levels = (
        forecasts.index
        .get_level_values("confidence_level")
        .to_numpy(dtype=float)
    )
    realized_losses = forecasts[
        "realized_loss"
    ].to_numpy(dtype=float)
    value_at_risk = forecasts[
        "value_at_risk"
    ].to_numpy(dtype=float)
    expected_shortfall = forecasts[
        "expected_shortfall"
    ].to_numpy(dtype=float)
    exceedance_indicators = forecasts[
        "var_exceedance"
    ].to_numpy(dtype=bool)

    if (expected_shortfall <= 0.0).any():
        raise ValueError(
            "Expected Shortfall forecasts must be strictly "
            "positive for FZ0 scoring."
        )

    tail_excess = np.where(
        exceedance_indicators,
        realized_losses - value_at_risk,
        0.0,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        fz0_losses = (
            tail_excess
            / (
                (1.0 - confidence_levels)
                * expected_shortfall
            )
            + value_at_risk / expected_shortfall
            + np.log(expected_shortfall)
            - 1.0
        )

    if not np.isfinite(fz0_losses).all():
        raise ValueError(
            "FZ0 losses must be finite."
        )

    return pd.DataFrame(
        {
            "fz0_loss": fz0_losses,
        },
        index=forecasts.index.copy(),
    )


def summarize_fz0_loss(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize average joint FZ0 loss by confidence level.

    Lower mean FZ0 loss indicates a better joint VaR-ES forecast.
    Scores should only be compared across models using the same
    confidence level, observations, and loss units.
    """

    fz0_losses = calculate_fz0_losses(
        forecasts=forecasts,
    )
    records: list[dict[str, float | int]] = []

    for confidence_level, level_losses in (
        fz0_losses.groupby(
            level="confidence_level",
            sort=False,
        )
    ):
        records.append(
            {
                "confidence_level": float(
                    confidence_level
                ),
                "observation_count": len(
                    level_losses
                ),
                "mean_fz0_loss": float(
                    level_losses["fz0_loss"].mean()
                ),
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("confidence_level")
    )


def evaluate_models_fz0_loss(
    forecasts_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare average joint FZ0 loss across aligned models.

    Models must use identical ordered forecast indexes and realized
    losses. Lower mean FZ0 loss receives a better descriptive rank
    within each confidence level.
    """

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if not forecasts_by_model:
        raise ValueError(
            "At least one model forecast table is required."
        )

    loss_tables: dict[str, pd.DataFrame] = {}
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

        loss_table = summarize_fz0_loss(
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

        loss_tables[model_name] = loss_table

    combined = pd.concat(
        loss_tables,
        names=["model"],
    )
    combined["fz0_loss_rank"] = (
        combined.groupby(
            level="confidence_level",
            sort=False,
        )["mean_fz0_loss"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    return combined


def calculate_var_es_identification_residuals(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate joint VaR-ES residuals for every forecast row.

    The identification residual equals the realized joint tail
    contribution minus the Expected Shortfall forecast. A correctly
    calibrated VaR-ES pair has an expected residual of zero.
    """

    # Reuse the established validation for the forecast index,
    # confidence levels, realized losses, strict VaR exceedances,
    # finite Expected Shortfall values, and ES >= VaR ordering.
    summarize_es_tail_severity(
        forecasts=forecasts,
    )

    confidence_levels = (
        forecasts.index
        .get_level_values("confidence_level")
        .to_numpy(dtype=float)
    )
    realized_losses = forecasts[
        "realized_loss"
    ].to_numpy(dtype=float)
    value_at_risk = forecasts[
        "value_at_risk"
    ].to_numpy(dtype=float)
    expected_shortfall = forecasts[
        "expected_shortfall"
    ].to_numpy(dtype=float)
    exceedance_indicators = forecasts[
        "var_exceedance"
    ].to_numpy(dtype=float)

    tail_contributions = (
        value_at_risk
        + (
            exceedance_indicators
            / (1.0 - confidence_levels)
        )
        * (realized_losses - value_at_risk)
    )
    identification_residuals = (
        tail_contributions - expected_shortfall
    )

    return pd.DataFrame(
        {
            "tail_contribution": tail_contributions,
            "var_es_identification_residual": (
                identification_residuals
            ),
        },
        index=forecasts.index.copy(),
    )






def summarize_var_es_identification(
    forecasts: pd.DataFrame,
    bandwidth: int | None = None,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Test joint VaR-ES calibration by confidence level.

    A positive mean residual indicates average tail-risk
    underestimation. A negative mean residual indicates average
    overestimation. Newey-West HAC uncertainty accounts for serial
    dependence and heteroskedasticity in the residual series.
    """

    residuals = (
        calculate_var_es_identification_residuals(
            forecasts=forecasts,
        )
    )

    analysis = residuals.copy()
    analysis["expected_shortfall"] = forecasts[
        "expected_shortfall"
    ].to_numpy(dtype=float)

    records: list[
        dict[str, float | int | bool]
    ] = []

    for confidence_level, level_analysis in (
        analysis.groupby(
            level="confidence_level",
            sort=False,
        )
    ):
        mean_tail_contribution = float(
            level_analysis[
                "tail_contribution"
            ].mean()
        )
        mean_expected_shortfall = float(
            level_analysis[
                "expected_shortfall"
            ].mean()
        )
        mean_identification_residual = float(
            level_analysis[
                "var_es_identification_residual"
            ].mean()
        )
        residual_series = level_analysis[
            "var_es_identification_residual"
        ].copy()
        residual_series.index = (
            residual_series.index.droplevel(
                "confidence_level"
            )
        )
        hac_result = calculate_newey_west_mean_test(
            observations=residual_series,
            bandwidth=bandwidth,
            significance_level=significance_level,
        )

        records.append(
            {
                "confidence_level": float(
                    confidence_level
                ),
                "observation_count": len(
                    level_analysis
                ),
                "mean_tail_contribution": (
                    mean_tail_contribution
                ),
                "mean_expected_shortfall": (
                    mean_expected_shortfall
                ),
                "mean_identification_residual": (
                    mean_identification_residual
                ),
                "hac_bandwidth": (
                    hac_result.bandwidth
                ),
                "hac_long_run_variance": (
                    hac_result.long_run_variance
                ),
                "hac_standard_error": (
                    hac_result.standard_error
                ),
                "hac_test_statistic": (
                    hac_result.test_statistic
                ),
                "hac_p_value": hac_result.p_value,
                "significance_level": (
                    hac_result.significance_level
                ),
                "hac_reject_null": (
                    hac_result.reject_null
                ),
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .set_index("confidence_level")
    )




def calculate_newey_west_mean_test(
    observations: pd.Series,
    bandwidth: int | None = None,
    significance_level: float = 0.05,
) -> NeweyWestMeanTestResult:
    """Test whether a time-series mean is zero using HAC uncertainty.

    Bartlett weights are applied to autocovariances through the selected
    bandwidth. If no bandwidth is supplied, a deterministic automatic
    bandwidth rule is used.
    """

    if not isinstance(observations, pd.Series):
        raise TypeError(
            "Observations must be a pandas Series."
        )

    if len(observations) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    if observations.index.has_duplicates:
        raise ValueError(
            "Observation index entries must be unique."
        )

    if not observations.index.is_monotonic_increasing:
        raise ValueError(
            "Observations must be ordered by increasing index."
        )

    if any(
        isinstance(value, (bool, np.bool_))
        for value in observations
    ):
        raise ValueError(
            "Observations must contain finite numbers."
        )

    try:
        values = observations.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Observations must contain finite numbers."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError(
            "Observations must contain finite numbers."
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

    observation_count = len(values)

    if bandwidth is None:
        selected_bandwidth = int(
            4.0
            * (
                observation_count / 100.0
            ) ** (2.0 / 9.0)
        )
        selected_bandwidth = min(
            selected_bandwidth,
            observation_count - 1,
        )
    else:
        if (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, Integral)
            or bandwidth < 0
            or bandwidth >= observation_count
        ):
            raise ValueError(
                "Bandwidth must be an integer between zero "
                "and one less than the observation count."
            )

        selected_bandwidth = int(bandwidth)

    sample_mean = float(values.mean())
    centered_values = values - sample_mean

    lag_zero_covariance = float(
        np.dot(
            centered_values,
            centered_values,
        )
        / observation_count
    )

    long_run_variance = lag_zero_covariance

    for lag in range(
        1,
        selected_bandwidth + 1,
    ):
        lagged_covariance = float(
            np.dot(
                centered_values[lag:],
                centered_values[:-lag],
            )
            / observation_count
        )
        bartlett_weight = (
            1.0
            - lag / (selected_bandwidth + 1.0)
        )

        long_run_variance += (
            2.0
            * bartlett_weight
            * lagged_covariance
        )

    # Bartlett-weighted HAC variance is nonnegative in theory.
    # This protects against tiny negative floating-point artifacts.
    long_run_variance = max(
        0.0,
        float(long_run_variance),
    )
    mean_variance = (
        long_run_variance / observation_count
    )
    standard_error = sqrt(mean_variance)

    if standard_error == 0.0:
        if sample_mean == 0.0:
            test_statistic = 0.0
            p_value = 1.0
        else:
            test_statistic = (
                float("inf")
                if sample_mean > 0.0
                else -float("inf")
            )
            p_value = 0.0
    else:
        test_statistic = (
            sample_mean / standard_error
        )
        p_value = erfc(
            abs(test_statistic) / sqrt(2.0)
        )

    significance_level = float(
        significance_level
    )

    return NeweyWestMeanTestResult(
        observation_count=observation_count,
        bandwidth=selected_bandwidth,
        sample_mean=sample_mean,
        long_run_variance=long_run_variance,
        mean_variance=mean_variance,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=p_value,
        significance_level=significance_level,
        reject_null=p_value < significance_level,
    )










def calculate_diebold_mariano_test(
    model_a_scores: pd.Series,
    model_b_scores: pd.Series,
    bandwidth: int | None = None,
    significance_level: float = 0.05,
) -> DieboldMarianoTestResult:
    """Compare two aligned daily score series with a HAC DM test.

    The score differential is model A minus model B. Because lower
    forecast scores are better, a negative mean difference favors
    model A and a positive mean difference favors model B.
    """

    if not isinstance(model_a_scores, pd.Series):
        raise TypeError(
            "Model A scores must be a pandas Series."
        )

    if not isinstance(model_b_scores, pd.Series):
        raise TypeError(
            "Model B scores must be a pandas Series."
        )

    if not model_a_scores.index.equals(
        model_b_scores.index
    ):
        raise ValueError(
            "Model score series must use the same ordered index."
        )

    for scores, label in (
        (model_a_scores, "Model A scores"),
        (model_b_scores, "Model B scores"),
    ):
        if any(
            isinstance(value, (bool, np.bool_))
            for value in scores
        ):
            raise ValueError(
                f"{label} must contain finite numbers."
            )

        try:
            score_values = scores.to_numpy(
                dtype=float
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} must contain finite numbers."
            ) from error

        if not np.isfinite(score_values).all():
            raise ValueError(
                f"{label} must contain finite numbers."
            )

    model_a_values = model_a_scores.to_numpy(
        dtype=float
    )
    model_b_values = model_b_scores.to_numpy(
        dtype=float
    )

    with np.errstate(over="ignore", invalid="ignore"):
        score_differences = (
            model_a_values - model_b_values
        )

    if not np.isfinite(score_differences).all():
        raise ValueError(
            "Model score differences must be finite."
        )

    hac_result = calculate_newey_west_mean_test(
        observations=pd.Series(
            score_differences,
            index=model_a_scores.index.copy(),
            name="score_difference",
        ),
        bandwidth=bandwidth,
        significance_level=significance_level,
    )

    return DieboldMarianoTestResult(
        observation_count=(
            hac_result.observation_count
        ),
        bandwidth=hac_result.bandwidth,
        mean_score_difference=(
            hac_result.sample_mean
        ),
        long_run_variance=(
            hac_result.long_run_variance
        ),
        mean_variance=hac_result.mean_variance,
        standard_error=hac_result.standard_error,
        test_statistic=hac_result.test_statistic,
        p_value=hac_result.p_value,
        significance_level=(
            hac_result.significance_level
        ),
        reject_null=hac_result.reject_null,
    )





def evaluate_models_diebold_mariano(
    forecasts_by_model: dict[str, pd.DataFrame],
    bandwidth: int | None = None,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Compare aligned models using quantile-loss and FZ0 DM tests.

    Every unordered model pair is tested separately at each confidence
    level. Holm adjustment is applied independently to the quantile-loss
    family and the joint VaR-ES FZ0 family.
    """

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if len(forecasts_by_model) < 2:
        raise ValueError(
            "At least two model forecast tables are required."
        )

    scores_by_model: dict[
        str,
        dict[str, pd.Series],
    ] = {}
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

        quantile_losses = calculate_var_quantile_losses(
            forecasts=forecasts,
        )["var_quantile_loss"]
        fz0_losses = calculate_fz0_losses(
            forecasts=forecasts,
        )["fz0_loss"]

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

        scores_by_model[model_name] = {
            "var_quantile_loss": quantile_losses,
            "fz0_loss": fz0_losses,
        }

    model_names = list(scores_by_model)
    confidence_levels = (
        reference_index
        .get_level_values("confidence_level")
        .unique()
    )
    records: list[
        dict[str, float | int | bool | str]
    ] = []

    for score_name in (
        "var_quantile_loss",
        "fz0_loss",
    ):
        for confidence_level in confidence_levels:
            for model_a_position, model_a in enumerate(
                model_names[:-1]
            ):
                for model_b in model_names[
                    model_a_position + 1:
                ]:
                    model_a_scores = scores_by_model[
                        model_a
                    ][score_name].xs(
                        confidence_level,
                        level="confidence_level",
                    )
                    model_b_scores = scores_by_model[
                        model_b
                    ][score_name].xs(
                        confidence_level,
                        level="confidence_level",
                    )
                    dm_result = (
                        calculate_diebold_mariano_test(
                            model_a_scores=(
                                model_a_scores
                            ),
                            model_b_scores=(
                                model_b_scores
                            ),
                            bandwidth=bandwidth,
                            significance_level=(
                                significance_level
                            ),
                        )
                    )

                    records.append(
                        {
                            "score_name": score_name,
                            "confidence_level": float(
                                confidence_level
                            ),
                            "model_a": model_a,
                            "model_b": model_b,
                            "observation_count": (
                                dm_result.observation_count
                            ),
                            "mean_score_difference": (
                                dm_result
                                .mean_score_difference
                            ),
                            "hac_long_run_variance": (
                                dm_result
                                .long_run_variance
                            ),
                            "hac_mean_variance": (
                                dm_result.mean_variance
                            ),
                            "hac_standard_error": (
                                dm_result.standard_error
                            ),
                            "hac_bandwidth": (
                                dm_result.bandwidth
                            ),
                            "dm_statistic": (
                                dm_result.test_statistic
                            ),
                            "dm_p_value": (
                                dm_result.p_value
                            ),
                            "significance_level": (
                                dm_result
                                .significance_level
                            ),
                            "dm_reject_equal_accuracy": (
                                dm_result.reject_null
                            ),
                        }
                    )

    combined = (
        pd.DataFrame.from_records(records)
        .set_index(
            [
                "score_name",
                "confidence_level",
                "model_a",
                "model_b",
            ]
        )
    )
    holm_p_values = pd.Series(
        index=combined.index,
        dtype=float,
    )
    holm_rejections = pd.Series(
        False,
        index=combined.index,
        dtype=bool,
    )

    for score_name in (
        "var_quantile_loss",
        "fz0_loss",
    ):
        family_mask = (
            combined.index.get_level_values(
                "score_name"
            )
            == score_name
        )
        family_adjustment = adjust_p_values_holm(
            p_values=combined.loc[
                family_mask,
                "dm_p_value",
            ],
            significance_level=significance_level,
        )
        holm_p_values.loc[family_mask] = (
            family_adjustment[
                "holm_adjusted_p_value"
            ].to_numpy()
        )
        holm_rejections.loc[family_mask] = (
            family_adjustment[
                "reject_null_holm"
            ].to_numpy(dtype=bool)
        )

    combined["holm_adjusted_p_value"] = (
        holm_p_values
    )
    combined[
        "holm_reject_equal_accuracy"
    ] = holm_rejections

    return combined





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






def evaluate_models_var_es_identification(
    forecasts_by_model: dict[str, pd.DataFrame],
    bandwidth: int | None = None,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Test joint VaR-ES calibration across aligned models.

    Each model receives one row per confidence level. Holm adjustment
    is applied to the complete family of HAC mean-test p-values.
    """

    if not isinstance(forecasts_by_model, dict):
        raise TypeError(
            "Forecasts by model must be a dictionary."
        )

    if not forecasts_by_model:
        raise ValueError(
            "At least one model forecast table is required."
        )

    calibration_tables: dict[str, pd.DataFrame] = {}
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

        calibration_table = (
            summarize_var_es_identification(
                forecasts=forecasts,
                bandwidth=bandwidth,
                significance_level=significance_level,
            )
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

        calibration_tables[model_name] = (
            calibration_table
        )

    combined = pd.concat(
        calibration_tables,
        names=["model"],
    )
    holm_results = adjust_p_values_holm(
        p_values=combined["hac_p_value"],
        significance_level=significance_level,
    )

    combined["hac_holm_p_value"] = holm_results[
        "holm_adjusted_p_value"
    ]
    combined["hac_holm_reject_null"] = (
        holm_results["reject_null_holm"]
    )

    return combined


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
