"""One-day portfolio risk forecasting models."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from statistics import NormalDist


from tailrisk.covariance import (
    calculate_sample_portfolio_moments,
)


import pandas as pd


_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class GaussianRiskForecast:
    """One-day Gaussian VaR and Expected Shortfall."""

    confidence_level: float
    value_at_risk: float
    expected_shortfall: float


def calculate_gaussian_var_es(
    mean_return: float,
    volatility: float,
    confidence_level: float,
) -> GaussianRiskForecast:
    """Calculate Gaussian VaR and ES under the return-loss convention.

    If the return is normally distributed with mean mu and volatility
    sigma, the loss L = -R has:

        VaR_alpha = -mu + sigma * z_alpha

        ES_alpha = -mu
                   + sigma * phi(z_alpha) / (1 - alpha)

    Here, z_alpha is the alpha quantile of the standard normal
    distribution and phi is its probability density function.
    """

    if (
        isinstance(mean_return, bool)
        or not isinstance(mean_return, Real)
        or not isfinite(mean_return)
    ):
        raise ValueError(
            "Mean return must be a finite number."
        )

    if (
        isinstance(volatility, bool)
        or not isinstance(volatility, Real)
        or not isfinite(volatility)
        or volatility <= 0.0
    ):
        raise ValueError(
            "Volatility must be a positive finite number."
        )

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError(
            "Confidence level must be a finite number "
            "strictly between 0 and 1."
        )

    z_score = _STANDARD_NORMAL.inv_cdf(
        confidence_level
    )
    density_at_z = _STANDARD_NORMAL.pdf(
        z_score
    )

    value_at_risk = (
        -mean_return
        + volatility * z_score
    )
    expected_shortfall = (
        -mean_return
        + volatility
        * density_at_z
        / (1.0 - confidence_level)
    )

    return GaussianRiskForecast(
        confidence_level=float(confidence_level),
        value_at_risk=float(value_at_risk),
        expected_shortfall=float(expected_shortfall),
    )



def calculate_rolling_gaussian_forecasts(
    portfolio_returns: pd.Series,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
) -> pd.DataFrame:
    """Calculate Gaussian forecasts from rolling return windows.

    For each forecast date t, the schedule identifies the return window
    ending at t - 1. The function estimates that window's sample mean and
    volatility, then calculates VaR and ES for each confidence level.

    Results are indexed by forecast date and confidence level.
    """

    if not confidence_levels:
        raise ValueError(
        "At least one confidence level is required."
    )

    records: list[dict[str, float]] = []
    forecast_keys: list[
        tuple[pd.Timestamp, float]
    ] = []

    for forecast_date, schedule_row in (
        forecast_schedule.iterrows()
    ):
        start_date = schedule_row[
            "estimation_start_date"
        ]
        end_date = schedule_row[
            "estimation_end_date"
        ]

        if end_date >= forecast_date:
            raise ValueError(
                "Each estimation window must end "
                "before its forecast date."
            )

        estimation_returns = portfolio_returns.loc[
            start_date:end_date
        ]

        mean_return = float(
            estimation_returns.mean()
        )
        volatility = float(
            estimation_returns.std(ddof=1)
        )

        for confidence_level in confidence_levels:
            forecast = calculate_gaussian_var_es(
                mean_return=mean_return,
                volatility=volatility,
                confidence_level=confidence_level,
            )

            forecast_keys.append(
                (
                    pd.Timestamp(forecast_date),
                    forecast.confidence_level,
                )
            )
            records.append(
                {
                    "mean_return": mean_return,
                    "volatility": volatility,
                    "value_at_risk": (
                        forecast.value_at_risk
                    ),
                    "expected_shortfall": (
                        forecast.expected_shortfall
                    ),
                }
            )

    forecast_index = pd.MultiIndex.from_tuples(
        forecast_keys,
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    return pd.DataFrame(
        records,
        index=forecast_index,
    )



def calculate_rolling_multivariate_gaussian_forecasts(
    asset_returns: pd.DataFrame,
    beginning_weights: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
) -> pd.DataFrame:
    """Calculate rolling Gaussian forecasts from asset-level returns.

    Each estimation window ends before its forecast date. The function
    estimates asset means and covariances, combines them using the
    beginning-of-day portfolio weights, and calculates Gaussian VaR and ES.
    """

    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError(
            "Asset returns must be a pandas DataFrame."
        )

    if not isinstance(beginning_weights, pd.DataFrame):
        raise TypeError(
            "Beginning weights must be a pandas DataFrame."
        )

    if not isinstance(forecast_schedule, pd.DataFrame):
        raise TypeError(
            "Forecast schedule must be a pandas DataFrame."
        )

    if not confidence_levels:
        raise ValueError(
            "At least one confidence level is required."
        )

    required_schedule_columns = {
        "estimation_start_date",
        "estimation_end_date",
    }

    if not required_schedule_columns.issubset(
        forecast_schedule.columns
    ):
        raise ValueError(
            "Forecast schedule must contain estimation "
            "start and end dates."
        )

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise ValueError(
            "Asset returns must use a DatetimeIndex."
        )

    if not isinstance(beginning_weights.index, pd.DatetimeIndex):
        raise ValueError(
            "Beginning weights must use a DatetimeIndex."
        )

    if not isinstance(forecast_schedule.index, pd.DatetimeIndex):
        raise ValueError(
            "Forecast schedule must use a DatetimeIndex."
        )

    if asset_returns.index.has_duplicates:
        raise ValueError(
            "Asset-return dates must be unique."
        )

    if beginning_weights.index.has_duplicates:
        raise ValueError(
            "Beginning-weight dates must be unique."
        )

    if forecast_schedule.index.has_duplicates:
        raise ValueError(
            "Forecast dates must be unique."
        )

    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError(
            "Asset returns must be ordered by increasing date."
        )

    if not beginning_weights.index.is_monotonic_increasing:
        raise ValueError(
            "Beginning weights must be ordered by increasing date."
        )

    if set(asset_returns.columns) != set(
        beginning_weights.columns
    ):
        raise ValueError(
            "Asset returns and beginning weights "
            "must use the same tickers."
        )

    records: list[dict[str, float]] = []
    forecast_keys: list[
        tuple[pd.Timestamp, float]
    ] = []

    for forecast_date, schedule_row in (
        forecast_schedule.iterrows()
    ):
        forecast_date = pd.Timestamp(forecast_date)
        start_date = pd.Timestamp(
            schedule_row["estimation_start_date"]
        )
        end_date = pd.Timestamp(
            schedule_row["estimation_end_date"]
        )

        if end_date >= forecast_date:
            raise ValueError(
                "Each estimation window must end "
                "before its forecast date."
            )

        if (
            start_date not in asset_returns.index
            or end_date not in asset_returns.index
        ):
            raise ValueError(
                "Estimation-window dates must exist "
                "in the asset returns."
            )

        if forecast_date not in beginning_weights.index:
            raise ValueError(
                "Beginning weights are required "
                "for every forecast date."
            )

        estimation_returns = asset_returns.loc[
            start_date:end_date
        ]
        forecast_weights = beginning_weights.loc[
            forecast_date
        ]

        moments = calculate_sample_portfolio_moments(
            asset_returns=estimation_returns,
            weights=forecast_weights,
        )

        for confidence_level in confidence_levels:
            forecast = calculate_gaussian_var_es(
                mean_return=moments.portfolio_mean,
                volatility=moments.portfolio_volatility,
                confidence_level=confidence_level,
            )

            forecast_keys.append(
                (
                    forecast_date,
                    forecast.confidence_level,
                )
            )
            records.append(
                {
                    "mean_return": moments.portfolio_mean,
                    "volatility": (
                        moments.portfolio_volatility
                    ),
                    "value_at_risk": (
                        forecast.value_at_risk
                    ),
                    "expected_shortfall": (
                        forecast.expected_shortfall
                    ),
                }
            )

    forecast_index = pd.MultiIndex.from_tuples(
        forecast_keys,
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    return pd.DataFrame(
        records,
        index=forecast_index,
    )