"""One-day portfolio risk forecasting models."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from statistics import NormalDist

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