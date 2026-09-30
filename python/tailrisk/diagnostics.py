"""Descriptive statistics for realized portfolio returns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnDiagnostics:
    """Summary statistics for a portfolio-return sample."""

    observation_count: int
    mean_daily_return: float
    daily_variance: float
    daily_volatility: float
    annualized_arithmetic_mean: float
    annualized_volatility: float
    skewness: float
    excess_kurtosis: float
    loss_quantile_95: float
    loss_quantile_975: float
    loss_quantile_99: float
    worst_daily_return: float
    worst_return_date: pd.Timestamp
    best_daily_return: float
    best_return_date: pd.Timestamp



def calculate_return_diagnostics(
        portfolio_returns: pd.Series,
        sessions_per_year: int = 252,
)-> ReturnDiagnostics:
    """Calculate basic statistics for realized portfolio returns.

    Mathematical conventions:

        Sample mean:
            R_bar = (1 / n) * sum(R_t)

        Sample variance:
            s^2 = sum((R_t - R_bar)^2) / (n - 1)

        Daily volatility:
            s = sqrt(s^2)

        Annualized arithmetic mean:
            R_bar_annual = m * R_bar

        Annualized volatility:
            s_annual = sqrt(m) * s

    Here, n is the number of observations and m is the assumed
    number of trading sessions per year.

    Square-root-of-time annualization assumes uncorrelated returns
    and stable daily variance. Annualized arithmetic mean is not CAGR.
    """


    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        raise ValueError(
        "Portfolio returns must use a DatetimeIndex."
    )

    if len(portfolio_returns) < 4:
        raise ValueError(
            "At least four portfolio returns are required."
        )

    if sessions_per_year <= 0:
        raise ValueError(
            "Sessions per year must be positive."
        )

    values = portfolio_returns.to_numpy(
        dtype = float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Portfolio returns must be finite numbers."
        )

    observation_count = len(portfolio_returns)

    mean_daily_return = float(
        portfolio_returns.mean()
    )

    # ddof=1 applies the n - 1 sample-variance denominator.
    daily_variance = float(
        portfolio_returns.var(ddof = 1)
    )

    if daily_variance <= 0.0:
        raise ValueError(
            "Portfolio returns must have positive variability."
        )

    daily_volatility = float(
        np.sqrt(daily_variance)
    )

    # Arithmetic annualization; this is not compounded CAGR.
    annualized_arithmetic_mean = float(
        sessions_per_year * mean_daily_return
    )

    # Square-root-of-time scaling under stable, uncorrelated returns.
    annualized_volatility = float(
        np.sqrt(sessions_per_year) * daily_volatility
    )

    # Pandas uses bias-corrected sample estimators.
    skewness = float(
        portfolio_returns.skew()
    )

    # Fisher excess kurtosis: a Gaussian distribution has value zero.
    excess_kurtosis = float(
        portfolio_returns.kurt()
    )

    # Convert returns to positive-loss convention before taking quantiles.
    losses = -portfolio_returns

    loss_quantiles = losses.quantile(
        [0.95, 0.975, 0.99],
        interpolation="linear",
    )

    loss_quantile_95 = float(
        loss_quantiles.loc[0.95]
    )

    loss_quantile_975 = float(
        loss_quantiles.loc[0.975]
    )

    loss_quantile_99 = float(
        loss_quantiles.loc[0.99]
    )

    worst_daily_return = float(
        portfolio_returns.min()
    )

    best_daily_return = float(
        portfolio_returns.max()
    )

    # Taking the minimum matching date makes the tie rule chronological,
    # rather than dependent on the original row order.
    worst_return_date = pd.Timestamp(
        portfolio_returns.index[
            portfolio_returns == worst_daily_return
        ].min()
    )

    best_return_date = pd.Timestamp(
        portfolio_returns.index[
            portfolio_returns == best_daily_return
        ].min()
    )


    return ReturnDiagnostics(
        observation_count=observation_count,
        mean_daily_return=mean_daily_return,
        daily_variance=daily_variance,
        daily_volatility=daily_volatility,
        annualized_arithmetic_mean=annualized_arithmetic_mean,
        annualized_volatility=annualized_volatility,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        loss_quantile_95=loss_quantile_95,
        loss_quantile_975=loss_quantile_975,
        loss_quantile_99=loss_quantile_99,
        worst_daily_return=worst_daily_return,
        worst_return_date=worst_return_date,
        best_daily_return=best_daily_return,
        best_return_date=best_return_date
    )
