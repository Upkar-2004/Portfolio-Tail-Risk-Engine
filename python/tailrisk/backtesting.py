"""Time-ordered VaR and Expected Shortfall backtesting."""

import numpy as np
import pandas as pd


def create_forecast_schedule(
    portfolio_returns: pd.Series,
    estimation_window: int,
) -> pd.DataFrame:
    """Create a common schedule for one-day-ahead forecasts.

    Notation
    --------
    R_t = portfolio return on forecast date t
    W   = number of returns in the estimation window
    L_t = realized portfolio loss on date t

    For each forecast date t:

        estimation window = R_(t-W), ..., R_(t-1)
        realized return   = R_t
        realized loss     = -R_t

    The forecast-date return is never included in its own estimation
    window. The returned table records the inclusive start and end dates
    of every window so all models use identical information and outcomes.
    """

    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError(
            "Portfolio returns must be a pandas Series."
        )

    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        raise ValueError(
            "Portfolio returns must use a DatetimeIndex."
        )

    if portfolio_returns.index.hasnans:
        raise ValueError(
            "Portfolio return dates must not be missing."
        )

    if portfolio_returns.index.has_duplicates:
        raise ValueError(
            "Portfolio return dates must be unique."
        )

    if not portfolio_returns.index.is_monotonic_increasing:
        raise ValueError(
            "Portfolio returns must be ordered by increasing date."
        )

    if (
        isinstance(estimation_window, bool)
        or not isinstance(estimation_window, int)
        or estimation_window <= 0
    ):
        raise ValueError(
            "Estimation window must be a positive integer."
        )

    if len(portfolio_returns) <= estimation_window:
        raise ValueError(
            "Portfolio returns must contain more observations "
            "than the estimation window."
        )

    values = portfolio_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Portfolio returns must be finite numbers."
        )

    forecast_dates = pd.DatetimeIndex(
        portfolio_returns.index[estimation_window:],
        name="forecast_date",
    )
    realized_returns = values[estimation_window:]

    return pd.DataFrame(
        {
            "estimation_start_date": (
                portfolio_returns.index[:-estimation_window]
            ),
            "estimation_end_date": (
                portfolio_returns.index[estimation_window - 1:-1]
            ),
            "realized_return": realized_returns,
            "realized_loss": -realized_returns,
        },
        index=forecast_dates,
    )
