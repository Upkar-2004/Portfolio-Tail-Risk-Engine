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




_REQUIRED_BACKTEST_COLUMNS = {
    "estimation_start_date",
    "estimation_end_date",
    "realized_return",
    "realized_loss",
    "value_at_risk",
    "expected_shortfall",
    "var_exceedance",
}


def validate_backtest_forecasts(
    forecasts: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
) -> None:
    """Validate the shared structure of model backtest forecasts."""

    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError(
            "Forecasts must be a pandas DataFrame."
        )

    if not isinstance(forecast_schedule, pd.DataFrame):
        raise TypeError(
            "Forecast schedule must be a pandas DataFrame."
        )

    if not confidence_levels:
        raise ValueError(
            "At least one confidence level is required."
        )

    if not isinstance(forecasts.index, pd.MultiIndex):
        raise ValueError(
            "Forecasts must use a MultiIndex."
        )

    if forecasts.index.nlevels != 2:
        raise ValueError(
            "Forecast index must contain exactly two levels."
        )

    expected_index_names = [
        "forecast_date",
        "confidence_level",
    ]

    if list(forecasts.index.names) != expected_index_names:
        raise ValueError(
            "Forecast index levels must be named "
            "'forecast_date' and 'confidence_level'."
        )

    if forecasts.index.has_duplicates:
        raise ValueError(
            "Forecast index entries must be unique."
        )

    missing_columns = (
        _REQUIRED_BACKTEST_COLUMNS
        - set(forecasts.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Forecasts are missing required column(s): "
            f"{missing_names}."
        )

    expected_index = pd.MultiIndex.from_product(
        [
            forecast_schedule.index,
            [
                float(level)
                for level in confidence_levels
            ],
        ],
        names=expected_index_names,
    )

    if not forecasts.index.equals(expected_index):
        raise ValueError(
            "Forecasts must contain the complete ordered grid "
            "of forecast dates and confidence levels."
        )


    schedule_columns = {
        "estimation_start_date",
        "estimation_end_date",
        "realized_return",
        "realized_loss",
    }

    missing_schedule_columns = (
        schedule_columns
        - set(forecast_schedule.columns)
    )

    if missing_schedule_columns:
        raise ValueError(
            "Forecast schedule is missing required columns."
        )

    if not isinstance(
        forecast_schedule.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Forecast schedule must use a DatetimeIndex."
        )

    if (
        forecast_schedule.index.hasnans
        or forecast_schedule.index.has_duplicates
        or not forecast_schedule.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Forecast schedule dates must be complete, "
            "unique, and ordered."
        )

    forecast_dates = forecasts.index.get_level_values(
        "forecast_date"
    )

    if not isinstance(forecast_dates, pd.DatetimeIndex):
        raise ValueError(
            "Forecast dates must be datetime values."
        )

    date_columns = [
        "estimation_start_date",
        "estimation_end_date",
    ]

    for column in date_columns:
        if not pd.api.types.is_datetime64_any_dtype(
            forecasts[column]
        ):
            raise ValueError(
                f"{column} must contain datetime values."
            )

        if forecasts[column].isna().any():
            raise ValueError(
                "Estimation dates must not be missing."
            )

    numeric_columns = [
        "realized_return",
        "realized_loss",
        "value_at_risk",
        "expected_shortfall",
    ]

    try:
        numeric_values = forecasts[
            numeric_columns
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Forecast numeric columns must contain numbers."
        ) from error

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Forecast numeric columns must contain "
            "finite numbers."
        )

    if not pd.api.types.is_bool_dtype(
        forecasts["var_exceedance"]
    ):
        raise ValueError(
            "VaR exceedance indicators must be boolean."
        )

    estimation_start_dates = pd.DatetimeIndex(
        forecasts["estimation_start_date"]
    )
    estimation_end_dates = pd.DatetimeIndex(
        forecasts["estimation_end_date"]
    )

    if (
        estimation_start_dates
        > estimation_end_dates
    ).any():
        raise ValueError(
            "Each estimation window must start on or "
            "before its end date."
        )

    if (
        estimation_end_dates
        >= forecast_dates
    ).any():
        raise ValueError(
            "Each estimation window must end before "
            "its forecast date."
        )

    aligned_schedule = forecast_schedule.reindex(
        forecast_dates
    )

    for column in date_columns:
        if not np.array_equal(
            forecasts[column].to_numpy(),
            aligned_schedule[column].to_numpy(),
        ):
            raise ValueError(
                "Forecast estimation dates must match "
                "the forecast schedule."
            )

    for column in [
        "realized_return",
        "realized_loss",
    ]:
        if not np.allclose(
            forecasts[column].to_numpy(dtype=float),
            aligned_schedule[column].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "Forecast realized outcomes must match "
                "the forecast schedule."
            )

    realized_returns = forecasts[
        "realized_return"
    ].to_numpy(dtype=float)
    realized_losses = forecasts[
        "realized_loss"
    ].to_numpy(dtype=float)

    if not np.allclose(
        realized_losses,
        -realized_returns,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Each realized loss must equal the negative "
            "of its realized return."
        )

    value_at_risk = forecasts[
        "value_at_risk"
    ].to_numpy(dtype=float)
    expected_shortfall = forecasts[
        "expected_shortfall"
    ].to_numpy(dtype=float)

    if (expected_shortfall < value_at_risk).any():
        raise ValueError(
            "Expected Shortfall must be at least as "
            "large as VaR."
        )

    expected_exceedances = (
        realized_losses > value_at_risk
    )
    observed_exceedances = forecasts[
        "var_exceedance"
    ].to_numpy(dtype=bool)

    if not np.array_equal(
        observed_exceedances,
        expected_exceedances,
    ):
        raise ValueError(
            "VaR exceedance indicators are inconsistent "
            "with realized losses and VaR."
        )