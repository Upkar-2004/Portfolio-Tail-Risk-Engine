"""Covariance estimators used by the risk models."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplePortfolioMoments:
    """Sample moments for assets and their weighted portfolio."""

    mean_vector: pd.Series
    covariance_matrix: pd.DataFrame
    portfolio_mean: float
    portfolio_variance: float
    portfolio_volatility: float


def calculate_initial_ewma_covariance(
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Initialize EWMA with an ordinary sample covariance matrix."""

    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError(
            "Asset returns must be a pandas DataFrame."
        )

    if asset_returns.empty:
        raise ValueError(
            "Asset returns must not be empty."
        )

    if len(asset_returns) < 2:
        raise ValueError(
            "At least two return observations are required."
        )

    if asset_returns.columns.has_duplicates:
        raise ValueError(
            "Asset-return tickers must be unique."
        )

    try:
        numeric_returns = asset_returns.astype(
            float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Asset returns must contain numeric values."
        ) from error

    return_values = numeric_returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Asset returns must contain finite numbers."
        )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        covariance_matrix = numeric_returns.cov(
            ddof=1
        )

    covariance_values = covariance_matrix.to_numpy(
        dtype=float
    )

    if not np.isfinite(covariance_values).all():
        raise ValueError(
            "Initial EWMA covariance must contain "
            "finite numbers."
        )

    if not np.allclose(
        covariance_values,
        covariance_values.T,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Initial EWMA covariance must be symmetric."
        )

    covariance_scale = max(
        1.0,
        float(np.abs(covariance_values).max()),
    )
    numerical_tolerance = (
        1e-12 * covariance_scale
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance_values
    )

    if float(eigenvalues.min()) < -numerical_tolerance:
        raise ValueError(
            "Initial EWMA covariance must be "
            "positive semidefinite."
        )

    return covariance_matrix


def calculate_ewma_covariance_update(
    previous_covariance: pd.DataFrame,
    innovation: pd.Series,
    decay_factor: float,
) -> pd.DataFrame:
    """Update an EWMA covariance matrix with one return innovation."""

    if not isinstance(previous_covariance, pd.DataFrame):
        raise TypeError(
            "Previous covariance must be a pandas DataFrame."
        )

    if not isinstance(innovation, pd.Series):
        raise TypeError(
            "Innovation must be a pandas Series."
        )

    if (
        isinstance(decay_factor, bool)
        or not isinstance(decay_factor, Real)
        or not isfinite(decay_factor)
        or not 0.0 < decay_factor < 1.0
    ):
        raise ValueError(
            "Decay factor must be a finite real number "
            "strictly between 0 and 1."
        )

    if previous_covariance.empty:
        raise ValueError(
            "Previous covariance must not be empty."
        )

    if innovation.empty:
        raise ValueError(
            "Innovation must not be empty."
        )

    if (
        previous_covariance.index.has_duplicates
        or previous_covariance.columns.has_duplicates
    ):
        raise ValueError(
            "Covariance tickers must be unique."
        )

    if innovation.index.has_duplicates:
        raise ValueError(
            "Innovation tickers must be unique."
        )

    if set(previous_covariance.index) != set(
        previous_covariance.columns
    ):
        raise ValueError(
            "Covariance rows and columns "
            "must use the same tickers."
        )

    if set(innovation.index) != set(
        previous_covariance.columns
    ):
        raise ValueError(
            "Previous covariance and innovation "
            "must use the same tickers."
        )

    aligned_covariance = previous_covariance.reindex(
        index=previous_covariance.columns,
        columns=previous_covariance.columns,
    )
    aligned_innovation = innovation.reindex(
        previous_covariance.columns
    )

    try:
        numeric_covariance = aligned_covariance.astype(
            float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Previous covariance must contain numeric values."
        ) from error

    try:
        numeric_innovation = aligned_innovation.astype(
            float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Innovation must contain numeric values."
        ) from error

    previous_values = numeric_covariance.to_numpy(
        dtype=float
    )
    innovation_values = numeric_innovation.to_numpy(
        dtype=float
    )

    if not np.isfinite(previous_values).all():
        raise ValueError(
            "Previous covariance must contain finite numbers."
        )

    if not np.isfinite(innovation_values).all():
        raise ValueError(
            "Innovation must contain finite numbers."
        )

    if not np.allclose(
        previous_values,
        previous_values.T,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Previous covariance must be symmetric."
        )

    covariance_scale = max(
        1.0,
        float(np.abs(previous_values).max()),
    )
    numerical_tolerance = (
        1e-12 * covariance_scale
    )

    eigenvalues = np.linalg.eigvalsh(
        previous_values
    )

    if float(eigenvalues.min()) < -numerical_tolerance:
        raise ValueError(
            "Previous covariance must be "
            "positive semidefinite."
        )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        innovation_outer_product = np.outer(
            innovation_values,
            innovation_values,
        )

        updated_values = (
            decay_factor * previous_values
            + (1.0 - decay_factor)
            * innovation_outer_product
        )

    if not np.isfinite(updated_values).all():
        raise ValueError(
            "Updated covariance must contain finite numbers."
        )

    return pd.DataFrame(
        updated_values,
        index=aligned_covariance.index,
        columns=aligned_covariance.columns,
    )


def calculate_ewma_covariance_sequence(
    asset_returns: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    decay_factor: float,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Calculate EWMA covariance matrices for scheduled forecast dates."""

    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError(
            "Asset returns must be a pandas DataFrame."
        )

    if not isinstance(forecast_schedule, pd.DataFrame):
        raise TypeError(
            "Forecast schedule must be a pandas DataFrame."
        )

    if (
        isinstance(decay_factor, bool)
        or not isinstance(decay_factor, Real)
        or not isfinite(decay_factor)
        or not 0.0 < decay_factor < 1.0
    ):
        raise ValueError(
            "Decay factor must be a finite real number "
            "strictly between 0 and 1."
        )

    if asset_returns.empty:
        raise ValueError(
            "Asset returns must not be empty."
        )

    if forecast_schedule.empty:
        raise ValueError(
            "Forecast schedule must not be empty."
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

    if not isinstance(
        asset_returns.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Asset returns must use a DatetimeIndex."
        )

    if not isinstance(
        forecast_schedule.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Forecast schedule must use a DatetimeIndex."
        )

    if forecast_schedule.index.has_duplicates:
        raise ValueError(
            "Forecast dates must be unique."
        )

    if not forecast_schedule.index.is_monotonic_increasing:
        raise ValueError(
            "Forecast dates must be ordered "
            "by increasing date."
        )

    if asset_returns.index.has_duplicates:
        raise ValueError(
            "Asset-return dates must be unique."
        )

    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError(
            "Asset returns must be ordered "
            "by increasing date."
        )

    if asset_returns.columns.has_duplicates:
        raise ValueError(
            "Asset-return tickers must be unique."
        )

    try:
        numeric_returns = asset_returns.astype(
            float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Asset returns must contain numeric values."
        ) from error

    return_values = numeric_returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Asset returns must contain finite numbers."
        )

    forecast_positions = asset_returns.index.get_indexer(
        forecast_schedule.index
    )

    if np.any(forecast_positions < 0):
        raise ValueError(
            "Forecast dates must exist in asset returns."
        )

    if (
        len(forecast_positions) > 1
        and np.any(np.diff(forecast_positions) != 1)
    ):
        raise ValueError(
            "Forecast dates must be consecutive "
            "asset-return sessions."
        )

    expected_window_length: int | None = None

    for forecast_date, schedule_row in (
        forecast_schedule.iterrows()
    ):
        forecast_date = pd.Timestamp(
            forecast_date
        )
        start_date = pd.Timestamp(
            schedule_row["estimation_start_date"]
        )
        end_date = pd.Timestamp(
            schedule_row["estimation_end_date"]
        )

        if (
            start_date not in asset_returns.index
            or end_date not in asset_returns.index
        ):
            raise ValueError(
                "Estimation-window dates must exist "
                "in asset returns."
            )

        if end_date >= forecast_date:
            raise ValueError(
                "Each estimation window must end "
                "before its forecast date."
            )

        forecast_position = asset_returns.index.get_loc(
            forecast_date
        )

        start_position = asset_returns.index.get_loc(
            start_date
        )
        end_position = asset_returns.index.get_loc(
            end_date
        )

        if end_position != forecast_position - 1:
            raise ValueError(
                "Each estimation window must end "
                "immediately before its forecast date."
            )

        if start_position > end_position:
            raise ValueError(
                "Each estimation window must start "
                "on or before its end date."
            )

        window_length = (
            end_position
            - start_position
            + 1
        )

        if window_length < 2:
            raise ValueError(
                "Each estimation window must contain "
                "at least two observations."
            )

        if expected_window_length is None:
            expected_window_length = window_length
        elif window_length != expected_window_length:
            raise ValueError(
                "Every estimation window must contain "
                "the same number of observations."
            )

    first_schedule_row = forecast_schedule.iloc[
        0
    ]
    first_estimation_returns = numeric_returns.loc[
        first_schedule_row["estimation_start_date"]:
        first_schedule_row["estimation_end_date"]
    ]

    current_covariance = (
        calculate_initial_ewma_covariance(
            first_estimation_returns
        )
    )

    covariance_sequence: dict[
        pd.Timestamp,
        pd.DataFrame,
    ] = {}

    final_position = len(forecast_schedule) - 1

    for position, (
        forecast_date,
        schedule_row,
    ) in enumerate(
        forecast_schedule.iterrows()
    ):
        forecast_date = pd.Timestamp(
            forecast_date
        )

        estimation_returns = numeric_returns.loc[
            schedule_row["estimation_start_date"]:
            schedule_row["estimation_end_date"]
        ]
        rolling_mean = estimation_returns.mean()

        covariance_sequence[forecast_date] = (
            current_covariance.copy()
        )

        if position == final_position:
            continue

        realized_return = numeric_returns.loc[
            forecast_date
        ]
        innovation = (
            realized_return
            - rolling_mean
        )

        current_covariance = (
            calculate_ewma_covariance_update(
                previous_covariance=(
                    current_covariance
                ),
                innovation=innovation,
                decay_factor=decay_factor,
            )
        )

    return covariance_sequence


def calculate_sample_portfolio_moments(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> SamplePortfolioMoments:
    """Estimate asset moments and combine them using portfolio weights.

    For asset mean vector mu, sample covariance matrix Sigma, and
    portfolio weights w:

        portfolio mean       = w.T @ mu
        portfolio variance   = w.T @ Sigma @ w
        portfolio volatility = sqrt(portfolio variance)

    The covariance matrix uses the sample denominator n - 1.
    """

    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError(
            "Asset returns must be a pandas DataFrame."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "Portfolio weights must be a pandas Series."
        )

    if asset_returns.empty:
        raise ValueError(
            "Asset returns must not be empty."
        )

    if len(asset_returns) < 2:
        raise ValueError(
            "At least two return observations are required."
        )

    if asset_returns.columns.has_duplicates:
        raise ValueError(
            "Asset-return tickers must be unique."
        )

    if weights.index.has_duplicates:
        raise ValueError(
            "Portfolio-weight tickers must be unique."
        )

    if set(asset_returns.columns) != set(weights.index):
        raise ValueError(
            "Asset returns and portfolio weights "
            "must use the same tickers."
        )

    try:
        numeric_returns = asset_returns.astype(float)
        aligned_weights = (
            weights
            .reindex(asset_returns.columns)
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Asset returns and portfolio weights "
            "must be numeric."
        ) from error

    if not np.isfinite(
        numeric_returns.to_numpy()
    ).all():
        raise ValueError(
            "Asset returns must be finite numbers."
        )

    if not np.isfinite(
        aligned_weights.to_numpy()
    ).all():
        raise ValueError(
            "Portfolio weights must be finite numbers."
        )

    mean_vector = (
        numeric_returns
        .mean()
        .rename("mean_return")
    )
    covariance_matrix = numeric_returns.cov(
        ddof=1
    )

    covariance_values = covariance_matrix.to_numpy(
        dtype=float
    )

    if not np.isfinite(covariance_values).all():
        raise ValueError(
            "Covariance matrix must contain finite numbers."
        )

    if not np.allclose(
        covariance_values,
        covariance_values.T,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Covariance matrix must be symmetric."
        )

    covariance_scale = max(
        1.0,
        float(np.abs(covariance_values).max()),
    )
    numerical_tolerance = (
        1e-12 * covariance_scale
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance_values
    )

    if float(eigenvalues.min()) < -numerical_tolerance:
        raise ValueError(
            "Covariance matrix must be "
            "positive semidefinite."
        )

    portfolio_mean = float(
        aligned_weights @ mean_vector
    )
    portfolio_variance = float(
        aligned_weights
        @ covariance_matrix
        @ aligned_weights
    )

    if portfolio_variance < -numerical_tolerance:
        raise ValueError(
            "Portfolio variance cannot be negative."
        )

    # Roundoff can produce a tiny negative value for a theoretically
    # zero-variance portfolio.
    portfolio_variance = max(
        portfolio_variance,
        0.0,
    )
    portfolio_volatility = float(
        np.sqrt(portfolio_variance)
    )

    return SamplePortfolioMoments(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        portfolio_mean=portfolio_mean,
        portfolio_variance=portfolio_variance,
        portfolio_volatility=portfolio_volatility,
    )
