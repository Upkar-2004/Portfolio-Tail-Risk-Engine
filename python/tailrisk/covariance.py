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
