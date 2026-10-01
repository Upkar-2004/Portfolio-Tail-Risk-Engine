"""Covariance estimators used by the risk models."""

from dataclasses import dataclass

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