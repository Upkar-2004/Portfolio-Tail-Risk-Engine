"""Gaussian asset-return simulation."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class GaussianMonteCarloForecast:
    """Summary of one Gaussian Monte Carlo risk calculation."""

    confidence_level: float
    scenario_count: int
    random_seed: int
    tail_scenario_count: int
    simulated_mean_return: float
    simulated_volatility: float
    value_at_risk: float
    expected_shortfall: float



def generate_gaussian_return_scenarios(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    scenario_count: int,
    random_seed: int,
) -> np.ndarray:
    """Generate reproducible correlated Gaussian return scenarios.

    If Sigma = L @ L.T and z contains independent standard-normal
    shocks, then:

        simulated return = mu + L @ z

    Each returned row represents one possible next-session outcome.
    """

    if not isinstance(mean_vector, pd.Series):
        raise TypeError(
            "Mean vector must be a pandas Series."
        )

    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError(
            "Covariance matrix must be a pandas DataFrame."
        )

    if mean_vector.empty:
        raise ValueError(
            "Mean vector must not be empty."
        )

    if mean_vector.index.has_duplicates:
        raise ValueError(
            "Mean-vector tickers must be unique."
        )

    if (
        covariance_matrix.index.has_duplicates
        or covariance_matrix.columns.has_duplicates
    ):
        raise ValueError(
            "Covariance-matrix tickers must be unique."
        )

    if set(covariance_matrix.index) != set(
        mean_vector.index
    ):
        raise ValueError(
            "Covariance rows and mean vector "
            "must use the same tickers."
        )

    if set(covariance_matrix.columns) != set(
        mean_vector.index
    ):
        raise ValueError(
            "Covariance columns and mean vector "
            "must use the same tickers."
        )

    if (
        isinstance(scenario_count, bool)
        or not isinstance(scenario_count, Integral)
        or scenario_count <= 0
    ):
        raise ValueError(
            "Scenario count must be a positive integer."
        )

    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError(
            "Random seed must be a non-negative integer."
        )

    try:
        numeric_mean = mean_vector.astype(float)
        aligned_covariance = (
            covariance_matrix
            .reindex(
                index=mean_vector.index,
                columns=mean_vector.index,
            )
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Mean vector and covariance matrix "
            "must contain numeric values."
        ) from error

    mean_values = numeric_mean.to_numpy(
        dtype=float
    )
    covariance_values = aligned_covariance.to_numpy(
        dtype=float
    )

    if not np.isfinite(mean_values).all():
        raise ValueError(
            "Mean vector must contain finite numbers."
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

    try:
        cholesky_factor = np.linalg.cholesky(
            covariance_values
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Covariance matrix must be positive "
            "definite for Cholesky simulation."
        ) from error

    random_generator = np.random.default_rng(
        int(random_seed)
    )

    independent_shocks = random_generator.standard_normal(
        size=(
            int(scenario_count),
            len(mean_values),
        )
    )

    correlated_shocks = (
        independent_shocks
        @ cholesky_factor.T
    )

    return correlated_shocks + mean_values




def calculate_gaussian_monte_carlo_var_es(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
    scenario_count: int,
    random_seed: int,
) -> GaussianMonteCarloForecast:
    """Estimate portfolio VaR and ES using Gaussian scenarios.

    Asset-return scenarios are combined using ticker-aligned portfolio
    weights. Returns are converted into losses before the empirical VaR
    quantile and Expected Shortfall tail average are calculated.
    """

    if not isinstance(mean_vector, pd.Series):
        raise TypeError(
            "Mean vector must be a pandas Series."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "Portfolio weights must be a pandas Series."
        )

    if weights.empty:
        raise ValueError(
            "Portfolio weights must not be empty."
        )

    if weights.index.has_duplicates:
        raise ValueError(
            "Portfolio-weight tickers must be unique."
        )

    if set(weights.index) != set(mean_vector.index):
        raise ValueError(
            "Portfolio weights and mean vector "
            "must use the same tickers."
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

    if (
        isinstance(scenario_count, bool)
        or not isinstance(scenario_count, Integral)
        or scenario_count < 2
    ):
        raise ValueError(
            "Scenario count must be an integer "
            "of at least two."
        )

    try:
        aligned_weights = (
            weights
            .reindex(mean_vector.index)
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Portfolio weights must contain "
            "numeric values."
        ) from error

    weight_values = aligned_weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(weight_values).all():
        raise ValueError(
            "Portfolio weights must contain "
            "finite numbers."
        )

    asset_return_scenarios = (
        generate_gaussian_return_scenarios(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            scenario_count=int(scenario_count),
            random_seed=random_seed,
        )
    )

    simulated_portfolio_returns = (
        asset_return_scenarios
        @ weight_values
    )
    simulated_losses = (
        -simulated_portfolio_returns
    )

    value_at_risk = float(
        np.quantile(
            simulated_losses,
            float(confidence_level),
            method="linear",
        )
    )

    tail_losses = simulated_losses[
        simulated_losses >= value_at_risk
    ]

    expected_shortfall = float(
        tail_losses.mean()
    )

    simulated_mean_return = float(
        simulated_portfolio_returns.mean()
    )
    simulated_volatility = float(
        simulated_portfolio_returns.std(
            ddof=1
        )
    )

    return GaussianMonteCarloForecast(
        confidence_level=float(confidence_level),
        scenario_count=int(scenario_count),
        random_seed=int(random_seed),
        tail_scenario_count=len(tail_losses),
        simulated_mean_return=simulated_mean_return,
        simulated_volatility=simulated_volatility,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
    )