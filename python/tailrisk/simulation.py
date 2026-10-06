"""Gaussian asset-return simulation."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd

from tailrisk.covariance import (
    calculate_ewma_covariance_sequence,
    calculate_sample_portfolio_moments,
)

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


@dataclass(frozen=True)
class EmpiricalRiskEstimate:
    """VaR and Expected Shortfall estimated from observed losses."""

    confidence_level: float
    observation_count: int
    tail_observation_count: int
    value_at_risk: float
    expected_shortfall: float


def generate_standard_normal_shocks(
    scenario_count: int,
    asset_count: int,
    random_seed: int,
) -> np.ndarray:
    """Generate reproducible independent standard-normal shocks."""

    if (
        isinstance(scenario_count, bool)
        or not isinstance(scenario_count, Integral)
        or scenario_count <= 0
    ):
        raise ValueError(
            "Scenario count must be a positive integer."
        )

    if (
        isinstance(asset_count, bool)
        or not isinstance(asset_count, Integral)
        or asset_count <= 0
    ):
        raise ValueError(
            "Asset count must be a positive integer."
        )

    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError(
            "Random seed must be a non-negative integer."
        )

    random_generator = np.random.default_rng(
        int(random_seed)
    )

    return random_generator.standard_normal(
        size=(
            int(scenario_count),
            int(asset_count),
        )
    )


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

    independent_shocks = generate_standard_normal_shocks(
        scenario_count=scenario_count,
        asset_count=len(mean_values),
        random_seed=random_seed,
    )

    correlated_shocks = (
        independent_shocks
        @ cholesky_factor.T
    )

    return correlated_shocks + mean_values




def generate_gaussian_portfolio_losses_from_shocks(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    weights: pd.Series,
    standard_normal_shocks: np.ndarray,
) -> np.ndarray:
    """Project shared Gaussian shocks into portfolio-loss scenarios.

    If covariance_matrix = L @ L.T and Z contains independent
    standard-normal shocks, portfolio returns are calculated as:

        portfolio return
        = weights.T @ mean_vector
          + Z @ (L.T @ weights)

    This avoids constructing a full asset-return scenario matrix.
    """

    if not isinstance(mean_vector, pd.Series):
        raise TypeError(
            "Mean vector must be a pandas Series."
        )

    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError(
            "Covariance matrix must be a pandas DataFrame."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "Portfolio weights must be a pandas Series."
        )

    if not isinstance(standard_normal_shocks, np.ndarray):
        raise TypeError(
            "Standard-normal shocks must be a NumPy array."
        )

    if mean_vector.empty:
        raise ValueError(
            "Mean vector must not be empty."
        )

    if weights.empty:
        raise ValueError(
            "Portfolio weights must not be empty."
        )

    if mean_vector.index.has_duplicates:
        raise ValueError(
            "Mean-vector tickers must be unique."
        )

    if weights.index.has_duplicates:
        raise ValueError(
            "Portfolio-weight tickers must be unique."
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

    if set(weights.index) != set(mean_vector.index):
        raise ValueError(
            "Portfolio weights and mean vector "
            "must use the same tickers."
        )

    if (
        standard_normal_shocks.ndim != 2
        or standard_normal_shocks.shape[0] == 0
    ):
        raise ValueError(
            "Standard-normal shocks must be a nonempty "
            "two-dimensional array."
        )

    if standard_normal_shocks.shape[1] != len(
        mean_vector
    ):
        raise ValueError(
            "Shock columns must match the number of assets."
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
        aligned_weights = (
            weights
            .reindex(mean_vector.index)
            .astype(float)
        )
        numeric_shocks = standard_normal_shocks.astype(
            float,
            copy=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Simulation inputs must contain numeric values."
        ) from error

    mean_values = numeric_mean.to_numpy(dtype=float)
    covariance_values = aligned_covariance.to_numpy(
        dtype=float
    )
    weight_values = aligned_weights.to_numpy(
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

    if not np.isfinite(weight_values).all():
        raise ValueError(
            "Portfolio weights must contain finite numbers."
        )

    if not np.isfinite(numeric_shocks).all():
        raise ValueError(
            "Standard-normal shocks must contain finite numbers."
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
    numerical_tolerance = 1e-12 * covariance_scale

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

    portfolio_mean = float(
        weight_values @ mean_values
    )

    portfolio_shock_loadings = (
        cholesky_factor.T @ weight_values
    )

    simulated_portfolio_returns = (
        portfolio_mean
        + numeric_shocks @ portfolio_shock_loadings
    )

    return -simulated_portfolio_returns




def generate_gaussian_portfolio_losses(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    weights: pd.Series,
    scenario_count: int,
    random_seed: int,
) -> np.ndarray:
    """Generate one-session Gaussian portfolio-loss scenarios."""

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
            scenario_count=scenario_count,
            random_seed=random_seed,
        )
    )

    simulated_portfolio_returns = (
        asset_return_scenarios
        @ weight_values
    )

    return -simulated_portfolio_returns


def calculate_empirical_var_es(
    losses: np.ndarray,
    confidence_level: float,
) -> EmpiricalRiskEstimate:
    """Calculate empirical VaR and ES from a one-dimensional loss sample."""

    if not isinstance(losses, np.ndarray):
        raise TypeError(
            "Losses must be a NumPy array."
        )

    if losses.ndim != 1:
        raise ValueError(
            "Losses must be one-dimensional."
        )

    if losses.size == 0:
        raise ValueError(
            "Losses must not be empty."
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

    try:
        numeric_losses = losses.astype(
            float,
            copy=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Losses must contain numeric values."
        ) from error

    if not np.isfinite(numeric_losses).all():
        raise ValueError(
            "Losses must contain finite numbers."
        )

    value_at_risk = float(
        np.quantile(
            numeric_losses,
            float(confidence_level),
            method="linear",
        )
    )

    tail_losses = numeric_losses[
        numeric_losses >= value_at_risk
    ]

    expected_shortfall = float(
        tail_losses.mean()
    )

    return EmpiricalRiskEstimate(
        confidence_level=float(confidence_level),
        observation_count=len(numeric_losses),
        tail_observation_count=len(tail_losses),
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
    )


def calculate_gaussian_monte_carlo_var_es(
    mean_vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
    scenario_count: int,
    random_seed: int,
) -> GaussianMonteCarloForecast:
    """Estimate portfolio VaR and ES using Gaussian scenarios.

    The same simulated loss sample supplies the empirical VaR quantile,
    Expected Shortfall tail average, and simulated portfolio moments.
    """

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

    simulated_losses = generate_gaussian_portfolio_losses(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        weights=weights,
        scenario_count=int(scenario_count),
        random_seed=random_seed,
    )

    simulated_portfolio_returns = (
        -simulated_losses
    )

    risk_estimate = calculate_empirical_var_es(
        losses=simulated_losses,
        confidence_level=confidence_level,
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
        tail_scenario_count=(
            risk_estimate.tail_observation_count
        ),
        simulated_mean_return=simulated_mean_return,
        simulated_volatility=simulated_volatility,
        value_at_risk=risk_estimate.value_at_risk,
        expected_shortfall=(
            risk_estimate.expected_shortfall
        ),
    )


def _calculate_gaussian_monte_carlo_forecasts(
    asset_returns: pd.DataFrame,
    beginning_weights: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
    scenario_count: int,
    random_seed: int,
    covariance_sequence: (
        dict[pd.Timestamp, pd.DataFrame] | None
    ),
) -> pd.DataFrame:
    """Calculate Gaussian forecasts using the supplied covariances.

    One matrix of independent standard-normal shocks is generated and
    reused for every forecast date. Each date uses only its scheduled
    historical estimation window and its beginning-of-period weights.
    Results are indexed by forecast date and confidence level.
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

    if asset_returns.empty:
        raise ValueError(
            "Asset returns must not be empty."
        )

    if beginning_weights.empty:
        raise ValueError(
            "Beginning weights must not be empty."
        )

    if forecast_schedule.empty:
        raise ValueError(
            "Forecast schedule must not be empty."
        )

    if not confidence_levels:
        raise ValueError(
            "At least one confidence level is required."
        )

    for confidence_level in confidence_levels:
        if (
            isinstance(confidence_level, bool)
            or not isinstance(confidence_level, Real)
            or not isfinite(confidence_level)
            or not 0.0 < confidence_level < 1.0
        ):
            raise ValueError(
                "Confidence levels must be finite numbers "
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

    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError(
            "Random seed must be a non-negative integer."
        )

    required_schedule_columns = {
        "estimation_start_date",
        "estimation_end_date",
        "realized_return",
        "realized_loss",
    }

    if not required_schedule_columns.issubset(
        forecast_schedule.columns
    ):
        raise ValueError(
            "Forecast schedule must contain estimation dates "
            "and realized return and loss columns."
        )

    for frame, label in (
        (asset_returns, "Asset returns"),
        (beginning_weights, "Beginning weights"),
        (forecast_schedule, "Forecast schedule"),
    ):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(
                f"{label} must use a DatetimeIndex."
            )

        if frame.index.hasnans:
            raise ValueError(
                f"{label} dates must not be missing."
            )

        if frame.index.has_duplicates:
            raise ValueError(
                f"{label} dates must be unique."
            )

        if not frame.index.is_monotonic_increasing:
            raise ValueError(
                f"{label} dates must be ordered by increasing date."
            )

    if asset_returns.columns.has_duplicates:
        raise ValueError(
            "Asset-return tickers must be unique."
        )

    if beginning_weights.columns.has_duplicates:
        raise ValueError(
            "Beginning-weight tickers must be unique."
        )

    if set(asset_returns.columns) != set(
        beginning_weights.columns
    ):
        raise ValueError(
            "Asset returns and beginning weights "
            "must use the same tickers."
        )

    standard_normal_shocks = generate_standard_normal_shocks(
        scenario_count=int(scenario_count),
        asset_count=len(asset_returns.columns),
        random_seed=int(random_seed),
    )

    records: list[dict[str, object]] = []
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

        if start_date > end_date:
            raise ValueError(
                "Each estimation window must start on or "
                "before its end date."
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

        if forecast_date not in asset_returns.index:
            raise ValueError(
                "Every forecast date must exist "
                "in the asset returns."
            )

        if forecast_date not in beginning_weights.index:
            raise ValueError(
                "Beginning weights are required "
                "for every forecast date."
            )

        try:
            realized_return = float(
                schedule_row["realized_return"]
            )
            realized_loss = float(
                schedule_row["realized_loss"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Realized returns and losses must be numeric."
            ) from error

        if not (
            isfinite(realized_return)
            and isfinite(realized_loss)
        ):
            raise ValueError(
                "Realized returns and losses must be finite."
            )

        if not np.isclose(
            realized_loss,
            -realized_return,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "Each realized loss must equal the negative "
                "of its realized return."
            )

        estimation_returns = asset_returns.loc[
            start_date:end_date
        ]
        forecast_weights = beginning_weights.loc[
            forecast_date
        ]

        if covariance_sequence is None:
            moments = calculate_sample_portfolio_moments(
                asset_returns=estimation_returns,
                weights=forecast_weights,
            )
            mean_vector = moments.mean_vector
            covariance_matrix = moments.covariance_matrix
        else:
            mean_vector = estimation_returns.mean()
            covariance_matrix = covariance_sequence[
                forecast_date
            ]

        simulated_losses = (
            generate_gaussian_portfolio_losses_from_shocks(
                mean_vector=mean_vector,
                covariance_matrix=covariance_matrix,
                weights=forecast_weights,
                standard_normal_shocks=(
                    standard_normal_shocks
                ),
            )
        )
        simulated_returns = -simulated_losses
        simulated_mean_return = float(
            simulated_returns.mean()
        )
        simulated_volatility = float(
            simulated_returns.std(ddof=1)
        )

        for confidence_level in confidence_levels:
            estimate = calculate_empirical_var_es(
                losses=simulated_losses,
                confidence_level=confidence_level,
            )

            forecast_keys.append(
                (
                    forecast_date,
                    estimate.confidence_level,
                )
            )
            records.append(
                {
                    "estimation_start_date": start_date,
                    "estimation_end_date": end_date,
                    "realized_return": realized_return,
                    "realized_loss": realized_loss,
                    "simulated_mean_return": (
                        simulated_mean_return
                    ),
                    "simulated_volatility": (
                        simulated_volatility
                    ),
                    "scenario_count": int(scenario_count),
                    "random_seed": int(random_seed),
                    "tail_scenario_count": (
                        estimate.tail_observation_count
                    ),
                    "value_at_risk": estimate.value_at_risk,
                    "expected_shortfall": (
                        estimate.expected_shortfall
                    ),
                    "var_exceedance": bool(
                        realized_loss
                        > estimate.value_at_risk
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


def calculate_rolling_gaussian_monte_carlo_forecasts(
    asset_returns: pd.DataFrame,
    beginning_weights: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
    scenario_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Calculate rolling Gaussian Monte Carlo forecasts."""

    return _calculate_gaussian_monte_carlo_forecasts(
        asset_returns=asset_returns,
        beginning_weights=beginning_weights,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        scenario_count=scenario_count,
        random_seed=random_seed,
        covariance_sequence=None,
    )


def calculate_ewma_gaussian_monte_carlo_forecasts(
    asset_returns: pd.DataFrame,
    beginning_weights: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
    scenario_count: int,
    random_seed: int,
    decay_factor: float,
) -> pd.DataFrame:
    """Calculate EWMA Gaussian Monte Carlo forecasts."""

    covariance_sequence = calculate_ewma_covariance_sequence(
        asset_returns=asset_returns,
        forecast_schedule=forecast_schedule,
        decay_factor=decay_factor,
    )

    return _calculate_gaussian_monte_carlo_forecasts(
        asset_returns=asset_returns,
        beginning_weights=beginning_weights,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        scenario_count=scenario_count,
        random_seed=random_seed,
        covariance_sequence=covariance_sequence,
    )
