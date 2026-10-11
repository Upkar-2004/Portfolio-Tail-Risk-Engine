"""Filtered historical simulation calculations."""


from numbers import Integral

import numpy as np
import pandas as pd

from tailrisk.covariance import (
    calculate_ewma_covariance_sequence,
)
from tailrisk.simulation import (
    EmpiricalRiskEstimate,
    calculate_empirical_var_es,
)


def _prepare_cholesky_transform(
    vector: pd.Series,
    covariance_matrix: pd.DataFrame,
    vector_label: str,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Validate and align one vector and its covariance matrix."""

    if vector.empty:
        raise ValueError(
            f"{vector_label} must not be empty."
        )

    if covariance_matrix.empty:
        raise ValueError(
            "Covariance matrix must not be empty."
        )

    if vector.index.has_duplicates:
        raise ValueError(
            f"{vector_label} tickers must be unique."
        )

    if (
        covariance_matrix.index.has_duplicates
        or covariance_matrix.columns.has_duplicates
    ):
        raise ValueError(
            "Covariance tickers must be unique."
        )

    if set(covariance_matrix.index) != set(
        covariance_matrix.columns
    ):
        raise ValueError(
            "Covariance rows and columns must use "
            "the same tickers."
        )

    if set(vector.index) != set(
        covariance_matrix.columns
    ):
        raise ValueError(
            f"{vector_label} and covariance matrix must use "
            "the same tickers."
        )

    ticker_order = covariance_matrix.columns

    try:
        numeric_covariance = (
            covariance_matrix
            .reindex(
                index=ticker_order,
                columns=ticker_order,
            )
            .astype(float)
        )
        numeric_vector = (
            vector
            .reindex(ticker_order)
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{vector_label} and covariance matrix must "
            "contain numeric values."
        ) from error

    covariance_values = numeric_covariance.to_numpy(
        dtype=float
    )
    vector_values = numeric_vector.to_numpy(
        dtype=float
    )

    if not np.isfinite(covariance_values).all():
        raise ValueError(
            "Covariance matrix must contain finite numbers."
        )

    if not np.isfinite(vector_values).all():
        raise ValueError(
            f"{vector_label} must contain finite numbers."
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

    if float(eigenvalues.min()) <= numerical_tolerance:
        raise ValueError(
            "Covariance matrix must be positive definite."
        )

    try:
        cholesky_factor = np.linalg.cholesky(
            covariance_values
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Covariance matrix must be positive definite."
        ) from error

    return (
        ticker_order,
        vector_values,
        cholesky_factor,
    )


def calculate_standardized_residual(
    innovation: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """Remove the covariance scale from one innovation vector.

    If covariance_matrix = L @ L.T, solve:

        L @ standardized_residual = innovation

    This is equivalent to L inverse times the innovation, but solving
    the linear system is more numerically stable than forming an inverse.
    """

    if not isinstance(innovation, pd.Series):
        raise TypeError(
            "Innovation must be a pandas Series."
        )

    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError(
            "Covariance matrix must be a pandas DataFrame."
        )

    (
        ticker_order,
        innovation_values,
        cholesky_factor,
    ) = _prepare_cholesky_transform(
        vector=innovation,
        covariance_matrix=covariance_matrix,
        vector_label="Innovation",
    )

    standardized_values = np.linalg.solve(
        cholesky_factor,
        innovation_values,
    )

    if not np.isfinite(standardized_values).all():
        raise ValueError(
            "Standardized residual must contain "
            "finite numbers."
        )

    return pd.Series(
        standardized_values,
        index=ticker_order,
        name="standardized_residual",
    )


def calculate_rescaled_innovation(
    standardized_residual: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """Apply a forecast covariance scale to a standardized residual.

    If covariance_matrix = L @ L.T, calculate:

        rescaled_innovation = L @ standardized_residual
    """

    if not isinstance(standardized_residual, pd.Series):
        raise TypeError(
            "Standardized residual must be a pandas Series."
        )

    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError(
            "Covariance matrix must be a pandas DataFrame."
        )

    (
        ticker_order,
        standardized_values,
        cholesky_factor,
    ) = _prepare_cholesky_transform(
        vector=standardized_residual,
        covariance_matrix=covariance_matrix,
        vector_label="Standardized residual",
    )

    rescaled_values = (
        cholesky_factor @ standardized_values
    )

    if not np.isfinite(rescaled_values).all():
        raise ValueError(
            "Rescaled innovation must contain finite numbers."
        )

    return pd.Series(
        rescaled_values,
        index=ticker_order,
        name="rescaled_innovation",
    )


def calculate_filtered_return_scenario(
    forecast_mean: pd.Series,
    standardized_residual: pd.Series,
    forecast_covariance: pd.DataFrame,
) -> pd.Series:
    """Construct one filtered historical asset-return scenario.

    The scenario is:

        forecast_mean
        + forecast_cholesky_factor @ standardized_residual
    """

    if not isinstance(forecast_mean, pd.Series):
        raise TypeError(
            "Forecast mean must be a pandas Series."
        )

    if forecast_mean.empty:
        raise ValueError(
            "Forecast mean must not be empty."
        )

    if forecast_mean.index.has_duplicates:
        raise ValueError(
            "Forecast-mean tickers must be unique."
        )

    rescaled_innovation = calculate_rescaled_innovation(
        standardized_residual=standardized_residual,
        covariance_matrix=forecast_covariance,
    )

    if set(forecast_mean.index) != set(
        rescaled_innovation.index
    ):
        raise ValueError(
            "Forecast mean and rescaled innovation "
            "must use the same tickers."
        )

    try:
        aligned_mean = (
            forecast_mean
            .reindex(rescaled_innovation.index)
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Forecast mean must contain numeric values."
        ) from error

    mean_values = aligned_mean.to_numpy(
        dtype=float
    )

    if not np.isfinite(mean_values).all():
        raise ValueError(
            "Forecast mean must contain finite numbers."
        )

    scenario = (
        aligned_mean + rescaled_innovation
    )

    if not np.isfinite(
        scenario.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Filtered return scenario must contain "
            "finite numbers."
        )

    return scenario.rename(
        "filtered_return_scenario"
    )


def calculate_filtered_portfolio_loss(
    return_scenario: pd.Series,
    weights: pd.Series,
) -> float:
    """Convert one asset-return scenario into portfolio loss."""

    if not isinstance(return_scenario, pd.Series):
        raise TypeError(
            "Return scenario must be a pandas Series."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "Portfolio weights must be a pandas Series."
        )

    if return_scenario.empty:
        raise ValueError(
            "Return scenario must not be empty."
        )

    if weights.empty:
        raise ValueError(
            "Portfolio weights must not be empty."
        )

    if return_scenario.index.has_duplicates:
        raise ValueError(
            "Return-scenario tickers must be unique."
        )

    if weights.index.has_duplicates:
        raise ValueError(
            "Portfolio-weight tickers must be unique."
        )

    if set(return_scenario.index) != set(weights.index):
        raise ValueError(
            "Return scenario and portfolio weights "
            "must use the same tickers."
        )

    try:
        numeric_scenario = return_scenario.astype(float)
        aligned_weights = (
            weights
            .reindex(return_scenario.index)
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Return scenario and portfolio weights "
            "must contain numeric values."
        ) from error

    scenario_values = numeric_scenario.to_numpy(
        dtype=float
    )
    weight_values = aligned_weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(scenario_values).all():
        raise ValueError(
            "Return scenario must contain finite numbers."
        )

    if not np.isfinite(weight_values).all():
        raise ValueError(
            "Portfolio weights must contain finite numbers."
        )

    portfolio_return = float(
        aligned_weights @ numeric_scenario
    )

    if not np.isfinite(portfolio_return):
        raise ValueError(
            "Portfolio return must be finite."
        )

    return -portfolio_return




def resample_standardized_residuals(
    residual_pool: pd.DataFrame,
    scenario_count: int,
    random_seed: int,
) -> pd.DataFrame:
    """Resample complete standardized-residual vectors with replacement."""

    if not isinstance(residual_pool, pd.DataFrame):
        raise TypeError(
            "Residual pool must be a pandas DataFrame."
        )

    if residual_pool.empty:
        raise ValueError(
            "Residual pool must not be empty."
        )

    if residual_pool.columns.has_duplicates:
        raise ValueError(
            "Residual-pool tickers must be unique."
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
        numeric_pool = residual_pool.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Residual pool must contain numeric values."
        ) from error

    pool_values = numeric_pool.to_numpy(
        dtype=float
    )

    if not np.isfinite(pool_values).all():
        raise ValueError(
            "Residual pool must contain finite numbers."
        )

    random_generator = np.random.default_rng(
        int(random_seed)
    )
    selected_positions = random_generator.integers(
        low=0,
        high=len(numeric_pool),
        size=int(scenario_count),
    )

    sampled_residuals = (
        numeric_pool
        .iloc[selected_positions]
        .copy()
    )
    sampled_residuals.index = pd.RangeIndex(
        start=0,
        stop=int(scenario_count),
        name="scenario",
    )

    return sampled_residuals




def calculate_initial_standardized_residual_pool(
    estimation_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the initial in-sample standardized-residual pool."""

    if not isinstance(estimation_returns, pd.DataFrame):
        raise TypeError(
            "Estimation returns must be a pandas DataFrame."
        )

    if estimation_returns.empty:
        raise ValueError(
            "Estimation returns must not be empty."
        )

    if len(estimation_returns) < 2:
        raise ValueError(
            "Estimation returns must contain at least "
            "two observations."
        )

    if estimation_returns.columns.has_duplicates:
        raise ValueError(
            "Estimation-return tickers must be unique."
        )

    if not isinstance(
        estimation_returns.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Estimation returns must use a DatetimeIndex."
        )

    if estimation_returns.index.has_duplicates:
        raise ValueError(
            "Estimation-return dates must be unique."
        )

    if not estimation_returns.index.is_monotonic_increasing:
        raise ValueError(
            "Estimation-return dates must be chronological."
        )

    try:
        numeric_returns = estimation_returns.astype(
            float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Estimation returns must contain numeric values."
        ) from error

    return_values = numeric_returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Estimation returns must contain finite numbers."
        )

    initial_mean = numeric_returns.mean()
    initial_covariance = numeric_returns.cov(
        ddof=1
    )

    (
        ticker_order,
        mean_values,
        cholesky_factor,
    ) = _prepare_cholesky_transform(
        vector=initial_mean,
        covariance_matrix=initial_covariance,
        vector_label="Initial mean",
    )

    aligned_returns = (
        numeric_returns
        .reindex(columns=ticker_order)
        .to_numpy(dtype=float)
    )

    innovation_values = (
        aligned_returns - mean_values
    )

    standardized_values = np.linalg.solve(
        cholesky_factor,
        innovation_values.T,
    ).T

    if not np.isfinite(standardized_values).all():
        raise ValueError(
            "Initial standardized residuals must "
            "contain finite numbers."
        )

    return pd.DataFrame(
        standardized_values,
        index=numeric_returns.index.copy(),
        columns=ticker_order.copy(),
    )


def update_standardized_residual_pool(
    residual_pool: pd.DataFrame,
    realized_return: pd.Series,
    forecast_mean: pd.Series,
    forecast_covariance: pd.DataFrame,
    observation_date: pd.Timestamp,
) -> pd.DataFrame:
    """Replace the oldest residual with one newly observed shock."""

    if not isinstance(residual_pool, pd.DataFrame):
        raise TypeError(
            "Residual pool must be a pandas DataFrame."
        )

    if not isinstance(realized_return, pd.Series):
        raise TypeError(
            "Realized return must be a pandas Series."
        )

    if not isinstance(forecast_mean, pd.Series):
        raise TypeError(
            "Forecast mean must be a pandas Series."
        )

    if not isinstance(
        forecast_covariance,
        pd.DataFrame,
    ):
        raise TypeError(
            "Forecast covariance must be a pandas DataFrame."
        )

    if not isinstance(observation_date, pd.Timestamp):
        raise TypeError(
            "Observation date must be a pandas Timestamp."
        )

    if residual_pool.empty:
        raise ValueError(
            "Residual pool must not be empty."
        )

    if not isinstance(
        residual_pool.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Residual pool must use a DatetimeIndex."
        )

    if residual_pool.index.has_duplicates:
        raise ValueError(
            "Residual-pool dates must be unique."
        )

    if not residual_pool.index.is_monotonic_increasing:
        raise ValueError(
            "Residual-pool dates must be chronological."
        )

    if observation_date <= residual_pool.index[-1]:
        raise ValueError(
            "Observation date must be later than every "
            "date in the residual pool."
        )

    if residual_pool.columns.has_duplicates:
        raise ValueError(
            "Residual-pool tickers must be unique."
        )

    if realized_return.index.has_duplicates:
        raise ValueError(
            "Realized-return tickers must be unique."
        )

    if forecast_mean.index.has_duplicates:
        raise ValueError(
            "Forecast-mean tickers must be unique."
        )

    ticker_order = residual_pool.columns

    if (
        set(realized_return.index) != set(ticker_order)
        or set(forecast_mean.index) != set(ticker_order)
    ):
        raise ValueError(
            "Residual pool, realized return, and forecast "
            "mean must use the same tickers."
        )

    try:
        numeric_pool = residual_pool.astype(float)
        aligned_return = (
            realized_return
            .reindex(ticker_order)
            .astype(float)
        )
        aligned_mean = (
            forecast_mean
            .reindex(ticker_order)
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Residual pool, realized return, and forecast "
            "mean must contain numeric values."
        ) from error

    if not np.isfinite(
        numeric_pool.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Residual pool must contain finite numbers."
        )

    if not np.isfinite(
        aligned_return.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Realized return must contain finite numbers."
        )

    if not np.isfinite(
        aligned_mean.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Forecast mean must contain finite numbers."
        )

    innovation = (
        aligned_return - aligned_mean
    ).rename("innovation")

    new_residual = calculate_standardized_residual(
        innovation=innovation,
        covariance_matrix=forecast_covariance,
    ).reindex(ticker_order)

    new_row = pd.DataFrame(
        [new_residual.to_numpy(dtype=float)],
        index=pd.DatetimeIndex(
            [observation_date],
            name=residual_pool.index.name,
        ),
        columns=ticker_order,
    )

    return pd.concat(
        [
            numeric_pool.iloc[1:].copy(),
            new_row,
        ]
    )






def calculate_filtered_portfolio_losses(
    sampled_residuals: pd.DataFrame,
    forecast_mean: pd.Series,
    forecast_covariance: pd.DataFrame,
    weights: pd.Series,
) -> np.ndarray:
    """Transform sampled residuals into filtered portfolio losses."""

    if not isinstance(sampled_residuals, pd.DataFrame):
        raise TypeError(
            "Sampled residuals must be a pandas DataFrame."
        )

    if not isinstance(forecast_mean, pd.Series):
        raise TypeError(
            "Forecast mean must be a pandas Series."
        )

    if not isinstance(
        forecast_covariance,
        pd.DataFrame,
    ):
        raise TypeError(
            "Forecast covariance must be a pandas DataFrame."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "Portfolio weights must be a pandas Series."
        )

    if sampled_residuals.empty:
        raise ValueError(
            "Sampled residuals must not be empty."
        )

    if weights.empty:
        raise ValueError(
            "Portfolio weights must not be empty."
        )

    if sampled_residuals.columns.has_duplicates:
        raise ValueError(
            "Sampled-residual tickers must be unique."
        )

    if weights.index.has_duplicates:
        raise ValueError(
            "Portfolio-weight tickers must be unique."
        )

    (
        ticker_order,
        mean_values,
        forecast_cholesky_factor,
    ) = _prepare_cholesky_transform(
        vector=forecast_mean,
        covariance_matrix=forecast_covariance,
        vector_label="Forecast mean",
    )

    if set(sampled_residuals.columns) != set(
        ticker_order
    ):
        raise ValueError(
            "Sampled residuals and forecast covariance "
            "must use the same tickers."
        )

    if set(weights.index) != set(ticker_order):
        raise ValueError(
            "Portfolio weights and forecast covariance "
            "must use the same tickers."
        )

    try:
        aligned_residuals = (
            sampled_residuals
            .reindex(columns=ticker_order)
            .astype(float)
        )
        aligned_weights = (
            weights
            .reindex(ticker_order)
            .astype(float)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Sampled residuals and portfolio weights "
            "must contain numeric values."
        ) from error

    residual_values = aligned_residuals.to_numpy(
        dtype=float
    )
    weight_values = aligned_weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(residual_values).all():
        raise ValueError(
            "Sampled residuals must contain finite numbers."
        )

    if not np.isfinite(weight_values).all():
        raise ValueError(
            "Portfolio weights must contain finite numbers."
        )

    return_scenarios = (
        mean_values
        + residual_values
        @ forecast_cholesky_factor.T
    )

    portfolio_returns = (
        return_scenarios @ weight_values
    )
    portfolio_losses = -portfolio_returns

    if not np.isfinite(portfolio_losses).all():
        raise ValueError(
            "Filtered portfolio losses must contain "
            "finite numbers."
        )

    return portfolio_losses



def calculate_filtered_historical_var_es(
    residual_pool: pd.DataFrame,
    forecast_mean: pd.Series,
    forecast_covariance: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
    scenario_count: int,
    random_seed: int,
) -> EmpiricalRiskEstimate:
    """Calculate one filtered historical VaR and ES forecast."""

    sampled_residuals = resample_standardized_residuals(
        residual_pool=residual_pool,
        scenario_count=scenario_count,
        random_seed=random_seed,
    )

    portfolio_losses = (
        calculate_filtered_portfolio_losses(
            sampled_residuals=sampled_residuals,
            forecast_mean=forecast_mean,
            forecast_covariance=forecast_covariance,
            weights=weights,
        )
    )

    return calculate_empirical_var_es(
        losses=portfolio_losses,
        confidence_level=confidence_level,
    )


def calculate_rolling_filtered_historical_forecasts(
    asset_returns: pd.DataFrame,
    beginning_weights: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
    scenario_count: int,
    random_seed: int,
    decay_factor: float,
) -> pd.DataFrame:
    """Calculate rolling filtered-historical VaR and ES forecasts."""

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
            "Asset returns and beginning weights must use "
            "the same tickers."
        )

    covariance_sequence = calculate_ewma_covariance_sequence(
        asset_returns=asset_returns,
        forecast_schedule=forecast_schedule,
        decay_factor=decay_factor,
    )

    first_schedule_row = forecast_schedule.iloc[0]
    first_estimation_returns = asset_returns.loc[
        first_schedule_row["estimation_start_date"]:
        first_schedule_row["estimation_end_date"]
    ]
    residual_pool = (
        calculate_initial_standardized_residual_pool(
            estimation_returns=first_estimation_returns,
        )
    )

    records: list[dict[str, object]] = []
    forecast_keys: list[
        tuple[pd.Timestamp, float]
    ] = []
    final_position = len(forecast_schedule) - 1

    for position, (
        forecast_date,
        schedule_row,
    ) in enumerate(forecast_schedule.iterrows()):
        forecast_date = pd.Timestamp(forecast_date)
        start_date = pd.Timestamp(
            schedule_row["estimation_start_date"]
        )
        end_date = pd.Timestamp(
            schedule_row["estimation_end_date"]
        )

        if forecast_date not in asset_returns.index:
            raise ValueError(
                "Every forecast date must exist in the asset returns."
            )

        if forecast_date not in beginning_weights.index:
            raise ValueError(
                "Beginning weights are required for every forecast date."
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
            np.isfinite(realized_return)
            and np.isfinite(realized_loss)
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

        if not residual_pool.index.equals(
            estimation_returns.index
        ):
            raise ValueError(
                "Residual-pool dates must match the current "
                "estimation window."
            )

        forecast_mean = estimation_returns.mean()
        forecast_covariance = covariance_sequence[
            forecast_date
        ]
        forecast_weights = beginning_weights.loc[
            forecast_date
        ]

        sampled_residuals = resample_standardized_residuals(
            residual_pool=residual_pool,
            scenario_count=scenario_count,
            random_seed=random_seed,
        )
        simulated_losses = calculate_filtered_portfolio_losses(
            sampled_residuals=sampled_residuals,
            forecast_mean=forecast_mean,
            forecast_covariance=forecast_covariance,
            weights=forecast_weights,
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
                    "simulated_mean_return": simulated_mean_return,
                    "simulated_volatility": simulated_volatility,
                    "scenario_count": int(scenario_count),
                    "random_seed": int(random_seed),
                    "tail_scenario_count": (
                        estimate.tail_observation_count
                    ),
                    "value_at_risk": estimate.value_at_risk,
                    "expected_shortfall": estimate.expected_shortfall,
                    "var_exceedance": bool(
                        realized_loss > estimate.value_at_risk
                    ),
                }
            )

        if position < final_position:
            residual_pool = update_standardized_residual_pool(
                residual_pool=residual_pool,
                realized_return=asset_returns.loc[
                    forecast_date
                ],
                forecast_mean=forecast_mean,
                forecast_covariance=forecast_covariance,
                observation_date=forecast_date,
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
