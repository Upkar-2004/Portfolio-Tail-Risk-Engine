"""Filtered historical simulation calculations."""

import numpy as np
import pandas as pd


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
