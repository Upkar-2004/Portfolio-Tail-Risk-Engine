"""Tests for filtered historical simulation calculations."""

import numpy as np
import pandas as pd
import pytest

from tailrisk.filtered_historical import (
    calculate_filtered_portfolio_loss,
    calculate_filtered_return_scenario,
    calculate_rescaled_innovation,
    calculate_standardized_residual,
)


def test_standardized_residual_matches_known_shock() -> None:
    """Whitening recovers a known standardized residual vector."""

    tickers = ["A", "B"]

    cholesky_factor = np.array(
        [
            [0.020, 0.000],
            [0.006, 0.010],
        ]
    )
    covariance_matrix = pd.DataFrame(
        cholesky_factor @ cholesky_factor.T,
        index=tickers,
        columns=tickers,
    )

    expected_residual = pd.Series(
        [-2.0, 1.5],
        index=tickers,
        name="standardized_residual",
    )
    innovation = pd.Series(
        cholesky_factor
        @ expected_residual.to_numpy(),
        index=tickers,
        name="innovation",
    )

    result = calculate_standardized_residual(
        innovation=innovation,
        covariance_matrix=covariance_matrix,
    )

    pd.testing.assert_series_equal(
        result,
        expected_residual,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )

    reconstructed_innovation = (
        cholesky_factor @ result.to_numpy()
    )

    np.testing.assert_allclose(
        reconstructed_innovation,
        innovation.to_numpy(),
        rtol=1e-12,
        atol=1e-15,
    )


def test_rescaled_innovation_uses_forecast_covariance() -> None:
    """Rescaling applies a new covariance to a historical shock."""

    tickers = ["A", "B"]
    forecast_cholesky_factor = np.array(
        [
            [0.015, 0.000],
            [0.004, 0.008],
        ]
    )
    forecast_covariance = pd.DataFrame(
        (
            forecast_cholesky_factor
            @ forecast_cholesky_factor.T
        ),
        index=tickers,
        columns=tickers,
    )
    standardized_residual = pd.Series(
        [-2.0, 1.5],
        index=tickers,
        name="standardized_residual",
    )
    expected = pd.Series(
        [-0.030, 0.004],
        index=tickers,
        name="rescaled_innovation",
    )

    result = calculate_rescaled_innovation(
        standardized_residual=standardized_residual,
        covariance_matrix=forecast_covariance,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )


def test_filtered_return_scenario_adds_forecast_mean() -> None:
    """A filtered scenario combines current mean and rescaled shock."""

    tickers = ["A", "B"]

    forecast_mean = pd.Series(
        [0.0005, 0.0003],
        index=tickers,
        name="forecast_mean",
    )
    forecast_cholesky_factor = np.array(
        [
            [0.015, 0.000],
            [0.004, 0.008],
        ]
    )
    forecast_covariance = pd.DataFrame(
        (
            forecast_cholesky_factor
            @ forecast_cholesky_factor.T
        ),
        index=tickers,
        columns=tickers,
    )
    standardized_residual = pd.Series(
        [-2.0, 1.5],
        index=tickers,
        name="standardized_residual",
    )

    expected = pd.Series(
        [-0.0295, 0.0043],
        index=tickers,
        name="filtered_return_scenario",
    )

    result = calculate_filtered_return_scenario(
        forecast_mean=forecast_mean,
        standardized_residual=standardized_residual,
        forecast_covariance=forecast_covariance,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )


def test_filtered_portfolio_loss_matches_manual_example() -> None:
    """Portfolio loss applies aligned weights and the loss sign."""

    return_scenario = pd.Series(
        [-0.0295, 0.0043],
        index=["A", "B"],
        name="filtered_return_scenario",
    )
    reversed_weights = pd.Series(
        [0.40, 0.60],
        index=["B", "A"],
        name="weight",
    )

    result = calculate_filtered_portfolio_loss(
        return_scenario=return_scenario,
        weights=reversed_weights,
    )

    assert result == pytest.approx(0.01598)


def test_standardized_residual_rejects_mismatched_tickers(
) -> None:
    """Innovation and covariance tickers must agree."""

    innovation = pd.Series(
        [0.01, -0.02],
        index=["A", "C"],
    )
    covariance = pd.DataFrame(
        [
            [0.0004, 0.0001],
            [0.0001, 0.0003],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        calculate_standardized_residual(
            innovation=innovation,
            covariance_matrix=covariance,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_standardized_residual_rejects_nonfinite_innovation(
    invalid_value: float,
) -> None:
    """Innovations must contain only finite numbers."""

    innovation = pd.Series(
        [invalid_value, -0.02],
        index=["A", "B"],
    )
    covariance = pd.DataFrame(
        [
            [0.0004, 0.0001],
            [0.0001, 0.0003],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Innovation must contain finite numbers",
    ):
        calculate_standardized_residual(
            innovation=innovation,
            covariance_matrix=covariance,
        )


def test_standardized_residual_rejects_singular_covariance(
) -> None:
    """Whitening requires a positive-definite covariance matrix."""

    innovation = pd.Series(
        [0.01, -0.02],
        index=["A", "B"],
    )
    singular_covariance = pd.DataFrame(
        [
            [0.0004, 0.0004],
            [0.0004, 0.0004],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="positive definite",
    ):
        calculate_standardized_residual(
            innovation=innovation,
            covariance_matrix=singular_covariance,
        )


def test_filtered_scenario_rejects_mismatched_mean_tickers(
) -> None:
    """Forecast means must cover the covariance tickers."""

    forecast_mean = pd.Series(
        [0.001, 0.002],
        index=["A", "C"],
    )
    standardized_residual = pd.Series(
        [-1.0, 0.5],
        index=["A", "B"],
    )
    covariance = pd.DataFrame(
        [
            [0.0004, 0.0001],
            [0.0001, 0.0003],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        calculate_filtered_return_scenario(
            forecast_mean=forecast_mean,
            standardized_residual=(
                standardized_residual
            ),
            forecast_covariance=covariance,
        )


def test_filtered_portfolio_loss_rejects_mismatched_tickers(
) -> None:
    """Return scenarios and weights must use the same assets."""

    return_scenario = pd.Series(
        [-0.02, 0.01],
        index=["A", "B"],
    )
    weights = pd.Series(
        [0.60, 0.40],
        index=["A", "C"],
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        calculate_filtered_portfolio_loss(
            return_scenario=return_scenario,
            weights=weights,
        )

    
