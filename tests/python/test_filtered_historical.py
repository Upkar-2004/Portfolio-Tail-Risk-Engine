"""Tests for filtered historical simulation calculations."""

import numpy as np
import pandas as pd
import pytest

from tailrisk.backtesting import (
    create_forecast_schedule,
    validate_backtest_forecasts,
)
from tailrisk.covariance import (
    calculate_ewma_covariance_sequence,
)
from tailrisk.filtered_historical import (
    calculate_filtered_historical_var_es,
    calculate_filtered_portfolio_loss,
    calculate_filtered_portfolio_losses,
    calculate_filtered_return_scenario,
    calculate_initial_standardized_residual_pool,
    calculate_rolling_filtered_historical_forecasts,
    calculate_rescaled_innovation,
    calculate_standardized_residual,
    resample_standardized_residuals,
    update_standardized_residual_pool,
)
from tailrisk.simulation import calculate_empirical_var_es


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



def _create_standardized_residual_pool() -> pd.DataFrame:
    """Create a controlled pool of joint historical shocks."""

    return pd.DataFrame(
        {
            "A": [-2.0, 0.4, 1.2],
            "B": [-1.5, 0.8, -0.3],
        },
        index=pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
            ]
        ),
    )


def test_resampled_residuals_preserve_complete_rows() -> None:
    """Resampling keeps each historical asset vector together."""

    residual_pool = (
        _create_standardized_residual_pool()
    )

    result = resample_standardized_residuals(
        residual_pool=residual_pool,
        scenario_count=20,
        random_seed=20261002,
    )

    assert result.shape == (20, 2)
    assert list(result.columns) == ["A", "B"]

    expected_index = pd.RangeIndex(
        start=0,
        stop=20,
        name="scenario",
    )
    pd.testing.assert_index_equal(
        result.index,
        expected_index,
    )

    historical_rows = {
        tuple(row)
        for row in residual_pool.to_numpy(
            dtype=float
        )
    }

    for sampled_row in result.to_numpy(
        dtype=float
    ):
        assert tuple(sampled_row) in historical_rows


def test_resampled_residuals_are_reproducible() -> None:
    """The same random seed produces identical samples."""

    residual_pool = (
        _create_standardized_residual_pool()
    )

    first_result = resample_standardized_residuals(
        residual_pool=residual_pool,
        scenario_count=20,
        random_seed=20261002,
    )
    second_result = resample_standardized_residuals(
        residual_pool=residual_pool,
        scenario_count=20,
        random_seed=20261002,
    )

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )


@pytest.mark.parametrize(
    "invalid_scenario_count",
    [
        0,
        -1,
        True,
        2.5,
    ],
)
def test_resampled_residuals_reject_invalid_scenario_count(
    invalid_scenario_count: object,
) -> None:
    """Scenario count must be a positive non-Boolean integer."""

    with pytest.raises(
        ValueError,
        match="Scenario count",
    ):
        resample_standardized_residuals(
            residual_pool=(
                _create_standardized_residual_pool()
            ),
            scenario_count=invalid_scenario_count,
            random_seed=20261002,
        )


@pytest.mark.parametrize(
    "invalid_random_seed",
    [
        -1,
        True,
        2.5,
    ],
)
def test_resampled_residuals_reject_invalid_seed(
    invalid_random_seed: object,
) -> None:
    """Random seed must be a non-negative integer."""

    with pytest.raises(
        ValueError,
        match="Random seed",
    ):
        resample_standardized_residuals(
            residual_pool=(
                _create_standardized_residual_pool()
            ),
            scenario_count=20,
            random_seed=invalid_random_seed,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_resampled_residuals_reject_nonfinite_pool(
    invalid_value: float,
) -> None:
    """The historical residual pool must be finite."""

    residual_pool = (
        _create_standardized_residual_pool()
    )
    residual_pool.loc[
        residual_pool.index[0],
        "A",
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        resample_standardized_residuals(
            residual_pool=residual_pool,
            scenario_count=20,
            random_seed=20261002,
        )




def test_filtered_portfolio_losses_match_single_scenario_pipeline(
) -> None:
    """Vectorized FHS losses match the tested one-scenario pipeline."""

    tickers = ["A", "B"]

    sampled_residuals = pd.DataFrame(
        [
            [-2.0, 1.5],
            [0.4, 0.8],
            [1.2, -0.3],
        ],
        columns=tickers,
        index=pd.RangeIndex(
            start=0,
            stop=3,
            name="scenario",
        ),
    )

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

    reversed_weights = pd.Series(
        [0.40, 0.60],
        index=["B", "A"],
        name="weight",
    )

    result = calculate_filtered_portfolio_losses(
        sampled_residuals=sampled_residuals,
        forecast_mean=forecast_mean,
        forecast_covariance=forecast_covariance,
        weights=reversed_weights,
    )

    expected_from_single_scenarios = np.array(
        [
            calculate_filtered_portfolio_loss(
                return_scenario=(
                    calculate_filtered_return_scenario(
                        forecast_mean=forecast_mean,
                        standardized_residual=residual,
                        forecast_covariance=(
                            forecast_covariance
                        ),
                    )
                ),
                weights=reversed_weights,
            )
            for _, residual in (
                sampled_residuals.iterrows()
            )
        ]
    )

    manual_expected = np.array(
        [
            0.01598,
            -0.00722,
            -0.01218,
        ]
    )

    np.testing.assert_allclose(
        result,
        expected_from_single_scenarios,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result,
        manual_expected,
        rtol=1e-12,
        atol=1e-15,
    )


def test_filtered_portfolio_losses_reject_mismatched_tickers(
) -> None:
    """Sampled residuals must cover all forecast assets."""

    sampled_residuals = pd.DataFrame(
        [[-2.0, 1.5]],
        columns=["A", "C"],
    )
    forecast_mean = pd.Series(
        [0.0005, 0.0003],
        index=["A", "B"],
    )
    forecast_covariance = pd.DataFrame(
        [
            [0.000225, 0.000060],
            [0.000060, 0.000080],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    weights = pd.Series(
        [0.60, 0.40],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        calculate_filtered_portfolio_losses(
            sampled_residuals=sampled_residuals,
            forecast_mean=forecast_mean,
            forecast_covariance=forecast_covariance,
            weights=weights,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_filtered_portfolio_losses_reject_nonfinite_residuals(
    invalid_value: float,
) -> None:
    """Sampled standardized residuals must be finite."""

    sampled_residuals = pd.DataFrame(
        [[invalid_value, 1.5]],
        columns=["A", "B"],
    )
    forecast_mean = pd.Series(
        [0.0005, 0.0003],
        index=["A", "B"],
    )
    forecast_covariance = pd.DataFrame(
        [
            [0.000225, 0.000060],
            [0.000060, 0.000080],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    weights = pd.Series(
        [0.60, 0.40],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        calculate_filtered_portfolio_losses(
            sampled_residuals=sampled_residuals,
            forecast_mean=forecast_mean,
            forecast_covariance=forecast_covariance,
            weights=weights,
        )




def test_filtered_historical_var_es_matches_manual_pipeline(
) -> None:
    """One-date FHS matches the separately tested calculation steps."""

    tickers = ["A", "B"]

    residual_pool = (
        _create_standardized_residual_pool()
    )

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

    weights = pd.Series(
        [0.60, 0.40],
        index=tickers,
        name="weight",
    )

    scenario_count = 100
    random_seed = 20261002
    confidence_level = 0.80

    result = calculate_filtered_historical_var_es(
        residual_pool=residual_pool,
        forecast_mean=forecast_mean,
        forecast_covariance=forecast_covariance,
        weights=weights,
        confidence_level=confidence_level,
        scenario_count=scenario_count,
        random_seed=random_seed,
    )

    sampled_residuals = resample_standardized_residuals(
        residual_pool=residual_pool,
        scenario_count=scenario_count,
        random_seed=random_seed,
    )
    losses = calculate_filtered_portfolio_losses(
        sampled_residuals=sampled_residuals,
        forecast_mean=forecast_mean,
        forecast_covariance=forecast_covariance,
        weights=weights,
    )
    expected = calculate_empirical_var_es(
        losses=losses,
        confidence_level=confidence_level,
    )

    assert result == expected
    assert result.observation_count == scenario_count
    assert (
        result.expected_shortfall
        >= result.value_at_risk - 1e-12
    )



def _create_initial_estimation_returns() -> pd.DataFrame:
    """Create controlled returns for initial-pool tests."""

    return pd.DataFrame(
        {
            "A": [0.010, -0.020, 0.030, -0.010],
            "B": [0.005, 0.015, -0.010, -0.020],
        },
        index=pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
                "2025-01-07",
            ]
        ),
    )


def test_initial_residual_pool_reconstructs_original_returns(
) -> None:
    """Initial standardized residuals reproduce their returns."""

    estimation_returns = (
        _create_initial_estimation_returns()
    )

    result = (
        calculate_initial_standardized_residual_pool(
            estimation_returns=estimation_returns,
        )
    )

    initial_mean = estimation_returns.mean()
    initial_covariance = estimation_returns.cov(
        ddof=1
    )
    cholesky_factor = np.linalg.cholesky(
        initial_covariance.to_numpy(dtype=float)
    )

    reconstructed_returns = (
        initial_mean.to_numpy(dtype=float)
        + result.to_numpy(dtype=float)
        @ cholesky_factor.T
    )

    np.testing.assert_allclose(
        reconstructed_returns,
        estimation_returns.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-15,
    )

    pd.testing.assert_index_equal(
        result.index,
        estimation_returns.index,
    )
    pd.testing.assert_index_equal(
        result.columns,
        estimation_returns.columns,
    )


def test_initial_residual_pool_is_centered_and_whitened(
) -> None:
    """Initial residuals have zero mean and identity covariance."""

    result = (
        calculate_initial_standardized_residual_pool(
            estimation_returns=(
                _create_initial_estimation_returns()
            ),
        )
    )

    np.testing.assert_allclose(
        result.mean().to_numpy(dtype=float),
        np.zeros(2),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.cov(ddof=1).to_numpy(dtype=float),
        np.eye(2),
        rtol=1e-12,
        atol=1e-15,
    )


def test_initial_residual_pool_requires_datetime_index(
) -> None:
    """Initial estimation returns must use session dates."""

    estimation_returns = (
        _create_initial_estimation_returns()
        .reset_index(drop=True)
    )

    with pytest.raises(
        TypeError,
        match="DatetimeIndex",
    ):
        calculate_initial_standardized_residual_pool(
            estimation_returns=estimation_returns,
        )


def test_initial_residual_pool_rejects_duplicate_dates(
) -> None:
    """Initial estimation-return dates must be unique."""

    estimation_returns = (
        _create_initial_estimation_returns()
    )
    estimation_returns.index = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-03",
            "2025-01-07",
        ]
    )

    with pytest.raises(
        ValueError,
        match="dates must be unique",
    ):
        calculate_initial_standardized_residual_pool(
            estimation_returns=estimation_returns,
        )


def test_initial_residual_pool_rejects_unordered_dates(
) -> None:
    """Initial estimation returns must be chronological."""

    estimation_returns = (
        _create_initial_estimation_returns()
        .iloc[[0, 2, 1, 3]]
    )

    with pytest.raises(
        ValueError,
        match="chronological",
    ):
        calculate_initial_standardized_residual_pool(
            estimation_returns=estimation_returns,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_initial_residual_pool_rejects_nonfinite_returns(
    invalid_value: float,
) -> None:
    """Initial estimation returns must be finite."""

    estimation_returns = (
        _create_initial_estimation_returns()
    )
    estimation_returns.iloc[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        calculate_initial_standardized_residual_pool(
            estimation_returns=estimation_returns,
        )


def test_initial_residual_pool_rejects_singular_covariance(
) -> None:
    """Initial whitening requires positive-definite covariance."""

    estimation_returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.02, 0.04, 0.06, 0.08],
        },
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="positive definite",
    ):
        calculate_initial_standardized_residual_pool(
            estimation_returns=estimation_returns,
        )


def test_residual_pool_update_drops_oldest_and_appends_new(
) -> None:
    """Pool updates retain size and append the latest residual."""

    residual_pool = _create_standardized_residual_pool()
    original_pool = residual_pool.copy(deep=True)
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
    expected_new_residual = pd.Series(
        [-2.0, 1.5],
        index=tickers,
    )
    realized_values = (
        forecast_mean.to_numpy(dtype=float)
        + forecast_cholesky_factor
        @ expected_new_residual.to_numpy(dtype=float)
    )
    reversed_realized_return = pd.Series(
        [
            realized_values[1],
            realized_values[0],
        ],
        index=["B", "A"],
        name="realized_return",
    )

    result = update_standardized_residual_pool(
        residual_pool=residual_pool,
        realized_return=reversed_realized_return,
        forecast_mean=forecast_mean,
        forecast_covariance=forecast_covariance,
        observation_date=pd.Timestamp("2025-01-07"),
    )

    assert len(result) == len(residual_pool)
    pd.testing.assert_frame_equal(
        result.iloc[:-1],
        residual_pool.iloc[1:].astype(float),
    )
    pd.testing.assert_series_equal(
        result.iloc[-1],
        expected_new_residual,
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )
    assert result.index[-1] == pd.Timestamp(
        "2025-01-07"
    )
    pd.testing.assert_frame_equal(
        residual_pool,
        original_pool,
    )


@pytest.mark.parametrize(
    "invalid_observation_date",
    [
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-01-03"),
    ],
)
def test_residual_pool_update_requires_later_date(
    invalid_observation_date: pd.Timestamp,
) -> None:
    """New residual dates must follow the existing pool."""

    residual_pool = _create_standardized_residual_pool()
    forecast_mean = pd.Series(
        [0.0005, 0.0003],
        index=["A", "B"],
    )
    forecast_covariance = pd.DataFrame(
        [
            [0.000225, 0.000060],
            [0.000060, 0.000080],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    realized_return = pd.Series(
        [-0.0295, 0.0043],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="later than",
    ):
        update_standardized_residual_pool(
            residual_pool=residual_pool,
            realized_return=realized_return,
            forecast_mean=forecast_mean,
            forecast_covariance=forecast_covariance,
            observation_date=invalid_observation_date,
        )


def test_residual_pool_update_rejects_mismatched_tickers(
) -> None:
    """Pool, realized returns, and forecasts must share assets."""

    residual_pool = _create_standardized_residual_pool()
    forecast_mean = pd.Series(
        [0.0005, 0.0003],
        index=["A", "B"],
    )
    forecast_covariance = pd.DataFrame(
        [
            [0.000225, 0.000060],
            [0.000060, 0.000080],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    realized_return = pd.Series(
        [-0.0295, 0.0043],
        index=["A", "C"],
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        update_standardized_residual_pool(
            residual_pool=residual_pool,
            realized_return=realized_return,
            forecast_mean=forecast_mean,
            forecast_covariance=forecast_covariance,
            observation_date=pd.Timestamp("2025-01-07"),
        )


def _create_rolling_fhs_inputs(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a controlled three-date FHS forecast problem."""

    dates = pd.date_range(
        "2025-01-02",
        periods=7,
        freq="B",
    )
    asset_returns = pd.DataFrame(
        {
            "A": [
                0.010,
                -0.020,
                0.030,
                -0.010,
                0.025,
                -0.015,
                0.005,
            ],
            "B": [
                0.005,
                0.015,
                -0.010,
                -0.020,
                0.010,
                0.025,
                -0.005,
            ],
        },
        index=dates,
    )
    beginning_weights = pd.DataFrame(
        {
            "A": [0.60] * len(dates),
            "B": [0.40] * len(dates),
        },
        index=dates,
    )
    portfolio_returns = (
        asset_returns * beginning_weights
    ).sum(axis=1)
    forecast_schedule = create_forecast_schedule(
        portfolio_returns=portfolio_returns,
        estimation_window=4,
    )

    return (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    )


def test_rolling_fhs_forecasts_match_first_date_pipeline(
) -> None:
    """Rolling FHS begins with the initial calibration pool."""

    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_fhs_inputs()
    confidence_levels = [0.80, 0.95]
    scenario_count = 500
    random_seed = 20261002
    decay_factor = 0.94

    result = calculate_rolling_filtered_historical_forecasts(
        asset_returns=asset_returns,
        beginning_weights=beginning_weights,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        scenario_count=scenario_count,
        random_seed=random_seed,
        decay_factor=decay_factor,
    )

    validate_backtest_forecasts(
        forecasts=result,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
    )
    expected_index = pd.MultiIndex.from_product(
        [
            forecast_schedule.index,
            confidence_levels,
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )
    pd.testing.assert_index_equal(
        result.index,
        expected_index,
    )

    first_forecast_date = forecast_schedule.index[0]
    first_schedule_row = forecast_schedule.iloc[0]
    first_estimation_returns = asset_returns.loc[
        first_schedule_row["estimation_start_date"]:
        first_schedule_row["estimation_end_date"]
    ]
    initial_pool = (
        calculate_initial_standardized_residual_pool(
            estimation_returns=first_estimation_returns,
        )
    )
    covariance_sequence = calculate_ewma_covariance_sequence(
        asset_returns=asset_returns,
        forecast_schedule=forecast_schedule,
        decay_factor=decay_factor,
    )
    sampled_residuals = resample_standardized_residuals(
        residual_pool=initial_pool,
        scenario_count=scenario_count,
        random_seed=random_seed,
    )
    expected_losses = calculate_filtered_portfolio_losses(
        sampled_residuals=sampled_residuals,
        forecast_mean=first_estimation_returns.mean(),
        forecast_covariance=covariance_sequence[
            first_forecast_date
        ],
        weights=beginning_weights.loc[
            first_forecast_date
        ],
    )
    expected_estimate = calculate_empirical_var_es(
        losses=expected_losses,
        confidence_level=0.80,
    )
    first_result = result.loc[
        (first_forecast_date, 0.80)
    ]

    assert first_result["value_at_risk"] == pytest.approx(
        expected_estimate.value_at_risk
    )
    assert first_result[
        "expected_shortfall"
    ] == pytest.approx(
        expected_estimate.expected_shortfall
    )
    assert first_result["scenario_count"] == scenario_count
    assert first_result["random_seed"] == random_seed


def test_rolling_fhs_does_not_use_return_in_its_own_forecast(
) -> None:
    """Changing R_t affects only forecasts strictly after t."""

    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_fhs_inputs()
    confidence_levels = [0.80, 0.95]
    common_arguments = {
        "beginning_weights": beginning_weights,
        "confidence_levels": confidence_levels,
        "scenario_count": 2_000,
        "random_seed": 20261002,
        "decay_factor": 0.94,
    }

    baseline = calculate_rolling_filtered_historical_forecasts(
        asset_returns=asset_returns,
        forecast_schedule=forecast_schedule,
        **common_arguments,
    )

    changed_returns = asset_returns.copy()
    changed_date = forecast_schedule.index[1]
    changed_returns.loc[changed_date, "A"] += 0.12
    changed_returns.loc[changed_date, "B"] -= 0.08
    changed_portfolio_returns = (
        changed_returns * beginning_weights
    ).sum(axis=1)
    changed_schedule = create_forecast_schedule(
        portfolio_returns=changed_portfolio_returns,
        estimation_window=4,
    )
    changed = calculate_rolling_filtered_historical_forecasts(
        asset_returns=changed_returns,
        forecast_schedule=changed_schedule,
        **common_arguments,
    )

    risk_columns = [
        "value_at_risk",
        "expected_shortfall",
    ]
    unchanged_dates = forecast_schedule.index[:2]
    pd.testing.assert_frame_equal(
        baseline.loc[
            (unchanged_dates, slice(None)),
            risk_columns,
        ],
        changed.loc[
            (unchanged_dates, slice(None)),
            risk_columns,
        ],
    )

    final_date = forecast_schedule.index[-1]
    baseline_final = baseline.loc[
        (final_date, slice(None)),
        risk_columns,
    ].to_numpy(dtype=float)
    changed_final = changed.loc[
        (final_date, slice(None)),
        risk_columns,
    ].to_numpy(dtype=float)

    assert not np.allclose(
        baseline_final,
        changed_final,
        rtol=1e-12,
        atol=1e-15,
    )


def test_rolling_fhs_forecasts_are_reproducible() -> None:
    """Identical inputs and seeds produce identical FHS forecasts."""

    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_fhs_inputs()
    arguments = {
        "asset_returns": asset_returns,
        "beginning_weights": beginning_weights,
        "forecast_schedule": forecast_schedule,
        "confidence_levels": [0.80, 0.95],
        "scenario_count": 500,
        "random_seed": 20261002,
        "decay_factor": 0.94,
    }

    first_result = (
        calculate_rolling_filtered_historical_forecasts(
            **arguments
        )
    )
    second_result = (
        calculate_rolling_filtered_historical_forecasts(
            **arguments
        )
    )

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )


def test_rolling_fhs_requires_forecast_weights() -> None:
    """Every FHS forecast date requires beginning weights."""

    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_fhs_inputs()
    missing_date = forecast_schedule.index[0]
    incomplete_weights = beginning_weights.drop(
        index=missing_date
    )

    with pytest.raises(
        ValueError,
        match="Beginning weights are required",
    ):
        calculate_rolling_filtered_historical_forecasts(
            asset_returns=asset_returns,
            beginning_weights=incomplete_weights,
            forecast_schedule=forecast_schedule,
            confidence_levels=[0.80],
            scenario_count=100,
            random_seed=20261002,
            decay_factor=0.94,
        )


def test_rolling_fhs_rejects_inconsistent_realized_loss(
) -> None:
    """A scheduled loss must equal negative scheduled return."""

    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_fhs_inputs()
    invalid_schedule = forecast_schedule.copy()
    invalid_schedule.iloc[
        0,
        invalid_schedule.columns.get_loc("realized_loss"),
    ] = 0.50

    with pytest.raises(
        ValueError,
        match="negative of its realized return",
    ):
        calculate_rolling_filtered_historical_forecasts(
            asset_returns=asset_returns,
            beginning_weights=beginning_weights,
            forecast_schedule=invalid_schedule,
            confidence_levels=[0.80],
            scenario_count=100,
            random_seed=20261002,
            decay_factor=0.94,
        )


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"confidence_levels": []},
            "At least one confidence level",
        ),
        (
            {"confidence_levels": [1.0]},
            "Confidence level",
        ),
        (
            {"scenario_count": 0},
            "Scenario count",
        ),
        (
            {"random_seed": -1},
            "Random seed",
        ),
        (
            {"decay_factor": 1.0},
            "Decay factor",
        ),
    ],
)
def test_rolling_fhs_rejects_invalid_model_settings(
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    """Rolling FHS rejects invalid model and simulation settings."""

    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_fhs_inputs()
    arguments: dict[str, object] = {
        "asset_returns": asset_returns,
        "beginning_weights": beginning_weights,
        "forecast_schedule": forecast_schedule,
        "confidence_levels": [0.80],
        "scenario_count": 100,
        "random_seed": 20261002,
        "decay_factor": 0.94,
    }
    arguments.update(overrides)

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        calculate_rolling_filtered_historical_forecasts(
            **arguments
        )

    
