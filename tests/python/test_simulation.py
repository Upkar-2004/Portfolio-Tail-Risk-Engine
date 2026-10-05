"""Tests for the Python Gaussian simulation reference."""

import numpy as np
import pandas as pd
import pytest

import tailrisk.simulation as simulation_module
from tailrisk.backtesting import (
    create_forecast_schedule,
    validate_backtest_forecasts,
)
from tailrisk.covariance import (
    calculate_sample_portfolio_moments,
)
from tailrisk.models import (
    calculate_gaussian_var_es,
)

from tailrisk.simulation import (
    calculate_empirical_var_es,
    calculate_gaussian_monte_carlo_var_es,
    calculate_rolling_gaussian_monte_carlo_forecasts,
    generate_gaussian_portfolio_losses,
    generate_gaussian_portfolio_losses_from_shocks,
    generate_gaussian_return_scenarios,
    generate_standard_normal_shocks,
)


def test_standard_normal_shocks_have_requested_shape() -> None:
    shocks = generate_standard_normal_shocks(
        scenario_count=100,
        asset_count=3,
        random_seed=20261002,
    )

    assert shocks.shape == (100, 3)
    assert np.isfinite(shocks).all()


def test_standard_normal_shocks_are_reproducible() -> None:
    first_result = generate_standard_normal_shocks(
        scenario_count=100,
        asset_count=3,
        random_seed=20261002,
    )
    second_result = generate_standard_normal_shocks(
        scenario_count=100,
        asset_count=3,
        random_seed=20261002,
    )

    np.testing.assert_array_equal(
        first_result,
        second_result,
    )


def test_standard_normal_shocks_change_with_seed() -> None:
    first_result = generate_standard_normal_shocks(
        scenario_count=100,
        asset_count=3,
        random_seed=20261002,
    )
    second_result = generate_standard_normal_shocks(
        scenario_count=100,
        asset_count=3,
        random_seed=20261003,
    )

    assert not np.array_equal(
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
def test_standard_normal_shocks_reject_invalid_scenario_count(
    invalid_scenario_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Scenario count",
    ):
        generate_standard_normal_shocks(
            scenario_count=invalid_scenario_count,
            asset_count=3,
            random_seed=20261002,
        )


@pytest.mark.parametrize(
    "invalid_asset_count",
    [
        0,
        -1,
        True,
        2.5,
    ],
)
def test_standard_normal_shocks_reject_invalid_asset_count(
    invalid_asset_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Asset count",
    ):
        generate_standard_normal_shocks(
            scenario_count=100,
            asset_count=invalid_asset_count,
            random_seed=20261002,
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        -1,
        True,
        1.5,
    ],
)
def test_standard_normal_shocks_reject_invalid_seed(
    invalid_seed: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Random seed",
    ):
        generate_standard_normal_shocks(
            scenario_count=100,
            asset_count=3,
            random_seed=invalid_seed,
        )


def test_gaussian_scenarios_are_reproducible() -> None:
    """The same random seed should produce identical scenarios."""

    mean_vector = pd.Series(
        [0.001, -0.0005],
        index=["A", "B"],
    )
    covariance_matrix = pd.DataFrame(
        [
            [0.000400, 0.000120],
            [0.000120, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    first_result = generate_gaussian_return_scenarios(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        scenario_count=100,
        random_seed=20261002,
    )
    second_result = generate_gaussian_return_scenarios(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        scenario_count=100,
        random_seed=20261002,
    )

    np.testing.assert_array_equal(
        first_result,
        second_result,
    )


def test_gaussian_scenarios_match_target_moments() -> None:
    """Simulated means and covariances should approach their targets."""

    mean_vector = pd.Series(
        [0.001, -0.0005],
        index=["A", "B"],
    )
    covariance_matrix = pd.DataFrame(
        [
            [0.000400, 0.000120],
            [0.000120, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    scenarios = generate_gaussian_return_scenarios(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        scenario_count=200_000,
        random_seed=20261002,
    )

    assert scenarios.shape == (
        200_000,
        2,
    )

    simulated_mean = scenarios.mean(axis=0)
    simulated_covariance = np.cov(
        scenarios,
        rowvar=False,
        ddof=1,
    )

    np.testing.assert_allclose(
        simulated_mean,
        mean_vector.to_numpy(),
        rtol=0.0,
        atol=0.00015,
    )
    np.testing.assert_allclose(
        simulated_covariance,
        covariance_matrix.to_numpy(),
        rtol=0.0,
        atol=0.000005,
    )



def test_monte_carlo_var_es_agree_with_analytic_gaussian() -> None:
    """Monte Carlo risk estimates should approach analytic values."""

    mean_vector = pd.Series(
        [0.001, -0.0005],
        index=["A", "B"],
    )
    covariance_matrix = pd.DataFrame(
        [
            [0.000400, 0.000120],
            [0.000120, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    # Deliberately reverse the ticker order. The function must align
    # weights by ticker rather than relying on their position.
    weights = pd.Series(
        [0.40, 0.60],
        index=["B", "A"],
    )

    result = calculate_gaussian_monte_carlo_var_es(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        weights=weights,
        confidence_level=0.95,
        scenario_count=250_000,
        random_seed=20261002,
    )

    aligned_weights = weights.reindex(
        mean_vector.index
    )

    portfolio_mean = float(
        aligned_weights @ mean_vector
    )
    portfolio_variance = float(
        aligned_weights
        @ covariance_matrix
        @ aligned_weights
    )
    portfolio_volatility = float(
        np.sqrt(portfolio_variance)
    )

    analytic_result = calculate_gaussian_var_es(
        mean_return=portfolio_mean,
        volatility=portfolio_volatility,
        confidence_level=0.95,
    )

    assert result.confidence_level == pytest.approx(
        0.95
    )
    assert result.scenario_count == 250_000
    assert result.random_seed == 20261002

    assert result.simulated_mean_return == pytest.approx(
        portfolio_mean,
        abs=0.0001,
    )
    assert result.simulated_volatility == pytest.approx(
        portfolio_volatility,
        abs=0.0001,
    )
    assert result.value_at_risk == pytest.approx(
        analytic_result.value_at_risk,
        abs=0.0003,
    )
    assert result.expected_shortfall == pytest.approx(
        analytic_result.expected_shortfall,
        abs=0.0003,
    )

    assert (
        result.expected_shortfall
        > result.value_at_risk
    )




def _create_monte_carlo_inputs(
) -> tuple[
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    mean_vector = pd.Series(
        [0.001, -0.0005],
        index=["A", "B"],
    )
    covariance_matrix = pd.DataFrame(
        [
            [0.000400, 0.000120],
            [0.000120, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    weights = pd.Series(
        [0.60, 0.40],
        index=["A", "B"],
    )

    return (
        mean_vector,
        covariance_matrix,
        weights,
    )


def test_monte_carlo_var_es_rejects_mismatched_weight_tickers(
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    weights = weights.rename(
        index={"B": "C"}
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        calculate_gaussian_monte_carlo_var_es(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            confidence_level=0.95,
            scenario_count=100,
            random_seed=20261002,
        )


@pytest.mark.parametrize(
    "invalid_weight",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_monte_carlo_var_es_rejects_nonfinite_weights(
    invalid_weight: float,
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    weights.loc["A"] = invalid_weight

    with pytest.raises(
        ValueError,
        match="finite numbers",
    ):
        calculate_gaussian_monte_carlo_var_es(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            confidence_level=0.95,
            scenario_count=100,
            random_seed=20261002,
        )


@pytest.mark.parametrize(
    "invalid_confidence_level",
    [
        0.0,
        1.0,
        np.nan,
        np.inf,
        True,
    ],
)
def test_monte_carlo_var_es_rejects_invalid_confidence(
    invalid_confidence_level: object,
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    with pytest.raises(
        ValueError,
        match="Confidence level",
    ):
        calculate_gaussian_monte_carlo_var_es(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            confidence_level=invalid_confidence_level,
            scenario_count=100,
            random_seed=20261002,
        )


@pytest.mark.parametrize(
    "invalid_scenario_count",
    [
        1,
        0,
        -1,
        True,
        2.5,
    ],
)
def test_monte_carlo_var_es_rejects_invalid_scenario_count(
    invalid_scenario_count: object,
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        calculate_gaussian_monte_carlo_var_es(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            confidence_level=0.95,
            scenario_count=invalid_scenario_count,
            random_seed=20261002,
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        -1,
        True,
        1.5,
    ],
)
def test_monte_carlo_var_es_rejects_invalid_seed(
    invalid_seed: object,
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    with pytest.raises(
        ValueError,
        match="non-negative integer",
    ):
        calculate_gaussian_monte_carlo_var_es(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            confidence_level=0.95,
            scenario_count=100,
            random_seed=invalid_seed,
        )


def test_gaussian_scenarios_reject_non_psd_covariance() -> None:
    (
        mean_vector,
        covariance_matrix,
        _,
    ) = _create_monte_carlo_inputs()

    covariance_matrix.loc["A", "B"] = 0.0005
    covariance_matrix.loc["B", "A"] = 0.0005

    with pytest.raises(
        ValueError,
        match="positive semidefinite",
    ):
        generate_gaussian_return_scenarios(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            scenario_count=100,
            random_seed=20261002,
        )


def test_gaussian_scenarios_require_positive_definite_covariance(
) -> None:
    mean_vector = pd.Series(
        [0.001, 0.0],
        index=["A", "B"],
    )
    singular_covariance = pd.DataFrame(
        [
            [0.0004, 0.0],
            [0.0, 0.0],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="positive definite",
    ):
        generate_gaussian_return_scenarios(
            mean_vector=mean_vector,
            covariance_matrix=singular_covariance,
            scenario_count=100,
            random_seed=20261002,
        )


def test_generate_gaussian_portfolio_losses_applies_aligned_weights(
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    # Reverse the order to prove that alignment uses ticker names.
    reversed_weights = weights.iloc[::-1]

    losses = generate_gaussian_portfolio_losses(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        weights=reversed_weights,
        scenario_count=1_000,
        random_seed=20261002,
    )

    asset_scenarios = generate_gaussian_return_scenarios(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        scenario_count=1_000,
        random_seed=20261002,
    )
    aligned_weights = reversed_weights.reindex(
        mean_vector.index
    )
    expected_losses = -(
        asset_scenarios
        @ aligned_weights.to_numpy()
    )

    assert losses.shape == (1_000,)

    np.testing.assert_array_equal(
        losses,
        expected_losses,
    )


def test_portfolio_losses_from_shocks_match_full_asset_scenarios(
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    # Reverse the order to prove that ticker labels, rather than
    # positional order, control the weight alignment.
    reversed_weights = weights.iloc[::-1]

    shocks = generate_standard_normal_shocks(
        scenario_count=1_000,
        asset_count=len(mean_vector),
        random_seed=20261002,
    )

    projected_losses = (
        generate_gaussian_portfolio_losses_from_shocks(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=reversed_weights,
            standard_normal_shocks=shocks,
        )
    )

    cholesky_factor = np.linalg.cholesky(
        covariance_matrix.to_numpy(dtype=float)
    )
    asset_scenarios = (
        mean_vector.to_numpy(dtype=float)
        + shocks @ cholesky_factor.T
    )
    aligned_weights = reversed_weights.reindex(
        mean_vector.index
    )
    expected_losses = -(
        asset_scenarios
        @ aligned_weights.to_numpy(dtype=float)
    )

    assert projected_losses.shape == (1_000,)

    np.testing.assert_allclose(
        projected_losses,
        expected_losses,
        rtol=1e-14,
        atol=1e-15,
    )


def test_portfolio_losses_from_shocks_reject_wrong_asset_count(
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    shocks = np.zeros((100, 3))

    with pytest.raises(
        ValueError,
        match="Shock columns",
    ):
        generate_gaussian_portfolio_losses_from_shocks(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            standard_normal_shocks=shocks,
        )


@pytest.mark.parametrize(
    "invalid_shocks",
    [
        np.array([0.0, 1.0]),
        np.empty((0, 2)),
    ],
)
def test_portfolio_losses_from_shocks_reject_invalid_shape(
    invalid_shocks: np.ndarray,
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    with pytest.raises(
        ValueError,
        match="nonempty two-dimensional",
    ):
        generate_gaussian_portfolio_losses_from_shocks(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            standard_normal_shocks=invalid_shocks,
        )


@pytest.mark.parametrize(
    "nonfinite_shock",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_portfolio_losses_from_shocks_reject_nonfinite_shocks(
    nonfinite_shock: float,
) -> None:
    (
        mean_vector,
        covariance_matrix,
        weights,
    ) = _create_monte_carlo_inputs()

    shocks = np.zeros((100, 2))
    shocks[0, 0] = nonfinite_shock

    with pytest.raises(
        ValueError,
        match="finite numbers",
    ):
        generate_gaussian_portfolio_losses_from_shocks(
            mean_vector=mean_vector,
            covariance_matrix=covariance_matrix,
            weights=weights,
            standard_normal_shocks=shocks,
        )


def test_calculate_empirical_var_es_matches_manual_losses() -> None:
    """Verify empirical VaR and ES against a known loss sample."""

    losses = np.array(
        [
            -0.02,
            -0.01,
            0.00,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
        ]
    )

    result = calculate_empirical_var_es(
        losses=losses,
        confidence_level=0.80,
    )

    assert result.confidence_level == pytest.approx(
        0.80
    )
    assert result.observation_count == 10
    assert result.tail_observation_count == 2
    assert result.value_at_risk == pytest.approx(
        0.052
    )
    assert result.expected_shortfall == pytest.approx(
        0.065
    )




def test_calculate_empirical_var_es_requires_numpy_array() -> None:
    with pytest.raises(
        TypeError,
        match="NumPy array",
    ):
        calculate_empirical_var_es(
            losses=[0.01, 0.02, 0.03],
            confidence_level=0.95,
        )


def test_calculate_empirical_var_es_requires_one_dimension() -> None:
    losses = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
        ]
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        calculate_empirical_var_es(
            losses=losses,
            confidence_level=0.95,
        )


def test_calculate_empirical_var_es_rejects_empty_losses() -> None:
    losses = np.array([])

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        calculate_empirical_var_es(
            losses=losses,
            confidence_level=0.95,
        )


@pytest.mark.parametrize(
    "nonfinite_loss",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_calculate_empirical_var_es_rejects_nonfinite_losses(
    nonfinite_loss: float,
) -> None:
    losses = np.array(
        [
            0.01,
            nonfinite_loss,
            0.03,
        ]
    )

    with pytest.raises(
        ValueError,
        match="finite numbers",
    ):
        calculate_empirical_var_es(
            losses=losses,
            confidence_level=0.95,
        )


def test_calculate_empirical_var_es_rejects_nonnumeric_losses(
) -> None:
    losses = np.array(
        [
            "small",
            "large",
        ]
    )

    with pytest.raises(
        ValueError,
        match="numeric values",
    ):
        calculate_empirical_var_es(
            losses=losses,
            confidence_level=0.95,
        )


@pytest.mark.parametrize(
    "invalid_confidence_level",
    [
        0.0,
        1.0,
        np.nan,
        np.inf,
        True,
    ],
)
def test_calculate_empirical_var_es_rejects_invalid_confidence(
    invalid_confidence_level: object,
) -> None:
    losses = np.array(
        [
            0.01,
            0.02,
            0.03,
        ]
    )

    with pytest.raises(
        ValueError,
        match="Confidence level",
    ):
        calculate_empirical_var_es(
            losses=losses,
            confidence_level=invalid_confidence_level,
        )


def _create_rolling_monte_carlo_inputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )
    asset_returns = pd.DataFrame(
        {
            "A": [0.02, 0.00, -0.01, 0.03, 0.01],
            "B": [0.01, -0.01, 0.00, 0.02, -0.02],
        },
        index=dates,
    )
    beginning_weights = pd.DataFrame(
        {
            "A": [0.50, 0.50, 0.50, 0.60, 0.25],
            "B": [0.50, 0.50, 0.50, 0.40, 0.75],
        },
        index=dates,
    )
    portfolio_returns = (
        asset_returns * beginning_weights
    ).sum(axis=1)
    forecast_schedule = create_forecast_schedule(
        portfolio_returns=portfolio_returns,
        estimation_window=3,
    )

    return (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    )


def test_rolling_monte_carlo_forecasts_align_results_and_losses(
) -> None:
    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_monte_carlo_inputs()

    confidence_levels = [0.80, 0.95]
    scenario_count = 2_000
    random_seed = 20261002

    result = calculate_rolling_gaussian_monte_carlo_forecasts(
        asset_returns=asset_returns,
        beginning_weights=beginning_weights,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        scenario_count=scenario_count,
        random_seed=random_seed,
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
    first_schedule_row = forecast_schedule.loc[
        first_forecast_date
    ]
    first_result = result.loc[
        (first_forecast_date, 0.80)
    ]

    assert first_result["realized_return"] == pytest.approx(
        first_schedule_row["realized_return"]
    )
    assert first_result["realized_loss"] == pytest.approx(
        first_schedule_row["realized_loss"]
    )
    assert first_result["scenario_count"] == scenario_count
    assert first_result["random_seed"] == random_seed

    estimation_returns = asset_returns.loc[
        first_schedule_row["estimation_start_date"]:
        first_schedule_row["estimation_end_date"]
    ]
    forecast_weights = beginning_weights.loc[
        first_forecast_date
    ]
    moments = calculate_sample_portfolio_moments(
        asset_returns=estimation_returns,
        weights=forecast_weights,
    )
    shocks = generate_standard_normal_shocks(
        scenario_count=scenario_count,
        asset_count=len(asset_returns.columns),
        random_seed=random_seed,
    )
    expected_losses = (
        generate_gaussian_portfolio_losses_from_shocks(
            mean_vector=moments.mean_vector,
            covariance_matrix=moments.covariance_matrix,
            weights=forecast_weights,
            standard_normal_shocks=shocks,
        )
    )
    expected_estimate = calculate_empirical_var_es(
        losses=expected_losses,
        confidence_level=0.80,
    )

    assert first_result["value_at_risk"] == pytest.approx(
        expected_estimate.value_at_risk
    )
    assert first_result["expected_shortfall"] == pytest.approx(
        expected_estimate.expected_shortfall
    )
    assert first_result["var_exceedance"] == (
        first_schedule_row["realized_loss"]
        > expected_estimate.value_at_risk
    )


def test_rolling_monte_carlo_forecasts_generate_shocks_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_monte_carlo_inputs()

    original_generator = (
        simulation_module.generate_standard_normal_shocks
    )
    call_count = 0

    def counting_generator(
        scenario_count: int,
        asset_count: int,
        random_seed: int,
    ) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original_generator(
            scenario_count=scenario_count,
            asset_count=asset_count,
            random_seed=random_seed,
        )

    monkeypatch.setattr(
        simulation_module,
        "generate_standard_normal_shocks",
        counting_generator,
    )

    calculate_rolling_gaussian_monte_carlo_forecasts(
        asset_returns=asset_returns,
        beginning_weights=beginning_weights,
        forecast_schedule=forecast_schedule,
        confidence_levels=[0.95],
        scenario_count=500,
        random_seed=20261002,
    )

    assert call_count == 1


def test_rolling_monte_carlo_forecasts_are_reproducible() -> None:
    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_monte_carlo_inputs()

    first_result = (
        calculate_rolling_gaussian_monte_carlo_forecasts(
            asset_returns=asset_returns,
            beginning_weights=beginning_weights,
            forecast_schedule=forecast_schedule,
            confidence_levels=[0.80, 0.95],
            scenario_count=500,
            random_seed=20261002,
        )
    )
    second_result = (
        calculate_rolling_gaussian_monte_carlo_forecasts(
            asset_returns=asset_returns,
            beginning_weights=beginning_weights,
            forecast_schedule=forecast_schedule,
            confidence_levels=[0.80, 0.95],
            scenario_count=500,
            random_seed=20261002,
        )
    )

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )


def test_rolling_monte_carlo_forecasts_reject_lookahead() -> None:
    (
        asset_returns,
        beginning_weights,
        forecast_schedule,
    ) = _create_rolling_monte_carlo_inputs()

    first_forecast_date = forecast_schedule.index[0]
    forecast_schedule.loc[
        first_forecast_date,
        "estimation_end_date",
    ] = first_forecast_date

    with pytest.raises(
        ValueError,
        match="must end before its forecast date",
    ):
        calculate_rolling_gaussian_monte_carlo_forecasts(
            asset_returns=asset_returns,
            beginning_weights=beginning_weights,
            forecast_schedule=forecast_schedule,
            confidence_levels=[0.95],
            scenario_count=500,
            random_seed=20261002,
        )
