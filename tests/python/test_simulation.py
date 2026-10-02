"""Tests for the Python Gaussian simulation reference."""

import numpy as np
import pandas as pd

from tailrisk.simulation import (
    generate_gaussian_return_scenarios,
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