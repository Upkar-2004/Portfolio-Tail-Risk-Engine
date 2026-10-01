"""Tests for sample covariance and portfolio moments."""

import numpy as np
import pandas as pd
import pytest

from tailrisk.covariance import (
    calculate_sample_portfolio_moments,
)


def test_sample_portfolio_moments_match_manual_example() -> None:
    """Verify multivariate moments against a two-asset example."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.02, 0.00, -0.01, 0.03],
            "B": [0.01, -0.01, 0.00, 0.02],
        },
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )
    weights = pd.Series(
        [0.60, 0.40],
        index=["A", "B"],
        name="weight",
    )

    result = calculate_sample_portfolio_moments(
        asset_returns=asset_returns,
        weights=weights,
    )

    expected_mean_vector = pd.Series(
        [0.01, 0.005],
        index=["A", "B"],
        name="mean_return",
    )
    expected_covariance = pd.DataFrame(
        [
            [0.0003333333333333333, 0.0002],
            [0.0002, 0.00016666666666666666],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    pd.testing.assert_series_equal(
        result.mean_vector,
        expected_mean_vector,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )
    pd.testing.assert_frame_equal(
        result.covariance_matrix,
        expected_covariance,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )

    assert result.portfolio_mean == pytest.approx(
        0.008
    )
    assert result.portfolio_variance == pytest.approx(
        0.00024266666666666667
    )
    assert result.portfolio_volatility == pytest.approx(
        0.015577761927397231
    )

    direct_portfolio_returns = (
        asset_returns @ weights
    )

    assert direct_portfolio_returns.mean() == pytest.approx(
        result.portfolio_mean
    )
    assert direct_portfolio_returns.var(ddof=1) == pytest.approx(
        result.portfolio_variance
    )

    pd.testing.assert_frame_equal(
        result.covariance_matrix,
        result.covariance_matrix.T,
    )

    eigenvalues = np.linalg.eigvalsh(
        result.covariance_matrix.to_numpy()
    )
    assert np.all(eigenvalues >= -1e-15)


def test_sample_portfolio_moments_align_weights_by_ticker() -> None:
    """Weight order should not change the portfolio calculation."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.02, 0.00, -0.01, 0.03],
            "B": [0.01, -0.01, 0.00, 0.02],
        }
    )
    ordered_weights = pd.Series(
        [0.60, 0.40],
        index=["A", "B"],
    )
    reversed_weights = ordered_weights.iloc[::-1]

    ordered_result = calculate_sample_portfolio_moments(
        asset_returns,
        ordered_weights,
    )
    reversed_result = calculate_sample_portfolio_moments(
        asset_returns,
        reversed_weights,
    )

    assert reversed_result.portfolio_mean == pytest.approx(
        ordered_result.portfolio_mean
    )
    assert reversed_result.portfolio_variance == pytest.approx(
        ordered_result.portfolio_variance
    )


def test_sample_portfolio_moments_reject_mismatched_tickers() -> None:
    asset_returns = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.00, -0.01],
        }
    )
    weights = pd.Series(
        [0.50, 0.50],
        index=["A", "C"],
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        calculate_sample_portfolio_moments(
            asset_returns,
            weights,
        )


def test_sample_portfolio_moments_require_two_observations() -> None:
    asset_returns = pd.DataFrame(
        {
            "A": [0.01],
            "B": [0.02],
        }
    )
    weights = pd.Series(
        [0.50, 0.50],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="At least two return observations",
    ):
        calculate_sample_portfolio_moments(
            asset_returns,
            weights,
        )


@pytest.mark.parametrize(
    "invalid_return",
    [np.nan, np.inf, -np.inf],
)
def test_sample_portfolio_moments_reject_nonfinite_returns(
    invalid_return: float,
) -> None:
    asset_returns = pd.DataFrame(
        {
            "A": [0.01, invalid_return],
            "B": [0.02, -0.01],
        }
    )
    weights = pd.Series(
        [0.50, 0.50],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Asset returns must be finite numbers",
    ):
        calculate_sample_portfolio_moments(
            asset_returns,
            weights,
        )


@pytest.mark.parametrize(
    "invalid_weight",
    [np.nan, np.inf, -np.inf],
)
def test_sample_portfolio_moments_reject_nonfinite_weights(
    invalid_weight: float,
) -> None:
    asset_returns = pd.DataFrame(
        {
            "A": [0.01, 0.00],
            "B": [0.02, -0.01],
        }
    )
    weights = pd.Series(
        [invalid_weight, 0.50],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Portfolio weights must be finite numbers",
    ):
        calculate_sample_portfolio_moments(
            asset_returns,
            weights,
        )

