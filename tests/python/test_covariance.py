"""Tests for sample covariance and portfolio moments."""

import numpy as np
import pandas as pd
import pytest

from tailrisk.covariance import (
    calculate_ewma_covariance_update,
    calculate_sample_portfolio_moments,
    calculate_initial_ewma_covariance,
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




def test_ewma_covariance_update_matches_manual_example() -> None:
    """Verify one EWMA update against a manual two-asset calculation."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
        name="innovation",
    )

    result = calculate_ewma_covariance_update(
        previous_covariance=previous_covariance,
        innovation=innovation,
        decay_factor=0.94,
    )

    expected = pd.DataFrame(
        [
            [0.000430, 0.000130],
            [0.000130, 0.0002355],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )



def test_ewma_covariance_update_aligns_innovation_by_ticker() -> None:
    """Verify that innovation order does not change the EWMA update."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    ordered_innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )
    reversed_innovation = ordered_innovation.iloc[
        ::-1
    ]

    ordered_result = calculate_ewma_covariance_update(
        previous_covariance=previous_covariance,
        innovation=ordered_innovation,
        decay_factor=0.94,
    )
    reversed_result = calculate_ewma_covariance_update(
        previous_covariance=previous_covariance,
        innovation=reversed_innovation,
        decay_factor=0.94,
    )

    pd.testing.assert_frame_equal(
        reversed_result,
        ordered_result,
    )



def test_ewma_covariance_update_rejects_mismatched_tickers() -> None:
    """Verify that covariance and innovation tickers must match."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "C"],
    )

    with pytest.raises(
        ValueError,
        match="same tickers",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )


def test_ewma_covariance_update_rejects_duplicate_innovation_tickers() -> None:
    """Verify that each innovation ticker appears exactly once."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "A"],
    )

    with pytest.raises(
        ValueError,
        match="Innovation tickers must be unique",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )



def test_ewma_covariance_update_rejects_mismatched_covariance_tickers() -> None:
    """Verify that covariance rows and columns use the same tickers."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "C"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "C"],
    )

    with pytest.raises(
        ValueError,
        match="Covariance rows and columns must use the same tickers",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )


def test_ewma_covariance_update_rejects_nonnumeric_covariance() -> None:
    """Verify that the previous covariance must be numeric."""

    previous_covariance = pd.DataFrame(
        [
            ["invalid", 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Previous covariance must contain numeric values",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )



def test_ewma_covariance_update_rejects_nonnumeric_innovation() -> None:
    """Verify that the innovation must be numeric."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        ["invalid", -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Innovation must contain numeric values",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_ewma_covariance_update_rejects_nonfinite_covariance(
    invalid_value: float,
) -> None:
    """Verify that the previous covariance must be finite."""

    previous_covariance = pd.DataFrame(
        [
            [invalid_value, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Previous covariance must contain finite numbers",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )



@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_ewma_covariance_update_rejects_nonfinite_innovation(
    invalid_value: float,
) -> None:
    """Verify that the innovation must be finite."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [invalid_value, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Innovation must contain finite numbers",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )



@pytest.mark.parametrize(
    "invalid_decay_factor",
    [
        0.0,
        1.0,
        -0.01,
        1.01,
        True,
        False,
        "0.94",
        None,
        0.94 + 0j,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_ewma_covariance_update_rejects_invalid_decay_factor(
    invalid_decay_factor: object,
) -> None:
    """Verify that decay must be finite, real, and strictly bounded."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Decay factor must be a finite real number",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=invalid_decay_factor,
        )



def test_ewma_covariance_update_rejects_nonsymmetric_covariance() -> None:
    """Verify that the previous covariance must be symmetric."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000200, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Previous covariance must be symmetric",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )



def test_ewma_covariance_update_rejects_non_psd_covariance() -> None:
    """Verify that the previous covariance must be positive semidefinite."""

    previous_covariance = pd.DataFrame(
        [
            [0.000100, 0.000200],
            [0.000200, 0.000100],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="positive semidefinite",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )


def test_ewma_covariance_update_preserves_covariance_properties() -> None:
    """Verify that an EWMA update remains symmetric and PSD."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    result = calculate_ewma_covariance_update(
        previous_covariance=previous_covariance,
        innovation=innovation,
        decay_factor=0.94,
    )

    result_values = result.to_numpy(
        dtype=float
    )

    assert np.allclose(
        result_values,
        result_values.T,
        rtol=1e-12,
        atol=1e-15,
    )

    eigenvalues = np.linalg.eigvalsh(
        result_values
    )

    assert np.all(eigenvalues >= -1e-15)



def test_ewma_covariance_update_rejects_nonfinite_result() -> None:
    """Verify that numerical overflow cannot produce a saved covariance."""

    previous_covariance = pd.DataFrame(
        [[1.0]],
        index=["A"],
        columns=["A"],
    )
    innovation = pd.Series(
        [1e308],
        index=["A"],
    )

    with pytest.raises(
        ValueError,
        match="Updated covariance must contain finite numbers",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )


def test_ewma_covariance_update_requires_dataframe() -> None:
    """Verify that the previous covariance must be a DataFrame."""

    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        TypeError,
        match="Previous covariance must be a pandas DataFrame",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=np.eye(2),
            innovation=innovation,
            decay_factor=0.94,
        )


def test_ewma_covariance_update_requires_series() -> None:
    """Verify that the innovation must be a Series."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        TypeError,
        match="Innovation must be a pandas Series",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=np.array([-0.03, -0.02]),
            decay_factor=0.94,
        )


def test_ewma_covariance_update_rejects_empty_covariance() -> None:
    """Verify that the previous covariance must not be empty."""

    innovation = pd.Series(
        [-0.03],
        index=["A"],
    )

    with pytest.raises(
        ValueError,
        match="Previous covariance must not be empty",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=pd.DataFrame(),
            innovation=innovation,
            decay_factor=0.94,
        )


def test_ewma_covariance_update_rejects_empty_innovation() -> None:
    """Verify that the innovation must not be empty."""

    previous_covariance = pd.DataFrame(
        [[0.000400]],
        index=["A"],
        columns=["A"],
    )

    with pytest.raises(
        ValueError,
        match="Innovation must not be empty",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=pd.Series(dtype=float),
            decay_factor=0.94,
        )


def test_ewma_covariance_update_rejects_duplicate_covariance_tickers() -> None:
    """Verify that covariance tickers must be unique."""

    previous_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "A"],
        columns=["A", "B"],
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Covariance tickers must be unique",
    ):
        calculate_ewma_covariance_update(
            previous_covariance=previous_covariance,
            innovation=innovation,
            decay_factor=0.94,
        )


def test_ewma_covariance_update_aligns_covariance_rows() -> None:
    """Verify that covariance row order does not change the update."""

    ordered_covariance = pd.DataFrame(
        [
            [0.000400, 0.000100],
            [0.000100, 0.000225],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )
    reordered_covariance = ordered_covariance.reindex(
        index=["B", "A"]
    )
    innovation = pd.Series(
        [-0.03, -0.02],
        index=["A", "B"],
    )

    ordered_result = calculate_ewma_covariance_update(
        previous_covariance=ordered_covariance,
        innovation=innovation,
        decay_factor=0.94,
    )
    reordered_result = calculate_ewma_covariance_update(
        previous_covariance=reordered_covariance,
        innovation=innovation,
        decay_factor=0.94,
    )

    pd.testing.assert_frame_equal(
        reordered_result,
        ordered_result,
    )



def test_initial_ewma_covariance_matches_manual_example() -> None:
    """Verify EWMA initialization against a manual sample covariance."""

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

    result = calculate_initial_ewma_covariance(
        asset_returns
    )

    expected = pd.DataFrame(
        [
            [0.0003333333333333333, 0.0002],
            [0.0002, 0.00016666666666666666],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )




def test_initial_ewma_covariance_requires_dataframe() -> None:
    """Verify that initialization requires a return DataFrame."""

    with pytest.raises(
        TypeError,
        match="Asset returns must be a pandas DataFrame",
    ):
        calculate_initial_ewma_covariance(
            np.array(
                [
                    [0.01, 0.02],
                    [-0.01, 0.00],
                ]
            )
        )



def test_initial_ewma_covariance_rejects_empty_returns() -> None:
    """Verify that initialization rejects an empty return matrix."""

    with pytest.raises(
        ValueError,
        match="Asset returns must not be empty",
    ):
        calculate_initial_ewma_covariance(
            pd.DataFrame()
        )



def test_initial_ewma_covariance_requires_two_observations() -> None:
    """Verify that sample covariance requires two observations."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.01],
            "B": [-0.02],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two return observations are required",
    ):
        calculate_initial_ewma_covariance(
            asset_returns
        )



def test_initial_ewma_covariance_requires_unique_tickers() -> None:
    """Verify that each asset-return ticker appears exactly once."""

    asset_returns = pd.DataFrame(
        [
            [0.01, 0.02],
            [-0.01, 0.00],
        ],
        columns=["A", "A"],
    )

    with pytest.raises(
        ValueError,
        match="Asset-return tickers must be unique",
    ):
        calculate_initial_ewma_covariance(
            asset_returns
        )



def test_initial_ewma_covariance_rejects_nonnumeric_returns() -> None:
    """Verify that initialization requires numeric returns."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.01, "invalid"],
            "B": [0.02, -0.01],
        }
    )

    with pytest.raises(
        ValueError,
        match="Asset returns must contain numeric values",
    ):
        calculate_initial_ewma_covariance(
            asset_returns
        )



@pytest.mark.parametrize(
    "invalid_return",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_initial_ewma_covariance_rejects_nonfinite_returns(
    invalid_return: float,
) -> None:
    """Verify that initialization requires finite returns."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.01, invalid_return],
            "B": [0.02, -0.01],
        }
    )

    with pytest.raises(
        ValueError,
        match="Asset returns must contain finite numbers",
    ):
        calculate_initial_ewma_covariance(
            asset_returns
        )



def test_initial_ewma_covariance_has_valid_properties() -> None:
    """Verify that initialization produces a finite symmetric PSD matrix."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "B": [0.02, 0.04, 0.06],
        }
    )

    result = calculate_initial_ewma_covariance(
        asset_returns
    )

    result_values = result.to_numpy(
        dtype=float
    )

    assert np.isfinite(result_values).all()

    assert np.allclose(
        result_values,
        result_values.T,
        rtol=1e-12,
        atol=1e-15,
    )

    eigenvalues = np.linalg.eigvalsh(
        result_values
    )

    assert np.all(eigenvalues >= -1e-15)



def test_initial_ewma_covariance_rejects_nonfinite_result() -> None:
    """Verify that numerical overflow cannot produce an initial covariance."""

    asset_returns = pd.DataFrame(
        {
            "A": [1e308, -1e308],
            "B": [-1e308, 1e308],
        }
    )

    with pytest.raises(
        ValueError,
        match="Initial EWMA covariance must contain finite numbers",
    ):
        calculate_initial_ewma_covariance(
            asset_returns
        )




def test_initial_ewma_covariance_matches_rolling_gaussian() -> None:
    """Verify that EWMA and rolling Gaussian start with one covariance."""

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
    )

    ewma_covariance = (
        calculate_initial_ewma_covariance(
            asset_returns
        )
    )
    rolling_gaussian_covariance = (
        calculate_sample_portfolio_moments(
            asset_returns=asset_returns,
            weights=weights,
        ).covariance_matrix
    )

    pd.testing.assert_frame_equal(
        ewma_covariance,
        rolling_gaussian_covariance,
    )