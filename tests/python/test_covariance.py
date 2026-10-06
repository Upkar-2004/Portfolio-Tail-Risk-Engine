"""Tests for sample covariance and portfolio moments."""

import numpy as np
import pandas as pd
import pytest

from tailrisk.covariance import (
    calculate_ewma_covariance_update,
    calculate_sample_portfolio_moments,
    calculate_initial_ewma_covariance,
    calculate_ewma_covariance_sequence,
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



def test_ewma_covariance_sequence_matches_manual_example() -> None:
    """Verify initialization and one recursive update manually."""

    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
        ]
    )
    asset_returns = pd.DataFrame(
        {
            "A": [
                0.01,
                0.03,
                -0.01,
                0.50,
            ],
        },
        index=dates,
    )

    forecast_dates = dates[2:]
    forecast_schedule = pd.DataFrame(
        {
            "estimation_start_date": [
                dates[0],
                dates[1],
            ],
            "estimation_end_date": [
                dates[1],
                dates[2],
            ],
        },
        index=pd.DatetimeIndex(
            forecast_dates,
            name="forecast_date",
        ),
    )

    result = calculate_ewma_covariance_sequence(
        asset_returns=asset_returns,
        forecast_schedule=forecast_schedule,
        decay_factor=0.50,
    )

    assert list(result) == list(
        forecast_schedule.index
    )

    expected_first_covariance = pd.DataFrame(
        [[0.0002]],
        index=["A"],
        columns=["A"],
    )
    expected_second_covariance = pd.DataFrame(
        [[0.00055]],
        index=["A"],
        columns=["A"],
    )

    pd.testing.assert_frame_equal(
        result[forecast_dates[0]],
        expected_first_covariance,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )
    pd.testing.assert_frame_equal(
        result[forecast_dates[1]],
        expected_second_covariance,
        check_exact=False,
        rtol=1e-12,
        atol=1e-15,
    )


def _valid_ewma_sequence_inputs(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid returns and a two-date forecast schedule."""

    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
        ]
    )
    asset_returns = pd.DataFrame(
        {
            "A": [
                0.01,
                0.03,
                -0.01,
                0.50,
            ],
        },
        index=dates,
    )
    forecast_schedule = pd.DataFrame(
        {
            "estimation_start_date": [
                dates[0],
                dates[1],
            ],
            "estimation_end_date": [
                dates[1],
                dates[2],
            ],
        },
        index=pd.DatetimeIndex(
            dates[2:],
            name="forecast_date",
        ),
    )

    return asset_returns, forecast_schedule



def test_ewma_covariance_sequence_uses_return_only_next_day() -> None:
    """Verify that a return first affects the next forecast covariance."""

    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
        ]
    )
    baseline_returns = pd.DataFrame(
        {
            "A": [
                0.01,
                0.03,
                -0.01,
                0.50,
            ],
        },
        index=dates,
    )
    forecast_schedule = pd.DataFrame(
        {
            "estimation_start_date": [
                dates[0],
                dates[1],
            ],
            "estimation_end_date": [
                dates[1],
                dates[2],
            ],
        },
        index=pd.DatetimeIndex(
            dates[2:],
            name="forecast_date",
        ),
    )

    baseline_sequence = (
        calculate_ewma_covariance_sequence(
            asset_returns=baseline_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )
    )

    changed_first_forecast_return = (
        baseline_returns.copy()
    )
    changed_first_forecast_return.loc[
        dates[2],
        "A",
    ] = 0.20

    changed_first_sequence = (
        calculate_ewma_covariance_sequence(
            asset_returns=(
                changed_first_forecast_return
            ),
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )
    )

    pd.testing.assert_frame_equal(
        baseline_sequence[dates[2]],
        changed_first_sequence[dates[2]],
    )

    assert not np.allclose(
        baseline_sequence[
            dates[3]
        ].to_numpy(),
        changed_first_sequence[
            dates[3]
        ].to_numpy(),
    )

    changed_final_return = (
        baseline_returns.copy()
    )
    changed_final_return.loc[
        dates[3],
        "A",
    ] = -0.50

    changed_final_sequence = (
        calculate_ewma_covariance_sequence(
            asset_returns=changed_final_return,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )
    )

    for forecast_date in forecast_schedule.index:
        pd.testing.assert_frame_equal(
            baseline_sequence[forecast_date],
            changed_final_sequence[forecast_date],
        )



@pytest.mark.parametrize(
    "invalid_input",
    [
        "asset_returns",
        "forecast_schedule",
    ],
)
def test_ewma_covariance_sequence_requires_dataframes(
    invalid_input: str,
) -> None:
    """Verify that sequence inputs must be DataFrames."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )

    if invalid_input == "asset_returns":
        asset_returns = np.array(
            [
                [0.01],
                [0.03],
                [-0.01],
                [0.50],
            ]
        )
        expected_message = (
            "Asset returns must be a pandas DataFrame"
        )
    else:
        forecast_schedule = []
        expected_message = (
            "Forecast schedule must be a pandas DataFrame"
        )

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


@pytest.mark.parametrize(
    "empty_input",
    [
        "asset_returns",
        "forecast_schedule",
    ],
)
def test_ewma_covariance_sequence_rejects_empty_inputs(
    empty_input: str,
) -> None:
    """Verify that sequence inputs must not be empty."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )

    if empty_input == "asset_returns":
        asset_returns = pd.DataFrame()
        expected_message = (
            "Asset returns must not be empty"
        )
    else:
        forecast_schedule = pd.DataFrame()
        expected_message = (
            "Forecast schedule must not be empty"
        )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )



@pytest.mark.parametrize(
    "missing_column",
    [
        "estimation_start_date",
        "estimation_end_date",
    ],
)
def test_ewma_covariance_sequence_requires_schedule_columns(
    missing_column: str,
) -> None:
    """Verify that both estimation-window columns are required."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    forecast_schedule = forecast_schedule.drop(
        columns=[missing_column]
    )

    with pytest.raises(
        ValueError,
        match="estimation start and end dates",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


@pytest.mark.parametrize(
    "invalid_index",
    [
        "asset_returns",
        "forecast_schedule",
    ],
)
def test_ewma_covariance_sequence_requires_datetime_indexes(
    invalid_index: str,
) -> None:
    """Verify that returns and forecasts use date indexes."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )

    if invalid_index == "asset_returns":
        asset_returns = asset_returns.reset_index(
            drop=True
        )
        expected_message = (
            "Asset returns must use a DatetimeIndex"
        )
    else:
        forecast_schedule = forecast_schedule.reset_index(
            drop=True
        )
        expected_message = (
            "Forecast schedule must use a DatetimeIndex"
        )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_requires_unique_forecast_dates() -> None:
    """Verify that every forecast date appears exactly once."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    forecast_schedule.index = pd.DatetimeIndex(
        [
            forecast_schedule.index[0],
            forecast_schedule.index[0],
        ],
        name="forecast_date",
    )

    with pytest.raises(
        ValueError,
        match="Forecast dates must be unique",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )



def test_ewma_covariance_sequence_requires_ordered_forecast_dates() -> None:
    """Verify that forecast dates are chronologically ordered."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    forecast_schedule = forecast_schedule.iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError,
        match="Forecast dates must be ordered",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_rejects_duplicate_return_dates() -> None:
    """Verify that every asset-return date appears exactly once."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    asset_returns.index = pd.DatetimeIndex(
        [
            asset_returns.index[0],
            asset_returns.index[0],
            asset_returns.index[2],
            asset_returns.index[3],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Asset-return dates must be unique",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_rejects_unordered_returns() -> None:
    """Verify that asset returns are chronologically ordered."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    asset_returns = asset_returns.iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError,
        match="Asset returns must be ordered",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_requires_window_dates() -> None:
    """Verify that estimation-window dates exist in asset returns."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    forecast_schedule.loc[
        forecast_schedule.index[0],
        "estimation_start_date",
    ] = pd.Timestamp("2024-12-31")

    with pytest.raises(
        ValueError,
        match="Estimation-window dates must exist",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_rejects_lookahead() -> None:
    """Verify that each estimation window ends before its forecast."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    first_forecast_date = (
        forecast_schedule.index[0]
    )
    forecast_schedule.loc[
        first_forecast_date,
        "estimation_end_date",
    ] = first_forecast_date

    with pytest.raises(
        ValueError,
        match="must end before its forecast date",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_requires_prior_session_end() -> None:
    """Verify that each window ends on the prior trading session."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    first_forecast_date = (
        forecast_schedule.index[0]
    )
    forecast_schedule.loc[
        first_forecast_date,
        "estimation_end_date",
    ] = asset_returns.index[0]

    with pytest.raises(
        ValueError,
        match="immediately before its forecast date",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_requires_consecutive_forecasts() -> None:
    """Verify that forecast dates are consecutive return sessions."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    final_date = pd.Timestamp("2025-01-08")
    asset_returns.loc[final_date, "A"] = 0.02

    forecast_schedule.index = pd.DatetimeIndex(
        [
            asset_returns.index[2],
            final_date,
        ],
        name="forecast_date",
    )
    forecast_schedule.iloc[
        1,
        forecast_schedule.columns.get_loc(
            "estimation_end_date"
        ),
    ] = asset_returns.index[3]

    with pytest.raises(
        ValueError,
        match="Forecast dates must be consecutive",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )



def test_ewma_covariance_sequence_rejects_reversed_window() -> None:
    """Verify that a window cannot start after it ends."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    first_forecast_date = (
        forecast_schedule.index[0]
    )
    forecast_schedule.loc[
        first_forecast_date,
        "estimation_start_date",
    ] = asset_returns.index[2]

    with pytest.raises(
        ValueError,
        match="must start on or before its end date",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_requires_two_window_observations() -> None:
    """Verify that every estimation window contains two observations."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    first_forecast_date = (
        forecast_schedule.index[0]
    )
    forecast_schedule.loc[
        first_forecast_date,
        "estimation_start_date",
    ] = asset_returns.index[1]

    with pytest.raises(
        ValueError,
        match="at least two observations",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_requires_equal_window_lengths() -> None:
    """Verify that every estimation window has one consistent length."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    second_forecast_date = (
        forecast_schedule.index[1]
    )
    forecast_schedule.loc[
        second_forecast_date,
        "estimation_start_date",
    ] = asset_returns.index[0]

    with pytest.raises(
        ValueError,
        match="same number of observations",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_requires_unique_tickers() -> None:
    """Verify that each asset-return ticker appears exactly once."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    asset_returns = pd.concat(
        [
            asset_returns,
            asset_returns,
        ],
        axis="columns",
    )
    asset_returns.columns = ["A", "A"]

    with pytest.raises(
        ValueError,
        match="Asset-return tickers must be unique",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


def test_ewma_covariance_sequence_rejects_nonnumeric_returns() -> None:
    """Verify that every return in the sequence input is numeric."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    asset_returns = asset_returns.astype(
        object
    )
    asset_returns.loc[
        asset_returns.index[-1],
        "A",
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="Asset returns must contain numeric values",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
        )


@pytest.mark.parametrize(
    "invalid_return",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_ewma_covariance_sequence_rejects_nonfinite_returns(
    invalid_return: float,
) -> None:
    """Verify that every return in the sequence input is finite."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    asset_returns.loc[
        asset_returns.index[-1],
        "A",
    ] = invalid_return

    with pytest.raises(
        ValueError,
        match="Asset returns must contain finite numbers",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=forecast_schedule,
            decay_factor=0.50,
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
def test_ewma_covariance_sequence_rejects_invalid_decay_factor(
    invalid_decay_factor: object,
) -> None:
    """Verify decay validation even for a one-date sequence."""

    asset_returns, forecast_schedule = (
        _valid_ewma_sequence_inputs()
    )
    one_date_schedule = forecast_schedule.iloc[
        [0]
    ]

    with pytest.raises(
        ValueError,
        match="Decay factor must be a finite real number",
    ):
        calculate_ewma_covariance_sequence(
            asset_returns=asset_returns,
            forecast_schedule=one_date_schedule,
            decay_factor=invalid_decay_factor,
        )


def test_ewma_covariance_sequence_preserves_matrix_properties() -> None:
    """Verify every sequence matrix is labelled, finite, symmetric, and PSD."""

    dates = pd.bdate_range(
        "2025-01-02",
        periods=5,
    )
    asset_returns = pd.DataFrame(
        {
            "A": [
                0.01,
                0.03,
                -0.01,
                0.02,
                -0.04,
            ],
            "B": [
                0.02,
                0.01,
                0.00,
                -0.03,
                0.05,
            ],
        },
        index=dates,
    )
    forecast_dates = dates[2:]
    forecast_schedule = pd.DataFrame(
        {
            "estimation_start_date": dates[:3],
            "estimation_end_date": dates[1:4],
        },
        index=pd.DatetimeIndex(
            forecast_dates,
            name="forecast_date",
        ),
    )

    result = calculate_ewma_covariance_sequence(
        asset_returns=asset_returns,
        forecast_schedule=forecast_schedule,
        decay_factor=0.94,
    )

    assert list(result) == list(forecast_dates)

    for covariance_matrix in result.values():
        assert list(covariance_matrix.index) == [
            "A",
            "B",
        ]
        assert list(covariance_matrix.columns) == [
            "A",
            "B",
        ]

        covariance_values = covariance_matrix.to_numpy(
            dtype=float
        )

        assert np.isfinite(covariance_values).all()
        assert np.allclose(
            covariance_values,
            covariance_values.T,
            rtol=1e-12,
            atol=1e-15,
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

        assert float(eigenvalues.min()) >= (
            -numerical_tolerance
        )


def test_ewma_covariance_sequence_uses_504_session_window() -> None:
    """Verify the baseline window initializes and advances correctly."""

    dates = pd.bdate_range(
        "2023-01-02",
        periods=506,
    )
    asset_returns = pd.DataFrame(
        {
            "A": np.linspace(
                -0.02,
                0.02,
                len(dates),
            ),
            "B": np.linspace(
                0.015,
                -0.015,
                len(dates),
            ),
        },
        index=dates,
    )
    forecast_dates = dates[504:]
    forecast_schedule = pd.DataFrame(
        {
            "estimation_start_date": [
                dates[0],
                dates[1],
            ],
            "estimation_end_date": [
                dates[503],
                dates[504],
            ],
        },
        index=pd.DatetimeIndex(
            forecast_dates,
            name="forecast_date",
        ),
    )

    result = calculate_ewma_covariance_sequence(
        asset_returns=asset_returns,
        forecast_schedule=forecast_schedule,
        decay_factor=0.94,
    )
    expected_initial_covariance = (
        calculate_initial_ewma_covariance(
            asset_returns.iloc[:504]
        )
    )

    assert list(result) == list(forecast_dates)
    pd.testing.assert_frame_equal(
        result[forecast_dates[0]],
        expected_initial_covariance,
    )
