"""Tests for asset and portfolio return calculations."""

import pandas as pd

from tailrisk.returns import (
    calculate_simple_returns,
    flag_large_returns,
)


def test_calculate_simple_returns_handles_basic_price_movements() -> None:
    """Verify positive, zero, and negative one-session returns."""

    adjusted_close = pd.DataFrame(
        {
            "GOOGL": [
                100.0,
                105.0,
                105.0,
                99.75,
            ]
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

    returns = calculate_simple_returns(adjusted_close)

    expected = pd.DataFrame(
        {
            "GOOGL": [
                float("nan"),
                0.05,
                0.00,
                -0.05,
            ]
        },
        index=adjusted_close.index,
    )

    pd.testing.assert_frame_equal(
        returns,
        expected,
    )



def test_calculate_simple_returns_does_not_fill_missing_prices() -> None:
    """Verify that missing prices produce missing one-session returns."""

    adjusted_close = pd.DataFrame(
        {
            "GOOGL": [
                100.0,
                float("nan"),
                110.0,
                121.0,
            ]
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

    returns = calculate_simple_returns(adjusted_close)

    expected = pd.DataFrame(
        {
            "GOOGL": [
                float("nan"),
                float("nan"),
                float("nan"),
                0.10,
            ]
        },
        index=adjusted_close.index,
    )

    pd.testing.assert_frame_equal(
        returns,
        expected,
    )



def test_flag_large_returns_uses_absolute_values() -> None:
    """Verify that unusually large gains and losses are both flagged."""

    returns = pd.DataFrame(
        {
            "GOOGL": [0.05, -0.25],
            "AMZN": [0.30, -0.10],
        }
    )

    flags = flag_large_returns(
        returns,
        threshold=0.20,
    )

    expected = pd.DataFrame(
        {
            "GOOGL": [False, True],
            "AMZN": [True, False],
        }
    )

    pd.testing.assert_frame_equal(
        flags,
        expected,
    )