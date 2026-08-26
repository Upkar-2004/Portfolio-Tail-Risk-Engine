"""Asset and portfolio return calculations."""

import pandas as pd


def calculate_simple_returns(
    adjusted_close: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate one-session simple returns from adjusted closing prices."""

    return adjusted_close.pct_change(
        fill_method=None,
    )


def flag_large_returns(
    returns: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Flag observed returns whose absolute value reaches a threshold."""

    if threshold <= 0:
        raise ValueError(
            "Large-return threshold must be positive."
        )

    return (
        returns.abs().ge(threshold)
        & returns.notna()
    )