"""Asset and portfolio return calculations."""

from pathlib import Path

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


def save_processed_returns(
    returns: pd.DataFrame,
    output_root: str | Path,
    snapshot_id: str,
) -> Path:
    """Save asset returns in a new processed-data snapshot."""

    snapshot_directory = (
        Path(output_root) / snapshot_id
    )
    snapshot_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    data_path = (
        snapshot_directory
        / "asset_returns.csv"
    )
    returns.to_csv(data_path)

    return data_path
