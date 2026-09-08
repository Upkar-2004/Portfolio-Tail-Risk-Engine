"""Tests for asset and portfolio return calculations."""

from pathlib import Path

import pandas as pd
import pytest

from tailrisk.returns import (
    calculate_simple_returns,
    flag_large_returns,
    save_processed_returns,
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


def test_save_processed_returns_creates_csv(
    tmp_path: Path,
) -> None:
    """Verify that processed returns are saved in a snapshot directory."""

    returns = pd.DataFrame(
        {
            "GOOGL": [0.05, -0.02],
            "AMZN": [0.03, 0.01],
        },
        index=pd.to_datetime(
            [
                "2025-01-03",
                "2025-01-06",
            ]
        ),
    )
    returns.index.name = "Date"

    data_path = save_processed_returns(
        returns=returns,
        output_root=tmp_path,
        snapshot_id="test_snapshot",
    )

    expected_path = (
        tmp_path
        / "test_snapshot"
        / "asset_returns.csv"
    )

    assert data_path == expected_path
    assert data_path.is_file()

    saved_returns = pd.read_csv(
        data_path,
        index_col="Date",
        parse_dates=["Date"],
    )

    pd.testing.assert_frame_equal(
        saved_returns,
        returns,
    )


def test_save_processed_returns_rejects_existing_snapshot(
    tmp_path: Path,
) -> None:
    """Verify that an existing processed snapshot is not overwritten."""

    returns = pd.DataFrame(
        {"GOOGL": [0.05]},
        index=pd.to_datetime(["2025-01-03"]),
    )
    returns.index.name = "Date"

    save_processed_returns(
        returns=returns,
        output_root=tmp_path,
        snapshot_id="existing_snapshot",
    )

    with pytest.raises(FileExistsError):
        save_processed_returns(
            returns=returns,
            output_root=tmp_path,
            snapshot_id="existing_snapshot",
        )
