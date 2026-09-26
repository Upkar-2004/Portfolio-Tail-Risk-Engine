"""Tests for asset and portfolio return calculations."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest
import tailrisk.returns as returns_module
from tailrisk.data import calculate_file_sha256


from tailrisk.returns import (
    calculate_simple_returns,
    flag_large_returns,
    save_processed_metadata,
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


def test_save_processed_metadata_records_lineage_and_flags(
    tmp_path: Path,
) -> None:
    """Verify that processed metadata records its source and validation."""

    returns = pd.DataFrame(
        {
            "GOOGL": [0.05, -0.25],
            "AMZN": [0.30, -0.10],
        },
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )
    returns.index.name = "Date"
    flags = flag_large_returns(returns, threshold=0.20)

    data_path = tmp_path / "asset_returns.csv"
    returns.to_csv(data_path)
    source_metadata = {
        "files": {
            "market_data": {
                "name": "market_data.csv",
                "sha256": "raw-checksum",
            }
        }
    }

    metadata_path = save_processed_metadata(
        data_path=data_path,
        source_snapshot_id="raw_snapshot",
        source_metadata=source_metadata,
        returns=returns,
        flags=flags,
        threshold=0.20,
        processed_at=datetime(
            2025, 1, 7, 12, 0, tzinfo=timezone.utc
        ),
        checksum="processed-checksum",
        flagged_return_decision="retain",
    )

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    assert metadata["source"] == {
        "snapshot_id": "raw_snapshot",
        "market_data_file": "market_data.csv",
        "market_data_sha256": "raw-checksum",
    }
    assert metadata["calculation"]["implicit_fill"] is False
    assert metadata["output"]["rows"] == 2
    assert metadata["output"]["file"]["sha256"] == (
        "processed-checksum"
    )
    assert metadata["validation"]["flagged_returns"] == [
        {
            "date": "2025-01-03",
            "ticker": "AMZN",
            "return": 0.30,
            "decision": "retain",
        },
        {
            "date": "2025-01-06",
            "ticker": "GOOGL",
            "return": -0.25,
            "decision": "retain",
        },
    ]


def test_load_processed_returns_verifies_checksum(
    tmp_path: Path,
) -> None:
    """Verify that a valid processed-return snapshot can be loaded."""

    snapshot_directory = tmp_path / "processed_snapshot"
    snapshot_directory.mkdir()

    expected_returns = pd.DataFrame(
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
    expected_returns.index.name = "Date"

    data_path = snapshot_directory / "asset_returns.csv"
    expected_returns.to_csv(data_path)

    metadata = {
        "output": {
            "file": {
                "name": data_path.name,
                "sha256": calculate_file_sha256(data_path),
            }
        }
    }

    metadata_path = snapshot_directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    loaded_returns, loaded_metadata = (
        returns_module.load_processed_returns(
            snapshot_directory
        )
    )

    pd.testing.assert_frame_equal(
        loaded_returns,
        expected_returns,
    )
    assert loaded_metadata == metadata


def test_load_processed_returns_rejects_checksum_mismatch(
    tmp_path: Path,
) -> None:
    """Verify that modified return data is rejected."""

    snapshot_directory = tmp_path / "processed_snapshot"
    snapshot_directory.mkdir()

    data_path = snapshot_directory / "asset_returns.csv"
    data_path.write_text(
        "Date,GOOGL\n2025-01-03,0.05\n",
        encoding="utf-8",
    )

    metadata = {
        "output": {
            "file": {
                "name": data_path.name,
                "sha256": "incorrect-checksum",
            }
        }
    }

    metadata_path = snapshot_directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="checksum",
    ):
        returns_module.load_processed_returns(
            snapshot_directory
        )
