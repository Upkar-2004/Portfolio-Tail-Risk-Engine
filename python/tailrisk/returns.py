"""Asset and portfolio return calculations."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from tailrisk.data import calculate_file_sha256

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


def save_processed_metadata(
    data_path: str | Path,
    source_snapshot_id: str,
    source_metadata: dict[str, Any],
    returns: pd.DataFrame,
    flags: pd.DataFrame,
    threshold: float,
    processed_at: datetime,
    checksum: str,
    flagged_return_decision: str,
) -> Path:
    """Save lineage, calculation, and validation metadata for returns."""

    if processed_at.utcoffset() is None:
        raise ValueError(
            "Processing timestamp must include a time zone."
        )

    flagged_returns = []
    for date, ticker_flags in flags.iterrows():
        for ticker in flags.columns[ticker_flags]:
            flagged_returns.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": str(ticker),
                    "return": float(returns.loc[date, ticker]),
                    "decision": flagged_return_decision,
                }
            )

    metadata = {
        "schema_version": 1,
        "processed_at_utc": processed_at.astimezone(
            timezone.utc
        ).isoformat(),
        "source": {
            "snapshot_id": source_snapshot_id,
            "market_data_file": source_metadata["files"][
                "market_data"
            ]["name"],
            "market_data_sha256": source_metadata["files"][
                "market_data"
            ]["sha256"],
        },
        "calculation": {
            "price_field": "Adj Close",
            "return_type": "simple",
            "formula": "P_t / P_t-1 - 1",
            "implicit_fill": False,
        },
        "output": {
            "rows": len(returns),
            "columns": len(returns.columns),
            "first_date": returns.index.min().date().isoformat(),
            "last_date": returns.index.max().date().isoformat(),
            "tickers": [str(ticker) for ticker in returns.columns],
            "missing_returns": {
                str(ticker): int(count)
                for ticker, count in returns.isna().sum().items()
            },
            "file": {
                "name": Path(data_path).name,
                "sha256": checksum,
            },
        },
        "validation": {
            "large_return_threshold": threshold,
            "flagged_returns": flagged_returns,
        },
    }

    metadata_path = Path(data_path).parent / "metadata.json"
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return metadata_path


def load_processed_returns(
    snapshot_directory: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load processed asset returns after verifying their checksum."""

    snapshot_path = Path(snapshot_directory)
    metadata_path = snapshot_path / "metadata.json"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    file_metadata = metadata["output"]["file"]
    data_path = snapshot_path / file_metadata["name"]

    expected_checksum = file_metadata["sha256"]
    actual_checksum = calculate_file_sha256(data_path)

    if actual_checksum != expected_checksum:
        raise ValueError(
            "Processed asset-return snapshot checksum does not match."
        )

    asset_returns = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=[0],
    )

    return asset_returns, metadata
