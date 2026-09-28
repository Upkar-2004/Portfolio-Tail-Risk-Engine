"""Tests for portfolio construction and accounting."""

import pandas as pd
import pytest
import json


import tailrisk.portfolio as portfolio
from datetime import datetime, timezone
from pathlib import Path
from tailrisk.data import calculate_file_sha256
from tailrisk.portfolio import (
    calculate_portfolio_step,
    create_equal_weights,
)


def test_create_equal_weights_allocates_investable_value() -> None:
    tickers = ["GOOGL", "AMZN", "WMT"]

    weights = create_equal_weights(
        tickers=tickers,
        cash_weight=0.10,
    )

    expected = pd.Series(
        [0.30, 0.30, 0.30],
        index=tickers,
        name="weight",
    )

    pd.testing.assert_series_equal(
        weights,
        expected,
    )
    assert weights.sum() == pytest.approx(0.90)


def test_create_equal_weights_rejects_empty_universe() -> None:
    with pytest.raises(
        ValueError,
        match="At least one ticker",
    ):
        create_equal_weights(
            tickers=[],
            cash_weight=0.0,
        )


def test_create_equal_weights_rejects_duplicate_tickers() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        create_equal_weights(
            tickers=["GOOGL", "GOOGL"],
            cash_weight=0.0,
        )


@pytest.mark.parametrize(
    "cash_weight",
    [
        -0.10,
        1.00,
        float("nan"),
    ],
)
def test_create_equal_weights_rejects_invalid_cash_weight(
    cash_weight: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Cash weight",
    ):
        create_equal_weights(
            tickers=["GOOGL", "AMZN"],
            cash_weight=cash_weight,
        )


def test_calculate_portfolio_step_matches_manual_example() -> None:
    beginning_weights = pd.Series(
        [0.50, 0.30, 0.20],
        index=["A", "B", "C"],
        name="weight",
    )
    asset_returns = pd.Series(
        [-0.04, 0.02, -0.01],
        index=["A", "B", "C"],
    )

    result = calculate_portfolio_step(
        beginning_value=1_000_000.0,
        beginning_weights=beginning_weights,
        asset_returns=asset_returns,
    )

    assert result.portfolio_return == pytest.approx(-0.016)
    assert result.pnl == pytest.approx(-16_000.0)
    assert result.loss == pytest.approx(16_000.0)
    assert result.ending_value == pytest.approx(984_000.0)

    expected_weights = pd.Series(
        [
            480_000.0 / 984_000.0,
            306_000.0 / 984_000.0,
            198_000.0 / 984_000.0,
        ],
        index=["A", "B", "C"],
        name="weight",
    )
    pd.testing.assert_series_equal(
        result.ending_weights,
        expected_weights,
    )
    assert result.ending_cash_weight == pytest.approx(0.0)


def test_calculate_portfolio_step_keeps_uninvested_weight_as_cash() -> None:
    beginning_weights = pd.Series(
        [0.45, 0.45],
        index=["A", "B"],
        name="weight",
    )
    asset_returns = pd.Series(
        [0.10, 0.00],
        index=["A", "B"],
    )

    result = calculate_portfolio_step(
        beginning_value=1_000.0,
        beginning_weights=beginning_weights,
        asset_returns=asset_returns,
    )

    assert result.ending_value == pytest.approx(1_045.0)
    assert result.portfolio_return == pytest.approx(0.045)
    assert result.ending_cash_weight == pytest.approx(100.0 / 1_045.0)
    assert (
        result.ending_weights.sum()
        + result.ending_cash_weight
    ) == pytest.approx(1.0)


def test_calculate_portfolio_step_aligns_assets_by_ticker() -> None:
    beginning_weights = pd.Series(
        [0.60, 0.40],
        index=["GOOGL", "AMZN"],
        name="weight",
    )
    asset_returns = pd.Series(
        [0.10, -0.10],
        index=["AMZN", "GOOGL"],
    )

    result = calculate_portfolio_step(
        beginning_value=1_000.0,
        beginning_weights=beginning_weights,
        asset_returns=asset_returns,
    )

    assert result.portfolio_return == pytest.approx(-0.02)


def test_run_portfolio_path_rebalances_after_month_end() -> None:
    """Drift during January, then restore targets before February."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.10, 0.00, 0.00],
            "B": [0.00, 0.00, 0.10],
        },
        index=pd.to_datetime(
            [
                "2025-01-30",
                "2025-01-31",
                "2025-02-03",
            ]
        ),
    )
    target_weights = pd.Series(
        [0.50, 0.50],
        index=["A", "B"],
        name="weight",
    )

    result = portfolio.run_portfolio_path(
        asset_returns=asset_returns,
        initial_value=1_000.0,
        target_weights=target_weights,
        rebalancing_frequency="monthly",
    )

    january_30 = pd.Timestamp("2025-01-30")
    january_31 = pd.Timestamp("2025-01-31")
    february_3 = pd.Timestamp("2025-02-03")

    # A's January gain makes it overweight before the rebalance.
    assert result.pre_rebalance_weights.loc[january_31, "A"] == pytest.approx(
        550.0 / 1_050.0
    )
    assert result.daily.loc[january_30, "portfolio_return"] == pytest.approx(
        0.05
    )
    assert not bool(
        result.daily.loc[january_30, "rebalanced_after_close"]
    )

    # The final January session is rebalanced after its return is applied.
    assert bool(
        result.daily.loc[january_31, "rebalanced_after_close"]
    )
    assert result.daily.loc[january_31, "turnover"] == pytest.approx(
        1.0 / 42.0
    )

    # February therefore starts at 50/50 and earns a 5% portfolio return.
    assert result.beginning_weights.loc[february_3, "A"] == pytest.approx(0.50)
    assert result.beginning_weights.loc[february_3, "B"] == pytest.approx(0.50)
    assert result.daily.loc[february_3, "beginning_value"] == pytest.approx(
        1_050.0
    )
    assert result.daily.loc[february_3, "portfolio_return"] == pytest.approx(
        0.05
    )
    assert result.daily.loc[february_3, "ending_value"] == pytest.approx(
        1_102.50
    )



def test_save_portfolio_path_creates_csv_files(
    tmp_path: Path,
) -> None:
    """Verify that all portfolio-path tables are saved together."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.10, 0.00],
            "B": [0.00, 0.10],
        },
        index=pd.to_datetime(
            [
                "2025-01-30",
                "2025-01-31",
            ]
        ),
    )
    target_weights = pd.Series(
        [0.50, 0.50],
        index=["A", "B"],
        name="weight",
    )

    result = portfolio.run_portfolio_path(
        asset_returns=asset_returns,
        initial_value=1_000.0,
        target_weights=target_weights,
        rebalancing_frequency="monthly",
    )

    files = portfolio.save_portfolio_path(
        result=result,
        output_root=tmp_path,
        snapshot_id="test_portfolio",
    )

    expected_directory = tmp_path / "test_portfolio"

    assert files.daily == (
        expected_directory / "portfolio_daily.csv"
    )
    assert files.beginning_weights == (
        expected_directory / "beginning_weights.csv"
    )
    assert files.pre_rebalance_weights == (
        expected_directory / "pre_rebalance_weights.csv"
    )

    saved_daily = pd.read_csv(
        files.daily,
        index_col="Date",
        parse_dates=["Date"],
    )
    saved_beginning_weights = pd.read_csv(
        files.beginning_weights,
        index_col="Date",
        parse_dates=["Date"],
    )
    saved_pre_rebalance_weights = pd.read_csv(
        files.pre_rebalance_weights,
        index_col="Date",
        parse_dates=["Date"],
    )

    expected_daily = result.daily.copy()
    expected_daily.index.name = "Date"

    expected_beginning_weights = (
        result.beginning_weights.copy()
    )
    expected_beginning_weights.index.name = "Date"

    expected_pre_rebalance_weights = (
        result.pre_rebalance_weights.copy()
    )
    expected_pre_rebalance_weights.index.name = "Date"

    pd.testing.assert_frame_equal(
        saved_daily,
        expected_daily,
    )
    pd.testing.assert_frame_equal(
        saved_beginning_weights,
        expected_beginning_weights,
    )
    pd.testing.assert_frame_equal(
        saved_pre_rebalance_weights,
        expected_pre_rebalance_weights,
    )



def test_save_portfolio_metadata_records_research_lineage(
    tmp_path: Path,
) -> None:
    """Verify portfolio metadata records inputs, settings, and outputs."""

    asset_returns = pd.DataFrame(
        {
            "A": [0.10, 0.00, 0.00],
            "B": [0.00, 0.00, 0.10],
        },
        index=pd.to_datetime(
            [
                "2025-01-30",
                "2025-01-31",
                "2025-02-03",
            ]
        ),
    )
    target_weights = pd.Series(
        [0.50, 0.50],
        index=["A", "B"],
        name="weight",
    )

    result = portfolio.run_portfolio_path(
        asset_returns=asset_returns,
        initial_value=1_000.0,
        target_weights=target_weights,
        rebalancing_frequency="monthly",
    )

    files = portfolio.save_portfolio_path(
        result=result,
        output_root=tmp_path,
        snapshot_id="test_portfolio",
    )

    source_metadata = {
        "output": {
            "file": {
                "name": "asset_returns.csv",
                "sha256": "source-return-checksum",
            }
        }
    }

    portfolio_config = {
        "weighting_method": "equal",
        "rebalancing_frequency": "monthly",
        "long_only": True,
        "cash_weight": 0.0,
        "initial_value": 1_000.0,
        "transaction_cost_bps": 0.0,
    }

    processed_at = datetime(
        2025,
        2,
        4,
        12,
        0,
        tzinfo=timezone.utc,
    )

    metadata_path = portfolio.save_portfolio_metadata(
        files=files,
        source_snapshot_id="processed_returns_snapshot",
        source_metadata=source_metadata,
        portfolio_config=portfolio_config,
        target_weights=target_weights,
        result=result,
        processed_at=processed_at,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["source"] == {
        "snapshot_id": "processed_returns_snapshot",
        "asset_returns_file": "asset_returns.csv",
        "asset_returns_sha256": "source-return-checksum",
    }

    assert metadata["portfolio"]["weighting_method"] == "equal"
    assert metadata["portfolio"]["rebalancing_frequency"] == "monthly"
    assert metadata["portfolio"]["target_weights"] == {
        "A": 0.50,
        "B": 0.50,
    }

    assert metadata["output"]["rows"] == 3
    assert metadata["output"]["first_date"] == "2025-01-30"
    assert metadata["output"]["last_date"] == "2025-02-03"
    assert metadata["output"]["rebalance_count"] == 1
    assert metadata["output"]["ending_value"] == pytest.approx(
        1_102.50
    )

    daily_file_metadata = metadata["output"]["files"][
        "portfolio_daily"
    ]

    assert daily_file_metadata["name"] == (
        "portfolio_daily.csv"
    )
    assert daily_file_metadata["sha256"] == (
        calculate_file_sha256(files.daily)
    )


def test_load_portfolio_path_verifies_and_loads_snapshot(
    tmp_path: Path,
) -> None:
    """Verify that a valid portfolio snapshot can be loaded."""

    snapshot_directory = tmp_path / "portfolio_snapshot"
    snapshot_directory.mkdir()

    dates = pd.to_datetime(
        [
            "2025-01-30",
            "2025-01-31",
        ]
    )

    expected_daily = pd.DataFrame(
        {
            "beginning_value": [1_000.0, 1_050.0],
            "portfolio_return": [0.05, 0.00],
            "pnl": [50.0, 0.0],
            "loss": [-50.0, 0.0],
            "ending_value": [1_050.0, 1_050.0],
            "rebalanced_after_close": [False, False],
            "turnover": [0.0, 0.0],
        },
        index=dates,
    )

    expected_beginning_weights = pd.DataFrame(
        {
            "A": [0.50, 550.0 / 1_050.0],
            "B": [0.50, 500.0 / 1_050.0],
        },
        index=dates,
    )

    expected_pre_rebalance_weights = pd.DataFrame(
        {
            "A": [550.0 / 1_050.0, 550.0 / 1_050.0],
            "B": [500.0 / 1_050.0, 500.0 / 1_050.0],
        },
        index=dates,
    )

    for table in (
        expected_daily,
        expected_beginning_weights,
        expected_pre_rebalance_weights,
    ):
        table.index.name = "Date"

    paths = {
        "portfolio_daily": (
            snapshot_directory / "portfolio_daily.csv"
        ),
        "beginning_weights": (
            snapshot_directory / "beginning_weights.csv"
        ),
        "pre_rebalance_weights": (
            snapshot_directory / "pre_rebalance_weights.csv"
        ),
    }

    expected_daily.to_csv(paths["portfolio_daily"])
    expected_beginning_weights.to_csv(
        paths["beginning_weights"]
    )
    expected_pre_rebalance_weights.to_csv(
        paths["pre_rebalance_weights"]
    )

    metadata = {
        "output": {
            "files": {
                name: {
                    "name": path.name,
                    "sha256": calculate_file_sha256(path),
                }
                for name, path in paths.items()
            }
        }
    }

    (snapshot_directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    loaded_path, loaded_metadata = (
        portfolio.load_portfolio_path(
            snapshot_directory
        )
    )

    pd.testing.assert_frame_equal(
        loaded_path.daily,
        expected_daily,
    )
    pd.testing.assert_frame_equal(
        loaded_path.beginning_weights,
        expected_beginning_weights,
    )
    pd.testing.assert_frame_equal(
        loaded_path.pre_rebalance_weights,
        expected_pre_rebalance_weights,
    )
    assert loaded_metadata == metadata




@pytest.mark.parametrize(
    "tampered_file_key",
    [
        "portfolio_daily",
        "beginning_weights",
        "pre_rebalance_weights",
    ],
)
def test_load_portfolio_path_rejects_checksum_mismatch(
    tmp_path: Path,
    tampered_file_key: str,
) -> None:
    """Verify that modified portfolio files are rejected."""

    snapshot_directory = tmp_path / "portfolio_snapshot"
    snapshot_directory.mkdir()

    paths = {
        "portfolio_daily": (
            snapshot_directory / "portfolio_daily.csv"
        ),
        "beginning_weights": (
            snapshot_directory / "beginning_weights.csv"
        ),
        "pre_rebalance_weights": (
            snapshot_directory / "pre_rebalance_weights.csv"
        ),
    }

    paths["portfolio_daily"].write_text(
        (
            "Date,beginning_value,portfolio_return,pnl,loss,"
            "ending_value,rebalanced_after_close,turnover\n"
            "2025-01-30,1000.0,0.05,50.0,-50.0,"
            "1050.0,False,0.0\n"
        ),
        encoding="utf-8",
    )
    paths["beginning_weights"].write_text(
        "Date,A,B\n2025-01-30,0.5,0.5\n",
        encoding="utf-8",
    )
    paths["pre_rebalance_weights"].write_text(
        "Date,A,B\n2025-01-30,0.5238,0.4762\n",
        encoding="utf-8",
    )

    metadata = {
        "output": {
            "files": {
                name: {
                    "name": path.name,
                    "sha256": calculate_file_sha256(path),
                }
                for name, path in paths.items()
            }
        }
    }

    (snapshot_directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    # Simulate accidental editing or corruption after the snapshot
    # checksums have already been recorded.
    paths[tampered_file_key].write_text(
        "modified file contents\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="checksum",
    ):
        portfolio.load_portfolio_path(
            snapshot_directory
        )



@pytest.mark.parametrize(
    ("mismatch_type", "expected_message"),
    [
        ("dates", "same dates"),
        ("tickers", "same tickers"),
    ],
)
def test_load_portfolio_path_rejects_inconsistent_tables(
    tmp_path: Path,
    mismatch_type: str,
    expected_message: str,
) -> None:
    """Verify that portfolio tables must describe one portfolio path."""

    snapshot_directory = tmp_path / "portfolio_snapshot"
    snapshot_directory.mkdir()

    dates = pd.to_datetime(["2025-01-30"])

    daily = pd.DataFrame(
        {
            "portfolio_return": [0.0],
        },
        index=dates,
    )
    beginning_weights = pd.DataFrame(
        {
            "A": [0.50],
            "B": [0.50],
        },
        index=dates,
    )
    pre_rebalance_weights = beginning_weights.copy()

    if mismatch_type == "dates":
        pre_rebalance_weights.index = pd.to_datetime(
            ["2025-01-31"]
        )
    else:
        pre_rebalance_weights.columns = [
            "A",
            "C",
        ]

    tables = {
        "portfolio_daily": daily,
        "beginning_weights": beginning_weights,
        "pre_rebalance_weights": pre_rebalance_weights,
    }

    paths = {
        name: snapshot_directory / f"{name}.csv"
        for name in tables
    }

    for name, table in tables.items():
        table.index.name = "Date"
        table.to_csv(paths[name])

    metadata = {
        "output": {
            "files": {
                name: {
                    "name": path.name,
                    "sha256": calculate_file_sha256(path),
                }
                for name, path in paths.items()
            }
        }
    }

    (snapshot_directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        portfolio.load_portfolio_path(
            snapshot_directory
        )