"""Tests for rolling backtests and forecast evaluation."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from tailrisk.backtesting import (
    create_forecast_schedule,
    load_backtest_snapshot,
    save_backtest_forecasts,
    save_backtest_metadata,
    validate_backtest_forecasts,
)
from tailrisk.data import calculate_file_sha256


def test_create_forecast_schedule_uses_only_prior_returns() -> None:
    """Verify that each forecast excludes its realized return."""

    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )
    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04, 0.05],
        index=dates,
        name="portfolio_return",
    )

    result = create_forecast_schedule(
        portfolio_returns,
        estimation_window=3,
    )

    expected = pd.DataFrame(
        {
            "estimation_start_date": dates[:2],
            "estimation_end_date": dates[2:4],
            "realized_return": [-0.04, 0.05],
            "realized_loss": [0.04, -0.05],
        },
        index=pd.DatetimeIndex(
            dates[3:],
            name="forecast_date",
        ),
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )




def _create_valid_backtest_forecasts(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[float],
]:
    forecast_dates = pd.to_datetime(
        [
            "2025-01-07",
            "2025-01-08",
        ]
    )
    confidence_levels = [0.95, 0.99]

    forecast_schedule = pd.DataFrame(
        {
            "estimation_start_date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "estimation_end_date": pd.to_datetime(
                [
                    "2025-01-06",
                    "2025-01-07",
                ]
            ),
            "realized_return": [-0.03, 0.01],
            "realized_loss": [0.03, -0.01],
        },
        index=pd.DatetimeIndex(
            forecast_dates,
            name="forecast_date",
        ),
    )

    forecast_index = pd.MultiIndex.from_product(
        [
            forecast_dates,
            confidence_levels,
        ],
        names=[
            "forecast_date",
            "confidence_level",
        ],
    )

    forecasts = pd.DataFrame(
        {
            "estimation_start_date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                ]
            ),
            "estimation_end_date": pd.to_datetime(
                [
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                ]
            ),
            "realized_return": [
                -0.03,
                -0.03,
                0.01,
                0.01,
            ],
            "realized_loss": [
                0.03,
                0.03,
                -0.01,
                -0.01,
            ],
            "value_at_risk": [
                0.02,
                0.03,
                0.018,
                0.028,
            ],
            "expected_shortfall": [
                0.025,
                0.035,
                0.023,
                0.033,
            ],
            "var_exceedance": [
                True,
                False,
                False,
                False,
            ],
        },
        index=forecast_index,
    )

    return (
        forecasts,
        forecast_schedule,
        confidence_levels,
    )





@pytest.mark.parametrize(
    "estimation_window",
    [
        0,
        -1,
        True,
        2.5,
    ],
)
def test_create_forecast_schedule_rejects_invalid_window(
    estimation_window: object,
) -> None:
    """Verify that estimation windows must be positive integers."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=estimation_window,
        )


def test_create_forecast_schedule_requires_out_of_sample_return() -> None:
    """Verify that the schedule requires a return after its first window."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="more observations",
    ):
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=4,
        )


@pytest.mark.parametrize(
    ("dates", "error_message"),
    [
        (
            [
                "2025-01-02",
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
            ],
            "unique",
        ),
        (
            [
                "2025-01-03",
                "2025-01-02",
                "2025-01-06",
                "2025-01-07",
            ],
            "increasing date",
        ),
        (
            [
                "2025-01-02",
                None,
                "2025-01-06",
                "2025-01-07",
            ],
            "must not be missing",
        ),
    ],
)
def test_create_forecast_schedule_rejects_invalid_dates(
    dates: list[str | None],
    error_message: str,
) -> None:
    """Verify that forecast inputs require valid ordered unique dates."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=pd.to_datetime(dates),
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=2,
        )


@pytest.mark.parametrize(
    "nonfinite_return",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_create_forecast_schedule_rejects_nonfinite_returns(
    nonfinite_return: float,
) -> None:
    """Verify that forecast scheduling rejects nonfinite returns."""

    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03, nonfinite_return],
        index=pd.date_range(
            "2025-01-02",
            periods=4,
            freq="B",
        ),
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        create_forecast_schedule(
            portfolio_returns,
            estimation_window=2,
        )



def test_validate_backtest_forecasts_accepts_valid_results(
) -> None:
    """Verify that a complete and consistent forecast table is accepted."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    validate_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
    )


def test_validate_backtest_forecasts_requires_columns(
) -> None:
    """Verify that every shared backtest result column is required."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts = forecasts.drop(
        columns="expected_shortfall"
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        validate_backtest_forecasts(
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            confidence_levels=confidence_levels,
        )


def test_validate_backtest_forecasts_requires_complete_grid(
) -> None:
    """Verify that every forecast date has every confidence level."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts = forecasts.drop(
        index=(pd.Timestamp("2025-01-08"), 0.99)
    )

    with pytest.raises(
        ValueError,
        match="complete ordered grid",
    ):
        validate_backtest_forecasts(
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            confidence_levels=confidence_levels,
        )



def test_validate_backtest_forecasts_rejects_nonfinite_values(
) -> None:
    """Verify that numerical forecast results must be finite."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.95),
        "value_at_risk",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="finite numbers",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_requires_schedule_alignment(
) -> None:
    """Verify that realized outcomes agree with the shared schedule."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.99),
        "realized_loss",
    ] = 0.031

    with pytest.raises(
        ValueError,
        match="must match the forecast schedule",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_invalid_loss_sign(
) -> None:
    """Verify that realized loss is the negative of realized return."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecast_date = pd.Timestamp("2025-01-07")

    forecast_schedule.loc[
        forecast_date,
        "realized_loss",
    ] = 0.02
    forecasts.loc[
        (forecast_date, slice(None)),
        "realized_loss",
    ] = 0.02

    with pytest.raises(
        ValueError,
        match="negative of its realized return",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_es_below_var(
) -> None:
    """Verify that Expected Shortfall cannot be smaller than VaR."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.95),
        "expected_shortfall",
    ] = 0.019

    with pytest.raises(
        ValueError,
        match="Expected Shortfall",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_wrong_exceedance(
) -> None:
    """Verify that exceedance flags follow realized loss versus VaR."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts.loc[
        (pd.Timestamp("2025-01-07"), 0.95),
        "var_exceedance",
    ] = False

    with pytest.raises(
        ValueError,
        match="exceedance indicators",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_validate_backtest_forecasts_rejects_lookahead(
) -> None:
    """Verify that each estimation window ends before its forecast date."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecast_date = pd.Timestamp("2025-01-07")

    forecast_schedule.loc[
        forecast_date,
        "estimation_end_date",
    ] = forecast_date
    forecasts.loc[
        (forecast_date, slice(None)),
        "estimation_end_date",
    ] = forecast_date

    with pytest.raises(
        ValueError,
        match="must end before",
    ):
        validate_backtest_forecasts(
            forecasts,
            forecast_schedule,
            confidence_levels,
        )


def test_save_backtest_forecasts_creates_versioned_csv(
    tmp_path: Path,
) -> None:
    """Verify that valid forecasts are saved in a new snapshot directory."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts_path = save_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        output_root=tmp_path,
        snapshot_id="test_rolling_gaussian",
    )

    assert forecasts_path == (
        tmp_path
        / "test_rolling_gaussian"
        / "forecasts.csv"
    )
    assert forecasts_path.is_file()

    saved_forecasts = pd.read_csv(
        forecasts_path
    )

    assert len(saved_forecasts) == len(forecasts)
    assert list(saved_forecasts.columns[:2]) == [
        "forecast_date",
        "confidence_level",
    ]


def test_save_backtest_forecasts_does_not_overwrite_snapshot(
    tmp_path: Path,
) -> None:
    """Verify that an existing backtest snapshot cannot be overwritten."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    save_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        output_root=tmp_path,
        snapshot_id="test_rolling_gaussian",
    )

    with pytest.raises(FileExistsError):
        save_backtest_forecasts(
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            confidence_levels=confidence_levels,
            output_root=tmp_path,
            snapshot_id="test_rolling_gaussian",
        )


def test_save_backtest_forecasts_rejects_invalid_results(
    tmp_path: Path,
) -> None:
    """Verify that invalid forecasts are rejected before files are created."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()

    forecasts = forecasts.drop(
        columns="expected_shortfall"
    )

    snapshot_directory = (
        tmp_path / "invalid_snapshot"
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        save_backtest_forecasts(
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            confidence_levels=confidence_levels,
            output_root=tmp_path,
            snapshot_id="invalid_snapshot",
        )

    assert not snapshot_directory.exists()


def _create_backtest_metadata_inputs(
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Create configuration and lineage metadata for persistence tests."""

    config = {
        "experiment": {
            "name": "test_experiment",
        },
        "forecasting": {
            "horizon_sessions": 1,
            "estimation_window": 3,
            "confidence_levels": [0.95, 0.99],
        },
        "simulation": {
            "scenario_count": 1_000,
            "random_seed": 20261002,
            "reuse_standard_normal_shocks": True,
        },
    }
    returns_metadata = {
        "output": {
            "file": {
                "name": "asset_returns.csv",
                "sha256": "returns-checksum",
            }
        }
    }
    portfolio_metadata = {
        "source": {
            "asset_returns_sha256": "returns-checksum",
        },
        "output": {
            "files": {
                "portfolio_daily": {
                    "name": "portfolio_daily.csv",
                    "sha256": "portfolio-daily-checksum",
                },
                "beginning_weights": {
                    "name": "beginning_weights.csv",
                    "sha256": "weights-checksum",
                },
            }
        },
    }

    return (
        config,
        returns_metadata,
        portfolio_metadata,
    )


def test_save_backtest_metadata_records_settings_and_lineage(
    tmp_path: Path,
) -> None:
    """Verify that metadata records settings, lineage, and the CSV checksum."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()
    (
        config,
        returns_metadata,
        portfolio_metadata,
    ) = _create_backtest_metadata_inputs()

    forecasts_path = save_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        output_root=tmp_path,
        snapshot_id="test_rolling_gaussian",
    )
    generated_at = datetime(
        2026,
        10,
        4,
        18,
        30,
        tzinfo=timezone.utc,
    )

    metadata_path = save_backtest_metadata(
        forecasts_path=forecasts_path,
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        config=config,
        model_name="rolling_gaussian_monte_carlo",
        source_returns_snapshot_id="returns-snapshot",
        source_returns_metadata=returns_metadata,
        source_portfolio_snapshot_id="portfolio-snapshot",
        source_portfolio_metadata=portfolio_metadata,
        generated_at=generated_at,
    )

    assert metadata_path == (
        forecasts_path.parent / "metadata.json"
    )
    assert metadata_path.is_file()

    with metadata_path.open(
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["schema_version"] == 1
    assert metadata["experiment"] == "test_experiment"
    assert metadata["generated_at_utc"] == (
        "2026-10-04T18:30:00+00:00"
    )
    assert metadata["model"]["name"] == (
        "rolling_gaussian_monte_carlo"
    )
    assert metadata["model"]["confidence_levels"] == [
        0.95,
        0.99,
    ]
    assert metadata["simulation"] == {
        "scenario_count": 1_000,
        "random_seed": 20261002,
        "reuse_standard_normal_shocks": True,
    }
    assert metadata["source"]["asset_returns"] == {
        "snapshot_id": "returns-snapshot",
        "file": "asset_returns.csv",
        "sha256": "returns-checksum",
    }
    assert metadata["source"]["portfolio"] == {
        "snapshot_id": "portfolio-snapshot",
        "portfolio_daily_file": "portfolio_daily.csv",
        "portfolio_daily_sha256": (
            "portfolio-daily-checksum"
        ),
        "beginning_weights_file": "beginning_weights.csv",
        "beginning_weights_sha256": "weights-checksum",
    }
    assert metadata["output"]["rows"] == 4
    assert metadata["output"]["forecast_dates"] == 2
    assert metadata["output"]["first_forecast_date"] == (
        "2025-01-07"
    )
    assert metadata["output"]["last_forecast_date"] == (
        "2025-01-08"
    )
    assert metadata["output"]["exceedance_counts"] == {
        "0.95": 1,
        "0.99": 0,
    }
    assert metadata["output"]["file"] == {
        "name": "forecasts.csv",
        "sha256": calculate_file_sha256(
            forecasts_path
        ),
    }


def test_save_backtest_metadata_requires_timezone(
    tmp_path: Path,
) -> None:
    """Verify that backtest metadata timestamps include a time zone."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()
    (
        config,
        returns_metadata,
        portfolio_metadata,
    ) = _create_backtest_metadata_inputs()
    forecasts_path = save_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        output_root=tmp_path,
        snapshot_id="test_rolling_gaussian",
    )

    with pytest.raises(
        ValueError,
        match="include a time zone",
    ):
        save_backtest_metadata(
            forecasts_path=forecasts_path,
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            config=config,
            model_name="rolling_gaussian_monte_carlo",
            source_returns_snapshot_id="returns-snapshot",
            source_returns_metadata=returns_metadata,
            source_portfolio_snapshot_id="portfolio-snapshot",
            source_portfolio_metadata=portfolio_metadata,
            generated_at=datetime(2026, 10, 4, 18, 30),
        )


def test_save_backtest_metadata_rejects_mismatched_lineage(
    tmp_path: Path,
) -> None:
    """Verify that portfolio and return snapshots share one return source."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()
    (
        config,
        returns_metadata,
        portfolio_metadata,
    ) = _create_backtest_metadata_inputs()
    portfolio_metadata["source"][
        "asset_returns_sha256"
    ] = "different-checksum"

    forecasts_path = save_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        output_root=tmp_path,
        snapshot_id="test_rolling_gaussian",
    )

    with pytest.raises(
        ValueError,
        match="do not share",
    ):
        save_backtest_metadata(
            forecasts_path=forecasts_path,
            forecasts=forecasts,
            forecast_schedule=forecast_schedule,
            config=config,
            model_name="rolling_gaussian_monte_carlo",
            source_returns_snapshot_id="returns-snapshot",
            source_returns_metadata=returns_metadata,
            source_portfolio_snapshot_id="portfolio-snapshot",
            source_portfolio_metadata=portfolio_metadata,
            generated_at=datetime.now(timezone.utc),
        )


def _save_complete_test_backtest_snapshot(
    tmp_path: Path,
) -> tuple[
    Path,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Save one complete backtest snapshot for loader tests."""

    (
        forecasts,
        forecast_schedule,
        confidence_levels,
    ) = _create_valid_backtest_forecasts()
    (
        config,
        returns_metadata,
        portfolio_metadata,
    ) = _create_backtest_metadata_inputs()
    snapshot_id = "test_rolling_gaussian"

    forecasts_path = save_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
        output_root=tmp_path,
        snapshot_id=snapshot_id,
    )
    save_backtest_metadata(
        forecasts_path=forecasts_path,
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        config=config,
        model_name="rolling_gaussian_monte_carlo",
        source_returns_snapshot_id="returns-snapshot",
        source_returns_metadata=returns_metadata,
        source_portfolio_snapshot_id="portfolio-snapshot",
        source_portfolio_metadata=portfolio_metadata,
        generated_at=datetime.now(timezone.utc),
    )

    return (
        tmp_path / snapshot_id,
        forecasts,
        forecast_schedule,
    )


def test_load_backtest_snapshot_round_trips_forecasts(
    tmp_path: Path,
) -> None:
    """Verify that saved forecasts load with their index and values intact."""

    (
        snapshot_directory,
        forecasts,
        forecast_schedule,
    ) = _save_complete_test_backtest_snapshot(
        tmp_path
    )

    loaded_forecasts, metadata = load_backtest_snapshot(
        snapshot_directory=snapshot_directory,
        forecast_schedule=forecast_schedule,
    )

    pd.testing.assert_frame_equal(
        loaded_forecasts,
        forecasts,
    )
    assert metadata["model"]["name"] == (
        "rolling_gaussian_monte_carlo"
    )


def test_load_backtest_snapshot_rejects_modified_csv(
    tmp_path: Path,
) -> None:
    """Verify that checksum validation detects modified forecast data."""

    (
        snapshot_directory,
        _,
        forecast_schedule,
    ) = _save_complete_test_backtest_snapshot(
        tmp_path
    )
    forecasts_path = (
        snapshot_directory / "forecasts.csv"
    )

    with forecasts_path.open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write("modified\n")

    with pytest.raises(
        ValueError,
        match="checksum does not match",
    ):
        load_backtest_snapshot(
            snapshot_directory=snapshot_directory,
            forecast_schedule=forecast_schedule,
        )


def test_load_backtest_snapshot_rejects_incorrect_metadata_counts(
    tmp_path: Path,
) -> None:
    """Verify that loader checks the metadata summary against the CSV."""

    (
        snapshot_directory,
        _,
        forecast_schedule,
    ) = _save_complete_test_backtest_snapshot(
        tmp_path
    )
    metadata_path = snapshot_directory / "metadata.json"

    with metadata_path.open(
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    metadata["output"]["rows"] = 999

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    with pytest.raises(
        ValueError,
        match="row count",
    ):
        load_backtest_snapshot(
            snapshot_directory=snapshot_directory,
            forecast_schedule=forecast_schedule,
        )
