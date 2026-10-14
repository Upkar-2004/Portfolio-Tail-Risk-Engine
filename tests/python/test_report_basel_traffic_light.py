"""Tests for the Basel-style traffic-light reporting command."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from tailrisk.data import calculate_file_sha256


def _load_report_module() -> ModuleType:
    """Load the report script without making scripts a package."""

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "report_basel_traffic_light.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_basel_traffic_light",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load the Basel traffic-light report script."
        )

    module = importlib.util.module_from_spec(
        module_spec
    )
    module_spec.loader.exec_module(module)

    return module


report_module = _load_report_module()


def _create_metadata(
    model_name: str,
) -> dict[str, object]:
    """Create minimal verified backtest metadata."""

    return {
        "model": {
            "name": model_name,
            "estimation_window": 504,
            "confidence_levels": [
                0.95,
                0.975,
                0.99,
            ],
        },
        "source": {
            "portfolio": {
                "snapshot_id": "portfolio-snapshot",
            },
        },
        "output": {
            "file": {
                "name": "forecasts.csv",
                "sha256": "forecast-checksum",
            },
        },
    }


def _create_paths_and_summary(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create small compatible path and summary report tables."""

    dates = pd.to_datetime(
        [
            "2025-12-18",
            "2025-12-19",
        ]
    )
    models = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
    ]
    index = pd.MultiIndex.from_product(
        [
            models,
            dates,
        ],
        names=[
            "model",
            "forecast_date",
        ],
    )
    paths = pd.DataFrame(
        {
            "window_start_date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "window_observation_count": [250] * 4,
            "confidence_level": [0.99] * 4,
            "exception_count": [5, 4, 3, 3],
            "traffic_light_zone": [
                "amber",
                "green",
                "green",
                "green",
            ],
        },
        index=index,
    )
    summary = pd.DataFrame(
        {
            "window_count": [2, 2],
            "latest_forecast_date": [
                dates[-1],
                dates[-1],
            ],
            "latest_exception_count": [4, 3],
            "latest_zone": ["green", "green"],
            "maximum_exception_count": [5, 3],
            "maximum_exception_date": [
                dates[0],
                dates[0],
            ],
            "worst_zone": ["amber", "green"],
            "green_window_count": [1, 2],
            "amber_window_count": [1, 0],
            "red_window_count": [0, 0],
            "green_window_rate": [0.5, 1.0],
            "amber_window_rate": [0.5, 0.0],
            "red_window_rate": [0.0, 0.0],
            "first_amber_date": [dates[0], pd.NaT],
            "first_red_date": [pd.NaT, pd.NaT],
        },
        index=pd.Index(
            models,
            name="model",
        ),
    )

    return paths, summary


def test_validate_backtest_metadata_accepts_matching_source(
) -> None:
    """Matching model and experiment metadata are accepted."""

    metadata = _create_metadata(
        "rolling_gaussian_monte_carlo"
    )

    report_module.validate_backtest_metadata(
        metadata=metadata,
        expected_model_name=(
            "rolling_gaussian_monte_carlo"
        ),
        expected_portfolio_snapshot_id=(
            "portfolio-snapshot"
        ),
        expected_estimation_window=504,
        expected_confidence_levels=[
            0.95,
            0.975,
            0.99,
        ],
    )


def test_validate_backtest_metadata_rejects_wrong_model(
) -> None:
    """A source snapshot cannot be assigned to another model."""

    metadata = _create_metadata(
        "ewma_gaussian_monte_carlo"
    )

    with pytest.raises(
        ValueError,
        match="model mismatch",
    ):
        report_module.validate_backtest_metadata(
            metadata=metadata,
            expected_model_name=(
                "rolling_gaussian_monte_carlo"
            ),
            expected_portfolio_snapshot_id=(
                "portfolio-snapshot"
            ),
            expected_estimation_window=504,
            expected_confidence_levels=[
                0.95,
                0.975,
                0.99,
            ],
        )


def test_save_basel_traffic_light_report_writes_verified_files(
    tmp_path: Path,
) -> None:
    """The report saves path, summary, lineage, and checksums."""

    paths, summary = _create_paths_and_summary()
    model_names = list(summary.index)
    source_snapshots = {
        model_name: Path(model_name)
        for model_name in model_names
    }
    source_metadata = {
        model_name: _create_metadata(model_name)
        for model_name in model_names
    }

    (
        paths_path,
        summary_path,
        png_path,
        svg_path,
        metadata_path,
    ) = (
        report_module.save_basel_traffic_light_report(
            paths=paths,
            summary=summary,
            source_snapshots=source_snapshots,
            source_metadata=source_metadata,
            output_root=tmp_path,
            snapshot_id="test_basel_traffic_light",
            generated_at=datetime(
                2026,
                10,
                13,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
        )
    )

    assert paths_path.exists()
    assert summary_path.exists()
    assert png_path.exists()
    assert svg_path.exists()
    assert metadata_path.exists()
    assert len(pd.read_csv(paths_path)) == 4
    assert len(pd.read_csv(summary_path)) == 2

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["report"]["name"] == (
        "basel_traffic_light"
    )
    assert metadata["report"]["confidence_level"] == (
        pytest.approx(0.99)
    )
    assert metadata["report"]["window_observations"] == 250
    assert metadata["report"]["figure"]["thresholds"] == [
        5,
        10,
    ]
    assert metadata["report"]["figure"]["png_dpi"] == 300
    assert metadata["report"]["zones"] == {
        "green": {
            "minimum_exceptions": 0,
            "maximum_exceptions": 4,
        },
        "amber": {
            "minimum_exceptions": 5,
            "maximum_exceptions": 9,
        },
        "red": {
            "minimum_exceptions": 10,
            "maximum_exceptions": 250,
        },
    }
    assert metadata["output"]["path"]["rows"] == 4
    assert metadata["output"]["summary"]["rows"] == 2
    assert metadata[
        "output"
    ]["path"]["file"]["sha256"] == (
        calculate_file_sha256(paths_path)
    )
    assert metadata[
        "output"
    ]["summary"]["file"]["sha256"] == (
        calculate_file_sha256(summary_path)
    )
    assert metadata[
        "output"
    ]["figure"]["png"]["sha256"] == (
        calculate_file_sha256(png_path)
    )
    assert metadata[
        "output"
    ]["figure"]["svg"]["sha256"] == (
        calculate_file_sha256(svg_path)
    )


def test_save_basel_traffic_light_report_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Every documented summary column must be saved."""

    paths, summary = _create_paths_and_summary()
    summary = summary.drop(columns="worst_zone")

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        report_module.save_basel_traffic_light_report(
            paths=paths,
            summary=summary,
            source_snapshots={},
            source_metadata={},
            output_root=tmp_path,
            snapshot_id="missing_column",
            generated_at=datetime(
                2026,
                10,
                13,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
        )


def test_save_basel_traffic_light_report_rejects_model_mismatch(
    tmp_path: Path,
) -> None:
    """The detailed and compact tables must cover the same models."""

    paths, summary = _create_paths_and_summary()
    summary = summary.iloc[:1]

    with pytest.raises(
        ValueError,
        match="same ordered models",
    ):
        report_module.save_basel_traffic_light_report(
            paths=paths,
            summary=summary,
            source_snapshots={},
            source_metadata={},
            output_root=tmp_path,
            snapshot_id="model_mismatch",
            generated_at=datetime(
                2026,
                10,
                13,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
        )


def test_save_basel_traffic_light_report_does_not_overwrite(
    tmp_path: Path,
) -> None:
    """An existing traffic-light report is never overwritten."""

    paths, summary = _create_paths_and_summary()
    model_names = list(summary.index)
    arguments = {
        "paths": paths,
        "summary": summary,
        "source_snapshots": {
            model_name: Path(model_name)
            for model_name in model_names
        },
        "source_metadata": {
            model_name: _create_metadata(model_name)
            for model_name in model_names
        },
        "output_root": tmp_path,
        "snapshot_id": "test_basel_traffic_light",
        "generated_at": datetime(
            2026,
            10,
            13,
            tzinfo=timezone.utc,
        ),
        "experiment_name": "baseline_11_asset",
    }

    report_module.save_basel_traffic_light_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_basel_traffic_light_report(
            **arguments
        )
