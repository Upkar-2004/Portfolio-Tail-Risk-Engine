"""Tests for the ES tail-severity reporting command."""

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
        / "report_es_tail_severity.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_es_tail_severity",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load scripts/report_es_tail_severity.py."
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


def _create_severity_table() -> pd.DataFrame:
    """Create a small multi-model ES severity report."""

    index = pd.MultiIndex.from_tuples(
        [
            (
                "rolling_gaussian_monte_carlo",
                0.95,
            ),
            (
                "ewma_gaussian_monte_carlo",
                0.95,
            ),
        ],
        names=[
            "model",
            "confidence_level",
        ],
    )

    return pd.DataFrame(
        {
            "observation_count": [100, 100],
            "var_exceedance_count": [5, 6],
            "mean_exceedance_loss": [0.031, 0.030],
            "mean_var_on_exceedance_days": [
                0.020,
                0.021,
            ],
            "mean_es_on_exceedance_days": [
                0.027,
                0.028,
            ],
            "mean_var_excess": [0.011, 0.009],
            "median_var_excess": [0.008, 0.007],
            "maximum_var_excess": [0.030, 0.025],
            "mean_es_gap": [0.004, 0.002],
        },
        index=index,
    )


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
    """A snapshot cannot be assigned to the wrong model."""

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


def test_save_es_tail_severity_report_writes_verified_files(
    tmp_path: Path,
) -> None:
    """The report command saves a CSV and lineage metadata."""

    severity = _create_severity_table()
    model_names = list(
        severity.index.get_level_values(
            "model"
        ).unique()
    )
    source_snapshots = {
        model_name: Path(model_name)
        for model_name in model_names
    }
    source_metadata = {
        model_name: _create_metadata(model_name)
        for model_name in model_names
    }

    severity_path, metadata_path = (
        report_module.save_es_tail_severity_report(
            severity=severity,
            source_snapshots=source_snapshots,
            source_metadata=source_metadata,
            output_root=tmp_path,
            snapshot_id="test_es_tail_severity",
            generated_at=datetime(
                2026,
                10,
                13,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
        )
    )

    assert severity_path.exists()
    assert metadata_path.exists()

    saved_severity = pd.read_csv(
        severity_path
    )
    assert len(saved_severity) == 2
    assert list(saved_severity["model"]) == model_names

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["report"]["name"] == (
        "es_tail_severity"
    )
    assert metadata["report"]["classification"] == (
        "descriptive"
    )
    assert metadata[
        "report"
    ]["es_var_order_tolerance"] == pytest.approx(
        1e-12
    )
    assert metadata["output"]["rows"] == 2
    assert metadata["output"]["file"]["sha256"] == (
        calculate_file_sha256(severity_path)
    )


def test_save_es_tail_severity_report_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Every documented severity column must be saved."""

    severity = _create_severity_table().drop(
        columns="mean_es_gap"
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        report_module.save_es_tail_severity_report(
            severity=severity,
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


def test_save_es_tail_severity_report_does_not_overwrite(
    tmp_path: Path,
) -> None:
    """An existing report directory is never overwritten."""

    severity = _create_severity_table()
    model_names = list(
        severity.index.get_level_values(
            "model"
        ).unique()
    )
    arguments = {
        "severity": severity,
        "source_snapshots": {
            model_name: Path(model_name)
            for model_name in model_names
        },
        "source_metadata": {
            model_name: _create_metadata(model_name)
            for model_name in model_names
        },
        "output_root": tmp_path,
        "snapshot_id": "test_es_tail_severity",
        "generated_at": datetime(
            2026,
            10,
            13,
            tzinfo=timezone.utc,
        ),
        "experiment_name": "baseline_11_asset",
    }

    report_module.save_es_tail_severity_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_es_tail_severity_report(
            **arguments
        )
