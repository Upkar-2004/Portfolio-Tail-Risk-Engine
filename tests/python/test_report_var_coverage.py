"""Tests for the VaR coverage reporting command."""

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
        / "report_var_coverage.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_var_coverage",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load scripts/report_var_coverage.py."
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


def _create_coverage_table() -> pd.DataFrame:
    """Create a small multi-model coverage report."""

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
            "observation_count": [
                100,
                100,
            ],
            "exceedance_count": [
                5,
                6,
            ],
            "interval_confidence_level": [
                0.95,
                0.95,
            ],
            "exceedance_rate_ci_lower": [
                0.0164,
                0.0223,
            ],
            "exceedance_rate_ci_upper": [
                0.1128,
                0.1260,
            ],
            "conditional_coverage_p_value": [
                0.50,
                0.25,
            ],
            "conditional_coverage_holm_p_value": [
                0.50,
                0.50,
            ],
            "conditional_coverage_reject_null_holm": [
                False,
                False,
            ],
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


def test_save_var_coverage_report_writes_verified_files(
    tmp_path: Path,
) -> None:
    """The report command saves a CSV and lineage metadata."""

    coverage = _create_coverage_table()
    model_names = list(
        coverage.index.get_level_values(
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

    coverage_path, metadata_path = (
        report_module.save_var_coverage_report(
            coverage=coverage,
            source_snapshots=source_snapshots,
            source_metadata=source_metadata,
            output_root=tmp_path,
            snapshot_id="test_var_coverage",
            generated_at=datetime(
                2026,
                10,
                12,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
            significance_level=0.05,
        )
    )

    assert coverage_path.exists()
    assert metadata_path.exists()

    saved_coverage = pd.read_csv(
        coverage_path
    )
    assert len(saved_coverage) == 2
    assert list(saved_coverage["model"]) == model_names
    assert {
        "interval_confidence_level",
        "exceedance_rate_ci_lower",
        "exceedance_rate_ci_upper",
    }.issubset(saved_coverage.columns)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["report"]["name"] == "var_coverage"
    assert metadata[
        "report"
    ]["exceedance_rate_interval"] == {
        "method": "clopper_pearson_exact",
        "confidence_level": 0.95,
    }
    assert metadata[
        "report"
    ]["multiplicity_adjustment"]["method"] == "holm"
    assert metadata["output"]["rows"] == 2
    assert metadata["output"]["file"]["sha256"] == (
        calculate_file_sha256(coverage_path)
    )


def test_save_var_coverage_report_does_not_overwrite(
    tmp_path: Path,
) -> None:
    """An existing report directory is never overwritten."""

    coverage = _create_coverage_table()
    model_names = list(
        coverage.index.get_level_values(
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
    arguments = {
        "coverage": coverage,
        "source_snapshots": source_snapshots,
        "source_metadata": source_metadata,
        "output_root": tmp_path,
        "snapshot_id": "test_var_coverage",
        "generated_at": datetime(
            2026,
            10,
            12,
            tzinfo=timezone.utc,
        ),
        "experiment_name": "baseline_11_asset",
        "significance_level": 0.05,
    }

    report_module.save_var_coverage_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_var_coverage_report(
            **arguments
        )


def test_save_var_coverage_report_rejects_mixed_intervals(
    tmp_path: Path,
) -> None:
    """A report cannot mix interval levels across coverage rows."""

    coverage = _create_coverage_table()
    coverage.iloc[
        1,
        coverage.columns.get_loc(
            "interval_confidence_level"
        ),
    ] = 0.90
    model_names = list(
        coverage.index.get_level_values(
            "model"
        ).unique()
    )

    with pytest.raises(
        ValueError,
        match="one interval confidence level",
    ):
        report_module.save_var_coverage_report(
            coverage=coverage,
            source_snapshots={
                model_name: Path(model_name)
                for model_name in model_names
            },
            source_metadata={
                model_name: _create_metadata(model_name)
                for model_name in model_names
            },
            output_root=tmp_path,
            snapshot_id="mixed_intervals",
            generated_at=datetime(
                2026,
                10,
                12,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
            significance_level=0.05,
        )
