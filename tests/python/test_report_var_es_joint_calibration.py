"""Tests for the joint VaR-ES calibration reporting command."""

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
        / "report_var_es_joint_calibration.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_var_es_joint_calibration",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load the joint-calibration report script."
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


def _create_calibration_table() -> pd.DataFrame:
    """Create a small multi-model joint-calibration report."""

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
            "mean_tail_contribution": [
                0.022,
                0.021,
            ],
            "mean_expected_shortfall": [
                0.020,
                0.0215,
            ],
            "mean_identification_residual": [
                0.002,
                -0.0005,
            ],
            "hac_bandwidth": [4, 4],
            "hac_long_run_variance": [
                0.0004,
                0.0003,
            ],
            "hac_standard_error": [
                0.002,
                0.0017,
            ],
            "hac_test_statistic": [
                1.0,
                -0.2941176471,
            ],
            "hac_p_value": [0.3173, 0.7687],
            "significance_level": [0.05, 0.05],
            "hac_reject_null": [False, False],
            "hac_holm_p_value": [0.6346, 0.7687],
            "hac_holm_reject_null": [False, False],
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


def test_save_joint_calibration_report_writes_verified_files(
    tmp_path: Path,
) -> None:
    """The report command saves a CSV and inference metadata."""

    calibration = _create_calibration_table()
    model_names = list(
        calibration.index.get_level_values(
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

    calibration_path, metadata_path = (
        report_module.save_var_es_joint_calibration_report(
            calibration=calibration,
            source_snapshots=source_snapshots,
            source_metadata=source_metadata,
            output_root=tmp_path,
            snapshot_id="test_joint_calibration",
            generated_at=datetime(
                2026,
                10,
                13,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
            requested_bandwidth=None,
        )
    )

    assert calibration_path.exists()
    assert metadata_path.exists()

    saved_calibration = pd.read_csv(
        calibration_path
    )
    assert len(saved_calibration) == 2
    assert list(saved_calibration["model"]) == (
        model_names
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["report"]["name"] == (
        "var_es_joint_calibration"
    )
    assert metadata["report"]["hac"]["estimator"] == (
        "newey_west"
    )
    assert metadata["report"]["hac"]["kernel"] == (
        "bartlett"
    )
    assert metadata[
        "report"
    ]["hac"]["selected_bandwidths"] == [4]
    assert metadata[
        "report"
    ]["multiplicity_adjustment"]["method"] == "holm"
    assert metadata[
        "report"
    ]["multiplicity_adjustment"][
        "hypothesis_count"
    ] == 2
    assert metadata["output"]["rows"] == 2
    assert metadata["output"]["file"]["sha256"] == (
        calculate_file_sha256(calibration_path)
    )


def test_save_joint_calibration_report_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Every documented calibration column must be saved."""

    calibration = _create_calibration_table().drop(
        columns="hac_holm_p_value"
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        report_module.save_var_es_joint_calibration_report(
            calibration=calibration,
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
            requested_bandwidth=None,
        )


def test_save_joint_calibration_report_does_not_overwrite(
    tmp_path: Path,
) -> None:
    """An existing report directory is never overwritten."""

    calibration = _create_calibration_table()
    model_names = list(
        calibration.index.get_level_values(
            "model"
        ).unique()
    )
    arguments = {
        "calibration": calibration,
        "source_snapshots": {
            model_name: Path(model_name)
            for model_name in model_names
        },
        "source_metadata": {
            model_name: _create_metadata(model_name)
            for model_name in model_names
        },
        "output_root": tmp_path,
        "snapshot_id": "test_joint_calibration",
        "generated_at": datetime(
            2026,
            10,
            13,
            tzinfo=timezone.utc,
        ),
        "experiment_name": "baseline_11_asset",
        "requested_bandwidth": None,
    }

    report_module.save_var_es_joint_calibration_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_var_es_joint_calibration_report(
            **arguments
        )
