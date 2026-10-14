"""Tests for the predictive-accuracy reporting command."""

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
        / "report_predictive_accuracy.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_predictive_accuracy",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load the predictive-accuracy report script."
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


def _create_comparison_table() -> pd.DataFrame:
    """Create a small two-family predictive-accuracy table."""

    index = pd.MultiIndex.from_tuples(
        [
            (
                "var_quantile_loss",
                0.95,
                "rolling_gaussian_monte_carlo",
                "ewma_gaussian_monte_carlo",
            ),
            (
                "var_quantile_loss",
                0.95,
                "rolling_gaussian_monte_carlo",
                "filtered_historical_simulation",
            ),
            (
                "fz0_loss",
                0.95,
                "rolling_gaussian_monte_carlo",
                "ewma_gaussian_monte_carlo",
            ),
            (
                "fz0_loss",
                0.95,
                "rolling_gaussian_monte_carlo",
                "filtered_historical_simulation",
            ),
        ],
        names=[
            "score_name",
            "confidence_level",
            "model_a",
            "model_b",
        ],
    )

    return pd.DataFrame(
        {
            "observation_count": [100] * 4,
            "mean_score_difference": [
                -0.001,
                0.002,
                -0.10,
                0.20,
            ],
            "hac_long_run_variance": [
                0.0004,
                0.0005,
                0.04,
                0.05,
            ],
            "hac_mean_variance": [
                0.000004,
                0.000005,
                0.0004,
                0.0005,
            ],
            "hac_standard_error": [
                0.002,
                0.0022,
                0.02,
                0.022,
            ],
            "hac_bandwidth": [4] * 4,
            "dm_statistic": [
                -0.5,
                0.91,
                -5.0,
                9.09,
            ],
            "dm_p_value": [
                0.62,
                0.36,
                0.000001,
                0.0000001,
            ],
            "significance_level": [0.05] * 4,
            "dm_reject_equal_accuracy": [
                False,
                False,
                True,
                True,
            ],
            "holm_adjusted_p_value": [
                0.62,
                0.62,
                0.000002,
                0.0000002,
            ],
            "holm_reject_equal_accuracy": [
                False,
                False,
                True,
                True,
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


def test_save_predictive_accuracy_report_writes_verified_files(
    tmp_path: Path,
) -> None:
    """The report saves both score families and inference metadata."""

    comparisons = _create_comparison_table()
    model_names = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
        "filtered_historical_simulation",
    ]
    source_snapshots = {
        model_name: Path(model_name)
        for model_name in model_names
    }
    source_metadata = {
        model_name: _create_metadata(model_name)
        for model_name in model_names
    }

    comparisons_path, metadata_path = (
        report_module.save_predictive_accuracy_report(
            comparisons=comparisons,
            source_snapshots=source_snapshots,
            source_metadata=source_metadata,
            output_root=tmp_path,
            snapshot_id="test_predictive_accuracy",
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

    assert comparisons_path.exists()
    assert metadata_path.exists()

    saved_comparisons = pd.read_csv(
        comparisons_path
    )
    assert len(saved_comparisons) == 4
    assert set(saved_comparisons["score_name"]) == {
        "var_quantile_loss",
        "fz0_loss",
    }

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["report"]["name"] == (
        "predictive_accuracy"
    )
    assert metadata["report"]["hac"]["estimator"] == (
        "newey_west"
    )
    assert metadata[
        "report"
    ]["hac"]["selected_bandwidths"] == [4]
    adjustment = metadata[
        "report"
    ]["multiplicity_adjustment"]
    assert adjustment["method"] == "holm"
    assert adjustment[
        "applied_separately_by_score_family"
    ]
    assert adjustment["families"] == [
        {
            "score_name": "var_quantile_loss",
            "hypothesis_count": 2,
        },
        {
            "score_name": "fz0_loss",
            "hypothesis_count": 2,
        },
    ]
    assert adjustment["total_hypothesis_count"] == 4
    assert metadata["output"]["rows"] == 4
    assert metadata["output"]["file"]["sha256"] == (
        calculate_file_sha256(comparisons_path)
    )


def test_save_predictive_accuracy_report_rejects_missing_family(
    tmp_path: Path,
) -> None:
    """A saved report must contain both declared score families."""

    comparisons = _create_comparison_table().xs(
        "var_quantile_loss",
        level="score_name",
        drop_level=False,
    )

    with pytest.raises(
        ValueError,
        match="quantile-loss and FZ0",
    ):
        report_module.save_predictive_accuracy_report(
            comparisons=comparisons,
            source_snapshots={},
            source_metadata={},
            output_root=tmp_path,
            snapshot_id="missing_family",
            generated_at=datetime(
                2026,
                10,
                13,
                tzinfo=timezone.utc,
            ),
            experiment_name="baseline_11_asset",
            requested_bandwidth=None,
        )


def test_save_predictive_accuracy_report_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Every documented comparison column must be saved."""

    comparisons = _create_comparison_table().drop(
        columns="holm_adjusted_p_value"
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        report_module.save_predictive_accuracy_report(
            comparisons=comparisons,
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


def test_save_predictive_accuracy_report_does_not_overwrite(
    tmp_path: Path,
) -> None:
    """An existing predictive-accuracy report is never overwritten."""

    comparisons = _create_comparison_table()
    model_names = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
        "filtered_historical_simulation",
    ]
    arguments = {
        "comparisons": comparisons,
        "source_snapshots": {
            model_name: Path(model_name)
            for model_name in model_names
        },
        "source_metadata": {
            model_name: _create_metadata(model_name)
            for model_name in model_names
        },
        "output_root": tmp_path,
        "snapshot_id": "test_predictive_accuracy",
        "generated_at": datetime(
            2026,
            10,
            13,
            tzinfo=timezone.utc,
        ),
        "experiment_name": "baseline_11_asset",
        "requested_bandwidth": None,
    }

    report_module.save_predictive_accuracy_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_predictive_accuracy_report(
            **arguments
        )
