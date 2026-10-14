"""Tests for the descriptive subperiod reporting command."""

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
        / "report_subperiod_analysis.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_subperiod_analysis",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load the subperiod report script."
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


def _create_report_tables(
) -> dict[str, pd.DataFrame]:
    """Create compatible examples of all five subperiod tables."""

    period_ids = [
        "calendar_2012",
        "nber_recession_2020",
    ]
    models = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
    ]
    confidence_levels = [
        0.95,
        0.99,
    ]
    detail_index = pd.MultiIndex.from_product(
        [
            period_ids,
            models,
            confidence_levels,
        ],
        names=[
            "period_id",
            "model",
            "confidence_level",
        ],
    )
    period_types = [
        period_id
        for period_id in detail_index.get_level_values(
            "period_id"
        )
    ]
    period_types = [
        (
            "calendar_year"
            if period_id == "calendar_2012"
            else "nber_recession"
        )
        for period_id in period_types
    ]
    common_columns = {
        "period_type": period_types,
        "first_forecast_date": pd.Timestamp(
            "2012-01-03"
        ),
        "last_forecast_date": pd.Timestamp(
            "2012-12-31"
        ),
        "observation_count": 250,
        "is_partial": False,
    }
    definitions = pd.DataFrame(
        {
            "period_type": [
                "calendar_year",
                "nber_recession",
            ],
            "label": [
                "Calendar year 2012",
                "NBER recession: February-April 2020",
            ],
            "requested_start_date": pd.to_datetime(
                [
                    "2012-01-01",
                    "2020-02-01",
                ]
            ),
            "requested_end_date": pd.to_datetime(
                [
                    "2012-12-31",
                    "2020-04-30",
                ]
            ),
            "first_forecast_date": pd.to_datetime(
                [
                    "2012-01-03",
                    "2020-02-03",
                ]
            ),
            "last_forecast_date": pd.to_datetime(
                [
                    "2012-12-31",
                    "2020-04-30",
                ]
            ),
            "observation_count": [250, 63],
            "is_partial": [False, False],
            "source": [
                "predeclared_calendar_year_rule",
                (
                    "https://www.nber.org/research/"
                    "business-cycle-dating"
                ),
            ],
        },
        index=pd.Index(
            period_ids,
            name="period_id",
        ),
    )
    var_coverage = pd.DataFrame(
        {
            **common_columns,
            "nominal_exceedance_rate": 0.05,
            "expected_exceedance_count": 12.5,
            "observed_exceedance_count": 11,
            "observed_exceedance_rate": 0.044,
            "observed_minus_expected_count": -1.5,
            "observed_minus_expected_rate": -0.006,
            "interval_confidence_level": 0.95,
            "exact_interval_lower_bound": 0.02,
            "exact_interval_upper_bound": 0.08,
            "mean_value_at_risk": 0.02,
            "mean_expected_shortfall": 0.03,
            "sparse_expected_tail": False,
            "sparse_observed_tail": False,
        },
        index=detail_index,
    )
    es_tail_severity = pd.DataFrame(
        {
            **common_columns,
            "nominal_exceedance_rate": 0.05,
            "expected_exceedance_count": 12.5,
            "var_exceedance_count": 11,
            "mean_exceedance_loss": 0.04,
            "mean_var_on_exceedance_days": 0.02,
            "mean_es_on_exceedance_days": 0.03,
            "mean_var_excess": 0.02,
            "median_var_excess": 0.018,
            "maximum_var_excess": 0.06,
            "mean_es_gap": 0.01,
            "sparse_expected_tail": False,
            "sparse_observed_tail": False,
        },
        index=detail_index,
    )
    joint_calibration = pd.DataFrame(
        {
            **common_columns,
            "expected_exceedance_count": 12.5,
            "var_exceedance_count": 11,
            "mean_tail_contribution": 0.031,
            "mean_expected_shortfall": 0.03,
            "mean_identification_residual": 0.001,
            "sparse_expected_tail": False,
            "sparse_observed_tail": False,
        },
        index=detail_index,
    )
    predictive_accuracy = pd.DataFrame(
        {
            **common_columns,
            "expected_exceedance_count": 12.5,
            "var_exceedance_count": 11,
            "mean_var_quantile_loss": 0.001,
            "var_quantile_loss_rank": 1,
            "mean_fz0_loss": -3.0,
            "fz0_loss_rank": 1,
            "sparse_expected_tail": False,
            "sparse_observed_tail": False,
        },
        index=detail_index,
    )

    return {
        "period_definitions": definitions,
        "var_coverage": var_coverage,
        "es_tail_severity": es_tail_severity,
        "joint_calibration": joint_calibration,
        "predictive_accuracy": predictive_accuracy,
    }


def _create_save_arguments(
    tmp_path: Path,
) -> dict[str, object]:
    """Create complete arguments for saving a test report."""

    tables = _create_report_tables()
    model_names = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
    ]

    return {
        **tables,
        "source_snapshots": {
            model_name: Path(model_name)
            for model_name in model_names
        },
        "source_metadata": {
            model_name: _create_metadata(model_name)
            for model_name in model_names
        },
        "output_root": tmp_path,
        "snapshot_id": "test_subperiod_analysis",
        "generated_at": datetime(
            2026,
            10,
            14,
            tzinfo=timezone.utc,
        ),
        "experiment_name": "baseline_11_asset",
    }


def test_validate_backtest_metadata_accepts_matching_source(
) -> None:
    """Matching model and experiment metadata are accepted."""

    report_module.validate_backtest_metadata(
        metadata=_create_metadata(
            "rolling_gaussian_monte_carlo"
        ),
        expected_model_name=(
            "rolling_gaussian_monte_carlo"
        ),
        expected_portfolio_snapshot_id=(
            "portfolio-snapshot"
        ),
        expected_estimation_window=504,
        expected_confidence_levels=[
            0.95,
            0.99,
        ],
    )


def test_validate_backtest_metadata_rejects_wrong_model(
) -> None:
    """A snapshot cannot be assigned to the wrong model."""

    with pytest.raises(
        ValueError,
        match="model mismatch",
    ):
        report_module.validate_backtest_metadata(
            metadata=_create_metadata(
                "ewma_gaussian_monte_carlo"
            ),
            expected_model_name=(
                "rolling_gaussian_monte_carlo"
            ),
            expected_portfolio_snapshot_id=(
                "portfolio-snapshot"
            ),
            expected_estimation_window=504,
            expected_confidence_levels=[
                0.95,
                0.99,
            ],
        )


def test_save_subperiod_report_writes_all_verified_tables(
    tmp_path: Path,
) -> None:
    """One snapshot saves five CSVs with matching SHA-256 metadata."""

    table_paths, metadata_path = (
        report_module.save_subperiod_analysis_report(
            **_create_save_arguments(tmp_path)
        )
    )

    assert set(table_paths) == {
        "subperiod_definitions",
        "subperiod_var_coverage",
        "subperiod_es_tail_severity",
        "subperiod_joint_calibration",
        "subperiod_predictive_accuracy",
    }
    assert all(
        path.exists()
        for path in table_paths.values()
    )
    assert metadata_path.exists()

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    assert metadata["report"]["classification"] == (
        "descriptive_secondary_analysis"
    )
    assert metadata[
        "report"
    ]["inferential_status"]["hypothesis_tests"] == "none"
    assert metadata["output"]["periods"] == [
        "calendar_2012",
        "nber_recession_2020",
    ]

    for table_name, table_path in table_paths.items():
        output = metadata[
            "output"
        ]["tables"][table_name]
        assert output["file"]["name"] == table_path.name
        assert output["file"]["sha256"] == (
            calculate_file_sha256(table_path)
        )


def test_save_subperiod_report_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Every documented subperiod column must be present."""

    arguments = _create_save_arguments(tmp_path)
    arguments["predictive_accuracy"] = arguments[
        "predictive_accuracy"
    ].drop(
        columns="mean_fz0_loss"
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        report_module.save_subperiod_analysis_report(
            **arguments
        )


def test_save_subperiod_report_rejects_misaligned_tables(
    tmp_path: Path,
) -> None:
    """Every detailed table must use the same ordered rows."""

    arguments = _create_save_arguments(tmp_path)
    arguments["joint_calibration"] = arguments[
        "joint_calibration"
    ].iloc[:-1]

    with pytest.raises(
        ValueError,
        match="same ordered period-model-confidence index",
    ):
        report_module.save_subperiod_analysis_report(
            **arguments
        )


def test_save_subperiod_report_does_not_overwrite(
    tmp_path: Path,
) -> None:
    """An existing subperiod snapshot is never overwritten."""

    arguments = _create_save_arguments(tmp_path)
    report_module.save_subperiod_analysis_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_subperiod_analysis_report(
            **arguments
        )
