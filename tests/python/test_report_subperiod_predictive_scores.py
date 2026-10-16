"""Tests for the subperiod predictive-score figure command."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from tailrisk.data import calculate_file_sha256


def _load_report_module() -> ModuleType:
    """Load the report script without making scripts a package."""

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "report_subperiod_predictive_scores.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_subperiod_predictive_scores",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load the subperiod predictive-score script."
        )

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


report_module = _load_report_module()


def _create_predictive_accuracy() -> pd.DataFrame:
    """Create a complete table with the official plotting schema."""

    period_ids = [
        *(f"calendar_{year}" for year in range(2012, 2026)),
        "nber_recession_2020",
    ]
    models = [
        "rolling_gaussian_monte_carlo",
        "ewma_gaussian_monte_carlo",
        "filtered_historical_simulation",
    ]
    confidence_levels = [0.95, 0.975, 0.99]
    records = []

    for period_index, period_id in enumerate(period_ids):
        for level_index, confidence_level in enumerate(
            confidence_levels
        ):
            ql_order = np.roll(
                np.arange(1, 4),
                period_index + level_index,
            )
            fz0_order = np.roll(
                np.arange(1, 4),
                period_index + level_index + 1,
            )

            for model_index, model in enumerate(models):
                ql_rank = int(ql_order[model_index])
                fz0_rank = int(fz0_order[model_index])
                records.append(
                    {
                        "period_id": period_id,
                        "model": model,
                        "confidence_level": confidence_level,
                        "period_type": (
                            "nber_recession"
                            if period_id == "nber_recession_2020"
                            else "calendar_year"
                        ),
                        "first_forecast_date": "2012-01-03",
                        "last_forecast_date": "2012-12-31",
                        "observation_count": 250,
                        "is_partial": False,
                        "expected_exceedance_count": (
                            250 * (1.0 - confidence_level)
                        ),
                        "var_exceedance_count": 5,
                        "mean_var_quantile_loss": (
                            0.001 * ql_rank
                        ),
                        "var_quantile_loss_rank": ql_rank,
                        "mean_fz0_loss": float(fz0_rank - 5),
                        "fz0_loss_rank": fz0_rank,
                        "sparse_expected_tail": False,
                        "sparse_observed_tail": False,
                    }
                )

    return pd.DataFrame.from_records(records).set_index(
        [
            "period_id",
            "model",
            "confidence_level",
        ]
    )


def _write_source_snapshot(
    root: Path,
) -> tuple[Path, pd.DataFrame, dict[str, object]]:
    """Write a minimal checksum-verified subperiod source."""

    snapshot_path = root / "source_subperiod_analysis"
    snapshot_path.mkdir()
    predictive_accuracy = _create_predictive_accuracy()
    table_path = (
        snapshot_path / "subperiod_predictive_accuracy.csv"
    )
    predictive_accuracy.to_csv(table_path, index=True)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "experiment": "baseline_11_asset",
        "report": {
            "name": "subperiod_analysis",
        },
        "output": {
            "tables": {
                "subperiod_predictive_accuracy": {
                    "rows": len(predictive_accuracy),
                    "index_names": list(
                        predictive_accuracy.index.names
                    ),
                    "columns": list(predictive_accuracy.columns),
                    "file": {
                        "name": table_path.name,
                        "sha256": calculate_file_sha256(
                            table_path
                        ),
                    },
                },
            },
        },
    }

    with (snapshot_path / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(metadata, stream)

    return snapshot_path, predictive_accuracy, metadata


def test_load_predictive_accuracy_snapshot_verifies_table(
    tmp_path: Path,
) -> None:
    """A matching source table and checksum load without alteration."""

    snapshot_path, expected, expected_metadata = (
        _write_source_snapshot(tmp_path)
    )
    observed, observed_metadata = (
        report_module.load_subperiod_predictive_accuracy_snapshot(
            snapshot_path
        )
    )

    pd.testing.assert_frame_equal(observed, expected)
    assert observed_metadata == expected_metadata


def test_load_predictive_accuracy_snapshot_rejects_tampering(
    tmp_path: Path,
) -> None:
    """A source table changed after metadata creation is rejected."""

    snapshot_path, _, _ = _write_source_snapshot(tmp_path)
    table_path = (
        snapshot_path / "subperiod_predictive_accuracy.csv"
    )

    with table_path.open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    with pytest.raises(ValueError, match="checksum"):
        report_module.load_subperiod_predictive_accuracy_snapshot(
            snapshot_path
        )


def test_save_predictive_score_report_writes_lineage(
    tmp_path: Path,
) -> None:
    """The report saves immutable and stable figures with checksums."""

    snapshot_path, predictive_accuracy, source_metadata = (
        _write_source_snapshot(tmp_path)
    )
    snapshot_paths, metadata_path, publication_paths = (
        report_module.save_subperiod_predictive_score_report(
            predictive_accuracy=predictive_accuracy,
            source_snapshot=snapshot_path,
            source_metadata=source_metadata,
            output_root=tmp_path / "evaluations",
            publication_root=tmp_path / "publication",
            snapshot_id="test_predictive_score_figure",
            generated_at=datetime(
                2026,
                10,
                16,
                tzinfo=timezone.utc,
            ),
        )
    )

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    assert set(snapshot_paths) == {
        "first_place_summary_png",
        "first_place_summary_svg",
        "annual_rank_grid_png",
        "annual_rank_grid_svg",
        "predictive_score_companion_csv",
    }
    assert set(publication_paths) == set(snapshot_paths)

    for output_name, snapshot_figure_path in (
        snapshot_paths.items()
    ):
        assert snapshot_figure_path.is_file()
        assert publication_paths[
            output_name
        ].read_bytes() == snapshot_figure_path.read_bytes()

    assert metadata["source"]["snapshot_id"] == (
        snapshot_path.name
    )
    assert metadata["output"]["figures"][
        "first_place_summary"
    ]["png"]["sha256"] == calculate_file_sha256(
        snapshot_paths["first_place_summary_png"]
    )
    assert metadata["report"]["annual_grid_encoding"][
        "shading"
    ] == "rank_1_model_house_color"
    assert metadata["report"]["annual_grid_encoding"][
        "hac_standardized_shading"
    ] is False
    assert "no subperiod hypothesis tests" in metadata[
        "report"
    ]["inferential_status"]
    copied_table = metadata_path.parent / metadata["output"][
        "source_table_copy"
    ]["name"]
    assert copied_table.is_file()
    assert calculate_file_sha256(copied_table) == metadata[
        "output"
    ]["source_table_copy"]["sha256"]
    assert metadata["report"]["model_specifications"][
        "all_models"
    ]["scenario_count_per_forecast"] == 100000
    companion_table = pd.read_csv(
        snapshot_paths["predictive_score_companion_csv"]
    )
    assert len(companion_table) == 135
    assert set(companion_table.columns) == {
        "period_id",
        "period_type",
        "model",
        "confidence_level",
        "observation_count",
        "mean_var_quantile_loss",
        "var_quantile_loss_rank",
        "mean_fz0_loss",
        "fz0_loss_rank",
    }


def test_save_predictive_score_report_never_overwrites(
    tmp_path: Path,
) -> None:
    """An existing immutable figure snapshot cannot be replaced."""

    snapshot_path, predictive_accuracy, source_metadata = (
        _write_source_snapshot(tmp_path)
    )
    arguments = {
        "predictive_accuracy": predictive_accuracy,
        "source_snapshot": snapshot_path,
        "source_metadata": source_metadata,
        "output_root": tmp_path / "evaluations",
        "publication_root": tmp_path / "publication",
        "snapshot_id": "fixed_snapshot",
        "generated_at": datetime(
            2026,
            10,
            16,
            tzinfo=timezone.utc,
        ),
    }
    report_module.save_subperiod_predictive_score_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_subperiod_predictive_score_report(
            **arguments
        )
