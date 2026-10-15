"""Tests for the descriptive subperiod exception-rate figure command."""

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
        / "report_subperiod_exception_rates.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_subperiod_exception_rates",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load the subperiod figure script."
        )

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


report_module = _load_report_module()


def _create_var_coverage() -> pd.DataFrame:
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
    index = pd.MultiIndex.from_product(
        [period_ids, models, confidence_levels],
        names=[
            "period_id",
            "model",
            "confidence_level",
        ],
    )
    nominal_rates = (
        1.0
        - index.get_level_values(
            "confidence_level"
        ).to_numpy(dtype=float)
    )
    observation_counts = np.full(len(index), 250)
    exception_counts = (
        np.arange(len(index), dtype=int) % 15
    ) + 1
    observed_rates = (
        exception_counts / observation_counts
    )
    period_types = [
        (
            "nber_recession"
            if period_id == "nber_recession_2020"
            else "calendar_year"
        )
        for period_id in index.get_level_values("period_id")
    ]

    return pd.DataFrame(
        {
            "period_type": period_types,
            "observation_count": observation_counts,
            "nominal_exceedance_rate": nominal_rates,
            "observed_exceedance_count": exception_counts,
            "observed_exceedance_rate": observed_rates,
            "observed_minus_expected_rate": (
                observed_rates - nominal_rates
            ),
            "interval_confidence_level": 0.95,
            "exact_interval_lower_bound": np.maximum(
                0.0,
                observed_rates - 0.01,
            ),
            "exact_interval_upper_bound": np.minimum(
                1.0,
                observed_rates + 0.02,
            ),
            "sparse_expected_tail": False,
            "sparse_observed_tail": False,
        },
        index=index,
    )


def _write_source_snapshot(
    root: Path,
) -> tuple[Path, pd.DataFrame, dict[str, object]]:
    """Write a minimal checksum-verified subperiod source."""

    snapshot_path = root / "source_subperiod_analysis"
    snapshot_path.mkdir()
    var_coverage = _create_var_coverage()
    table_path = snapshot_path / "subperiod_var_coverage.csv"
    var_coverage.to_csv(table_path, index=True)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "experiment": "baseline_11_asset",
        "report": {
            "name": "subperiod_analysis",
        },
        "output": {
            "tables": {
                "subperiod_var_coverage": {
                    "rows": len(var_coverage),
                    "index_names": list(
                        var_coverage.index.names
                    ),
                    "columns": list(var_coverage.columns),
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

    return snapshot_path, var_coverage, metadata


def test_load_subperiod_var_coverage_snapshot_verifies_table(
    tmp_path: Path,
) -> None:
    """A matching source table and checksum load without alteration."""

    snapshot_path, expected, expected_metadata = (
        _write_source_snapshot(tmp_path)
    )

    observed, observed_metadata = (
        report_module.load_subperiod_var_coverage_snapshot(
            snapshot_path
        )
    )

    pd.testing.assert_frame_equal(observed, expected)
    assert observed_metadata == expected_metadata


def test_load_subperiod_var_coverage_snapshot_rejects_tampering(
    tmp_path: Path,
) -> None:
    """A source table changed after metadata creation is rejected."""

    snapshot_path, _, _ = _write_source_snapshot(
        tmp_path
    )
    table_path = snapshot_path / "subperiod_var_coverage.csv"

    with table_path.open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    with pytest.raises(ValueError, match="checksum"):
        report_module.load_subperiod_var_coverage_snapshot(
            snapshot_path
        )


def test_save_subperiod_exception_rate_report_writes_lineage(
    tmp_path: Path,
) -> None:
    """The report saves immutable and stable figures with checksums."""

    snapshot_path, var_coverage, source_metadata = (
        _write_source_snapshot(tmp_path)
    )
    output_root = tmp_path / "evaluations"
    publication_root = tmp_path / "publication"
    snapshot_paths, metadata_path, publication_paths = (
        report_module.save_subperiod_exception_rate_report(
            var_coverage=var_coverage,
            source_snapshot=snapshot_path,
            source_metadata=source_metadata,
            output_root=output_root,
            publication_root=publication_root,
            snapshot_id=(
                "test_subperiod_exception_rate_figures"
            ),
            generated_at=datetime(
                2026,
                10,
                14,
                tzinfo=timezone.utc,
            ),
        )
    )

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    assert set(snapshot_paths) == {
        "heatmap_png",
        "heatmap_svg",
        "interval_png",
        "interval_svg",
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
    assert metadata["output"]["figures"]["heatmap"][
        "png"
    ]["sha256"] == calculate_file_sha256(
        snapshot_paths["heatmap_png"]
    )
    assert metadata["output"]["figures"][
        "exact_interval"
    ]["svg"]["sha256"] == calculate_file_sha256(
        snapshot_paths["interval_svg"]
    )
    assert metadata["report"]["exact_intervals"][
        "method"
    ] == "clopper_pearson"
    assert metadata["report"]["exact_intervals"][
        "calendar_x_axis_upper_bounds"
    ] == {
        "0.95": 0.20,
        "0.975": 0.16,
        "0.99": 0.125,
    }
    assert metadata["report"]["exact_intervals"][
        "binomial_reference_bands"
    ]["reference_observation_counts"] == {
        "calendar": 252,
        "nber_recession": 62,
    }
    assert metadata["report"]["inferential_status"] == (
        "descriptive_only; no visual pass/fail rule"
    )


def test_save_subperiod_exception_rate_report_never_overwrites(
    tmp_path: Path,
) -> None:
    """An existing immutable figure snapshot cannot be replaced."""

    snapshot_path, var_coverage, source_metadata = (
        _write_source_snapshot(tmp_path)
    )
    arguments = {
        "var_coverage": var_coverage,
        "source_snapshot": snapshot_path,
        "source_metadata": source_metadata,
        "output_root": tmp_path / "evaluations",
        "publication_root": tmp_path / "publication",
        "snapshot_id": "fixed_snapshot",
        "generated_at": datetime(
            2026,
            10,
            14,
            tzinfo=timezone.utc,
        ),
    }
    report_module.save_subperiod_exception_rate_report(
        **arguments
    )

    with pytest.raises(FileExistsError):
        report_module.save_subperiod_exception_rate_report(
            **arguments
        )
