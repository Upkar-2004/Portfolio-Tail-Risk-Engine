"""Generate the descriptive subperiod VaR exception-rate figure."""

import argparse
from datetime import datetime, timezone
import json
from math import ceil
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from tailrisk.data import calculate_file_sha256
from tailrisk.visualization import (
    SUBPERIOD_EXCEPTION_RATE_PALETTE,
    save_subperiod_exception_rate_figure,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "evaluations"
_PUBLICATION_ROOT = _PROJECT_ROOT / "reports" / "figures"
_FIGURE_STEM = "subperiod_exception_rate_heatmap"
_INDEX_NAMES = [
    "period_id",
    "model",
    "confidence_level",
]
_REQUIRED_COLUMNS = {
    "period_type",
    "observation_count",
    "nominal_exceedance_rate",
    "observed_exceedance_count",
    "observed_exceedance_rate",
    "observed_minus_expected_rate",
    "sparse_expected_tail",
    "sparse_observed_tail",
}


def parse_args() -> argparse.Namespace:
    """Parse the verified table snapshot and figure destinations."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a checksum-traced heatmap of calendar-year "
            "and 2020 NBER-period VaR exception-rate deviations."
        )
    )
    parser.add_argument(
        "subperiod_snapshot",
        type=Path,
        help=(
            "Path to the verified subperiod-analysis snapshot "
            "containing metadata.json and the VaR coverage CSV."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_OUTPUT_ROOT,
        help="Root directory for immutable versioned figure reports.",
    )
    parser.add_argument(
        "--publication-root",
        type=Path,
        default=_PUBLICATION_ROOT,
        help=(
            "Directory for stable PNG and SVG copies updated by "
            "each successful run."
        ),
    )
    parser.add_argument(
        "--snapshot-id",
        type=str,
        default=None,
        help=(
            "Optional figure snapshot ID. A timestamped ID is "
            "generated when omitted."
        ),
    )

    return parser.parse_args()


def load_subperiod_var_coverage_snapshot(
    snapshot_directory: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the subperiod VaR table after checksum verification."""

    snapshot_path = Path(snapshot_directory)
    metadata_path = snapshot_path / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    if metadata.get("report", {}).get("name") != (
        "subperiod_analysis"
    ):
        raise ValueError(
            "Source snapshot is not a subperiod-analysis report."
        )

    try:
        table_metadata = metadata[
            "output"
        ]["tables"]["subperiod_var_coverage"]
        file_metadata = table_metadata["file"]
        table_path = snapshot_path / file_metadata["name"]
        expected_checksum = file_metadata["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Source metadata does not describe the subperiod "
            "VaR coverage table."
        ) from error

    actual_checksum = calculate_file_sha256(table_path)

    if actual_checksum != expected_checksum:
        raise ValueError(
            "Subperiod VaR coverage checksum does not match "
            "its source metadata."
        )

    var_coverage = pd.read_csv(
        table_path,
        index_col=_INDEX_NAMES,
    )

    if list(var_coverage.index.names) != _INDEX_NAMES:
        raise ValueError(
            "Subperiod VaR coverage uses unexpected index levels."
        )

    if len(var_coverage) != table_metadata.get("rows"):
        raise ValueError(
            "Subperiod VaR coverage row count does not match "
            "its source metadata."
        )

    recorded_columns = table_metadata.get("columns")

    if (
        not isinstance(recorded_columns, list)
        or list(var_coverage.columns) != recorded_columns
    ):
        raise ValueError(
            "Subperiod VaR coverage columns do not match "
            "its source metadata."
        )

    missing_columns = (
        _REQUIRED_COLUMNS - set(var_coverage.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Subperiod VaR coverage is missing required "
            f"column(s): {missing_names}."
        )

    return var_coverage, metadata


def save_subperiod_exception_rate_report(
    var_coverage: pd.DataFrame,
    source_snapshot: str | Path,
    source_metadata: dict[str, Any],
    output_root: str | Path,
    publication_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
) -> tuple[Path, Path, Path, Path, Path]:
    """Save immutable figures, metadata, and refreshed stable copies."""

    if not isinstance(snapshot_id, str):
        raise TypeError("Snapshot ID must be a string.")

    if not snapshot_id.strip():
        raise ValueError("Snapshot ID must not be empty.")

    source_snapshot_path = Path(source_snapshot)
    source_table_metadata = source_metadata[
        "output"
    ]["tables"]["subperiod_var_coverage"]
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    report_directory = output_root_path / snapshot_id
    report_directory.mkdir(
        parents=False,
        exist_ok=False,
    )
    png_path = report_directory / f"{_FIGURE_STEM}.png"
    svg_path = report_directory / f"{_FIGURE_STEM}.svg"
    save_subperiod_exception_rate_figure(
        var_coverage=var_coverage,
        png_output_path=png_path,
        svg_output_path=svg_path,
    )

    deviations = var_coverage[
        "observed_minus_expected_rate"
    ].to_numpy(dtype=float)
    color_limit = max(
        0.01,
        ceil(float(np.max(np.abs(deviations))) * 100.0)
        / 100.0,
    )
    png_checksum = calculate_file_sha256(png_path)
    svg_checksum = calculate_file_sha256(svg_path)
    publication_root_path = Path(publication_root)
    publication_png_path = (
        publication_root_path / f"{_FIGURE_STEM}.png"
    )
    publication_svg_path = (
        publication_root_path / f"{_FIGURE_STEM}.svg"
    )
    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "experiment": source_metadata.get("experiment"),
        "report": {
            "name": "subperiod_exception_rate_figure",
            "classification": (
                "descriptive_secondary_analysis"
            ),
            "measure": "observed_minus_nominal_exception_rate",
            "display_units": "percentage_points",
            "interpretation": {
                "positive": "more_exceptions_than_nominal",
                "negative": "fewer_exceptions_than_nominal",
                "zero": "observed_rate_equals_nominal_rate",
            },
            "layout": {
                "confidence_level_panels": [
                    0.95,
                    0.975,
                    0.99,
                ],
                "calendar_years": list(range(2012, 2026)),
                "nber_period": "2020-02-01_to_2020-04-30",
                "nber_period_is_overlapping_case_study": True,
            },
            "color_scale": {
                "center": 0.0,
                "symmetric_limit_rate": color_limit,
                "palette": SUBPERIOD_EXCEPTION_RATE_PALETTE,
            },
            "inferential_status": (
                "descriptive_only; no visual pass/fail rule"
            ),
            "sparse_tail_warning": (
                "Large deviations can arise from small expected "
                "or observed exception counts, especially at 99% "
                "VaR and in the 62-session NBER period."
            ),
            "png_dpi": 300,
        },
        "source": {
            "snapshot_id": source_snapshot_path.name,
            "report_name": source_metadata["report"]["name"],
            "var_coverage": {
                "name": source_table_metadata["file"]["name"],
                "sha256": source_table_metadata[
                    "file"
                ]["sha256"],
            },
        },
        "output": {
            "figure": {
                "png": {
                    "name": png_path.name,
                    "sha256": png_checksum,
                },
                "svg": {
                    "name": svg_path.name,
                    "sha256": svg_checksum,
                },
            },
            "publication_copies": {
                "png": {
                    "name": publication_png_path.name,
                    "sha256": png_checksum,
                },
                "svg": {
                    "name": publication_svg_path.name,
                    "sha256": svg_checksum,
                },
            },
        },
    }
    metadata_path = report_directory / "metadata.json"

    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    publication_root_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copyfile(png_path, publication_png_path)
    shutil.copyfile(svg_path, publication_svg_path)

    if (
        calculate_file_sha256(publication_png_path)
        != png_checksum
        or calculate_file_sha256(publication_svg_path)
        != svg_checksum
    ):
        raise RuntimeError(
            "Stable publication copies do not match the immutable "
            "figure snapshot."
        )

    return (
        png_path,
        svg_path,
        metadata_path,
        publication_png_path,
        publication_svg_path,
    )


def main() -> None:
    """Verify the source table and generate the figure report."""

    args = parse_args()
    generated_at = datetime.now(timezone.utc)
    var_coverage, source_metadata = (
        load_subperiod_var_coverage_snapshot(
            args.subperiod_snapshot
        )
    )
    snapshot_id = args.snapshot_id

    if snapshot_id is None:
        timestamp = generated_at.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        experiment_name = source_metadata.get(
            "experiment",
            "experiment",
        )
        snapshot_id = (
            f"{timestamp}_{experiment_name}_"
            "subperiod_exception_rate_figure"
        )

    (
        png_path,
        svg_path,
        metadata_path,
        publication_png_path,
        publication_svg_path,
    ) = save_subperiod_exception_rate_report(
        var_coverage=var_coverage,
        source_snapshot=args.subperiod_snapshot,
        source_metadata=source_metadata,
        output_root=args.output_root,
        publication_root=args.publication_root,
        snapshot_id=snapshot_id,
        generated_at=generated_at,
    )

    print("Descriptive subperiod exception-rate figure")
    print("Saved immutable PNG:", png_path)
    print("Saved immutable SVG:", svg_path)
    print("Saved metadata:", metadata_path)
    print("Updated publication PNG:", publication_png_path)
    print("Updated publication SVG:", publication_svg_path)


if __name__ == "__main__":
    main()
