"""Generate the full-sample predictive-accuracy inference figure."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from tailrisk.data import calculate_file_sha256
from tailrisk.figures.inference import (
    prepare_predictive_inference_table,
    save_predictive_inference_figure,
)
from tailrisk.figures.predictive import (
    prepare_subperiod_predictive_rank_matrices,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "evaluations"
_PUBLICATION_ROOT = _PROJECT_ROOT / "reports" / "figures"
_TABLE_PUBLICATION_ROOT = _PROJECT_ROOT / "reports" / "tables"
_FIGURE_STEM = "predictive_accuracy_dm_comparison"
_COMPANION_TABLE_NAME = "predictive_accuracy_dm_comparisons.csv"
_DM_INDEX_NAMES = [
    "score_name",
    "confidence_level",
    "model_a",
    "model_b",
]
_SUBPERIOD_INDEX_NAMES = [
    "period_id",
    "model",
    "confidence_level",
]


def parse_args() -> argparse.Namespace:
    """Parse verified inference inputs and report destinations."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a checksum-traced full-sample predictive-"
            "accuracy figure, magnitude table, and bandwidth-"
            "sensitivity summary."
        )
    )
    parser.add_argument(
        "predictive_accuracy_snapshot",
        type=Path,
        help=(
            "Verified automatic-bandwidth predictive-accuracy "
            "snapshot."
        ),
    )
    parser.add_argument(
        "subperiod_snapshot",
        type=Path,
        help=(
            "Verified subperiod-analysis snapshot used to recover "
            "full-sample model mean scores."
        ),
    )
    parser.add_argument(
        "sensitivity_snapshot",
        type=Path,
        help=(
            "Verified bandwidth-zero predictive-accuracy snapshot."
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
        help="Directory for stable PNG and SVG publication copies.",
    )
    parser.add_argument(
        "--table-publication-root",
        type=Path,
        default=_TABLE_PUBLICATION_ROOT,
        help="Directory for the stable companion CSV.",
    )
    parser.add_argument(
        "--snapshot-id",
        type=str,
        default=None,
        help=(
            "Optional report snapshot ID. A timestamped ID is "
            "generated when omitted."
        ),
    )

    return parser.parse_args()


def load_predictive_accuracy_snapshot(
    snapshot_directory: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and checksum-verify one predictive-accuracy snapshot."""

    snapshot_path = Path(snapshot_directory)
    metadata_path = snapshot_path / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    if metadata.get("report", {}).get("name") != (
        "predictive_accuracy"
    ):
        raise ValueError(
            "Source snapshot is not a predictive-accuracy report."
        )

    try:
        file_metadata = metadata["output"]["file"]
        table_path = snapshot_path / file_metadata["name"]
        expected_checksum = file_metadata["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Predictive-accuracy metadata does not describe its "
            "result table."
        ) from error

    if calculate_file_sha256(table_path) != expected_checksum:
        raise ValueError(
            "Predictive-accuracy table checksum does not match "
            "its source metadata."
        )

    comparisons = pd.read_csv(
        table_path,
        index_col=_DM_INDEX_NAMES,
    )

    if list(comparisons.index.names) != _DM_INDEX_NAMES:
        raise ValueError(
            "Predictive-accuracy results use unexpected index levels."
        )

    if len(comparisons) != metadata["output"].get("rows"):
        raise ValueError(
            "Predictive-accuracy row count does not match metadata."
        )

    if list(comparisons.columns) != metadata["output"].get(
        "columns"
    ):
        raise ValueError(
            "Predictive-accuracy columns do not match metadata."
        )

    return comparisons, metadata


def load_subperiod_predictive_snapshot(
    snapshot_directory: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the checksum-verified subperiod predictive-score table."""

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
        ]["tables"]["subperiod_predictive_accuracy"]
        file_metadata = table_metadata["file"]
        table_path = snapshot_path / file_metadata["name"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Subperiod metadata does not describe the predictive-"
            "accuracy table."
        ) from error

    if (
        calculate_file_sha256(table_path)
        != file_metadata["sha256"]
    ):
        raise ValueError(
            "Subperiod predictive-accuracy checksum does not match "
            "its source metadata."
        )

    predictive_accuracy = pd.read_csv(
        table_path,
        index_col=_SUBPERIOD_INDEX_NAMES,
    )

    if len(predictive_accuracy) != table_metadata.get("rows"):
        raise ValueError(
            "Subperiod predictive-accuracy row count does not "
            "match metadata."
        )

    if list(predictive_accuracy.columns) != table_metadata.get(
        "columns"
    ):
        raise ValueError(
            "Subperiod predictive-accuracy columns do not match "
            "metadata."
        )

    prepare_subperiod_predictive_rank_matrices(
        predictive_accuracy
    )

    return predictive_accuracy, metadata


def calculate_full_sample_model_scores(
    subperiod_predictive_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    """Recover full-sample means from disjoint calendar-year rows."""

    calendar_rows = subperiod_predictive_accuracy.loc[
        subperiod_predictive_accuracy["period_type"].astype(str)
        == "calendar_year"
    ]
    records = []

    for (model, confidence_level), group in calendar_rows.groupby(
        level=["model", "confidence_level"],
        sort=False,
    ):
        observation_counts = group[
            "observation_count"
        ].to_numpy(dtype=float)

        if (
            len(group) != 14
            or not np.isfinite(observation_counts).all()
            or (observation_counts <= 0.0).any()
        ):
            raise ValueError(
                "Full-sample model means require fourteen positive "
                "calendar-year observation counts per model and level."
            )

        records.append(
            {
                "model": model,
                "confidence_level": float(confidence_level),
                "observation_count": int(observation_counts.sum()),
                "mean_var_quantile_loss": float(
                    np.average(
                        group[
                            "mean_var_quantile_loss"
                        ].to_numpy(dtype=float),
                        weights=observation_counts,
                    )
                ),
                "mean_fz0_loss": float(
                    np.average(
                        group["mean_fz0_loss"].to_numpy(
                            dtype=float
                        ),
                        weights=observation_counts,
                    )
                ),
            }
        )

    return pd.DataFrame.from_records(records).set_index(
        ["model", "confidence_level"]
    )


def validate_inference_lineage(
    baseline_metadata: dict[str, Any],
    sensitivity_metadata: dict[str, Any],
    subperiod_metadata: dict[str, Any],
) -> None:
    """Require common experiment lineage and predeclared inference rules."""

    experiments = {
        baseline_metadata.get("experiment"),
        sensitivity_metadata.get("experiment"),
        subperiod_metadata.get("experiment"),
    }

    if len(experiments) != 1:
        raise ValueError(
            "Inference inputs must use the same experiment."
        )

    if baseline_metadata.get("source_backtests") != (
        sensitivity_metadata.get("source_backtests")
    ):
        raise ValueError(
            "Baseline and sensitivity reports must use identical "
            "backtest snapshots."
        )

    baseline_report = baseline_metadata["report"]
    sensitivity_report = sensitivity_metadata["report"]

    if baseline_report["hac"].get("requested_bandwidth") is not None:
        raise ValueError(
            "The primary predictive-accuracy report must use the "
            "predeclared automatic HAC bandwidth."
        )

    if sensitivity_report["hac"].get("requested_bandwidth") != 0:
        raise ValueError(
            "The sensitivity predictive-accuracy report must use "
            "HAC bandwidth zero."
        )

    for report in (baseline_report, sensitivity_report):
        adjustment = report["multiplicity_adjustment"]
        families = adjustment.get("families")

        if (
            adjustment.get("method") != "holm"
            or not adjustment.get(
                "applied_separately_by_score_family"
            )
            or families
            != [
                {
                    "hypothesis_count": 9,
                    "score_name": "var_quantile_loss",
                },
                {
                    "hypothesis_count": 9,
                    "score_name": "fz0_loss",
                },
            ]
        ):
            raise ValueError(
                "Inference inputs must use separate nine-test Holm "
                "families for QL and FZ0."
            )


def save_predictive_accuracy_figure_report(
    inference_table: pd.DataFrame,
    baseline_snapshot: str | Path,
    baseline_metadata: dict[str, Any],
    subperiod_snapshot: str | Path,
    subperiod_metadata: dict[str, Any],
    sensitivity_snapshot: str | Path,
    sensitivity_metadata: dict[str, Any],
    output_root: str | Path,
    publication_root: str | Path,
    table_publication_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
) -> tuple[dict[str, Path], Path, dict[str, Path]]:
    """Save immutable and stable figure-report artifacts."""

    if not isinstance(snapshot_id, str):
        raise TypeError("Snapshot ID must be a string.")

    if not snapshot_id.strip():
        raise ValueError("Snapshot ID must not be empty.")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    report_directory = output_root_path / snapshot_id
    report_directory.mkdir(parents=False, exist_ok=False)
    source_specs = {
        "baseline": (
            Path(baseline_snapshot),
            baseline_metadata,
            "baseline_predictive_accuracy.csv",
        ),
        "subperiod": (
            Path(subperiod_snapshot),
            subperiod_metadata,
            "subperiod_predictive_accuracy.csv",
        ),
        "bandwidth_zero": (
            Path(sensitivity_snapshot),
            sensitivity_metadata,
            "bandwidth_zero_predictive_accuracy.csv",
        ),
    }
    source_records: dict[str, dict[str, object]] = {}

    for source_name, (
        source_directory,
        metadata,
        copied_name,
    ) in source_specs.items():
        if source_name == "subperiod":
            file_metadata = metadata[
                "output"
            ]["tables"]["subperiod_predictive_accuracy"]["file"]
        else:
            file_metadata = metadata["output"]["file"]

        source_path = source_directory / file_metadata["name"]
        copied_path = report_directory / copied_name
        shutil.copyfile(source_path, copied_path)

        if calculate_file_sha256(copied_path) != file_metadata["sha256"]:
            raise RuntimeError(
                f"Copied {source_name} source table failed checksum "
                "verification."
            )

        source_records[source_name] = {
            "snapshot_id": source_directory.name,
            "source_file": file_metadata["name"],
            "source_sha256": file_metadata["sha256"],
            "copied_file": copied_path.name,
            "copied_sha256": calculate_file_sha256(copied_path),
        }

    png_path = report_directory / f"{_FIGURE_STEM}.png"
    svg_path = report_directory / f"{_FIGURE_STEM}.svg"
    table_path = report_directory / _COMPANION_TABLE_NAME
    save_predictive_inference_figure(
        inference_table=inference_table,
        png_output_path=png_path,
        svg_output_path=svg_path,
    )
    inference_table.reset_index().to_csv(table_path, index=False)
    decision_changes = inference_table.loc[
        inference_table["sensitivity_decision_changed"].astype(bool)
    ]

    if (
        not decision_changes.empty
        and (
            decision_changes[
                "holm_reject_equal_accuracy"
            ].astype(bool).any()
            or not decision_changes[
                "sensitivity_holm_reject_equal_accuracy"
            ].astype(bool).all()
        )
    ):
        raise ValueError(
            "The dagger definition requires every bandwidth-zero "
            "decision change to be a primary non-rejection that "
            "becomes a rejection."
        )

    changed_comparisons = [
        {
            "score_name": score_name,
            "confidence_level": float(confidence_level),
            "comparison": comparison,
            "baseline_holm_reject": bool(
                row["holm_reject_equal_accuracy"]
            ),
            "bandwidth_zero_holm_reject": bool(
                row[
                    "sensitivity_holm_reject_equal_accuracy"
                ]
            ),
        }
        for (
            score_name,
            confidence_level,
            comparison,
        ), row in decision_changes.iterrows()
    ]
    output_checksums = {
        "png": calculate_file_sha256(png_path),
        "svg": calculate_file_sha256(svg_path),
        "csv": calculate_file_sha256(table_path),
    }
    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "experiment": baseline_metadata["experiment"],
        "report": {
            "name": "predictive_accuracy_figure",
            "classification": "formal_model_comparison_figure",
            "orientation": {
                "comparisons": [
                    "EWMA - RG",
                    "FHS - RG",
                    "FHS - EWMA",
                ],
                "negative_statistic": "first_model_better",
                "positive_statistic": "second_model_better",
            },
            "primary_inference": {
                "statistic": "diebold_mariano_form",
                "interpretation": (
                    "Giacomini-White unconditional predictive-"
                    "ability comparison of forecasting methods under "
                    "fixed 504-session rolling estimation windows"
                ),
                "hac_estimator": "newey_west_bartlett",
                "hac_bandwidth_rule": baseline_metadata[
                    "report"
                ]["hac"]["bandwidth_rule"],
                "hac_bandwidth": baseline_metadata[
                    "report"
                ]["hac"]["selected_bandwidths"],
                "holm_families": (
                    "nine QL tests and nine FZ0 tests, adjusted "
                    "separately"
                ),
            },
            "encoding": {
                "horizontal_position": "dm_statistic",
                "color": "lower_mean_loss_model_house_color",
                "marker": "lower_mean_loss_model_house_marker",
                "filled": "holm_adjusted_rejection_at_0.05",
                "open": "no_holm_adjusted_rejection_at_0.05",
                "right_column": "holm_adjusted_p_value",
                "dagger": (
                    "significant_at_0.05_with_bandwidth_zero_but_"
                    "not_with_primary_bandwidth"
                ),
                "critical_value_lines": False,
            },
            "robustness": {
                "alternative_hac_bandwidth": 0,
                "decision_change_count": len(changed_comparisons),
                "changed_comparisons": changed_comparisons,
            },
            "magnitude_table": {
                "ql_relative_difference_definition": (
                    "100 * (mean_score_model_a - mean_score_model_b) "
                    "/ mean_score_model_b"
                ),
                "fz0_relative_difference_reported": False,
            },
            "caption": (
                "Full-sample Diebold-Mariano statistics for pairwise "
                "differences in mean loss, oriented as first model "
                "minus second model. Marker shape and color identify "
                "the model with lower mean loss; filled markers denote "
                "Holm-adjusted p < 0.05 within each nine-test score "
                "family. Holm-adjusted p-values are monotone, so ties "
                "occur. Standard errors use Newey-West HAC with "
                "Bartlett weights and the automatic bandwidth rule, "
                "which selects K = 8 for the 3,519-session baseline. "
                "The dagger marks comparisons significant at 5% with "
                "HAC lag zero but not under the primary specification. "
                "With fixed 504-session rolling estimation windows, "
                "the statistic is interpreted as a Giacomini-White "
                "unconditional test of forecasting methods."
            ),
        },
        "source": source_records,
        "output": {
            "figure": {
                "png": {
                    "name": png_path.name,
                    "sha256": output_checksums["png"],
                },
                "svg": {
                    "name": svg_path.name,
                    "sha256": output_checksums["svg"],
                },
            },
            "companion_table": {
                "name": table_path.name,
                "sha256": output_checksums["csv"],
                "rows": len(inference_table),
                "columns": list(inference_table.reset_index().columns),
            },
        },
    }
    metadata_path = report_directory / "metadata.json"

    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    publication_root_path = Path(publication_root)
    table_publication_root_path = Path(table_publication_root)
    publication_root_path.mkdir(parents=True, exist_ok=True)
    table_publication_root_path.mkdir(parents=True, exist_ok=True)
    publication_paths = {
        "png": publication_root_path / png_path.name,
        "svg": publication_root_path / svg_path.name,
        "csv": table_publication_root_path / table_path.name,
    }
    snapshot_paths = {
        "png": png_path,
        "svg": svg_path,
        "csv": table_path,
    }

    for output_name, snapshot_path in snapshot_paths.items():
        shutil.copyfile(
            snapshot_path,
            publication_paths[output_name],
        )

        if calculate_file_sha256(
            publication_paths[output_name]
        ) != calculate_file_sha256(snapshot_path):
            raise RuntimeError(
                f"Stable {output_name} publication copy does not "
                "match its immutable snapshot."
            )

    return snapshot_paths, metadata_path, publication_paths


def main() -> None:
    """Verify sources and generate the inference figure report."""

    args = parse_args()
    generated_at = datetime.now(timezone.utc)
    baseline, baseline_metadata = load_predictive_accuracy_snapshot(
        args.predictive_accuracy_snapshot
    )
    subperiod, subperiod_metadata = (
        load_subperiod_predictive_snapshot(
            args.subperiod_snapshot
        )
    )
    sensitivity, sensitivity_metadata = (
        load_predictive_accuracy_snapshot(
            args.sensitivity_snapshot
        )
    )
    validate_inference_lineage(
        baseline_metadata=baseline_metadata,
        sensitivity_metadata=sensitivity_metadata,
        subperiod_metadata=subperiod_metadata,
    )
    model_mean_scores = calculate_full_sample_model_scores(
        subperiod
    )
    inference_table = prepare_predictive_inference_table(
        comparisons=baseline,
        sensitivity_comparisons=sensitivity,
        model_mean_scores=model_mean_scores,
    )
    snapshot_id = args.snapshot_id

    if snapshot_id is None:
        timestamp = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
        snapshot_id = (
            f"{timestamp}_{baseline_metadata['experiment']}_"
            "predictive_accuracy_figure"
        )

    snapshot_paths, metadata_path, publication_paths = (
        save_predictive_accuracy_figure_report(
            inference_table=inference_table,
            baseline_snapshot=args.predictive_accuracy_snapshot,
            baseline_metadata=baseline_metadata,
            subperiod_snapshot=args.subperiod_snapshot,
            subperiod_metadata=subperiod_metadata,
            sensitivity_snapshot=args.sensitivity_snapshot,
            sensitivity_metadata=sensitivity_metadata,
            output_root=args.output_root,
            publication_root=args.publication_root,
            table_publication_root=args.table_publication_root,
            snapshot_id=snapshot_id,
            generated_at=generated_at,
        )
    )

    print("Full-sample predictive-accuracy inference figure")

    for output_name, output_path in snapshot_paths.items():
        print(f"Saved immutable {output_name}:", output_path)

    print("Saved metadata:", metadata_path)

    for output_name, output_path in publication_paths.items():
        print(f"Updated publication {output_name}:", output_path)


if __name__ == "__main__":
    main()
