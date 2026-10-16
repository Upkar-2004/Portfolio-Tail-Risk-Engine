"""Generate descriptive subperiod predictive-score figures."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from tailrisk.data import calculate_file_sha256
from tailrisk.visualization import (
    SUBPERIOD_PREDICTIVE_RANK_PALETTE,
    save_subperiod_predictive_first_place_figure,
    save_subperiod_predictive_rank_figure,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "evaluations"
_PUBLICATION_ROOT = _PROJECT_ROOT / "reports" / "figures"
_FIGURE_STEMS = {
    "first_place_summary": (
        "subperiod_predictive_first_place_summary"
    ),
    "annual_rank_grid": "subperiod_predictive_annual_ranks",
}
_COMPANION_TABLE_NAME = "subperiod_predictive_scores.csv"
_INDEX_NAMES = [
    "period_id",
    "model",
    "confidence_level",
]
_REQUIRED_COLUMNS = {
    "period_type",
    "first_forecast_date",
    "last_forecast_date",
    "observation_count",
    "is_partial",
    "mean_var_quantile_loss",
    "var_quantile_loss_rank",
    "mean_fz0_loss",
    "fz0_loss_rank",
}


def parse_args() -> argparse.Namespace:
    """Parse the verified table snapshot and figure destinations."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate checksum-traced summary and annual figures for "
            "calendar-year and 2020 NBER-period VaR quantile loss "
            "and joint VaR-Expected-Shortfall FZ0 loss."
        )
    )
    parser.add_argument(
        "subperiod_snapshot",
        type=Path,
        help=(
            "Path to the verified subperiod-analysis snapshot "
            "containing metadata.json and the predictive-accuracy "
            "CSV."
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


def load_subperiod_predictive_accuracy_snapshot(
    snapshot_directory: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load predictive scores after source checksum verification."""

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
        expected_checksum = file_metadata["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Source metadata does not describe the subperiod "
            "predictive-accuracy table."
        ) from error

    actual_checksum = calculate_file_sha256(table_path)

    if actual_checksum != expected_checksum:
        raise ValueError(
            "Subperiod predictive-accuracy checksum does not match "
            "its source metadata."
        )

    predictive_accuracy = pd.read_csv(
        table_path,
        index_col=_INDEX_NAMES,
    )

    if list(predictive_accuracy.index.names) != _INDEX_NAMES:
        raise ValueError(
            "Subperiod predictive accuracy uses unexpected index "
            "levels."
        )

    if len(predictive_accuracy) != table_metadata.get("rows"):
        raise ValueError(
            "Subperiod predictive-accuracy row count does not match "
            "its source metadata."
        )

    recorded_columns = table_metadata.get("columns")

    if (
        not isinstance(recorded_columns, list)
        or list(predictive_accuracy.columns) != recorded_columns
    ):
        raise ValueError(
            "Subperiod predictive-accuracy columns do not match "
            "its source metadata."
        )

    missing_columns = (
        _REQUIRED_COLUMNS - set(predictive_accuracy.columns)
    )

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Subperiod predictive accuracy is missing required "
            f"column(s): {missing_names}."
        )

    return predictive_accuracy, metadata


def create_subperiod_predictive_score_companion_table(
    predictive_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    """Select period-average scores and ranks for publication."""

    columns = [
        "period_id",
        "period_type",
        "model",
        "confidence_level",
        "observation_count",
        "mean_var_quantile_loss",
        "var_quantile_loss_rank",
        "mean_fz0_loss",
        "fz0_loss_rank",
    ]
    return (
        predictive_accuracy.reset_index()
        .loc[:, columns]
        .copy()
    )


def save_subperiod_predictive_score_report(
    predictive_accuracy: pd.DataFrame,
    source_snapshot: str | Path,
    source_metadata: dict[str, Any],
    output_root: str | Path,
    publication_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
) -> tuple[dict[str, Path], Path, dict[str, Path]]:
    """Save immutable figures, table, metadata, and stable copies."""

    if not isinstance(snapshot_id, str):
        raise TypeError("Snapshot ID must be a string.")

    if not snapshot_id.strip():
        raise ValueError("Snapshot ID must not be empty.")

    source_snapshot_path = Path(source_snapshot)
    source_table_metadata = source_metadata[
        "output"
    ]["tables"]["subperiod_predictive_accuracy"]
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    report_directory = output_root_path / snapshot_id
    report_directory.mkdir(parents=False, exist_ok=False)
    source_table_path = (
        source_snapshot_path
        / source_table_metadata["file"]["name"]
    )
    source_table_copy_path = (
        report_directory / source_table_path.name
    )
    shutil.copyfile(source_table_path, source_table_copy_path)

    if (
        calculate_file_sha256(source_table_copy_path)
        != source_table_metadata["file"]["sha256"]
    ):
        raise RuntimeError(
            "Copied predictive-accuracy table does not match its "
            "verified source checksum."
        )

    figure_paths: dict[str, dict[str, Path]] = {}

    for figure_name, figure_stem in _FIGURE_STEMS.items():
        figure_paths[figure_name] = {
            "png": report_directory / f"{figure_stem}.png",
            "svg": report_directory / f"{figure_stem}.svg",
        }

    save_subperiod_predictive_first_place_figure(
        predictive_accuracy=predictive_accuracy,
        png_output_path=figure_paths[
            "first_place_summary"
        ]["png"],
        svg_output_path=figure_paths[
            "first_place_summary"
        ]["svg"],
    )
    save_subperiod_predictive_rank_figure(
        predictive_accuracy=predictive_accuracy,
        png_output_path=figure_paths["annual_rank_grid"]["png"],
        svg_output_path=figure_paths["annual_rank_grid"]["svg"],
    )
    companion_table = (
        create_subperiod_predictive_score_companion_table(
            predictive_accuracy
        )
    )
    companion_table_path = (
        report_directory / _COMPANION_TABLE_NAME
    )
    companion_table.to_csv(companion_table_path, index=False)
    figure_checksums = {
        figure_name: {
            output_format: calculate_file_sha256(output_path)
            for output_format, output_path in outputs.items()
        }
        for figure_name, outputs in figure_paths.items()
    }
    companion_table_checksum = calculate_file_sha256(
        companion_table_path
    )
    publication_root_path = Path(publication_root)
    publication_paths: dict[str, Path] = {
        (
            f"{figure_name}_{output_format}"
        ): publication_root_path / output_path.name
        for figure_name, outputs in figure_paths.items()
        for output_format, output_path in outputs.items()
    }
    publication_table_path = (
        publication_root_path.parent
        / "tables"
        / _COMPANION_TABLE_NAME
    )
    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "experiment": source_metadata.get("experiment"),
        "report": {
            "name": "subperiod_predictive_score_figures",
            "classification": "descriptive_secondary_analysis",
            "scores": {
                "var_quantile_loss": {
                    "display_label": "QL",
                    "target": "VaR",
                },
                "fz0_loss": {
                    "display_label": "FZ0",
                    "target": "joint_VaR_ES",
                    "convention": "positive_loss",
                },
            },
            "score_direction": "lower_is_better",
            "rank_scope": "within_period_confidence_and_score",
            "tie_method": "minimum_rank",
            "layout": {
                "main_figure": "first_place_counts_by_score",
                "supporting_figure": (
                    "annual_ranks_grouped_by_score"
                ),
                "confidence_level_panels": [0.95, 0.975, 0.99],
                "calendar_years": list(range(2012, 2026)),
                "nber_period": "2020-02-01_to_2020-04-30",
                "nber_period_is_overlapping_case_study": True,
                "first_place_summary_scope": (
                    "calendar_years_2012_to_2025_only"
                ),
            },
            "annual_grid_encoding": {
                "text": "minimum_rank_1_to_3",
                "shading": "rank_1_model_house_color",
                "palette": SUBPERIOD_PREDICTIVE_RANK_PALETTE,
                "hac_standardized_shading": False,
            },
            "summary_encoding": {
                "measure": "calendar_year_first_place_count",
                "reference": "even_split_14_divided_by_3",
                "reference_is_inferential_test": False,
            },
            "data_description": {
                "portfolio": "equal_weight_11_stock_portfolio",
                "frequency": "daily_returns",
                "forecast_horizon": "one_session_ahead",
                "forecast_period": "2012-01_to_2025-12",
            },
            "model_specifications": {
                "rolling_gaussian": {
                    "estimation_window_sessions": 504,
                    "shock_distribution": "standard_normal",
                },
                "ewma_gaussian": {
                    "decay_factor": 0.94,
                    "shock_distribution": "standard_normal",
                },
                "filtered_historical_simulation": {
                    "covariance_filter": "ewma_0.94",
                    "residual_pool_sessions": 504,
                    "sampling": "complete_residual_vectors",
                },
                "all_models": {
                    "scenario_count_per_forecast": 100000,
                    "var_estimator": "empirical_linear_quantile",
                    "es_estimator": "empirical_tail_mean",
                },
            },
            "typography": {
                "preferred_font": "Helvetica",
                "fallback_fonts": ["Arial", "DejaVu Sans"],
                "figure_sizes_inches": {
                    "first_place_summary": [7.5, 3.35],
                    "annual_rank_grid": [7.5, 5.65],
                },
            },
            "inferential_status": (
                "descriptive_only; no subperiod hypothesis tests; "
                "formal inference remains the full-sample Holm-"
                "adjusted Diebold-Mariano analysis"
            ),
            "png_dpi": 300,
        },
        "source": {
            "snapshot_id": source_snapshot_path.name,
            "report_name": source_metadata["report"]["name"],
            "predictive_accuracy": {
                "name": source_table_metadata["file"]["name"],
                "sha256": source_table_metadata[
                    "file"
                ]["sha256"],
            },
        },
        "output": {
            "source_table_copy": {
                "name": source_table_copy_path.name,
                "sha256": source_table_metadata[
                    "file"
                ]["sha256"],
            },
            "figures": {
                figure_name: {
                    output_format: {
                        "name": output_path.name,
                        "sha256": figure_checksums[
                            figure_name
                        ][output_format],
                    }
                    for output_format, output_path in outputs.items()
                }
                for figure_name, outputs in figure_paths.items()
            },
            "tables": {
                "predictive_score_companion": {
                    "name": companion_table_path.name,
                    "rows": len(companion_table),
                    "columns": list(companion_table.columns),
                    "sha256": companion_table_checksum,
                },
            },
            "publication_copies": {
                **{
                    output_name: {
                        "name": publication_path.name,
                        "sha256": figure_checksums[
                            output_name.rsplit("_", 1)[0]
                        ][output_name.rsplit("_", 1)[1]],
                    }
                    for output_name, publication_path
                    in publication_paths.items()
                },
                "predictive_score_companion_csv": {
                    "name": publication_table_path.name,
                    "sha256": companion_table_checksum,
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

    publication_root_path.mkdir(parents=True, exist_ok=True)
    publication_table_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_paths = {
        f"{figure_name}_{output_format}": output_path
        for figure_name, outputs in figure_paths.items()
        for output_format, output_path in outputs.items()
    }
    snapshot_paths["predictive_score_companion_csv"] = (
        companion_table_path
    )
    publication_paths["predictive_score_companion_csv"] = (
        publication_table_path
    )

    for output_name, snapshot_path in snapshot_paths.items():
        shutil.copyfile(
            snapshot_path,
            publication_paths[output_name],
        )

        if calculate_file_sha256(
            publication_paths[output_name]
        ) != calculate_file_sha256(snapshot_path):
            raise RuntimeError(
                "Stable publication copy does not match the "
                f"immutable {output_name} snapshot."
            )

    return snapshot_paths, metadata_path, publication_paths


def main() -> None:
    """Verify the source table and generate the figure report."""

    args = parse_args()
    generated_at = datetime.now(timezone.utc)
    predictive_accuracy, source_metadata = (
        load_subperiod_predictive_accuracy_snapshot(
            args.subperiod_snapshot
        )
    )
    snapshot_id = args.snapshot_id

    if snapshot_id is None:
        timestamp = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
        experiment_name = source_metadata.get(
            "experiment",
            "experiment",
        )
        snapshot_id = (
            f"{timestamp}_{experiment_name}_"
            "subperiod_predictive_score_figures"
        )

    snapshot_paths, metadata_path, publication_paths = (
        save_subperiod_predictive_score_report(
            predictive_accuracy=predictive_accuracy,
            source_snapshot=args.subperiod_snapshot,
            source_metadata=source_metadata,
            output_root=args.output_root,
            publication_root=args.publication_root,
            snapshot_id=snapshot_id,
            generated_at=generated_at,
        )
    )

    print("Descriptive subperiod predictive-score figures")

    for label, path in snapshot_paths.items():
        print(f"Saved immutable {label}:", path)

    print("Saved metadata:", metadata_path)

    for label, path in publication_paths.items():
        print(f"Updated publication {label}:", path)


if __name__ == "__main__":
    main()
