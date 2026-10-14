"""Generate and save the complete descriptive subperiod report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tailrisk.backtesting import (
    create_forecast_schedule,
    load_backtest_snapshot,
)
from tailrisk.config import load_config
from tailrisk.data import calculate_file_sha256
from tailrisk.evaluation import (
    create_subperiod_definitions,
    evaluate_models_subperiod_es_tail_severity,
    evaluate_models_subperiod_predictive_accuracy,
    evaluate_models_subperiod_var_coverage,
    evaluate_models_subperiod_var_es_identification,
)
from tailrisk.portfolio import load_portfolio_path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "evaluations"
_INTERVAL_CONFIDENCE_LEVEL = 0.95

_MODEL_ARGUMENTS = (
    (
        "rolling_gaussian_monte_carlo",
        "rolling_backtest_snapshot",
    ),
    (
        "ewma_gaussian_monte_carlo",
        "ewma_backtest_snapshot",
    ),
    (
        "filtered_historical_simulation",
        "filtered_historical_backtest_snapshot",
    ),
)

_TABLE_SPECS = {
    "subperiod_definitions": {
        "filename": "subperiod_definitions.csv",
        "index_names": ["period_id"],
        "required_columns": [
            "period_type",
            "label",
            "requested_start_date",
            "requested_end_date",
            "first_forecast_date",
            "last_forecast_date",
            "observation_count",
            "is_partial",
            "source",
        ],
    },
    "subperiod_var_coverage": {
        "filename": "subperiod_var_coverage.csv",
        "index_names": [
            "period_id",
            "model",
            "confidence_level",
        ],
        "required_columns": [
            "period_type",
            "first_forecast_date",
            "last_forecast_date",
            "observation_count",
            "is_partial",
            "nominal_exceedance_rate",
            "expected_exceedance_count",
            "observed_exceedance_count",
            "observed_exceedance_rate",
            "observed_minus_expected_count",
            "observed_minus_expected_rate",
            "interval_confidence_level",
            "exact_interval_lower_bound",
            "exact_interval_upper_bound",
            "mean_value_at_risk",
            "mean_expected_shortfall",
            "sparse_expected_tail",
            "sparse_observed_tail",
        ],
    },
    "subperiod_es_tail_severity": {
        "filename": "subperiod_es_tail_severity.csv",
        "index_names": [
            "period_id",
            "model",
            "confidence_level",
        ],
        "required_columns": [
            "period_type",
            "first_forecast_date",
            "last_forecast_date",
            "observation_count",
            "is_partial",
            "nominal_exceedance_rate",
            "expected_exceedance_count",
            "var_exceedance_count",
            "mean_exceedance_loss",
            "mean_var_on_exceedance_days",
            "mean_es_on_exceedance_days",
            "mean_var_excess",
            "median_var_excess",
            "maximum_var_excess",
            "mean_es_gap",
            "sparse_expected_tail",
            "sparse_observed_tail",
        ],
    },
    "subperiod_joint_calibration": {
        "filename": "subperiod_joint_calibration.csv",
        "index_names": [
            "period_id",
            "model",
            "confidence_level",
        ],
        "required_columns": [
            "period_type",
            "first_forecast_date",
            "last_forecast_date",
            "observation_count",
            "is_partial",
            "expected_exceedance_count",
            "var_exceedance_count",
            "mean_tail_contribution",
            "mean_expected_shortfall",
            "mean_identification_residual",
            "sparse_expected_tail",
            "sparse_observed_tail",
        ],
    },
    "subperiod_predictive_accuracy": {
        "filename": "subperiod_predictive_accuracy.csv",
        "index_names": [
            "period_id",
            "model",
            "confidence_level",
        ],
        "required_columns": [
            "period_type",
            "first_forecast_date",
            "last_forecast_date",
            "observation_count",
            "is_partial",
            "expected_exceedance_count",
            "var_exceedance_count",
            "mean_var_quantile_loss",
            "var_quantile_loss_rank",
            "mean_fz0_loss",
            "fz0_loss_rank",
            "sparse_expected_tail",
            "sparse_observed_tail",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    """Parse source snapshots and report output options."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the five-table descriptive calendar-year and "
            "NBER subperiod analysis from three verified backtests."
        )
    )
    parser.add_argument(
        "portfolio_snapshot",
        type=Path,
        help="Path to the verified portfolio-path snapshot.",
    )
    parser.add_argument(
        "rolling_backtest_snapshot",
        type=Path,
        help="Path to the rolling Gaussian backtest snapshot.",
    )
    parser.add_argument(
        "ewma_backtest_snapshot",
        type=Path,
        help="Path to the EWMA Gaussian backtest snapshot.",
    )
    parser.add_argument(
        "filtered_historical_backtest_snapshot",
        type=Path,
        help="Path to the filtered historical backtest snapshot.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Configuration file. Defaults to configs/baseline.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_OUTPUT_ROOT,
        help="Root directory for saved evaluation reports.",
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


def validate_backtest_metadata(
    metadata: dict[str, Any],
    expected_model_name: str,
    expected_portfolio_snapshot_id: str,
    expected_estimation_window: int,
    expected_confidence_levels: list[float],
) -> None:
    """Validate a source backtest's identity and experiment lineage."""

    model_metadata = metadata["model"]

    if model_metadata["name"] != expected_model_name:
        raise ValueError(
            "Backtest model mismatch: expected "
            f"{expected_model_name}, received "
            f"{model_metadata['name']}."
        )

    if (
        model_metadata["estimation_window"]
        != expected_estimation_window
    ):
        raise ValueError(
            "Backtest estimation window does not match "
            "the evaluation configuration."
        )

    observed_levels = [
        float(level)
        for level in model_metadata["confidence_levels"]
    ]
    expected_levels = [
        float(level)
        for level in expected_confidence_levels
    ]

    if observed_levels != expected_levels:
        raise ValueError(
            "Backtest confidence levels do not match "
            "the evaluation configuration."
        )

    source_portfolio_id = metadata[
        "source"
    ]["portfolio"]["snapshot_id"]

    if source_portfolio_id != expected_portfolio_snapshot_id:
        raise ValueError(
            "Backtest does not use the requested "
            "portfolio snapshot."
        )


def _validate_report_table(
    table_name: str,
    table: pd.DataFrame,
) -> None:
    """Validate one table against the fixed report schema."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"{table_name} must be a pandas DataFrame."
        )

    if table.empty:
        raise ValueError(
            f"{table_name} must not be empty."
        )

    spec = _TABLE_SPECS[table_name]
    expected_index_names = spec["index_names"]

    if list(table.index.names) != expected_index_names:
        raise ValueError(
            f"{table_name} must use index levels: "
            f"{', '.join(expected_index_names)}."
        )

    missing_columns = (
        set(spec["required_columns"])
        - set(table.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            f"{table_name} is missing required "
            f"column(s): {missing_names}."
        )


def save_subperiod_analysis_report(
    period_definitions: pd.DataFrame,
    var_coverage: pd.DataFrame,
    es_tail_severity: pd.DataFrame,
    joint_calibration: pd.DataFrame,
    predictive_accuracy: pd.DataFrame,
    source_snapshots: dict[str, Path],
    source_metadata: dict[str, dict[str, Any]],
    output_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
    experiment_name: str,
) -> tuple[dict[str, Path], Path]:
    """Save five immutable subperiod tables and checksum metadata."""

    tables = {
        "subperiod_definitions": period_definitions,
        "subperiod_var_coverage": var_coverage,
        "subperiod_es_tail_severity": es_tail_severity,
        "subperiod_joint_calibration": joint_calibration,
        "subperiod_predictive_accuracy": predictive_accuracy,
    }

    for table_name, table in tables.items():
        _validate_report_table(
            table_name=table_name,
            table=table,
        )

    detail_tables = [
        var_coverage,
        es_tail_severity,
        joint_calibration,
        predictive_accuracy,
    ]
    reference_index = var_coverage.index

    if any(
        not table.index.equals(reference_index)
        for table in detail_tables[1:]
    ):
        raise ValueError(
            "All subperiod result tables must use the same "
            "ordered period-model-confidence index."
        )

    period_ids = list(period_definitions.index)
    detail_period_ids = list(
        reference_index.get_level_values(
            "period_id"
        ).unique()
    )

    if period_ids != detail_period_ids:
        raise ValueError(
            "Subperiod result tables must use the ordered period "
            "definitions."
        )

    models = list(
        reference_index.get_level_values(
            "model"
        ).unique()
    )
    confidence_levels = [
        float(level)
        for level in reference_index.get_level_values(
            "confidence_level"
        ).unique()
    ]

    if list(source_snapshots) != models:
        raise ValueError(
            "Source snapshots must use the same ordered models "
            "as the subperiod tables."
        )

    if set(source_metadata) != set(source_snapshots):
        raise ValueError(
            "Source metadata must be provided for every source "
            "snapshot."
        )

    if not isinstance(snapshot_id, str):
        raise TypeError(
            "Snapshot ID must be a string."
        )

    if not snapshot_id.strip():
        raise ValueError(
            "Snapshot ID must not be empty."
        )

    output_root_path = Path(output_root)
    output_root_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_directory = output_root_path / snapshot_id
    report_directory.mkdir(
        parents=False,
        exist_ok=False,
    )
    table_paths: dict[str, Path] = {}

    for table_name, table in tables.items():
        table_path = (
            report_directory
            / str(_TABLE_SPECS[table_name]["filename"])
        )
        table.to_csv(
            table_path,
            index=True,
        )
        table_paths[table_name] = table_path

    source_records: dict[str, dict[str, str]] = {}

    for model_name, snapshot_path in (
        source_snapshots.items()
    ):
        metadata = source_metadata[model_name]
        source_records[model_name] = {
            "snapshot_id": snapshot_path.name,
            "forecast_file": metadata[
                "output"
            ]["file"]["name"],
            "forecast_sha256": metadata[
                "output"
            ]["file"]["sha256"],
        }

    output_records: dict[str, dict[str, object]] = {}

    for table_name, table in tables.items():
        table_path = table_paths[table_name]
        output_records[table_name] = {
            "rows": len(table),
            "index_names": list(table.index.names),
            "columns": list(table.columns),
            "file": {
                "name": table_path.name,
                "sha256": calculate_file_sha256(
                    table_path
                ),
            },
        }

    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "experiment": experiment_name,
        "report": {
            "name": "subperiod_analysis",
            "classification": (
                "descriptive_secondary_analysis"
            ),
            "inferential_status": {
                "hypothesis_tests": "none",
                "rejection_decisions": "none",
                "multiplicity_adjustment": "not_applicable",
                "exact_intervals": (
                    "pointwise_non_simultaneous"
                ),
            },
            "period_definitions": {
                "calendar_years": {
                    "first_year": 2012,
                    "last_year": 2025,
                    "partial_year_threshold_fraction": 0.90,
                },
                "nber_recession_2020": {
                    "published_peak_month": "2020-02",
                    "published_trough_month": "2020-04",
                    "requested_start_date": "2020-02-01",
                    "requested_end_date": "2020-04-30",
                    "mapping": (
                        "all common forecast sessions in peak-"
                        "through-trough calendar months inclusive"
                    ),
                    "source": (
                        "https://www.nber.org/research/"
                        "business-cycle-dating"
                    ),
                },
            },
            "exception_rule": (
                "realized_loss > value_at_risk"
            ),
            "exact_interval": {
                "method": "clopper_pearson",
                "confidence_level": (
                    _INTERVAL_CONFIDENCE_LEVEL
                ),
            },
            "sparse_tail_flags": {
                "sparse_expected_tail": (
                    "expected_exceedance_count < 5"
                ),
                "sparse_observed_tail": (
                    "var_exceedance_count < 5"
                ),
            },
            "predictive_accuracy": {
                "scores": [
                    "var_quantile_loss",
                    "fz0_loss",
                ],
                "score_direction": "lower_is_better",
                "rank_scope": (
                    "within_period_and_confidence_level"
                ),
                "tie_method": "minimum_rank",
            },
        },
        "source_backtests": source_records,
        "output": {
            "periods": period_ids,
            "models": models,
            "confidence_levels": confidence_levels,
            "tables": output_records,
        },
    }
    metadata_path = report_directory / "metadata.json"

    with metadata_path.open(
        "x",
        encoding="utf-8",
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    return table_paths, metadata_path


def main() -> None:
    """Load verified forecasts and save all subperiod tables."""

    args = parse_args()
    generated_at = datetime.now(timezone.utc)
    config = load_config(args.config)

    portfolio_path, _ = load_portfolio_path(
        args.portfolio_snapshot
    )
    forecasting_config = config["forecasting"]
    forecast_schedule = create_forecast_schedule(
        portfolio_returns=portfolio_path.daily[
            "portfolio_return"
        ],
        estimation_window=forecasting_config[
            "estimation_window"
        ],
    )
    snapshot_paths = {
        model_name: getattr(args, argument_name)
        for model_name, argument_name in _MODEL_ARGUMENTS
    }
    forecasts_by_model: dict[str, pd.DataFrame] = {}
    metadata_by_model: dict[
        str,
        dict[str, Any],
    ] = {}

    for model_name, snapshot_path in (
        snapshot_paths.items()
    ):
        forecasts, metadata = load_backtest_snapshot(
            snapshot_directory=snapshot_path,
            forecast_schedule=forecast_schedule,
        )
        validate_backtest_metadata(
            metadata=metadata,
            expected_model_name=model_name,
            expected_portfolio_snapshot_id=(
                args.portfolio_snapshot.name
            ),
            expected_estimation_window=(
                forecasting_config["estimation_window"]
            ),
            expected_confidence_levels=(
                forecasting_config["confidence_levels"]
            ),
        )
        forecasts_by_model[model_name] = forecasts
        metadata_by_model[model_name] = metadata

    period_definitions = create_subperiod_definitions(
        forecasts_by_model=forecasts_by_model,
    )
    var_coverage = evaluate_models_subperiod_var_coverage(
        forecasts_by_model=forecasts_by_model,
        interval_confidence_level=(
            _INTERVAL_CONFIDENCE_LEVEL
        ),
    )
    es_tail_severity = (
        evaluate_models_subperiod_es_tail_severity(
            forecasts_by_model=forecasts_by_model,
        )
    )
    joint_calibration = (
        evaluate_models_subperiod_var_es_identification(
            forecasts_by_model=forecasts_by_model,
        )
    )
    predictive_accuracy = (
        evaluate_models_subperiod_predictive_accuracy(
            forecasts_by_model=forecasts_by_model,
        )
    )
    snapshot_id = args.snapshot_id

    if snapshot_id is None:
        timestamp = generated_at.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        snapshot_id = (
            f"{timestamp}_"
            f"{config['experiment']['name']}_"
            "subperiod_analysis"
        )

    table_paths, metadata_path = (
        save_subperiod_analysis_report(
            period_definitions=period_definitions,
            var_coverage=var_coverage,
            es_tail_severity=es_tail_severity,
            joint_calibration=joint_calibration,
            predictive_accuracy=predictive_accuracy,
            source_snapshots=snapshot_paths,
            source_metadata=metadata_by_model,
            output_root=args.output_root,
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            experiment_name=config[
                "experiment"
            ]["name"],
        )
    )

    print("Descriptive subperiod analysis")

    for table_name, table_path in table_paths.items():
        print(
            f"Saved {table_name}:",
            table_path,
        )

    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
