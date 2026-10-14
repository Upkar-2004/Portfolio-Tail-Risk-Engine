"""Generate and save the multi-model Basel-style traffic-light report."""

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
    evaluate_models_basel_traffic_light,
    summarize_models_basel_traffic_light,
)
from tailrisk.portfolio import load_portfolio_path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.yaml"
_OUTPUT_ROOT = _PROJECT_ROOT / "data" / "evaluations"

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

_PATH_COLUMNS = [
    "window_start_date",
    "window_observation_count",
    "confidence_level",
    "exception_count",
    "traffic_light_zone",
]
_SUMMARY_COLUMNS = [
    "window_count",
    "latest_forecast_date",
    "latest_exception_count",
    "latest_zone",
    "maximum_exception_count",
    "maximum_exception_date",
    "worst_zone",
    "green_window_count",
    "amber_window_count",
    "red_window_count",
    "green_window_rate",
    "amber_window_rate",
    "red_window_rate",
    "first_amber_date",
    "first_red_date",
]


def parse_args() -> argparse.Namespace:
    """Parse source snapshots and report output options."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate rolling 250-session Basel-style traffic-light "
            "diagnostics for the three baseline 99% VaR models."
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


def save_basel_traffic_light_report(
    paths: pd.DataFrame,
    summary: pd.DataFrame,
    source_snapshots: dict[str, Path],
    source_metadata: dict[str, dict[str, Any]],
    output_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
    experiment_name: str,
) -> tuple[Path, Path, Path]:
    """Save versioned traffic-light paths, summary, and metadata."""

    if not isinstance(paths, pd.DataFrame):
        raise TypeError(
            "Traffic-light paths must be a pandas DataFrame."
        )

    if paths.empty:
        raise ValueError(
            "Traffic-light paths must not be empty."
        )

    if (
        not isinstance(paths.index, pd.MultiIndex)
        or list(paths.index.names)
        != ["model", "forecast_date"]
    ):
        raise ValueError(
            "Traffic-light paths must use model and forecast_date "
            "index levels."
        )

    missing_path_columns = (
        set(_PATH_COLUMNS) - set(paths.columns)
    )

    if missing_path_columns:
        missing_names = ", ".join(
            sorted(missing_path_columns)
        )
        raise ValueError(
            "Traffic-light paths are missing required "
            f"column(s): {missing_names}."
        )

    if not isinstance(summary, pd.DataFrame):
        raise TypeError(
            "Traffic-light summary must be a pandas DataFrame."
        )

    if summary.empty:
        raise ValueError(
            "Traffic-light summary must not be empty."
        )

    if summary.index.name != "model":
        raise ValueError(
            "Traffic-light summary must use model as its index."
        )

    missing_summary_columns = (
        set(_SUMMARY_COLUMNS) - set(summary.columns)
    )

    if missing_summary_columns:
        missing_names = ", ".join(
            sorted(missing_summary_columns)
        )
        raise ValueError(
            "Traffic-light summary is missing required "
            f"column(s): {missing_names}."
        )

    path_models = list(
        paths.index.get_level_values("model").unique()
    )
    summary_models = list(summary.index)

    if path_models != summary_models:
        raise ValueError(
            "Traffic-light paths and summary must use the same "
            "ordered models."
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

    paths_path = (
        report_directory / "basel_traffic_light_path.csv"
    )
    summary_path = (
        report_directory / "basel_traffic_light_summary.csv"
    )
    paths.to_csv(
        paths_path,
        index=True,
    )
    summary.to_csv(
        summary_path,
        index=True,
    )

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

    forecast_dates = paths.index.get_level_values(
        "forecast_date"
    )
    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "experiment": experiment_name,
        "report": {
            "name": "basel_traffic_light",
            "classification": (
                "descriptive_monitoring_diagnostic"
            ),
            "regulatory_status": (
                "Basel-style research diagnostic; not a claim "
                "of regulatory compliance"
            ),
            "confidence_level": 0.99,
            "window_observations": 250,
            "exception_rule": (
                "realized_loss > value_at_risk"
            ),
            "partial_window_policy": "exclude",
            "zones": {
                "green": {
                    "minimum_exceptions": 0,
                    "maximum_exceptions": 4,
                },
                "amber": {
                    "minimum_exceptions": 5,
                    "maximum_exceptions": 9,
                },
                "red": {
                    "minimum_exceptions": 10,
                    "maximum_exceptions": 250,
                },
            },
            "reference": {
                "standard": (
                    "Basel Framework MAR32.9"
                ),
                "url": (
                    "https://www.bis.org/committees/bcbs/"
                    "basel-framework/standard/mar/32/"
                    "inforce/2022-01-01/published/2019-12-15"
                ),
            },
        },
        "source_backtests": source_records,
        "output": {
            "models": path_models,
            "first_window_end_date": (
                forecast_dates.min().isoformat()
            ),
            "last_window_end_date": (
                forecast_dates.max().isoformat()
            ),
            "path": {
                "rows": len(paths),
                "index_names": list(paths.index.names),
                "columns": list(paths.columns),
                "file": {
                    "name": paths_path.name,
                    "sha256": calculate_file_sha256(
                        paths_path
                    ),
                },
            },
            "summary": {
                "rows": len(summary),
                "index_name": summary.index.name,
                "columns": list(summary.columns),
                "file": {
                    "name": summary_path.name,
                    "sha256": calculate_file_sha256(
                        summary_path
                    ),
                },
            },
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

    return paths_path, summary_path, metadata_path


def main() -> None:
    """Load verified forecasts and save traffic-light results."""

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

    paths = evaluate_models_basel_traffic_light(
        forecasts_by_model=forecasts_by_model,
    )
    summary = summarize_models_basel_traffic_light(
        traffic_light_paths=paths,
    )

    snapshot_id = args.snapshot_id

    if snapshot_id is None:
        timestamp = generated_at.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        snapshot_id = (
            f"{timestamp}_"
            f"{config['experiment']['name']}_"
            "basel_traffic_light"
        )

    paths_path, summary_path, metadata_path = (
        save_basel_traffic_light_report(
            paths=paths,
            summary=summary,
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

    print("Basel-style rolling 99% VaR traffic-light summary")
    print(
        summary[_SUMMARY_COLUMNS].to_string(
            float_format=lambda value: f"{value:.6g}"
        )
    )
    print()
    print("Saved traffic-light path:", paths_path)
    print("Saved traffic-light summary:", summary_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
