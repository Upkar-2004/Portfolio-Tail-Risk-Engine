"""Generate and save the multi-model VaR coverage report."""

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
    evaluate_models_var_coverage,
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

_DISPLAY_COLUMNS = [
    "observation_count",
    "exceedance_count",
    "expected_exceedance_count",
    "observed_exceedance_rate",
    "transition_n00",
    "transition_n01",
    "transition_n10",
    "transition_n11",
    "kupiec_p_value",
    "independence_p_value",
    "conditional_coverage_p_value",
    "conditional_coverage_holm_p_value",
    "conditional_coverage_reject_null_holm",
]


def parse_args() -> argparse.Namespace:
    """Parse portfolio, backtest, and report destinations."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VaR coverage across the rolling Gaussian, "
            "EWMA Gaussian, and filtered historical models."
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
        "--significance-level",
        type=float,
        default=0.05,
        help="Statistical significance level. Defaults to 0.05.",
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
    """Validate a backtest's identity and experiment lineage."""

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

    if (
        source_portfolio_id
        != expected_portfolio_snapshot_id
    ):
        raise ValueError(
            "Backtest does not use the requested "
            "portfolio snapshot."
        )


def save_var_coverage_report(
    coverage: pd.DataFrame,
    source_snapshots: dict[str, Path],
    source_metadata: dict[str, dict[str, Any]],
    output_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
    experiment_name: str,
    significance_level: float,
) -> tuple[Path, Path]:
    """Save a versioned coverage table and its lineage metadata."""

    if not isinstance(coverage, pd.DataFrame):
        raise TypeError(
            "Coverage results must be a pandas DataFrame."
        )

    if coverage.empty:
        raise ValueError(
            "Coverage results must not be empty."
        )

    if (
        not isinstance(coverage.index, pd.MultiIndex)
        or list(coverage.index.names)
        != ["model", "confidence_level"]
    ):
        raise ValueError(
            "Coverage results must use model and "
            "confidence-level index levels."
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

    report_directory = (
        output_root_path / snapshot_id
    )
    report_directory.mkdir(
        parents=False,
        exist_ok=False,
    )

    coverage_path = (
        report_directory / "var_coverage.csv"
    )
    coverage.to_csv(
        coverage_path,
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

    confidence_levels = [
        float(level)
        for level in coverage.index.get_level_values(
            "confidence_level"
        ).unique()
    ]

    metadata = {
        "schema_version": 1,
        "generated_at_utc": (
            generated_at.isoformat()
        ),
        "experiment": experiment_name,
        "report": {
            "name": "var_coverage",
            "significance_level": (
                float(significance_level)
            ),
            "var_exceedance_rule": (
                "realized_loss > value_at_risk"
            ),
            "tests": [
                "kupiec_unconditional_coverage",
                "christoffersen_independence",
                "christoffersen_conditional_coverage",
            ],
            "multiplicity_adjustment": {
                "method": "holm",
                "family": (
                    "christoffersen_conditional_coverage"
                ),
                "test_count": len(coverage),
            },
        },
        "source_backtests": source_records,
        "output": {
            "rows": len(coverage),
            "models": list(source_snapshots),
            "confidence_levels": confidence_levels,
            "index_names": list(
                coverage.index.names
            ),
            "columns": list(coverage.columns),
            "file": {
                "name": coverage_path.name,
                "sha256": calculate_file_sha256(
                    coverage_path
                ),
            },
        },
    }

    metadata_path = (
        report_directory / "metadata.json"
    )

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

    return coverage_path, metadata_path


def main() -> None:
    """Load verified forecasts and save the coverage report."""

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
                forecasting_config[
                    "estimation_window"
                ]
            ),
            expected_confidence_levels=(
                forecasting_config[
                    "confidence_levels"
                ]
            ),
        )

        forecasts_by_model[model_name] = forecasts
        metadata_by_model[model_name] = metadata

    coverage = evaluate_models_var_coverage(
        forecasts_by_model=forecasts_by_model,
        significance_level=args.significance_level,
    )

    snapshot_id = args.snapshot_id

    if snapshot_id is None:
        timestamp = generated_at.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        snapshot_id = (
            f"{timestamp}_"
            f"{config['experiment']['name']}_"
            "var_coverage"
        )

    coverage_path, metadata_path = (
        save_var_coverage_report(
            coverage=coverage,
            source_snapshots=snapshot_paths,
            source_metadata=metadata_by_model,
            output_root=args.output_root,
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            experiment_name=config[
                "experiment"
            ]["name"],
            significance_level=(
                args.significance_level
            ),
        )
    )

    print("Multi-model VaR coverage evaluation")
    print(
        coverage[_DISPLAY_COLUMNS].to_string(
            float_format=lambda value: (
                f"{value:.6g}"
            )
        )
    )
    print()
    print("Saved coverage table:", coverage_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()