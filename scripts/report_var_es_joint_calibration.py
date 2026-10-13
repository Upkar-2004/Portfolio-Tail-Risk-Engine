"""Generate and save the multi-model joint VaR-ES calibration report."""

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
    evaluate_models_var_es_identification,
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
    "mean_tail_contribution",
    "mean_expected_shortfall",
    "mean_identification_residual",
    "hac_bandwidth",
    "hac_long_run_variance",
    "hac_standard_error",
    "hac_test_statistic",
    "hac_p_value",
    "hac_reject_null",
    "hac_holm_p_value",
    "hac_holm_reject_null",
]
_REQUIRED_COLUMNS = [
    *_DISPLAY_COLUMNS,
    "significance_level",
]


def parse_args() -> argparse.Namespace:
    """Parse portfolio, backtest, inference, and report arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Test joint VaR-Expected-Shortfall calibration across "
            "the rolling Gaussian, EWMA Gaussian, and filtered "
            "historical models."
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
        "--bandwidth",
        type=int,
        default=None,
        help=(
            "Optional fixed Newey-West bandwidth. The documented "
            "automatic rule is used when omitted."
        ),
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

    if source_portfolio_id != expected_portfolio_snapshot_id:
        raise ValueError(
            "Backtest does not use the requested "
            "portfolio snapshot."
        )


def save_var_es_joint_calibration_report(
    calibration: pd.DataFrame,
    source_snapshots: dict[str, Path],
    source_metadata: dict[str, dict[str, Any]],
    output_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
    experiment_name: str,
    requested_bandwidth: int | None,
) -> tuple[Path, Path]:
    """Save a versioned joint-calibration table and metadata."""

    if not isinstance(calibration, pd.DataFrame):
        raise TypeError(
            "Calibration results must be a pandas DataFrame."
        )

    if calibration.empty:
        raise ValueError(
            "Calibration results must not be empty."
        )

    if (
        not isinstance(calibration.index, pd.MultiIndex)
        or list(calibration.index.names)
        != ["model", "confidence_level"]
    ):
        raise ValueError(
            "Calibration results must use model and "
            "confidence-level index levels."
        )

    missing_columns = (
        set(_REQUIRED_COLUMNS) - set(calibration.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Calibration results are missing required "
            f"column(s): {missing_names}."
        )

    if not isinstance(snapshot_id, str):
        raise TypeError(
            "Snapshot ID must be a string."
        )

    if not snapshot_id.strip():
        raise ValueError(
            "Snapshot ID must not be empty."
        )

    significance_levels = calibration[
        "significance_level"
    ].astype(float).unique()

    if len(significance_levels) != 1:
        raise ValueError(
            "Calibration results must use one significance level."
        )

    significance_level = float(
        significance_levels[0]
    )
    selected_bandwidths = sorted(
        {
            int(value)
            for value in calibration["hac_bandwidth"]
        }
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

    calibration_path = (
        report_directory
        / "var_es_joint_calibration.csv"
    )
    calibration.to_csv(
        calibration_path,
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
        for level in calibration.index.get_level_values(
            "confidence_level"
        ).unique()
    ]
    bandwidth_rule = (
        "floor(4 * (T / 100) ** (2 / 9)), capped at T - 1"
        if requested_bandwidth is None
        else "explicit"
    )

    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "experiment": experiment_name,
        "report": {
            "name": "var_es_joint_calibration",
            "classification": "formal_calibration_test",
            "null_hypothesis": (
                "E[var_es_identification_residual] = 0"
            ),
            "alternative": "two_sided",
            "significance_level": significance_level,
            "identification": {
                "tail_contribution": (
                    "VaR + I(loss > VaR) * (loss - VaR) "
                    "/ (1 - confidence_level)"
                ),
                "residual": (
                    "tail_contribution - expected_shortfall"
                ),
            },
            "hac": {
                "estimator": "newey_west",
                "kernel": "bartlett",
                "autocovariance_denominator": (
                    "observation_count"
                ),
                "bandwidth_rule": bandwidth_rule,
                "requested_bandwidth": requested_bandwidth,
                "selected_bandwidths": selected_bandwidths,
                "reference_distribution": (
                    "asymptotic_standard_normal"
                ),
            },
            "multiplicity_adjustment": {
                "method": "holm",
                "family": (
                    "joint_var_es_calibration_model_"
                    "confidence_pairs"
                ),
                "hypothesis_count": len(calibration),
            },
        },
        "source_backtests": source_records,
        "output": {
            "rows": len(calibration),
            "models": list(source_snapshots),
            "confidence_levels": confidence_levels,
            "index_names": list(calibration.index.names),
            "columns": list(calibration.columns),
            "file": {
                "name": calibration_path.name,
                "sha256": calculate_file_sha256(
                    calibration_path
                ),
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

    return calibration_path, metadata_path


def main() -> None:
    """Load verified forecasts and save the calibration report."""

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

    calibration = (
        evaluate_models_var_es_identification(
            forecasts_by_model=forecasts_by_model,
            bandwidth=args.bandwidth,
            significance_level=(
                args.significance_level
            ),
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
            "var_es_joint_calibration"
        )

    calibration_path, metadata_path = (
        save_var_es_joint_calibration_report(
            calibration=calibration,
            source_snapshots=snapshot_paths,
            source_metadata=metadata_by_model,
            output_root=args.output_root,
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            experiment_name=config[
                "experiment"
            ]["name"],
            requested_bandwidth=args.bandwidth,
        )
    )

    print("Multi-model joint VaR-ES calibration")
    print(
        calibration[_DISPLAY_COLUMNS].to_string(
            float_format=lambda value: f"{value:.6g}"
        )
    )
    print()
    print("Saved calibration table:", calibration_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
