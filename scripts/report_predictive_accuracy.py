"""Generate and save pairwise predictive-accuracy comparisons."""

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
    evaluate_models_diebold_mariano,
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

_SCORE_FAMILIES = (
    "var_quantile_loss",
    "fz0_loss",
)

_DISPLAY_COLUMNS = [
    "observation_count",
    "mean_score_difference",
    "hac_bandwidth",
    "hac_standard_error",
    "dm_statistic",
    "dm_p_value",
    "dm_reject_equal_accuracy",
    "holm_adjusted_p_value",
    "holm_reject_equal_accuracy",
]
_REQUIRED_COLUMNS = [
    *_DISPLAY_COLUMNS,
    "hac_long_run_variance",
    "hac_mean_variance",
    "significance_level",
]


def parse_args() -> argparse.Namespace:
    """Parse source snapshots, inference settings, and output options."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare rolling Gaussian, EWMA Gaussian, and filtered "
            "historical forecasts using quantile-loss and FZ0 "
            "Diebold-Mariano tests."
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


def save_predictive_accuracy_report(
    comparisons: pd.DataFrame,
    source_snapshots: dict[str, Path],
    source_metadata: dict[str, dict[str, Any]],
    output_root: str | Path,
    snapshot_id: str,
    generated_at: datetime,
    experiment_name: str,
    requested_bandwidth: int | None,
) -> tuple[Path, Path]:
    """Save a versioned predictive-accuracy table and metadata."""

    if not isinstance(comparisons, pd.DataFrame):
        raise TypeError(
            "Predictive-accuracy results must be a pandas DataFrame."
        )

    if comparisons.empty:
        raise ValueError(
            "Predictive-accuracy results must not be empty."
        )

    expected_index_names = [
        "score_name",
        "confidence_level",
        "model_a",
        "model_b",
    ]

    if (
        not isinstance(comparisons.index, pd.MultiIndex)
        or list(comparisons.index.names)
        != expected_index_names
    ):
        raise ValueError(
            "Predictive-accuracy results must use score, "
            "confidence-level, and ordered model-pair indexes."
        )

    if comparisons.index.has_duplicates:
        raise ValueError(
            "Predictive-accuracy result indexes must be unique."
        )

    missing_columns = (
        set(_REQUIRED_COLUMNS) - set(comparisons.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Predictive-accuracy results are missing required "
            f"column(s): {missing_names}."
        )

    observed_families = list(
        comparisons.index
        .get_level_values("score_name")
        .unique()
    )

    if set(observed_families) != set(_SCORE_FAMILIES):
        raise ValueError(
            "Predictive-accuracy results must contain the "
            "quantile-loss and FZ0 score families."
        )

    if not isinstance(snapshot_id, str):
        raise TypeError(
            "Snapshot ID must be a string."
        )

    if not snapshot_id.strip():
        raise ValueError(
            "Snapshot ID must not be empty."
        )

    significance_levels = comparisons[
        "significance_level"
    ].astype(float).unique()

    if len(significance_levels) != 1:
        raise ValueError(
            "Predictive-accuracy results must use one "
            "significance level."
        )

    significance_level = float(
        significance_levels[0]
    )
    selected_bandwidths = sorted(
        {
            int(value)
            for value in comparisons["hac_bandwidth"]
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

    comparisons_path = (
        report_directory / "predictive_accuracy.csv"
    )
    comparisons.to_csv(
        comparisons_path,
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
        for level in comparisons.index.get_level_values(
            "confidence_level"
        ).unique()
    ]
    bandwidth_rule = (
        "floor(4 * (T / 100) ** (2 / 9)), capped at T - 1"
        if requested_bandwidth is None
        else "explicit"
    )
    holm_families = [
        {
            "score_name": score_name,
            "hypothesis_count": int(
                (
                    comparisons.index.get_level_values(
                        "score_name"
                    )
                    == score_name
                ).sum()
            ),
        }
        for score_name in _SCORE_FAMILIES
    ]

    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "experiment": experiment_name,
        "report": {
            "name": "predictive_accuracy",
            "classification": (
                "formal_pairwise_model_comparison"
            ),
            "null_hypothesis": (
                "E[score_model_a - score_model_b] = 0"
            ),
            "alternative": "two_sided",
            "score_difference_order": (
                "model_a_minus_model_b"
            ),
            "lower_score_is_better": True,
            "significance_level": significance_level,
            "scores": {
                "var_quantile_loss": (
                    "strictly consistent VaR quantile loss"
                ),
                "fz0_loss": (
                    "jointly consistent positive-loss VaR-ES "
                    "FZ0 loss"
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
                "applied_separately_by_score_family": True,
                "families": holm_families,
                "total_hypothesis_count": len(comparisons),
            },
        },
        "source_backtests": source_records,
        "output": {
            "rows": len(comparisons),
            "models": list(source_snapshots),
            "confidence_levels": confidence_levels,
            "score_families": observed_families,
            "index_names": list(comparisons.index.names),
            "columns": list(comparisons.columns),
            "file": {
                "name": comparisons_path.name,
                "sha256": calculate_file_sha256(
                    comparisons_path
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

    return comparisons_path, metadata_path


def main() -> None:
    """Load verified forecasts and save predictive-accuracy results."""

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

    comparisons = evaluate_models_diebold_mariano(
        forecasts_by_model=forecasts_by_model,
        bandwidth=args.bandwidth,
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
            "predictive_accuracy"
        )

    comparisons_path, metadata_path = (
        save_predictive_accuracy_report(
            comparisons=comparisons,
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

    print("Pairwise predictive-accuracy comparisons")
    print(
        comparisons[_DISPLAY_COLUMNS].to_string(
            float_format=lambda value: f"{value:.6g}"
        )
    )
    print()
    print("Saved comparison table:", comparisons_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
