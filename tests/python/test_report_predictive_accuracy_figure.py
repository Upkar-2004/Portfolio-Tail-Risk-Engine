"""Tests for the full-sample predictive-inference figure command."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from tailrisk.data import calculate_file_sha256
from tailrisk.figures.inference import (
    prepare_predictive_inference_table,
)


MODELS = [
    "rolling_gaussian_monte_carlo",
    "ewma_gaussian_monte_carlo",
    "filtered_historical_simulation",
]
LEVELS = [0.95, 0.975, 0.99]
SCORES = ["var_quantile_loss", "fz0_loss"]
PAIRS = [
    (MODELS[0], MODELS[1]),
    (MODELS[0], MODELS[2]),
    (MODELS[1], MODELS[2]),
]


def _load_report_module() -> ModuleType:
    """Load the report script without making scripts a package."""

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "report_predictive_accuracy_figure.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "report_predictive_accuracy_figure",
        script_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(
            "Could not load the predictive-inference figure script."
        )

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


report_module = _load_report_module()


def _score_mean(
    model: str,
    confidence_level: float,
    score_name: str,
) -> float:
    """Return one controlled model-level mean score."""

    model_index = MODELS.index(model)
    level_index = LEVELS.index(confidence_level)

    if score_name == "var_quantile_loss":
        return 0.0015 - (0.00015 * model_index) - (
            0.0001 * level_index
        )

    return -3.0 - (0.2 * model_index) - (0.1 * level_index)


def _create_comparisons(
    bandwidth: int,
    flip_one_decision: bool,
) -> pd.DataFrame:
    """Create one complete predictive-accuracy table."""

    records = []

    for score_name in SCORES:
        for confidence_level in LEVELS:
            for pair_index, (model_a, model_b) in enumerate(PAIRS):
                difference = _score_mean(
                    model_a,
                    confidence_level,
                    score_name,
                ) - _score_mean(
                    model_b,
                    confidence_level,
                    score_name,
                )
                standard_error = abs(difference) / (2.0 + pair_index)
                statistic = difference / standard_error
                adjusted_p = 0.03 if pair_index == 0 else 0.20
                reject = adjusted_p <= 0.05

                if (
                    flip_one_decision
                    and score_name == "fz0_loss"
                    and confidence_level == 0.99
                    and pair_index == 1
                ):
                    adjusted_p = 0.04
                    reject = True

                records.append(
                    {
                        "score_name": score_name,
                        "confidence_level": confidence_level,
                        "model_a": model_a,
                        "model_b": model_b,
                        "observation_count": 1400,
                        "mean_score_difference": difference,
                        "hac_long_run_variance": 0.01,
                        "hac_mean_variance": standard_error**2,
                        "hac_standard_error": standard_error,
                        "hac_bandwidth": bandwidth,
                        "dm_statistic": statistic,
                        "dm_p_value": min(adjusted_p, 0.02),
                        "significance_level": 0.05,
                        "dm_reject_equal_accuracy": True,
                        "holm_adjusted_p_value": adjusted_p,
                        "holm_reject_equal_accuracy": reject,
                    }
                )

    return pd.DataFrame.from_records(records).set_index(
        [
            "score_name",
            "confidence_level",
            "model_a",
            "model_b",
        ]
    )


def _write_predictive_snapshot(
    root: Path,
    name: str,
    bandwidth: int,
    requested_bandwidth: int | None,
    flip_one_decision: bool,
) -> tuple[Path, pd.DataFrame, dict[str, object]]:
    """Write a minimal checksum-verified predictive snapshot."""

    snapshot_path = root / name
    snapshot_path.mkdir()
    comparisons = _create_comparisons(
        bandwidth=bandwidth,
        flip_one_decision=flip_one_decision,
    )
    table_path = snapshot_path / "predictive_accuracy.csv"
    comparisons.to_csv(table_path, index=True)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "experiment": "baseline_11_asset",
        "report": {
            "name": "predictive_accuracy",
            "hac": {
                "requested_bandwidth": requested_bandwidth,
                "selected_bandwidths": [bandwidth],
                "bandwidth_rule": (
                    "floor(4 * (T / 100) ** (2 / 9)), capped at T - 1"
                    if requested_bandwidth is None
                    else "explicit"
                ),
            },
            "multiplicity_adjustment": {
                "method": "holm",
                "applied_separately_by_score_family": True,
                "families": [
                    {
                        "hypothesis_count": 9,
                        "score_name": "var_quantile_loss",
                    },
                    {
                        "hypothesis_count": 9,
                        "score_name": "fz0_loss",
                    },
                ],
            },
        },
        "source_backtests": {
            model: {
                "snapshot_id": f"{model}_snapshot",
                "forecast_sha256": f"{model}_checksum",
            }
            for model in MODELS
        },
        "output": {
            "rows": len(comparisons),
            "columns": list(comparisons.columns),
            "file": {
                "name": table_path.name,
                "sha256": calculate_file_sha256(table_path),
            },
        },
    }

    with (snapshot_path / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(metadata, stream)

    return snapshot_path, comparisons, metadata


def _write_subperiod_snapshot(
    root: Path,
) -> tuple[Path, pd.DataFrame, dict[str, object]]:
    """Write disjoint annual means plus the predeclared stress period."""

    snapshot_path = root / "subperiod"
    snapshot_path.mkdir()
    records = []
    period_ids = [
        *(f"calendar_{year}" for year in range(2012, 2026)),
        "nber_recession_2020",
    ]

    for period_id in period_ids:
        for confidence_level in LEVELS:
            ql_means = {
                model: _score_mean(
                    model,
                    confidence_level,
                    "var_quantile_loss",
                )
                for model in MODELS
            }
            fz0_means = {
                model: _score_mean(
                    model,
                    confidence_level,
                    "fz0_loss",
                )
                for model in MODELS
            }
            ql_ranks = {
                model: rank
                for rank, (model, _) in enumerate(
                    sorted(ql_means.items(), key=lambda item: item[1]),
                    start=1,
                )
            }
            fz0_ranks = {
                model: rank
                for rank, (model, _) in enumerate(
                    sorted(fz0_means.items(), key=lambda item: item[1]),
                    start=1,
                )
            }

            for model in MODELS:
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
                        "observation_count": (
                            62
                            if period_id == "nber_recession_2020"
                            else 100
                        ),
                        "mean_var_quantile_loss": ql_means[model],
                        "var_quantile_loss_rank": ql_ranks[model],
                        "mean_fz0_loss": fz0_means[model],
                        "fz0_loss_rank": fz0_ranks[model],
                    }
                )

    predictive = pd.DataFrame.from_records(records).set_index(
        ["period_id", "model", "confidence_level"]
    )
    table_path = snapshot_path / "subperiod_predictive_accuracy.csv"
    predictive.to_csv(table_path, index=True)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "experiment": "baseline_11_asset",
        "report": {"name": "subperiod_analysis"},
        "output": {
            "tables": {
                "subperiod_predictive_accuracy": {
                    "rows": len(predictive),
                    "columns": list(predictive.columns),
                    "file": {
                        "name": table_path.name,
                        "sha256": calculate_file_sha256(table_path),
                    },
                }
            }
        },
    }

    with (snapshot_path / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(metadata, stream)

    return snapshot_path, predictive, metadata


def test_load_predictive_snapshot_rejects_tampering(
    tmp_path: Path,
) -> None:
    """A changed source table is rejected before plotting."""

    snapshot_path, _, _ = _write_predictive_snapshot(
        tmp_path,
        "baseline",
        bandwidth=8,
        requested_bandwidth=None,
        flip_one_decision=False,
    )

    with (snapshot_path / "predictive_accuracy.csv").open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write("tampered\n")

    with pytest.raises(ValueError, match="checksum"):
        report_module.load_predictive_accuracy_snapshot(
            snapshot_path
        )


def test_calculate_full_sample_scores_uses_calendar_weights(
    tmp_path: Path,
) -> None:
    """The overlapping stress period is excluded from full means."""

    _, predictive, _ = _write_subperiod_snapshot(tmp_path)
    means = report_module.calculate_full_sample_model_scores(
        predictive
    )

    assert (means["observation_count"] == 1400).all()
    assert means.loc[
        (MODELS[1], 0.975),
        "mean_var_quantile_loss",
    ] == pytest.approx(
        _score_mean(MODELS[1], 0.975, "var_quantile_loss")
    )


def test_save_inference_figure_report_preserves_lineage(
    tmp_path: Path,
) -> None:
    """The report saves verified figures, table, and robustness metadata."""

    baseline_path, baseline, baseline_metadata = (
        _write_predictive_snapshot(
            tmp_path,
            "baseline",
            bandwidth=8,
            requested_bandwidth=None,
            flip_one_decision=False,
        )
    )
    sensitivity_path, sensitivity, sensitivity_metadata = (
        _write_predictive_snapshot(
            tmp_path,
            "sensitivity",
            bandwidth=0,
            requested_bandwidth=0,
            flip_one_decision=True,
        )
    )
    subperiod_path, subperiod, subperiod_metadata = (
        _write_subperiod_snapshot(tmp_path)
    )
    report_module.validate_inference_lineage(
        baseline_metadata=baseline_metadata,
        sensitivity_metadata=sensitivity_metadata,
        subperiod_metadata=subperiod_metadata,
    )
    means = report_module.calculate_full_sample_model_scores(
        subperiod
    )
    inference_table = prepare_predictive_inference_table(
        comparisons=baseline,
        sensitivity_comparisons=sensitivity,
        model_mean_scores=means,
    )
    snapshot_paths, metadata_path, publication_paths = (
        report_module.save_predictive_accuracy_figure_report(
            inference_table=inference_table,
            baseline_snapshot=baseline_path,
            baseline_metadata=baseline_metadata,
            subperiod_snapshot=subperiod_path,
            subperiod_metadata=subperiod_metadata,
            sensitivity_snapshot=sensitivity_path,
            sensitivity_metadata=sensitivity_metadata,
            output_root=tmp_path / "evaluations",
            publication_root=tmp_path / "figures",
            table_publication_root=tmp_path / "tables",
            snapshot_id="test_inference_figure",
            generated_at=datetime(
                2026,
                10,
                16,
                tzinfo=timezone.utc,
            ),
        )
    )

    assert set(snapshot_paths) == {"png", "svg", "csv"}
    assert set(publication_paths) == set(snapshot_paths)

    for output_name, snapshot_path in snapshot_paths.items():
        assert publication_paths[output_name].read_bytes() == (
            snapshot_path.read_bytes()
        )

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    assert metadata["report"]["orientation"][
        "negative_statistic"
    ] == "first_model_better"
    assert metadata["report"]["robustness"][
        "decision_change_count"
    ] == 1
    assert metadata["report"]["primary_inference"][
        "hac_bandwidth"
    ] == [8]
    assert metadata["report"]["primary_inference"][
        "hac_bandwidth_rule"
    ].startswith("floor(4")
    assert "Holm-adjusted p-values are monotone" in metadata[
        "report"
    ]["caption"]
    assert metadata["report"]["encoding"]["dagger"] == (
        "significant_at_0.05_with_bandwidth_zero_but_not_with_"
        "primary_bandwidth"
    )
    assert metadata["output"]["companion_table"]["rows"] == 18
    companion = pd.read_csv(snapshot_paths["csv"])
    assert companion["ql_relative_difference_percent"].notna().sum() == 9
    assert companion.loc[
        companion["score_name"] == "fz0_loss",
        "ql_relative_difference_percent",
    ].isna().all()
