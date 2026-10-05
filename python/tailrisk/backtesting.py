"""Time-ordered VaR and Expected Shortfall backtesting."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tailrisk.data import calculate_file_sha256


def create_forecast_schedule(
    portfolio_returns: pd.Series,
    estimation_window: int,
) -> pd.DataFrame:
    """Create a common schedule for one-day-ahead forecasts.

    Notation
    --------
    R_t = portfolio return on forecast date t
    W   = number of returns in the estimation window
    L_t = realized portfolio loss on date t

    For each forecast date t:

        estimation window = R_(t-W), ..., R_(t-1)
        realized return   = R_t
        realized loss     = -R_t

    The forecast-date return is never included in its own estimation
    window. The returned table records the inclusive start and end dates
    of every window so all models use identical information and outcomes.
    """

    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError(
            "Portfolio returns must be a pandas Series."
        )

    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        raise ValueError(
            "Portfolio returns must use a DatetimeIndex."
        )

    if portfolio_returns.index.hasnans:
        raise ValueError(
            "Portfolio return dates must not be missing."
        )

    if portfolio_returns.index.has_duplicates:
        raise ValueError(
            "Portfolio return dates must be unique."
        )

    if not portfolio_returns.index.is_monotonic_increasing:
        raise ValueError(
            "Portfolio returns must be ordered by increasing date."
        )

    if (
        isinstance(estimation_window, bool)
        or not isinstance(estimation_window, int)
        or estimation_window <= 0
    ):
        raise ValueError(
            "Estimation window must be a positive integer."
        )

    if len(portfolio_returns) <= estimation_window:
        raise ValueError(
            "Portfolio returns must contain more observations "
            "than the estimation window."
        )

    values = portfolio_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Portfolio returns must be finite numbers."
        )

    forecast_dates = pd.DatetimeIndex(
        portfolio_returns.index[estimation_window:],
        name="forecast_date",
    )
    realized_returns = values[estimation_window:]

    return pd.DataFrame(
        {
            "estimation_start_date": (
                portfolio_returns.index[:-estimation_window]
            ),
            "estimation_end_date": (
                portfolio_returns.index[estimation_window - 1:-1]
            ),
            "realized_return": realized_returns,
            "realized_loss": -realized_returns,
        },
        index=forecast_dates,
    )




_REQUIRED_BACKTEST_COLUMNS = {
    "estimation_start_date",
    "estimation_end_date",
    "realized_return",
    "realized_loss",
    "value_at_risk",
    "expected_shortfall",
    "var_exceedance",
}


def validate_backtest_forecasts(
    forecasts: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
) -> None:
    """Validate the shared structure of model backtest forecasts."""

    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError(
            "Forecasts must be a pandas DataFrame."
        )

    if not isinstance(forecast_schedule, pd.DataFrame):
        raise TypeError(
            "Forecast schedule must be a pandas DataFrame."
        )

    if not confidence_levels:
        raise ValueError(
            "At least one confidence level is required."
        )

    if not isinstance(forecasts.index, pd.MultiIndex):
        raise ValueError(
            "Forecasts must use a MultiIndex."
        )

    if forecasts.index.nlevels != 2:
        raise ValueError(
            "Forecast index must contain exactly two levels."
        )

    expected_index_names = [
        "forecast_date",
        "confidence_level",
    ]

    if list(forecasts.index.names) != expected_index_names:
        raise ValueError(
            "Forecast index levels must be named "
            "'forecast_date' and 'confidence_level'."
        )

    if forecasts.index.has_duplicates:
        raise ValueError(
            "Forecast index entries must be unique."
        )

    missing_columns = (
        _REQUIRED_BACKTEST_COLUMNS
        - set(forecasts.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Forecasts are missing required column(s): "
            f"{missing_names}."
        )

    expected_index = pd.MultiIndex.from_product(
        [
            forecast_schedule.index,
            [
                float(level)
                for level in confidence_levels
            ],
        ],
        names=expected_index_names,
    )

    if not forecasts.index.equals(expected_index):
        raise ValueError(
            "Forecasts must contain the complete ordered grid "
            "of forecast dates and confidence levels."
        )


    schedule_columns = {
        "estimation_start_date",
        "estimation_end_date",
        "realized_return",
        "realized_loss",
    }

    missing_schedule_columns = (
        schedule_columns
        - set(forecast_schedule.columns)
    )

    if missing_schedule_columns:
        raise ValueError(
            "Forecast schedule is missing required columns."
        )

    if not isinstance(
        forecast_schedule.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Forecast schedule must use a DatetimeIndex."
        )

    if (
        forecast_schedule.index.hasnans
        or forecast_schedule.index.has_duplicates
        or not forecast_schedule.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Forecast schedule dates must be complete, "
            "unique, and ordered."
        )

    forecast_dates = forecasts.index.get_level_values(
        "forecast_date"
    )

    if not isinstance(forecast_dates, pd.DatetimeIndex):
        raise ValueError(
            "Forecast dates must be datetime values."
        )

    date_columns = [
        "estimation_start_date",
        "estimation_end_date",
    ]

    for column in date_columns:
        if not pd.api.types.is_datetime64_any_dtype(
            forecasts[column]
        ):
            raise ValueError(
                f"{column} must contain datetime values."
            )

        if forecasts[column].isna().any():
            raise ValueError(
                "Estimation dates must not be missing."
            )

    numeric_columns = [
        "realized_return",
        "realized_loss",
        "value_at_risk",
        "expected_shortfall",
    ]

    try:
        numeric_values = forecasts[
            numeric_columns
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Forecast numeric columns must contain numbers."
        ) from error

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Forecast numeric columns must contain "
            "finite numbers."
        )

    if not pd.api.types.is_bool_dtype(
        forecasts["var_exceedance"]
    ):
        raise ValueError(
            "VaR exceedance indicators must be boolean."
        )

    estimation_start_dates = pd.DatetimeIndex(
        forecasts["estimation_start_date"]
    )
    estimation_end_dates = pd.DatetimeIndex(
        forecasts["estimation_end_date"]
    )

    if (
        estimation_start_dates
        > estimation_end_dates
    ).any():
        raise ValueError(
            "Each estimation window must start on or "
            "before its end date."
        )

    if (
        estimation_end_dates
        >= forecast_dates
    ).any():
        raise ValueError(
            "Each estimation window must end before "
            "its forecast date."
        )

    aligned_schedule = forecast_schedule.reindex(
        forecast_dates
    )

    for column in date_columns:
        if not np.array_equal(
            forecasts[column].to_numpy(),
            aligned_schedule[column].to_numpy(),
        ):
            raise ValueError(
                "Forecast estimation dates must match "
                "the forecast schedule."
            )

    for column in [
        "realized_return",
        "realized_loss",
    ]:
        if not np.allclose(
            forecasts[column].to_numpy(dtype=float),
            aligned_schedule[column].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "Forecast realized outcomes must match "
                "the forecast schedule."
            )

    realized_returns = forecasts[
        "realized_return"
    ].to_numpy(dtype=float)
    realized_losses = forecasts[
        "realized_loss"
    ].to_numpy(dtype=float)

    if not np.allclose(
        realized_losses,
        -realized_returns,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Each realized loss must equal the negative "
            "of its realized return."
        )

    value_at_risk = forecasts[
        "value_at_risk"
    ].to_numpy(dtype=float)
    expected_shortfall = forecasts[
        "expected_shortfall"
    ].to_numpy(dtype=float)

    if (expected_shortfall < value_at_risk).any():
        raise ValueError(
            "Expected Shortfall must be at least as "
            "large as VaR."
        )

    expected_exceedances = (
        realized_losses > value_at_risk
    )
    observed_exceedances = forecasts[
        "var_exceedance"
    ].to_numpy(dtype=bool)

    if not np.array_equal(
        observed_exceedances,
        expected_exceedances,
    ):
        raise ValueError(
            "VaR exceedance indicators are inconsistent "
            "with realized losses and VaR."
        )


def save_backtest_forecasts(
    forecasts: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    confidence_levels: list[float],
    output_root: str | Path,
    snapshot_id: str,
) -> Path:
    """Save validated forecasts in a new versioned snapshot."""

    validate_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
    )

    if not isinstance(snapshot_id, str):
        raise TypeError(
            "Snapshot ID must be a string."
        )

    if not snapshot_id.strip():
        raise ValueError(
            "Snapshot ID must not be empty."
        )

    snapshot_directory = (
        Path(output_root) / snapshot_id
    )

    snapshot_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    forecasts_path = (
        snapshot_directory / "forecasts.csv"
    )

    forecasts.to_csv(forecasts_path)

    return forecasts_path


def save_backtest_metadata(
    forecasts_path: str | Path,
    forecasts: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    config: dict[str, Any],
    model_name: str,
    source_returns_snapshot_id: str,
    source_returns_metadata: dict[str, Any],
    source_portfolio_snapshot_id: str,
    source_portfolio_metadata: dict[str, Any],
    generated_at: datetime,
) -> Path:
    """Save backtest settings, lineage, validation, and output metadata."""

    confidence_levels = config["forecasting"][
        "confidence_levels"
    ]

    validate_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
    )

    output_path = Path(forecasts_path)

    if not output_path.is_file():
        raise ValueError(
            "Backtest forecast file does not exist."
        )

    if not isinstance(model_name, str):
        raise TypeError(
            "Model name must be a string."
        )

    if not model_name.strip():
        raise ValueError(
            "Model name must not be empty."
        )

    if generated_at.utcoffset() is None:
        raise ValueError(
            "Backtest generation timestamp must include a time zone."
        )

    returns_file = source_returns_metadata[
        "output"
    ]["file"]
    portfolio_source = source_portfolio_metadata[
        "source"
    ]
    portfolio_files = source_portfolio_metadata[
        "output"
    ]["files"]

    if (
        portfolio_source["asset_returns_sha256"]
        != returns_file["sha256"]
    ):
        raise ValueError(
            "Portfolio and return snapshots do not share "
            "the same asset-return source."
        )

    forecast_dates = pd.DatetimeIndex(
        forecasts.index.get_level_values(
            "forecast_date"
        )
    )
    exceedance_counts = (
        forecasts["var_exceedance"]
        .groupby(level="confidence_level")
        .sum()
    )

    metadata = {
        "schema_version": 1,
        "generated_at_utc": generated_at.astimezone(
            timezone.utc
        ).isoformat(),
        "experiment": config["experiment"]["name"],
        "model": {
            "name": model_name,
            "horizon_sessions": config["forecasting"][
                "horizon_sessions"
            ],
            "estimation_window": config["forecasting"][
                "estimation_window"
            ],
            "confidence_levels": [
                float(level)
                for level in confidence_levels
            ],
            "loss_unit": "portfolio_return",
            "loss_sign_convention": (
                "positive values represent losses"
            ),
            "var_exceedance_rule": (
                "realized_loss > value_at_risk"
            ),
        },
        "simulation": {
            "scenario_count": config["simulation"][
                "scenario_count"
            ],
            "random_seed": config["simulation"][
                "random_seed"
            ],
            "reuse_standard_normal_shocks": config[
                "simulation"
            ]["reuse_standard_normal_shocks"],
        },
        "source": {
            "asset_returns": {
                "snapshot_id": source_returns_snapshot_id,
                "file": returns_file["name"],
                "sha256": returns_file["sha256"],
            },
            "portfolio": {
                "snapshot_id": source_portfolio_snapshot_id,
                "portfolio_daily_file": portfolio_files[
                    "portfolio_daily"
                ]["name"],
                "portfolio_daily_sha256": portfolio_files[
                    "portfolio_daily"
                ]["sha256"],
                "beginning_weights_file": portfolio_files[
                    "beginning_weights"
                ]["name"],
                "beginning_weights_sha256": portfolio_files[
                    "beginning_weights"
                ]["sha256"],
            },
        },
        "output": {
            "rows": len(forecasts),
            "forecast_dates": forecast_dates.nunique(),
            "first_forecast_date": (
                forecast_dates.min().date().isoformat()
            ),
            "last_forecast_date": (
                forecast_dates.max().date().isoformat()
            ),
            "columns": [
                str(column)
                for column in forecasts.columns
            ],
            "exceedance_counts": {
                str(float(level)): int(count)
                for level, count in exceedance_counts.items()
            },
            "file": {
                "name": output_path.name,
                "sha256": calculate_file_sha256(
                    output_path
                ),
            },
        },
        "validation": {
            "shared_schedule_alignment": True,
            "finite_numeric_values": True,
            "realized_loss_sign_checked": True,
            "var_exceedance_rule_checked": True,
        },
    }

    metadata_path = output_path.parent / "metadata.json"

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

    return metadata_path


def load_backtest_snapshot(
    snapshot_directory: str | Path,
    forecast_schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, checksum-verify, and validate a backtest snapshot."""

    snapshot_path = Path(snapshot_directory)
    metadata_path = snapshot_path / "metadata.json"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    if metadata.get("schema_version") != 1:
        raise ValueError(
            "Unsupported backtest metadata schema version."
        )

    file_metadata = metadata["output"]["file"]
    forecasts_path = snapshot_path / file_metadata["name"]

    expected_checksum = file_metadata["sha256"]
    actual_checksum = calculate_file_sha256(
        forecasts_path
    )

    if actual_checksum != expected_checksum:
        raise ValueError(
            "Backtest forecast snapshot checksum does not match."
        )

    forecasts = pd.read_csv(
        forecasts_path,
        parse_dates=[
            "forecast_date",
            "estimation_start_date",
            "estimation_end_date",
        ],
    )

    forecasts = forecasts.set_index(
        [
            "forecast_date",
            "confidence_level",
        ]
    )

    confidence_levels = metadata["model"][
        "confidence_levels"
    ]

    validate_backtest_forecasts(
        forecasts=forecasts,
        forecast_schedule=forecast_schedule,
        confidence_levels=confidence_levels,
    )

    forecast_dates = pd.DatetimeIndex(
        forecasts.index.get_level_values(
            "forecast_date"
        )
    )
    output_metadata = metadata["output"]

    if len(forecasts) != output_metadata["rows"]:
        raise ValueError(
            "Backtest row count does not match its metadata."
        )

    if (
        forecast_dates.nunique()
        != output_metadata["forecast_dates"]
    ):
        raise ValueError(
            "Backtest forecast-date count does not match "
            "its metadata."
        )

    if (
        forecast_dates.min().date().isoformat()
        != output_metadata["first_forecast_date"]
        or forecast_dates.max().date().isoformat()
        != output_metadata["last_forecast_date"]
    ):
        raise ValueError(
            "Backtest forecast-date boundaries do not match "
            "their metadata."
        )

    if list(forecasts.columns) != output_metadata["columns"]:
        raise ValueError(
            "Backtest columns do not match their metadata."
        )

    observed_exceedance_counts = (
        forecasts["var_exceedance"]
        .groupby(level="confidence_level")
        .sum()
    )
    expected_exceedance_counts = {
        str(float(level)): int(count)
        for level, count in observed_exceedance_counts.items()
    }

    if (
        expected_exceedance_counts
        != output_metadata["exceedance_counts"]
    ):
        raise ValueError(
            "Backtest exceedance counts do not match "
            "their metadata."
        )

    return forecasts, metadata
