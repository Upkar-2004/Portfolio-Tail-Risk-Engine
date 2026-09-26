"""Portfolio construction and accounting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioStep:
    """Results from applying one session of asset returns."""

    portfolio_return: float
    pnl: float
    loss: float
    ending_value: float
    ending_weights: pd.Series
    ending_cash_weight: float


@dataclass(frozen=True)
class PortfolioSnapshotFiles:
    """Paths belonging to a saved portfolio snapshot."""

    daily: Path
    beginning_weights: Path
    pre_rebalance_weights: Path


@dataclass(frozen=True)
class PortfolioPath:
    """Daily accounting history for a portfolio over multiple sessions."""

    daily: pd.DataFrame
    beginning_weights: pd.DataFrame
    pre_rebalance_weights: pd.DataFrame


def create_equal_weights(
    tickers: list[str],
    cash_weight: float,
) -> pd.Series:
    """Create equal asset weights after reserving the cash allocation."""

    if not tickers:
        raise ValueError(
            "At least one ticker is required."
        )

    if len(tickers) != len(set(tickers)):
        raise ValueError(
            "Portfolio tickers must be unique."
        )

    if not 0.0 <= cash_weight < 1.0:
        raise ValueError(
            "Cash weight must be between "
            "0 inclusive and 1 exclusive."
        )

    asset_weight = (
        1.0 - cash_weight
    ) / len(tickers)

    return pd.Series(
        asset_weight,
        index=tickers,
        name="weight",
        dtype=float,
    )


def calculate_portfolio_step(
    beginning_value: float,
    beginning_weights: pd.Series,
    asset_returns: pd.Series,
) -> PortfolioStep:
    """Apply one session of returns to the portfolio."""

    if not np.isfinite(beginning_value) or beginning_value <= 0.0:
        raise ValueError(
            "Portfolio beginning_value must be a positive number."
        )

    if beginning_weights.index.has_duplicates:
        raise ValueError(
            "Portfolio beginning_weights must have unique tickers."
        )

    if asset_returns.index.has_duplicates:
        raise ValueError(
            "Portfolio asset_returns must have unique tickers."
        )

    if set(beginning_weights.index) != set(asset_returns.index):
        raise ValueError(
            "Portfolio beginning_weights and asset_returns "
            "must have the same tickers."
        )

    # Align by ticker label so column order cannot attach a return to the
    # wrong asset weight.
    aligned_returns = asset_returns.reindex(beginning_weights.index)

    if not np.isfinite(beginning_weights.to_numpy()).all():
        raise ValueError(
            "Portfolio beginning_weights must be finite numbers."
        )

    if not np.isfinite(aligned_returns.to_numpy()).all():
        raise ValueError(
            "Portfolio asset_returns must be finite numbers."
        )

    if (beginning_weights < 0.0).any():
        raise ValueError(
            "Portfolio beginning_weights must be non-negative."
        )

    if (aligned_returns < -1.0).any():
        raise ValueError(
            "Portfolio asset_returns must be greater than or equal to -1.0."
        )

    invested_weight = float(beginning_weights.sum())

    if (
        invested_weight > 1.0
        and not np.isclose(invested_weight, 1.0)
    ):
        raise ValueError(
            "Asset weights cannot sum to more than one."
        )

    cash_weight = max(
        0.0,
        1.0 - invested_weight,
    )

    # Convert beginning weights into actual dollar exposures. Any weight not
    # allocated to assets is cash, which earns zero return in the baseline.
    beginning_positions = beginning_weights * beginning_value
    cash_value = cash_weight * beginning_value

    # Each asset's ending value is its beginning exposure multiplied by its
    # one-session gross return, 1 + R_i,t.
    ending_positions = (
        beginning_positions
        * (1.0 + aligned_returns)
    )

    ending_value = float(
        ending_positions.sum()
        + cash_value
    )

    if ending_value <= 0.0:
        raise ValueError(
            "Ending portfolio value must remain positive."
        )

    # P&L is the change in wealth. Loss uses the opposite sign so adverse
    # outcomes are positive, matching the project's VaR and ES convention.
    pnl = ending_value - beginning_value
    loss = -pnl
    portfolio_return = pnl / beginning_value

    # Returns change each position by a different amount. Dividing the ending
    # positions by total ending wealth produces the naturally drifted weights.
    ending_weights = (
        ending_positions / ending_value
    ).rename("weight")
    ending_cash_weight = cash_value / ending_value

    return PortfolioStep(
        portfolio_return=portfolio_return,
        pnl=pnl,
        loss=loss,
        ending_value=ending_value,
        ending_weights=ending_weights,
        ending_cash_weight=ending_cash_weight,
    )


def run_portfolio_path(
    asset_returns: pd.DataFrame,
    initial_value: float,
    target_weights: pd.Series,
    rebalancing_frequency: str,
) -> PortfolioPath:
    """Apply asset returns through time and rebalance on schedule."""

    if asset_returns.empty:
        raise ValueError(
            "Portfolio asset_returns must not be empty."
        )

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise ValueError(
            "Portfolio asset_returns must use a DatetimeIndex."
        )

    if asset_returns.index.has_duplicates:
        raise ValueError(
            "Portfolio asset_returns dates must be unique."
        )

    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError(
            "Portfolio asset_returns dates must be sorted."
        )

    if asset_returns.columns.has_duplicates:
        raise ValueError(
            "Portfolio asset_returns must have unique ticker columns."
        )

    if target_weights.index.has_duplicates:
        raise ValueError(
            "Portfolio target_weights must have unique tickers."
        )

    if set(asset_returns.columns) != set(target_weights.index):
        raise ValueError(
            "Portfolio target_weights and asset_returns "
            "must have the same tickers."
        )

    if rebalancing_frequency not in {"daily", "monthly"}:
        raise ValueError(
            "Portfolio rebalancing_frequency must be daily or monthly."
        )

    # The return columns define the canonical ticker order used throughout
    # the simulation and in every output weight table.
    aligned_targets = target_weights.reindex(asset_returns.columns).rename(
        "weight"
    )

    if not np.isfinite(aligned_targets.to_numpy()).all():
        raise ValueError(
            "Portfolio target_weights must be finite numbers."
        )

    if (aligned_targets < 0.0).any():
        raise ValueError(
            "Portfolio target_weights must be non-negative."
        )

    target_asset_weight = float(aligned_targets.sum())

    if (
        target_asset_weight > 1.0
        and not np.isclose(target_asset_weight, 1.0)
    ):
        raise ValueError(
            "Portfolio target_weights cannot sum to more than one."
        )

    # calculate_portfolio_step performs the detailed value, weight, and
    # return validation for the initial state and for each daily row.
    current_value = initial_value
    current_weights = aligned_targets.copy()
    target_cash_weight = max(0.0, 1.0 - target_asset_weight)

    daily_rows: list[dict[str, float | bool]] = []
    beginning_weight_rows: list[pd.Series] = []
    pre_rebalance_weight_rows: list[pd.Series] = []

    dates = asset_returns.index

    for position, (date, daily_returns) in enumerate(
        asset_returns.iterrows()
    ):
        # These weights were known before this session's returns occurred.
        beginning_weight_rows.append(current_weights.copy())

        step = calculate_portfolio_step(
            beginning_value=current_value,
            beginning_weights=current_weights,
            asset_returns=daily_returns,
        )
        pre_rebalance_weight_rows.append(step.ending_weights.copy())

        has_next_session = position + 1 < len(dates)
        is_month_end = (
            has_next_session
            and date.to_period("M")
            != dates[position + 1].to_period("M")
        )
        rebalance_after_close = has_next_session and (
            rebalancing_frequency == "daily"
            or (
                rebalancing_frequency == "monthly"
                and is_month_end
            )
        )

        if rebalance_after_close:
            # One-way turnover measures the fraction of portfolio value moved
            # between assets (and cash) to restore the target allocation.
            turnover = 0.5 * (
                float(
                    (
                        aligned_targets
                        - step.ending_weights
                    ).abs().sum()
                )
                + abs(
                    target_cash_weight
                    - step.ending_cash_weight
                )
            )
            next_weights = aligned_targets.copy()
        else:
            turnover = 0.0
            next_weights = step.ending_weights.copy()

        daily_rows.append(
            {
                "beginning_value": current_value,
                "portfolio_return": step.portfolio_return,
                "pnl": step.pnl,
                "loss": step.loss,
                "ending_value": step.ending_value,
                "rebalanced_after_close": rebalance_after_close,
                "turnover": turnover,
            }
        )

        # With zero transaction costs, rebalancing redistributes wealth but
        # does not change total value. The ending state becomes tomorrow's
        # beginning state.
        current_value = step.ending_value
        current_weights = next_weights

    return PortfolioPath(
        daily=pd.DataFrame(
            daily_rows,
            index=dates,
        ),
        beginning_weights=pd.DataFrame(
            beginning_weight_rows,
            index=dates,
        ),
        pre_rebalance_weights=pd.DataFrame(
            pre_rebalance_weight_rows,
            index=dates,
        ),
    )


def save_portfolio_path(
    result: PortfolioPath,
    output_root: str | Path,
    snapshot_id: str,
) -> PortfolioSnapshotFiles:
    """Save a portfolio path in a new snapshot directory."""

    if result.daily.empty:
        raise ValueError(
            "Portfolio path must contain at least one session."
        )

    indexes_match = (
        result.daily.index.equals(
            result.beginning_weights.index
        )
        and result.daily.index.equals(
            result.pre_rebalance_weights.index
        )
    )

    if not indexes_match:
        raise ValueError(
            "Portfolio path tables must use the same dates."
        )

    weight_columns_match = (
        result.beginning_weights.columns.equals(
            result.pre_rebalance_weights.columns
        )
    )

    if not weight_columns_match:
        raise ValueError(
            "Portfolio weight tables must use the same tickers."
        )

    snapshot_directory = (
        Path(output_root) / snapshot_id
    )
    snapshot_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    files = PortfolioSnapshotFiles(
        daily=(
            snapshot_directory
            / "portfolio_daily.csv"
        ),
        beginning_weights=(
            snapshot_directory
            / "beginning_weights.csv"
        ),
        pre_rebalance_weights=(
            snapshot_directory
            / "pre_rebalance_weights.csv"
        ),
    )

    daily = result.daily.copy()
    beginning_weights = (
        result.beginning_weights.copy()
    )
    pre_rebalance_weights = (
        result.pre_rebalance_weights.copy()
    )

    daily.index.name = "Date"
    beginning_weights.index.name = "Date"
    pre_rebalance_weights.index.name = "Date"

    daily.to_csv(files.daily)
    beginning_weights.to_csv(
        files.beginning_weights
    )
    pre_rebalance_weights.to_csv(
        files.pre_rebalance_weights
    )

    return files
