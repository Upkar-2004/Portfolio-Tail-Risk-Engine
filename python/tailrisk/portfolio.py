"""Portfolio construction and accounting."""

from dataclasses import dataclass

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
