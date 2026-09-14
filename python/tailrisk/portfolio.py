"""Portfolio construction and accounting."""

import pandas as pd


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