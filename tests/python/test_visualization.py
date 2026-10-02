from pathlib import Path

import numpy as np
import pandas as pd

from tailrisk.models import calculate_gaussian_var_es
from tailrisk.simulation import calculate_empirical_var_es
from tailrisk.visualization import save_gaussian_tail_risk_figure


def test_save_gaussian_tail_risk_figure_creates_png(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(12345)

    analytic_mean_return = 0.001
    analytic_volatility = 0.02

    simulated_returns = (
        analytic_mean_return
        + analytic_volatility * rng.standard_normal(20_000)
    )
    simulated_losses = -simulated_returns

    confidence_levels = (0.95, 0.975, 0.99)

    analytic_forecasts = tuple(
        calculate_gaussian_var_es(
            mean_return=analytic_mean_return,
            volatility=analytic_volatility,
            confidence_level=confidence_level,
        )
        for confidence_level in confidence_levels
    )

    empirical_estimates = tuple(
        calculate_empirical_var_es(
            simulated_losses,
            confidence_level=confidence_level,
        )
        for confidence_level in confidence_levels
    )

    output_path = tmp_path / "gaussian_tail_risk.png"

    saved_path = save_gaussian_tail_risk_figure(
        losses=simulated_losses,
        analytic_mean_return=analytic_mean_return,
        analytic_volatility=analytic_volatility,
        analytic_forecasts=analytic_forecasts,
        empirical_estimates=empirical_estimates,
        realized_loss=0.025,
        forecast_date=pd.Timestamp("2025-12-31"),
        estimation_start_date=pd.Timestamp("2024-01-24"),
        estimation_end_date=pd.Timestamp("2025-12-30"),
        estimation_observation_count=504,
        asset_count=11,
        scenario_count=20_000,
        random_seed=12345,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")