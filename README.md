# Portfolio Tail-Risk Engine

## Project status

See the [project checkpoint list](docs/project-checkpoints.md) for completed work, the current phase, and upcoming milestones.

## Gaussian backtests

Run the configured 100,000-scenario rolling Gaussian baseline from the
project root. Rolling Gaussian remains the default model:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset
```

Run the EWMA Gaussian baseline over the same forecast timeline and with the
same simulation settings:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  --model ewma-gaussian
```

For a faster workflow check, override only the scenario count and write to a
temporary output directory:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  --model ewma-gaussian \
  --scenario-count 1000 \
  --output-root /tmp/portfolio-tail-risk-smoke
```

Every saved snapshot contains `forecasts.csv` and `metadata.json`. The command
reloads the snapshot and verifies its checksum and forecast structure before
reporting success.

Validate the saved EWMA Monte Carlo forecasts against closed-form Gaussian
VaR and Expected Shortfall across the complete forecast timeline:

```bash
.venv/bin/python scripts/report_ewma_validation.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  data/backtests/20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo
```
