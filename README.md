# Portfolio Tail-Risk Engine

## Project status

See the [project checkpoint list](docs/project-checkpoints.md) for completed work, the current phase, and upcoming milestones.

## Rolling Gaussian backtest

Run the configured 100,000-scenario baseline from the project root:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset
```

For a faster workflow check, override only the scenario count and write to a
temporary output directory:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  --scenario-count 1000 \
  --output-root /tmp/portfolio-tail-risk-smoke
```

Every saved snapshot contains `forecasts.csv` and `metadata.json`. The command
reloads the snapshot and verifies its checksum and forecast structure before
reporting success.
