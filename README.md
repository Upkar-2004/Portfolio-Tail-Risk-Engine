# Portfolio Tail-Risk Engine

## Project status

See the [project checkpoint list](docs/project-checkpoints.md) for completed work, the current phase, and upcoming milestones.

## Risk-model backtests

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

Run the filtered historical simulation baseline with the same EWMA covariance
filter, forecast timeline, and simulation settings:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  --model filtered-historical
```

For a faster workflow check, override only the scenario count and write to a
temporary output directory:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/20260825T162125895465Z_baseline_11_asset \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  --model filtered-historical \
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

## VaR coverage evaluation

Generate the shared VaR coverage report after all three baseline backtests
have been saved:

```bash
.venv/bin/python scripts/report_var_coverage.py \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  data/backtests/20261005T021450532620Z_baseline_11_asset_rolling_gaussian_monte_carlo \
  data/backtests/20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo \
  data/backtests/20261011T112940733142Z_baseline_11_asset_filtered_historical_simulation
```

The command checksum-verifies the three source backtests, requires identical
forecast dates and realized losses, and prints one row for each of the nine
model-confidence combinations. It reports exceedance counts, transition
counts, Kupiec unconditional coverage, Christoffersen independence,
Christoffersen conditional coverage, and Holm-adjusted conditional-coverage
decisions.

Each run saves a versioned `var_coverage.csv` and `metadata.json` under
`data/evaluations/`. These generated reports remain local and are excluded
from Git; the metadata records the source-backtest checksums and statistical
decision rules needed to reproduce the table.

## Expected Shortfall tail-severity evaluation

Generate the shared descriptive tail-severity report after all three baseline
backtests have been saved:

```bash
.venv/bin/python scripts/report_es_tail_severity.py \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  data/backtests/20261005T021450532620Z_baseline_11_asset_rolling_gaussian_monte_carlo \
  data/backtests/20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo \
  data/backtests/20261011T112940733142Z_baseline_11_asset_filtered_historical_simulation
```

The command checksum-verifies the source backtests, requires identical
ordered forecast rows and realized losses, and prints one row for each of the
nine model-confidence combinations. It reports the VaR-exceedance count,
mean realized exceedance loss, mean VaR and Expected Shortfall on exceedance
dates, mean, median, and maximum loss excess over VaR, and the mean realized
loss minus forecast Expected Shortfall.

Each run saves a versioned `es_tail_severity.csv` and `metadata.json` under
`data/evaluations/`. The metadata records the source-backtest checksums, the
strict VaR-exceedance rule, the ES-VaR ordering tolerance, and the descriptive
status of the results. These conditional summaries are diagnostics and must
not be interpreted as formal pairwise model rankings.

## Joint VaR-Expected-Shortfall calibration

Generate the formal joint-calibration report from the same three verified
baseline backtests:

```bash
.venv/bin/python scripts/report_var_es_joint_calibration.py \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  data/backtests/20261005T021450532620Z_baseline_11_asset_rolling_gaussian_monte_carlo \
  data/backtests/20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo \
  data/backtests/20261011T112940733142Z_baseline_11_asset_filtered_historical_simulation
```

The command verifies the source checksums and lineage, calculates the
all-session joint VaR-ES identification residual, and tests its mean using
Newey-West HAC uncertainty. It records the selected bandwidth and reports raw
and Holm-adjusted decisions across the nine model-confidence hypotheses.

Each run saves a versioned `var_es_joint_calibration.csv` and `metadata.json`
under `data/evaluations/`. The metadata records the residual definition, HAC
kernel and bandwidth rule, significance level, Holm family, source snapshots,
and output checksum required to reproduce the report.

## Pairwise predictive-accuracy evaluation

Generate the formal model-comparison report from the three verified baseline
backtests:

```bash
.venv/bin/python scripts/report_predictive_accuracy.py \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  data/backtests/20261005T021450532620Z_baseline_11_asset_rolling_gaussian_monte_carlo \
  data/backtests/20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo \
  data/backtests/20261011T112940733142Z_baseline_11_asset_filtered_historical_simulation
```

The command checksum-verifies the source forecasts and their common portfolio
lineage. It calculates daily VaR quantile loss and joint VaR-Expected-Shortfall
FZ0 loss, then runs two-sided pairwise Diebold-Mariano tests with Newey-West
HAC uncertainty for all three model pairs at all three confidence levels. A
negative model-A-minus-model-B mean score difference favors model A; a positive
difference favors model B.

The resulting `predictive_accuracy.csv` contains 18 rows: nine quantile-loss
comparisons and nine FZ0 comparisons. Holm adjustment is applied separately
within those two nine-test families. The accompanying `metadata.json` records
the score direction, null hypothesis, HAC kernel and bandwidth rule,
significance level, separate Holm families, source snapshot checksums, and
output checksum.
