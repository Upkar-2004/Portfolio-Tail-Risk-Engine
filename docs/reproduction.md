# Reproduction Guide

## 1. Purpose and artifact policy

This guide records the commands used to build, validate, evaluate, and plot
the `v0.1.0` Python baseline of the Portfolio Tail-Risk Engine.

Commands are run from the repository root. Generated artifacts are written to
timestamped, non-overwriting directories under `data/`. Those directories are
excluded from Git because raw market data, portfolio paths, forecasts, and
complete evaluation tables can be large or machine-specific. Every completed
snapshot includes metadata and SHA-256 checksums. Selected stable figures and
compact tables are copied to `reports/` and tracked by Git.

Canonical baseline snapshot identifiers and checksums are recorded in
[`project-checkpoints.md`](project-checkpoints.md). A new market-data download
receives a new timestamp and should not be expected to reproduce the original
raw bytes after a data provider revises its history.

## 2. Environment setup

Python 3.12 or newer is required.

```bash
git clone https://github.com/Upkar-2004/Portfolio-Tail-Risk-Engine.git
cd Portfolio-Tail-Risk-Engine
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Run the complete automated test suite:

```bash
.venv/bin/pytest -q
```

The current baseline contains 650 tests. A different count after later
extensions is acceptable if the complete suite still passes.

## 3. Configuration

The baseline experiment is defined in [`configs/baseline.yaml`](../configs/baseline.yaml).
Important settings include:

- 11 USD-denominated equities spanning 11 sectors;
- daily Yahoo Finance data from 2010-01-01 through the exclusive end date
  2026-01-01;
- an equal-weighted, long-only portfolio rebalanced monthly;
- a one-session forecast horizon and 504-session estimation window;
- 95%, 97.5%, and 99% VaR and ES;
- 100,000 scenarios and random seed 20261002; and
- EWMA decay factor 0.94.

Baseline conclusions should not be silently regenerated under edited settings.
Material changes belong in a new configuration and a new experiment snapshot.

## 4. Acquire and process data

The download command immediately contacts Yahoo Finance, validates the returned
fields and configured tickers, and saves a new raw snapshot:

```bash
.venv/bin/python scripts/download_data.py
```

The command prints the new snapshot directory. Use that identifier in the
following commands, replacing `SNAPSHOT_ID` with the generated directory name:

```bash
.venv/bin/python scripts/process_returns.py data/raw/SNAPSHOT_ID

.venv/bin/python scripts/process_portfolio.py data/processed/SNAPSHOT_ID

.venv/bin/python scripts/summarize_portfolio.py data/portfolio/SNAPSHOT_ID
```

The raw, processed-return, and portfolio directories for one lineage share the
same snapshot identifier. Loaders verify saved checksums before returning data.

## 5. Run the three baseline backtests

Set `SNAPSHOT_ID` in the paths below to the verified processed-return and
portfolio snapshot created in the previous stage.

Rolling Gaussian:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/SNAPSHOT_ID \
  data/portfolio/SNAPSHOT_ID
```

EWMA Gaussian:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/SNAPSHOT_ID \
  data/portfolio/SNAPSHOT_ID \
  --model ewma-gaussian
```

Filtered Historical Simulation:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/SNAPSHOT_ID \
  data/portfolio/SNAPSHOT_ID \
  --model filtered-historical
```

Each command creates a distinct model-specific directory under
`data/backtests/`, then reloads and verifies the saved forecasts. Record the
three printed backtest paths for the evaluation stage.

### Faster workflow check

The official baseline uses 100,000 scenarios. To verify orchestration without
waiting for a full run, use an explicit smaller scenario count and a temporary
output directory:

```bash
.venv/bin/python scripts/run_backtest.py \
  data/processed/SNAPSHOT_ID \
  data/portfolio/SNAPSHOT_ID \
  --model filtered-historical \
  --scenario-count 1000 \
  --output-root /tmp/portfolio-tail-risk-smoke
```

A smoke run validates the workflow but is not a substitute for the official
baseline.

## 6. Validate the Gaussian simulation

Generate a selected-date Gaussian loss-distribution report and figure:

```bash
.venv/bin/python scripts/report_gaussian_simulation.py \
  data/processed/SNAPSHOT_ID \
  data/portfolio/SNAPSHOT_ID \
  --figure-output reports/figures/gaussian_tail_risk_selected_date.png
```

Validate the saved EWMA Monte Carlo forecasts against closed-form Gaussian VaR
and ES across the complete forecast timeline:

```bash
.venv/bin/python scripts/report_ewma_validation.py \
  data/processed/SNAPSHOT_ID \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/EWMA_BACKTEST_ID
```

## 7. Generate the full-sample evaluation reports

In the commands below, replace the four path identifiers with the verified
portfolio and model snapshots produced above.

### VaR coverage

```bash
.venv/bin/python scripts/report_var_coverage.py \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/ROLLING_BACKTEST_ID \
  data/backtests/EWMA_BACKTEST_ID \
  data/backtests/FHS_BACKTEST_ID
```

The report contains Kupiec unconditional coverage, Christoffersen independence
and conditional coverage, transition counts, raw p-values, and the predeclared
Holm-adjusted conditional-coverage decisions.

### Expected Shortfall tail severity

```bash
.venv/bin/python scripts/report_es_tail_severity.py \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/ROLLING_BACKTEST_ID \
  data/backtests/EWMA_BACKTEST_ID \
  data/backtests/FHS_BACKTEST_ID
```

These conditional tail summaries are descriptive and are not formal pairwise
model rankings.

### Joint VaR-ES calibration

```bash
.venv/bin/python scripts/report_var_es_joint_calibration.py \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/ROLLING_BACKTEST_ID \
  data/backtests/EWMA_BACKTEST_ID \
  data/backtests/FHS_BACKTEST_ID
```

The report tests the mean joint identification residual with Newey-West HAC
uncertainty and Holm adjustment across the nine model-confidence hypotheses.

### Pairwise predictive accuracy

Generate the primary automatic-bandwidth report:

```bash
.venv/bin/python scripts/report_predictive_accuracy.py \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/ROLLING_BACKTEST_ID \
  data/backtests/EWMA_BACKTEST_ID \
  data/backtests/FHS_BACKTEST_ID
```

Generate the predeclared lag-zero sensitivity report from the same inputs:

```bash
.venv/bin/python scripts/report_predictive_accuracy.py \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/ROLLING_BACKTEST_ID \
  data/backtests/EWMA_BACKTEST_ID \
  data/backtests/FHS_BACKTEST_ID \
  --bandwidth 0
```

The two score families contain nine quantile-loss comparisons and nine FZ0
comparisons. Holm adjustment is applied separately within the two families.

### Basel-style rolling traffic light

```bash
.venv/bin/python scripts/report_basel_traffic_light.py \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/ROLLING_BACKTEST_ID \
  data/backtests/EWMA_BACKTEST_ID \
  data/backtests/FHS_BACKTEST_ID
```

This is a 99% VaR research diagnostic using complete 250-session windows. It
does not claim regulatory compliance.

## 8. Generate the descriptive subperiod tables

```bash
.venv/bin/python scripts/report_subperiod_analysis.py \
  data/portfolio/SNAPSHOT_ID \
  data/backtests/ROLLING_BACKTEST_ID \
  data/backtests/EWMA_BACKTEST_ID \
  data/backtests/FHS_BACKTEST_ID
```

The command produces the fixed 2012-2025 calendar definitions and the separate
February-April 2020 NBER case study, then saves five checksum-traced tables in
one immutable evaluation directory. The subperiod analysis is descriptive and
does not introduce additional local hypothesis tests.

## 9. Generate publication figures and compact tables

Replace `SUBPERIOD_REPORT_ID`, `PREDICTIVE_REPORT_ID`, and
`LAG_ZERO_REPORT_ID` with the verified evaluation snapshots created above.

### Exception-rate figures

```bash
.venv/bin/python scripts/report_subperiod_exception_rates.py \
  data/evaluations/SUBPERIOD_REPORT_ID
```

This refreshes the tracked heatmap and exact-interval PNG and SVG files under
`reports/figures/`.

### Predictive-score summary and supporting annual ranks

```bash
.venv/bin/python scripts/report_subperiod_predictive_scores.py \
  data/evaluations/SUBPERIOD_REPORT_ID
```

This refreshes the first-place summary, annual rank grid, and 135-row companion
score table.

### Full-sample predictive-inference figure

```bash
.venv/bin/python scripts/report_predictive_accuracy_figure.py \
  data/evaluations/PREDICTIVE_REPORT_ID \
  data/evaluations/SUBPERIOD_REPORT_ID \
  data/evaluations/LAG_ZERO_REPORT_ID
```

This refreshes the full-sample DM figure and 18-row companion magnitude table.
The command verifies all three input checksums and preserves another immutable
figure snapshot before replacing the stable publication copies.

## 10. Official `v0.1.0` baseline lineage

The publication figures currently tracked by Git were generated from these
verified local snapshots:

```text
Portfolio:
  20260825T162125895465Z_baseline_11_asset

Rolling Gaussian:
  20261005T021450532620Z_baseline_11_asset_rolling_gaussian_monte_carlo

EWMA Gaussian:
  20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo

Filtered Historical Simulation:
  20261011T112940733142Z_baseline_11_asset_filtered_historical_simulation

Primary predictive accuracy:
  20261014T014947595076Z_baseline_11_asset_predictive_accuracy

Lag-zero sensitivity:
  20261016T174625930461Z_baseline_11_asset_predictive_accuracy

Subperiod analysis:
  20261014T170317535380Z_baseline_11_asset_subperiod_analysis

Current full-sample inference figure:
  20261016T182508090910Z_baseline_11_asset_predictive_accuracy_figure
```

The authoritative output checksums and the remaining evaluation snapshot IDs
are recorded in [`project-checkpoints.md`](project-checkpoints.md).

## 11. Reproduction boundaries

- Yahoo Finance may revise historical records, corporate actions, or metadata.
  A later fresh download can reproduce the protocol without being byte-identical
  to the original raw snapshot.
- The complete baseline is computationally expensive. Smoke runs verify code
  paths but cannot reproduce the published estimates.
- Generated directories must not be edited manually or silently overwritten.
- Publication files under `reports/` must be regenerated from verified inputs
  and committed together with the code and interpretation that produced them.
- The C++ and pybind11 phases are future extensions and are not required to
  reproduce the completed Python baseline.
