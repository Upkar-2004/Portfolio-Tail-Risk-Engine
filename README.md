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

Generate the publication figure and companion magnitude table from the
verified baseline report, the verified subperiod score table, and the
bandwidth-zero sensitivity report:

```bash
.venv/bin/python scripts/report_predictive_accuracy_figure.py \
  data/evaluations/20261014T014947595076Z_baseline_11_asset_predictive_accuracy \
  data/evaluations/20261014T170317535380Z_baseline_11_asset_subperiod_analysis \
  data/evaluations/20261016T174625930461Z_baseline_11_asset_predictive_accuracy
```

The figure uses the fixed display order EWMA minus Rolling Gaussian, FHS minus
Rolling Gaussian, and FHS minus EWMA, so negative statistics always favor the
first named model. Marker color and shape identify the sample winner, filled
markers denote Holm-adjusted rejection, and a dagger flags decisions that
become significant when the Newey-West bandwidth is set to zero; the reported
automatic-bandwidth specification is more conservative in those cases. The
accompanying table reports raw mean score differences and, for quantile loss
only, the percentage difference relative to the second model's mean score. The
command verifies all source checksums and refreshes the stable files under
`reports/` while also preserving an immutable versioned snapshot under
`data/evaluations/`.

## Basel-style rolling traffic-light evaluation

Generate the rolling 99% VaR monitoring report from the three verified
baseline backtests:

```bash
.venv/bin/python scripts/report_basel_traffic_light.py \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  data/backtests/20261005T021450532620Z_baseline_11_asset_rolling_gaussian_monte_carlo \
  data/backtests/20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo \
  data/backtests/20261011T112940733142Z_baseline_11_asset_filtered_historical_simulation
```

The command verifies each source checksum and the shared portfolio lineage,
then counts strict 99% VaR exceptions over complete rolling 250-session
windows. It assigns the current Basel labels green for zero through four
exceptions, amber for five through nine, and red for ten or more. The output
is a Basel-style research diagnostic rather than a claim of regulatory
compliance.

Each run saves `basel_traffic_light_path.csv`,
`basel_traffic_light_summary.csv`, `basel_traffic_light.png`,
`basel_traffic_light.svg`, and `metadata.json` under a versioned directory in
`data/evaluations/`. The detailed path stores textual zone labels rather than
presentation colours. The compact summary reports latest and worst zones,
maximum exception counts and dates, zone counts and rates, and first amber and
red dates. The publication figure presents the three model paths in aligned
panels with shared axes, directly labelled zones, and visible five- and
ten-exception boundaries. Metadata records the thresholds, figure design,
official Basel reference, source snapshots, and checksums for both CSVs and
both figures.

## Descriptive subperiod analysis

Generate all five predeclared calendar-year and NBER-period tables together:

```bash
.venv/bin/python scripts/report_subperiod_analysis.py \
  data/portfolio/20260825T162125895465Z_baseline_11_asset \
  data/backtests/20261005T021450532620Z_baseline_11_asset_rolling_gaussian_monte_carlo \
  data/backtests/20261006T151840866317Z_baseline_11_asset_ewma_gaussian_monte_carlo \
  data/backtests/20261011T112940733142Z_baseline_11_asset_filtered_historical_simulation
```

The command verifies the source checksums and shared portfolio lineage before
creating `subperiod_definitions.csv`, `subperiod_var_coverage.csv`,
`subperiod_es_tail_severity.csv`, `subperiod_joint_calibration.csv`, and
`subperiod_predictive_accuracy.csv` in one new, non-overwriting directory
under `data/evaluations/`. Its metadata records every output checksum, the
fixed 2012–2025 calendar and February–April 2020 NBER rules, pointwise exact
interval status, sparse-tail thresholds, and within-period scoring rules.

This report is a descriptive secondary analysis. It adds no subperiod
hypothesis tests, rejection decisions, or multiple-testing family; the
completed full-sample inference remains primary.

Generate the publication subperiod exception-rate figures from that verified
snapshot:

```bash
.venv/bin/python scripts/report_subperiod_exception_rates.py \
  data/evaluations/20261014T170317535380Z_baseline_11_asset_subperiod_analysis
```

The command verifies the saved VaR-coverage checksum before plotting. It
creates a new immutable figure directory under `data/evaluations/`, records
PNG and SVG checksums plus source lineage, and refreshes four stable tracked
copies:

- `reports/figures/subperiod_exception_rate_heatmap.png` and `.svg`;
- `reports/figures/subperiod_exception_rate_intervals.png` and `.svg`.

The heatmap's three confidence-level panels show observed minus nominal annual
exception rates in percentage points. The interval figure shows observed
rates with pointwise 95% Clopper-Pearson exact binomial intervals and dashed
nominal-rate references. Its annual panels use confidence-level-specific scales
so the rarer tails remain legible. Pale-green central 95% binomial reference
ranges use 252 sessions in the annual panels and 62 sessions in the stress
panels. Both figures keep the overlapping February-April 2020 NBER case study
separate from the calendar-year panels. These displays are descriptive; their
pointwise intervals are not simultaneous or multiplicity-adjusted, the
reference ranges are not new decision rules, and neither figure replaces the
primary full-sample tests.

Generate the main predictive-score summary, supporting annual rank grid, and
companion score table from the same verified snapshot:

```bash
.venv/bin/python scripts/report_subperiod_predictive_scores.py \
  data/evaluations/20261014T170317535380Z_baseline_11_asset_subperiod_analysis
```

The command checksum-verifies `subperiod_predictive_accuracy.csv`, creates a
new immutable report directory with traced checksums, and refreshes these
stable publication files:

- `reports/figures/subperiod_predictive_first_place_summary.png` and `.svg`;
- `reports/figures/subperiod_predictive_annual_ranks.png` and `.svg`; and
- `reports/tables/subperiod_predictive_scores.csv`.

The main-text summary plots calendar-year first-place counts for QL and FZ0 by
confidence level. The dashed 14/3 line is an even-split visual reference, not a
test. The supporting grid groups confidence levels within QL and FZ0, colors
rank-one cells with the winning model's house color, and preserves the separate
February-April 2020 stress strip. The companion table reports all period-average
score magnitudes and ranks. No subperiod Diebold-Mariano statistics or
significance decisions are added.

The immutable report directory also stores an unchanged checksum-matching copy
of `subperiod_predictive_accuracy.csv` beside the figures, companion table, and
metadata. The tracked publication files remain generated outputs and must not
be edited by hand.

### Figure house style

Shared model identities, colors, markers, typography, semantic palettes, and
plotting helpers live in `python/tailrisk/style.py`. Production visualization
code imports those definitions instead of maintaining separate copies. The
module supports a paper genre and a report genre, plus a local style context so
figure settings do not leak between tests or report commands. Its behavioral
guards live in `tests/python/test_style.py`.
