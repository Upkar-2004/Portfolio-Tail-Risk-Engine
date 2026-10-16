# Project Checkpoints

Last updated: 2026-10-16

This checklist tracks the research, learning, and engineering progress of the Portfolio Tail-Risk Engine. A checkpoint is marked complete only when the relevant implementation or decision is documented, tested where applicable, and committed.

**Current phase:** Results visualization and interpretation

**Next checkpoint:** Build the concise publication-facing results summary and
GitHub figure gallery without changing the completed statistical protocol.

## 1. Project definition and research scope

- [x] Define the project purpose and research question.
- [x] Define the three-model comparison: rolling Gaussian, EWMA Gaussian, and filtered historical simulation.
- [x] Record the one-day forecast horizon and loss-sign convention.
- [x] Separate the Python research layer, C++ simulation core, and pybind11 bindings.
- [x] Create the public GitHub repository and initial project structure.
- [x] Document the initial methodology and data-source policies.
- [x] Select the 11-company baseline universe.
- [x] Fix the requested sample period from 2010-01-01 through the exclusive end date 2026-01-01.
- [ ] Finalize all experiment decisions listed in `docs/methodology.md` before the relevant backtests begin.

## 2. Python project foundation

- [x] Create a project-specific virtual environment.
- [x] Configure the Python package and dependencies in `pyproject.toml`.
- [x] Configure pytest and the Python test directory.
- [x] Create and validate `configs/baseline.yaml`.
- [x] Implement the YAML configuration loader.
- [x] Test valid, malformed, incomplete, and missing configuration files.

## 3. Data acquisition and validation

- [x] Confirm the installed `yfinance` response structure using a small sample.
- [x] Implement configuration-driven market-data downloading.
- [x] Extract fields from the `Price` and `Ticker` column levels.
- [x] Validate expected tickers and required fields.
- [x] Validate date type, uniqueness, and chronological ordering.
- [x] Validate that observed prices are positive and finite.
- [x] Preserve missing prices and count them by ticker.
- [x] Create the initial `scripts/download_data.py` command-line entry point.
- [x] Run the full baseline retrieval successfully: 4,024 sessions, 88 columns, 11 tickers, and zero missing adjusted-close prices.
- [x] Persist each raw download as a versioned snapshot without silent overwriting.
- [x] Record retrieval metadata, including timestamp, provider, package version, configuration, fields, and validation results.
- [x] Calculate and record a stable checksum for the raw snapshot.
- [x] Implement and test loading a saved raw snapshot.
- [x] Produce a retained validation summary for each retrieval run.

## 4. Asset-return pipeline

- [x] Review the mathematics of one-day simple returns.
- [x] Implement adjusted-close simple returns without implicit forward filling.
- [x] Ensure missing prices produce missing returns.
- [x] Test positive, negative, zero, and missing-return examples.
- [x] Verify that returns use consecutive chronological sessions.
- [x] Generate and validate the baseline asset-return matrix.
- [x] Flag unusually large returns for review without automatically deleting them.
- [x] Save the processed-return snapshot with lineage metadata and a SHA-256 checksum.
- [x] Implement and test checksum-verified loading of processed returns.

## 5. Portfolio construction and realized losses

- [x] Finalize and document the baseline portfolio-weight rule.
- [x] Finalize the rebalancing frequency, cash treatment, and initial portfolio value.
- [x] Add the portfolio decisions to the baseline configuration.
- [x] Implement beginning-of-period weights, natural drift, and monthly rebalancing.
- [x] Calculate portfolio returns, P&L, and losses using the documented sign convention.
- [x] Test return aggregation, P&L, loss, turnover, and rebalancing numerically.
- [x] Verify that no future information enters the portfolio weights.
- [x] Save portfolio paths, weight histories, lineage metadata, and SHA-256 checksums.
- [x] Implement and test a loader that verifies every portfolio-snapshot checksum.
- [x] Reject portfolio snapshots with inconsistent dates or ticker columns.

## 6. Descriptive portfolio-return diagnostics

- [x] Review sample mean, sample variance, volatility, and annualization.
- [x] Review skewness, excess kurtosis, empirical loss quantiles, and extreme returns.
- [x] Implement and test descriptive return diagnostics against a manual example.
- [x] Validate sample-size, date-index, finiteness, variability, and annualization inputs.
- [x] Add a command-line summary using the checksum-verified portfolio snapshot.
- [x] Reproduce the baseline portfolio statistics over all 4,023 sessions.

## 7. Backtest protocol decisions

- [x] Fix the primary forecast horizon at one trading session.
- [x] Fix the VaR and Expected Shortfall levels at 95%, 97.5%, and 99%.
- [x] Fix the baseline estimation window at 504 sessions.
- [x] Fix the first eligible baseline forecast date at 2012-01-04.
- [x] Validate the shared forecasting settings in the baseline configuration.
- [x] Fix covariance-estimation conventions.
- [x] Fix simulation counts and random seeds.
- [x] Require verified, finite, complete portfolio returns on eligible forecast dates.
- [x] Fix the core VaR statistical-test decision rules.
- [x] Record the core VaR testing decisions before formal comparative
  interpretation.

## 8. Rolling Gaussian reference model

- [x] Review and test scalar Gaussian return-loss VaR and Expected Shortfall.
- [x] Implement analytic one-day Gaussian VaR and Expected Shortfall.
- [x] Implement and test a univariate rolling portfolio-return benchmark.
- [x] Review mean vectors, sample covariance matrices, and portfolio variance.
- [x] Implement and test one-window multivariate portfolio moments.
- [x] Implement rolling asset-level sample mean and covariance estimation.
- [x] Apply forecast-date portfolio weights to asset-level moments without look-ahead.
- [x] Validate covariance symmetry, finiteness, and positive semidefiniteness.
- [x] Implement a Python Monte Carlo reference simulation.
- [x] Test multivariate portfolio mean, variance, VaR, and ES on controlled examples.
- [x] Verify agreement between analytic and simulated Gaussian results within a justified tolerance.
- [x] Generate reusable Gaussian portfolio-loss scenarios.
- [x] Add a real-data report comparing analytic and Monte Carlo Gaussian estimates.
- [x] Add a reproducible loss-distribution figure for a selected forecast date.
- [x] Generate the 100,000-scenario rolling Gaussian baseline across all 3,519 forecast dates.
- [x] Save and reload-verify Gaussian forecasts, realized losses, exceedances, lineage, and metadata.

## 9. EWMA Gaussian model

- [x] Review the mathematics of exponentially weighted covariance estimation.
- [x] Select and document the EWMA decay parameter.
- [x] Fix and document the initialization, rolling-mean, innovation, and
  forecast-timing conventions.
- [x] Add and validate the EWMA decay parameter in baseline configuration.
- [x] Implement and validate one EWMA covariance update in Python.
- [x] Implement first-date EWMA covariance initialization.
- [x] Build the recursive EWMA covariance sequence without look-ahead.
- [x] Test initialization, sequence timing, symmetry, and numerical stability.
- [x] Implement EWMA Gaussian VaR and ES forecasts.
- [x] Test first-date initialization, recursive forecasting, shared shocks,
  backtest alignment, reproducibility, and decay-factor validation.
- [x] Add and test a reproducible command for the configured EWMA Gaussian
  backtest.
- [x] Generate the 100,000-scenario EWMA Gaussian baseline across all eligible
  forecast dates.
- [x] Save and reload-verify EWMA forecasts, realized losses, exceedances,
  lineage, and metadata.
- [x] Validate EWMA Monte Carlo VaR and Expected Shortfall against closed-form
  Gaussian forecasts across the complete baseline timeline.

## 10. Filtered historical simulation

- [x] Review volatility filtering, standardized residuals, resampling, and volatility rescaling.
- [x] Finalize the filtering and residual-sampling conventions.
- [x] Implement and test single-scenario covariance whitening, forecast
  rescaling, return reconstruction, and portfolio-loss calculation.
- [x] Implement the Python reference model.
- [x] Test filtering, residual standardization, resampling, and forecast scaling.
- [x] Implement filtered-historical-simulation VaR and ES forecasts.
- [x] Add and test a reproducible command for the configured FHS backtest.
- [x] Generate the 100,000-scenario FHS baseline across all eligible forecast
  dates.
- [x] Save and reload-verify FHS forecasts, realized losses, exceedances,
  lineage, and metadata.

## 11. Rolling forecast and backtest engine

- [x] Implement a shared one-step-ahead forecast schedule with no look-ahead.
- [x] Test forecast-window boundaries, date ordering, and realized-loss alignment.
- [x] Use identical forecast dates and realized losses for all models.
- [x] Store forecasts, realized losses, exceedances, and model metadata.
- [x] Run the complete baseline backtest for all three models.

## 12. Statistical evaluation

- [x] Review VaR exceedances and nominal coverage mathematically.
- [x] Implement exceedance-rate summaries.
- [x] Implement the Kupiec unconditional-coverage test.
- [x] Implement the Christoffersen independence and conditional-coverage
  tests.
- [x] Apply Holm adjustment across the nine primary conditional-coverage
  hypotheses while retaining all raw p-values.
- [x] Add and test a reproducible command that verifies the three source
  backtests and saves the combined nine-row VaR coverage report.
- [x] Generate the baseline VaR coverage report as snapshot
  `20261013T040616407287Z_baseline_11_asset_var_coverage`; its CSV SHA-256 is
  `0bdbe92b544613ebeabc47f76defe13ce15bdd48c9aca876facc184d205defdb`.
- [x] Implement exact 95% binomial confidence intervals for observed
  exceedance rates.
- [x] Implement and test descriptive Expected Shortfall tail-severity
  summaries.
- [x] Add and test a reproducible command that verifies the three source
  backtests and saves the combined nine-row ES tail-severity report.
- [x] Generate the baseline ES tail-severity report as snapshot
  `20261013T185315140525Z_baseline_11_asset_es_tail_severity`; its CSV SHA-256
  is `206e690a10f37275e5bcd6d68b70cfdbd3b8c2eba5b9d0b9b8587b2612ab5214`.
- [x] Implement the joint VaR-Expected-Shortfall identification residual and
  its predeclared Newey-West HAC uncertainty procedure.
- [x] Combine aligned model-confidence calibration results and apply Holm
  adjustment across the joint nine-test family.
- [x] Add the reproducible joint-calibration report command and save the
  verified nine-row baseline report.
- [x] Generate the baseline joint VaR-Expected-Shortfall calibration report as
  snapshot
  `20261013T214919902521Z_baseline_11_asset_var_es_joint_calibration`; its CSV
  SHA-256 is
  `c1d0cbca94b77d76e8fc1c2a27b928bf6a3511b7bb69ca18d3d2daf1764c0dcf`.
- [x] Test the completed VaR coverage statistics against controlled manual
  examples, boundary cases, and integration checks.
- [x] Implement strictly consistent VaR quantile loss and the joint
  VaR-Expected-Shortfall FZ0 score in the project's positive-loss convention.
- [x] Implement pairwise Diebold-Mariano tests with Newey-West HAC uncertainty
  for every model pair and confidence level.
- [x] Apply Holm adjustment separately to the nine quantile-loss comparisons
  and the nine FZ0 comparisons.
- [x] Add and test the reproducible predictive-accuracy report command.
- [x] Generate the 18-row baseline predictive-accuracy report as snapshot
  `20261014T014947595076Z_baseline_11_asset_predictive_accuracy`; its CSV
  SHA-256 is
  `0793527ab2c0fc44687103a7d34149ee78bc1a07892fc107fd1f915b4f5e3f1e`.
- [x] Generate the predeclared bandwidth-zero predictive-accuracy sensitivity
  report as snapshot
  `20261016T174625930461Z_baseline_11_asset_predictive_accuracy`; its CSV
  SHA-256 is
  `7803a24f77f6d676e4b6b43f6433b134d48b0e0b75a47c20e103d331dd18699c`.
- [x] Create and visually verify the full-sample predictive-accuracy inference
  figure and companion magnitude table. The figure snapshot is
  `20261016T182508090910Z_baseline_11_asset_predictive_accuracy_figure`.
  The PNG and SVG SHA-256 values are
  `659442f9d1091618eeb08506c0f854f05b166640ed00d6492457e7757b4725c9`
  and
  `ed6d845a4bb4eb67edbbe1a59302e2b7c08ea237df01b3d748ba26d539d826e4`.
  The companion-table SHA-256 is
  `245da916583be865b86c550ef84fd42e0d8749d1b0eb0cb70451efd9bb5621a3`.
- [x] Compare all models without assuming that greater complexity performs
  better.
- [x] Implement and test the 99% VaR, 250-session Basel-style traffic-light
  classifier, rolling paths, and compact multi-model summary.
- [x] Add and test the checksum-verified traffic-light report command.
- [x] Generate the baseline traffic-light report as snapshot
  `20261014T144750865418Z_baseline_11_asset_basel_traffic_light`; the detailed
  path CSV SHA-256 is
  `849364f36415cb0f4c83be76d7a8e61630b4e4cc78e463aa3be6328c44589771`
  and the summary CSV SHA-256 is
  `336bc10f1687181a8c8193a261480ef608e17f6e4d1049d5507a46eb03f3cafd`;
  its 300-DPI PNG SHA-256 is
  `b99d08253bddc7e58a9bbf258e57d74137620209cd59a288d188effd0a21ff4a`
  and its SVG SHA-256 is
  `ec0c52e070ec41b6bf0f50ab1846da9496dfc2279286ace6711e1d81296df53e`.
- [x] Create and visually verify the professional three-panel traffic-light
  timeline figure with shared axes, direct zone labels, and threshold lines.
- [x] Predeclare the calendar-year and externally dated NBER subperiod rules,
  sparse-tail warnings, descriptive statistics, inferential status, and
  reproducible output design.
- [x] Implement and test common-date calendar-year and NBER-period
  construction.
- [x] Implement and test the descriptive subperiod VaR coverage table with
  exact intervals and sparse-tail warnings.
- [x] Implement and test the descriptive subperiod tail-severity table,
  including missing zero-tail statistics and sparse-tail warnings.
- [x] Implement and test the descriptive subperiod joint-calibration table
  without adding subperiod HAC inference or hypothesis decisions.
- [x] Implement and test the descriptive subperiod predictive-accuracy table.
- [x] Add and test a checksum-verified subperiod report command.
- [x] Generate and verify the baseline subperiod tables as snapshot
  `20261014T170317535380Z_baseline_11_asset_subperiod_analysis` before
  designing the associated figures. The definitions CSV SHA-256 is
  `b8a4426d4f3727dae8a296558785f7feb0e3d7e869a28d25f3d807afade3d09a`,
  the VaR-coverage CSV SHA-256 is
  `9606b71649b728819beb86ec40ff5b4ab98fb7670ef7934af31be43f6383421f`,
  the ES-tail-severity CSV SHA-256 is
  `6626e622b70b6628f5d048e0523a344ee6e6262e5acd02736c3fcfd58bf96ab5`,
  the joint-calibration CSV SHA-256 is
  `002476b17a5b278a4d3db2e92795e4c57c790e9e2c8fb2af1dae204796866d78`,
  and the predictive-accuracy CSV SHA-256 is
  `955d38d4e7659d96981bd2e535d3a50a70e7c937cbed59b03b425bdad79e795c`.
- [x] Create and visually verify the confidence-level-faceted subperiod VaR
  exception-rate heatmap. The initial figure snapshot is
  `20261014T192324342907Z_baseline_11_asset_subperiod_exception_rate_figure`;
  its 300-DPI PNG SHA-256 is
  `092aa43a56d9066d60509e65a1f8c8c2d2423b53672903416d7d440ffda14c62`
  and its SVG SHA-256 is
  `f36f073583105f8a8400917d0519fd015f1c0957b6b1f30cb810f64eee058f6d`.
- [x] Create and visually verify the observed exception-rate figure with
  pointwise 95% Clopper-Pearson exact intervals. The current house-style figure
  snapshot is
  `20261016T150151101497Z_baseline_11_asset_subperiod_exception_rate_figures`.
  The exact-interval figure's 300-DPI PNG SHA-256 is
  `902942e86679e0cf06a2a81a8d6de29ab3d174ecd8bfd4d8c74d161ee38f6710`
  and its SVG SHA-256 is
  `a576484abc6c81052c8a81a1156a1e1a6ce378e88475402a017d8f03a7554102`.
  The house-style heatmap PNG SHA-256 is
  `5958d864ec8b4e9cdd8e65cc961bf02e1e090f878953c3afac71e0260a25aca6`
  and its SVG SHA-256 is
  `ded741ad06b92c6a73b86d0eb15f29723fa9dab10e9ede7f33f774e2c5347424`.
- [x] Split the predictive-score display into a main first-place summary and a
  supporting annual rank grid, then generate the 135-row companion score
  table. The current house-style snapshot is
  `20261016T155533505550Z_baseline_11_asset_subperiod_predictive_score_figures`.
  The summary PNG and SVG SHA-256 values are
  `7e67cf06bb0c626b0dfe0a11ec4c2d15519b29f17c10227ebbf1312a7c4bca3e`
  and
  `9503da42108aa65b8fe1f6a2ea57fe0b927f5dc4b2636ccd5e84e23f55c49943`.
  The annual-grid PNG and SVG SHA-256 values are
  `68d4cebc200c4a46df38652ac6d526868169f231fbbdbc27e056c1f70810f018`
  and
  `848e99abf35a1548ebe6faf63d832d4af660800b23b5d76d5153f2d90554b082`.
  The companion-table SHA-256 is
  `ed7940aae0066a47cfd9b36773cf100e1ccf2a8a2899343b282b82dbd6f81775`.
- [x] Create and visually verify the professional subperiod figures.

## 13. C++ numerical simulation core

- [ ] Finalize the C++ numerical API and input contracts.
- [ ] Implement dimension, finiteness, and covariance validation.
- [ ] Implement Cholesky-based correlated scenario generation.
- [ ] Implement deterministic random-number handling.
- [ ] Implement portfolio P&L, VaR, and ES extraction.
- [ ] Add native C++ tests for invariants and controlled examples.
- [ ] Compare C++ outputs with the validated Python reference implementation.
- [ ] Benchmark only after correctness is established.

## 14. Python bindings

- [ ] Implement the thin pybind11 interface.
- [ ] Validate array shapes, data types, and ownership across the language boundary.
- [ ] Add Python integration tests for the compiled module.
- [ ] Confirm Python and C++ agreement on identical inputs and seeds.

## 15. Terminal interface and reproducibility workflow

- [ ] Design the terminal interface and command structure.
- [x] Add commands for raw-data retrieval, return processing, portfolio construction,
  and descriptive portfolio diagnostics.
- [ ] Provide commands for downloading, validating, processing, backtesting, and reporting.
- [ ] Support explicit configuration-file selection.
- [ ] Display readable progress, validation summaries, and actionable errors.
- [ ] Add a single reproducibility command for the permitted end-to-end workflow.
- [ ] Document setup and usage in `README.md`.

## 16. Results, robustness, and final delivery

- [ ] Generate reproducible result tables and figures.
- [ ] Interpret calibration, exceedance clustering, and tail-loss severity.
- [ ] Record limitations, unexpected findings, and negative results.
- [ ] Run robustness checks with alternative permitted settings.
- [ ] Consider expanding beyond 11 companies only after the baseline study is complete.
- [ ] Complete the final technical report and methodology record.
- [ ] Run all Python, C++, binding, and reproducibility checks from a clean checkout.
- [ ] Tag a final reproducible release.

## Update rule

After each completed checkpoint:

1. Mark the item complete only after its evidence is available.
2. Update the current phase and next checkpoint at the top of this file.
3. Add any newly discovered work to the appropriate phase.
4. Run the relevant tests before committing the checklist update with the implementation.
