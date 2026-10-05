# Project Checkpoints

Last updated: 2026-10-05

This checklist tracks the research, learning, and engineering progress of the Portfolio Tail-Risk Engine. A checkpoint is marked complete only when the relevant implementation or decision is documented, tested where applicable, and committed.

**Current phase:** EWMA Gaussian configuration

**Next checkpoint:** Add the EWMA decay parameter to configuration, validate
that it is a finite real number strictly between zero and one, and add focused
configuration tests.

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
- [ ] Fix statistical-test decision rules.
- [ ] Record all finalized decisions before examining comparative model results.

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
- [ ] Add and validate the EWMA decay parameter in baseline configuration.
- [ ] Implement the EWMA covariance recursion in Python.
- [ ] Test initialization, recursion, symmetry, and numerical stability.
- [ ] Implement EWMA Gaussian VaR and ES forecasts.

## 10. Filtered historical simulation

- [ ] Review volatility filtering, standardized residuals, resampling, and volatility rescaling.
- [ ] Finalize the filtering and residual-sampling conventions.
- [ ] Implement the Python reference model.
- [ ] Test filtering, residual standardization, resampling, and forecast scaling.
- [ ] Implement filtered-historical-simulation VaR and ES forecasts.

## 11. Rolling forecast and backtest engine

- [x] Implement a shared one-step-ahead forecast schedule with no look-ahead.
- [x] Test forecast-window boundaries, date ordering, and realized-loss alignment.
- [ ] Use identical forecast dates and realized losses for all models.
- [x] Store forecasts, realized losses, exceedances, and model metadata.
- [ ] Run the complete baseline backtest for all three models.

## 12. Statistical evaluation

- [ ] Review VaR exceedances and nominal coverage mathematically.
- [ ] Implement exceedance-rate summaries.
- [ ] Implement the Kupiec unconditional-coverage test.
- [ ] Implement the Christoffersen independence test.
- [ ] Implement Expected Shortfall diagnostics and tail-loss-severity summaries.
- [ ] Test statistical functions against controlled examples or independent calculations.
- [ ] Compare all models without assuming that greater complexity performs better.

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
