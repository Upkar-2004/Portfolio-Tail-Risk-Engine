# Project Checkpoints

Last updated: 2026-08-25

This checklist tracks the research, learning, and engineering progress of the Portfolio Tail-Risk Engine. A checkpoint is marked complete only when the relevant implementation or decision is documented, tested where applicable, and committed.

**Current phase:** Data acquisition and validation

**Next checkpoint:** Persist a versioned raw-data snapshot with retrieval metadata and a checksum.

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
- [ ] Persist each raw download as a versioned snapshot without silent overwriting.
- [ ] Record retrieval metadata, including timestamp, provider, package version, configuration, fields, and validation results.
- [ ] Calculate and record a stable checksum for the raw snapshot.
- [ ] Implement and test loading a saved raw snapshot.
- [ ] Produce a retained validation summary for each retrieval run.

## 4. Asset-return pipeline

- [ ] Review the mathematics of one-day simple returns.
- [ ] Implement adjusted-close simple returns without implicit forward filling.
- [ ] Ensure missing prices produce missing returns.
- [ ] Test positive, negative, zero, and missing-return examples.
- [ ] Verify that returns use consecutive chronological sessions.
- [ ] Generate and validate the baseline asset-return matrix.
- [ ] Flag unusually large returns for review without automatically deleting them.

## 5. Portfolio construction and realized losses

- [ ] Finalize and document the baseline portfolio-weight rule.
- [ ] Finalize the rebalancing frequency, cash treatment, and initial portfolio value.
- [ ] Add the portfolio decisions to the baseline configuration.
- [ ] Implement beginning-of-period portfolio weights.
- [ ] Calculate portfolio returns, P&L, and losses using the documented sign convention.
- [ ] Test return aggregation, P&L, and loss calculations numerically.
- [ ] Verify that no future information enters the portfolio weights.

## 6. Backtest protocol decisions

- [ ] Fix the VaR and Expected Shortfall confidence levels.
- [ ] Fix estimation-window lengths and the first forecast date.
- [ ] Fix covariance-estimation conventions.
- [ ] Fix simulation counts and random seeds.
- [ ] Fix missing-data and forecast-date eligibility rules.
- [ ] Fix statistical-test decision rules.
- [ ] Record all finalized decisions before examining comparative model results.

## 7. Rolling Gaussian reference model

- [ ] Review the mathematics of mean vectors, covariance matrices, portfolio variance, Gaussian VaR, and Gaussian ES.
- [ ] Implement rolling sample mean and covariance estimation in Python.
- [ ] Implement analytic one-day Gaussian VaR and ES.
- [ ] Implement a Python Monte Carlo reference simulation.
- [ ] Test covariance, portfolio variance, VaR, and ES on controlled examples.
- [ ] Verify agreement between analytic and simulated Gaussian results within a justified tolerance.

## 8. EWMA Gaussian model

- [ ] Review the mathematics of exponentially weighted covariance estimation.
- [ ] Select and document the EWMA decay parameter.
- [ ] Implement the EWMA covariance recursion in Python.
- [ ] Test initialization, recursion, symmetry, and numerical stability.
- [ ] Implement EWMA Gaussian VaR and ES forecasts.

## 9. Filtered historical simulation

- [ ] Review volatility filtering, standardized residuals, resampling, and volatility rescaling.
- [ ] Finalize the filtering and residual-sampling conventions.
- [ ] Implement the Python reference model.
- [ ] Test filtering, residual standardization, resampling, and forecast scaling.
- [ ] Implement filtered-historical-simulation VaR and ES forecasts.

## 10. Rolling forecast and backtest engine

- [ ] Implement one-step-ahead rolling forecasts with no look-ahead.
- [ ] Use identical forecast dates and realized losses for all models.
- [ ] Store forecasts, realized losses, exceedances, and model metadata.
- [ ] Test forecast-window boundaries and time alignment.
- [ ] Run the complete baseline backtest for all three models.

## 11. Statistical evaluation

- [ ] Review VaR exceedances and nominal coverage mathematically.
- [ ] Implement exceedance-rate summaries.
- [ ] Implement the Kupiec unconditional-coverage test.
- [ ] Implement the Christoffersen independence test.
- [ ] Implement Expected Shortfall diagnostics and tail-loss-severity summaries.
- [ ] Test statistical functions against controlled examples or independent calculations.
- [ ] Compare all models without assuming that greater complexity performs better.

## 12. C++ numerical simulation core

- [ ] Finalize the C++ numerical API and input contracts.
- [ ] Implement dimension, finiteness, and covariance validation.
- [ ] Implement Cholesky-based correlated scenario generation.
- [ ] Implement deterministic random-number handling.
- [ ] Implement portfolio P&L, VaR, and ES extraction.
- [ ] Add native C++ tests for invariants and controlled examples.
- [ ] Compare C++ outputs with the validated Python reference implementation.
- [ ] Benchmark only after correctness is established.

## 13. Python bindings

- [ ] Implement the thin pybind11 interface.
- [ ] Validate array shapes, data types, and ownership across the language boundary.
- [ ] Add Python integration tests for the compiled module.
- [ ] Confirm Python and C++ agreement on identical inputs and seeds.

## 14. Terminal interface and reproducibility workflow

- [ ] Design the terminal interface and command structure.
- [ ] Provide commands for downloading, validating, processing, backtesting, and reporting.
- [ ] Support explicit configuration-file selection.
- [ ] Display readable progress, validation summaries, and actionable errors.
- [ ] Add a single reproducibility command for the permitted end-to-end workflow.
- [ ] Document setup and usage in `README.md`.

## 15. Results, robustness, and final delivery

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
