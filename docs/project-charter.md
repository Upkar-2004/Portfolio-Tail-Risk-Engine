# Project Charter

## 1. Project overview

The Portfolio Tail-Risk Engine is a research and software-engineering project for studying one-day portfolio loss forecasts. It will compare a simple rolling Gaussian model with models that respond to changing volatility and non-Gaussian return behaviour.

The project has two equally important outcomes:

1. Produce a careful empirical comparison of the models.
2. Build the knowledge required to explain, test, and defend every important part of the implementation.

The result is not intended to be a production trading or regulatory-capital system. It is a focused learning project built with production-minded habits: explicit assumptions, reproducible experiments, independent correctness checks, and clear documentation.

## 2. Research question

> Do models that account for changing volatility and non-Gaussian returns produce better-calibrated one-day portfolio VaR and Expected Shortfall forecasts than a rolling Gaussian model?

This is a question to be tested, not a conclusion chosen in advance. A more complicated model will only be considered better if the backtest evidence supports that claim.

## 3. Project objectives

### Research objectives

- Build one-day Value at Risk (VaR) and Expected Shortfall (ES) forecasts for a clearly defined portfolio.
- Compare three model specifications using the same data, portfolio, forecast dates, and evaluation rules.
- Measure whether each model produces an appropriate number of VaR exceedances.
- Test whether exceedances are independent or clustered through time.
- Compare the severity of losses beyond VaR, not only the number of exceedances.
- Report uncertainty, limitations, and results that do not support the preferred narrative.

### Engineering objectives

- Keep data handling, modelling, simulation, and evaluation as separate components.
- Establish a correct and tested Python reference implementation before porting numerical work to C++.
- Use analytic formulas and mathematical invariants as correctness oracles where possible.
- Expose the C++ simulation core to Python through a small pybind11 interface.
- Make experiments reproducible from recorded data rules, configuration, and random seeds.
- Optimize only after correctness has been demonstrated and measured.

### Learning objectives

- Understand VaR, ES, covariance, volatility dynamics, tail behaviour, and rolling backtests.
- Improve practical Python skills in packaging, testing, type hints, and numerical programming.
- Learn modern C++20 through ownership, value semantics, const correctness, the STL, Eigen, CMake, and native tests.
- Learn how to diagnose numerical problems rather than hiding them.
- Develop the ability to explain assumptions, formulas, implementation choices, and empirical results in plain language.

## 4. Models in scope

The models will be developed in this order so that each stage answers a distinct question.

### 4.1 Rolling Gaussian model

This is the benchmark. It uses a rolling estimation window, a sample covariance matrix, and a multivariate normal return assumption. For a linear portfolio, its one-day VaR can also be calculated analytically, providing an important check on the simulation.

### 4.2 EWMA Gaussian model

This model retains the Gaussian return assumption but gives more weight to recent observations when estimating covariance. Comparing it with the rolling Gaussian model helps isolate the effect of changing volatility.

### 4.3 Filtered historical simulation

This model filters returns using time-varying volatility, resamples standardized historical residuals, and scales them using the current volatility estimate. Comparing it with the Gaussian models helps examine the additional effect of non-Gaussian residual behaviour.

No model is expected to perform best automatically. An overly conservative forecast is not well calibrated simply because it produces few exceedances.

## 5. Scope

### Included

- A portfolio of liquid assets with a documented selection rule.
- Daily market data and daily returns.
- A primary forecast horizon of one trading day.
- VaR and ES at confidence levels defined before the backtest is run.
- Rolling out-of-sample forecasts that use only information available at forecast time.
- Kupiec unconditional-coverage testing for VaR exceedances.
- Christoffersen independence testing for exceedance clustering.
- Diagnostics for losses beyond VaR and the quality of ES forecasts.
- Python research code, a C++ simulation core, pybind11 bindings, automated tests, and reproducible reports.
- Honest benchmarks against vectorized NumPy after the implementations are validated.

### Not included

- Machine-learning models.
- Intraday risk or horizons longer than one day as a primary research target.
- Options, exotic derivatives, or nonlinear pricing models.
- Credit risk, counterparty risk, XVA, or liquidity-risk modelling.
- Multiple asset-class expansion before the core study is complete.
- A web application or trading dashboard.
- GPU computing, custom allocators, advanced metaprogramming, or premature concurrency infrastructure.
- Claims about regulatory compliance or production readiness.

Possible extensions must be treated as separate work and must not delay the core comparison.

## 6. System boundaries

### Python research layer

Python owns:

- data acquisition, validation, cleaning, and alignment;
- return calculation;
- covariance estimation;
- exploratory analysis;
- model orchestration and rolling backtests;
- statistical evaluation;
- plots, tables, and reporting; and
- the initial reference implementation used to validate the C++ port.

### C++ simulation core

C++ owns:

- numerical input validation;
- Cholesky decomposition;
- deterministic random-number generation;
- correlated scenario generation;
- portfolio profit-and-loss calculation;
- VaR and ES extraction; and
- batch simulation and, later, measured parallel execution.

### Python bindings

The pybind11 layer should remain thin. It translates inputs and outputs across the language boundary but should not contain research logic or duplicate model behaviour.

## 7. Core conventions

The full definitions will be recorded in `methodology.md`, but the following rules apply throughout the project:

- Portfolio profit and loss (P&L) is positive for a gain and negative for a loss.
- Portfolio loss is defined as `-P&L`, so VaR and ES are normally reported as positive loss amounts.
- A forecast made at the end of day `t` may only use information available by the end of day `t`.
- That forecast is evaluated against the realized portfolio loss on day `t + 1`.
- Estimation windows, confidence levels, portfolio weights, rebalancing rules, missing-data rules, and transaction assumptions must be fixed or recorded before results are interpreted.
- Random experiments must use recorded seeds and reproducible sampling rules.
- Model comparisons must use matching forecast dates and the same realized portfolio returns.

These rules are intended to prevent look-ahead bias and make comparisons fair.

## 8. How success will be judged

Success is divided into four categories so that they are not confused with one another.

### Mathematical and software correctness

- Analytic Gaussian VaR agrees with Monte Carlo estimates within a justified simulation tolerance.
- Cholesky reconstruction satisfies `L L^T ≈ Σ` within a numerical tolerance.
- Simulated moments agree with their target moments within sampling error.
- Python and C++ implementations agree on controlled test cases.
- Invalid dimensions, non-finite values, and unsuitable covariance matrices are rejected clearly.
- Tests guard against look-ahead errors and incorrect forecast alignment.

### Statistical evidence

- VaR exceedance rates are compared with their nominal probabilities.
- Kupiec and Christoffersen results are reported with their assumptions and limitations.
- Exceedance clustering and tail-loss severity are shown, not hidden behind a single test result.
- A model is described as better calibrated only when several relevant diagnostics support the conclusion.
- Results that contradict the working expectation are retained and discussed.

### Reproducibility

- A documented workflow can rebuild the processed data and analysis from the permitted raw inputs.
- Experiment parameters and random seeds are stored in configuration.
- Tests can be run from a clean checkout using documented commands.
- Generated figures and tables can be reproduced without manual editing.

### Performance

- Performance work begins only after the single-threaded implementations are correct.
- Benchmarks compare C++ with vectorized NumPy rather than deliberately slow Python loops.
- Timing methodology, hardware, number of paths, repetitions, and summary statistics are reported.
- Parallel results must remain deterministic across supported thread counts before speedups are claimed.

## 9. Main deliverables

1. A documented and tested Python research package.
2. A validated C++20 simulation library.
3. A small pybind11 interface connecting the two layers.
4. A reproducible rolling-backtest pipeline.
5. Statistical comparisons of the three model specifications.
6. A data-exploration notebook and final figures that support the written analysis.
7. A report explaining the methodology, evidence, limitations, and conclusions.
8. A README that shows the research result before presenting setup instructions.

## 10. Development sequence

Work will proceed in the following order:

1. Inspect and understand the data.
2. Define returns, portfolio rules, and evaluation conventions.
3. Implement and validate the rolling Gaussian model in Python.
4. Build the rolling backtest and statistical tests.
5. Add the EWMA Gaussian model.
6. Add filtered historical simulation.
7. Compare the models using the same out-of-sample period.
8. Port validated simulation components to C++.
9. Add deterministic parallelism only after single-threaded correctness.
10. Benchmark, document, and reproduce the final results.

Each stage should leave behind tests and documentation before the next stage begins.

## 11. Main risks and limitations

- **Data quality:** missing observations, corporate actions, and inconsistent trading calendars can corrupt returns.
- **Look-ahead bias:** an incorrect date alignment can make a weak model appear successful.
- **Estimation error:** sample covariance estimates can be noisy or poorly conditioned, especially as the number of assets grows.
- **Limited tail observations:** extreme losses are rare, so VaR and particularly ES evaluation will have substantial uncertainty.
- **Model risk:** all three models simplify real markets and may fail during structural breaks.
- **Simulation error:** estimated quantiles vary with the random sample and number of scenarios.
- **Implementation agreement:** matching Python and C++ outputs does not prove correctness if both repeat the same conceptual mistake.
- **Scope growth:** extra assets, models, interfaces, or performance features could distract from the research question.

These limitations should be measured or discussed rather than quietly removed from the final report.

## 12. Definition of done

The core project is complete when:

- all three models produce one-day VaR and ES forecasts over the same valid backtest period;
- the forecasts are evaluated using documented statistical and graphical diagnostics;
- analytic and numerical validation tests pass;
- the C++ core is callable from Python and agrees with independent reference cases;
- the full experiment can be reproduced from documented commands and configuration;
- performance claims are supported by an honest benchmark;
- the final report answers the research question without overstating the evidence; and
- at least one unresolved weakness or inconvenient result is documented clearly.

## 13. Working principles

- Correctness before evaluation.
- Evaluation before optimization.
- Simple code before abstraction.
- Evidence before conclusions.
- Reproducibility before presentation.
- Every important formula and design choice should be explainable by the project owner.

This charter may evolve as the project reveals new constraints. Material changes to the research question, scope, model definitions, or evaluation rules must be recorded in `decisions.md` rather than changed silently.
