# Portfolio and Resume Copy

This document contains concise, reusable descriptions of release `v0.1.0`.
Update the numerical claims only when a later verified release changes the
baseline experiment or its results.

## Project-card summary

Built a reproducible portfolio tail-risk engine that compares Rolling
Gaussian, EWMA Gaussian, and Filtered Historical Simulation VaR and Expected
Shortfall forecasts using 3,519 out-of-sample sessions, formal backtests, and
checksum-traced research artifacts.

## Short portfolio description

The Portfolio Tail-Risk Engine is a research-grade Python system for comparing
one-day portfolio VaR and Expected Shortfall forecasts. It evaluates three
models on the same 11-asset portfolio using 100,000 scenarios per forecast
date, no-look-ahead rolling backtests, calibration tests, strictly consistent
scoring rules, dependence-robust predictive-accuracy inference, and
multiplicity adjustment. The completed baseline contains 3,519 out-of-sample
forecasts and 650 automated tests.

## Portfolio case study

### The problem

Traditional Gaussian VaR models can react slowly to volatility changes and can
miss non-Gaussian tail behavior. I wanted to test whether more adaptive methods
actually produce better forecasts rather than assuming that greater model
complexity must help.

### What I built

I developed a reproducible Python research pipeline that downloads and
validates market data, constructs a monthly rebalanced 11-asset portfolio, and
generates aligned one-day-ahead VaR and Expected Shortfall forecasts from:

- a 504-session Rolling Gaussian benchmark;
- an EWMA Gaussian model with decay factor 0.94; and
- Filtered Historical Simulation using filtered empirical innovations.

The baseline evaluates 95%, 97.5%, and 99% tail forecasts across 3,519 sessions
from 2012 through 2025. Each model uses 100,000 scenarios per forecast date and
the same realized losses, portfolio weights, forecast schedule, and confidence
levels.

### How I evaluated it

The evaluation goes beyond counting VaR exceptions. It includes Kupiec and
Christoffersen backtests, exact binomial intervals, Expected Shortfall
tail-severity diagnostics, joint VaR-ES calibration, VaR quantile loss, joint
FZ0 loss, and pairwise predictive-accuracy comparisons with Newey-West HAC
uncertainty. Holm adjustment is applied within predeclared hypothesis families.

Generated data, forecasts, and evaluations are stored as non-overwriting
snapshots with configuration, source lineage, and SHA-256 checksums. Selected
figures and compact tables are regenerated from verified results rather than
edited manually.

### What I found

The adaptive models significantly outperformed Rolling Gaussian in 6 of the 12
adaptive-model-versus-benchmark score-confidence comparisons after Holm
adjustment. EWMA and FHS were not statistically distinguishable from one
another in any full-sample comparison.

The descriptive results show why both adaptive models remain useful. EWMA most
often led annual VaR quantile-loss rankings at 95% and 97.5%, while FHS became
more competitive at 99% and under the joint VaR-ES score. Calibration also
changed materially through time. All three models underestimated 99% tail risk
during the separately defined February-April 2020 contraction, with Rolling
Gaussian showing the largest miss.

A lag-zero sensitivity analysis creates five additional rejections, so I retain
the automatic-bandwidth Newey-West results as the more conservative primary
evidence rather than presenting the stronger sensitivity result as definitive.

### Engineering decisions

- Separated data, portfolio, forecasting, evaluation, and figure layers.
- Used one shared forecast schedule and strict no-look-ahead validation.
- Added analytic Gaussian checks and controlled numerical invariants.
- Built 650 automated tests covering mathematics, timing, edge cases,
  integration, snapshot verification, and publication artifacts.
- Preserved large generated artifacts locally while tracking publication-ready
  outputs and their checksums.
- Structured models, statistical reports, documentation, and figures so later
  research extensions can be added without rewriting the baseline.

### Limitations and next research directions

The baseline is a research system rather than a production or regulatory risk
platform. It uses daily data, a linear long-only portfolio, one-day horizons,
and three initial model families. Planned extensions include additional
conditional-volatility and tail models, alternative portfolios, formal regime
analysis, paper-specific experiments, and an independently validated C++
acceleration layer.

## Resume bullets

Use two or three bullets depending on available space.

- Built a reproducible Python VaR and Expected Shortfall engine comparing
  Rolling Gaussian, EWMA Gaussian, and Filtered Historical Simulation across
  3,519 one-day-ahead forecasts for an 11-asset portfolio using 100,000
  scenarios per model-date.
- Implemented Kupiec and Christoffersen backtests, quantile and FZ0 scoring,
  Newey-West predictive-accuracy inference, Holm multiplicity adjustment, and
  checksum-traced experiment artifacts; validated the system with 650 tests.
- Found that adaptive models significantly outperformed Rolling Gaussian in 6
  of 12 score-confidence comparisons, while EWMA and FHS were not statistically
  distinguishable after multiplicity adjustment.

## One-line resume version

Developed and tested a reproducible Python portfolio tail-risk engine comparing
three VaR-ES methods across 3,519 forecasts, with 100,000-scenario simulation,
formal predictive-accuracy inference, and 650 automated tests.

## Suggested skills and keywords

Python, NumPy, pandas, SciPy, Matplotlib, Monte Carlo simulation, Value at Risk,
Expected Shortfall, filtered historical simulation, time-series backtesting,
Newey-West HAC inference, multiple testing, reproducible research, automated
testing, Git, and quantitative risk management.
