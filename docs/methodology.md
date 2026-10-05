# Methodology

## 1. Purpose

This document records the mathematical definitions, timing conventions, portfolio assumptions, model specifications, and evaluation rules used by the Portfolio Tail-Risk Engine.

The primary research question is:

> Do models that account for changing volatility and non-Gaussian returns produce better-calibrated one-day portfolio VaR and Expected Shortfall forecasts than a rolling Gaussian model?

This is a neutral comparison. No model is assumed to perform better before the out-of-sample evidence is evaluated.

The primary risk horizon is one trading day. All models will ultimately be compared using the same portfolio, realized losses, valid forecast dates, and evaluation rules.

## 2. Price and return convention

The project will use adjusted daily closing-price information according to the policy in `docs/data-sources.md`.

Let $P^{\mathrm{adj}}_{i,t}$ denote the adjusted closing price of asset $i$ for trading session $t$. Its one-day simple return is

```math
R_{i,t}
=
\frac{P^{\mathrm{adj}}_{i,t}}
     {P^{\mathrm{adj}}_{i,t-1}}
-1.
```

A positive return represents an increase in economic value, while a negative return represents a decrease.

The two prices must correspond to consecutive valid trading sessions. If either required price is missing, the one-day return will initially be treated as unavailable. A return spanning multiple sessions must not be labelled as a one-day return.

## 3. Why simple returns are primary

Simple returns will be the primary convention because they combine directly across assets in a linear portfolio.

Let $w_{i,t-1}$ denote the weight of asset $i$ established before the return for session $t$ is realized. The portfolio return is

```math
R_{p,t}
=
\sum_{i=1}^{N}
w_{i,t-1}R_{i,t}.
```

The use of beginning-of-period weights ensures that portfolio positions are fixed before the current session's returns become known. This prevents look-ahead bias.

Logarithmic returns are defined by

```math
r_{i,t}
=
\ln\left(
\frac{P^{\mathrm{adj}}_{i,t}}
     {P^{\mathrm{adj}}_{i,t-1}}
\right)
=
\ln(1+R_{i,t}).
```

Log returns add across consecutive time periods, which can be useful for exploratory analysis. However, weighted asset log returns do not directly equal the return of a rebalanced linear portfolio. Log returns will therefore not be the primary convention for portfolio P&L, VaR, or Expected Shortfall.

### 3.1 Baseline portfolio specification

The baseline experiment uses a fully invested, long-only, equal-weight portfolio containing all 11 assets. Each asset has the target weight

```math
w_i = \frac{1}{11}.
```

The portfolio is rebalanced to equal target weights after the final trading session of each month. The resulting weights apply beginning with the following trading session. Between rebalancing dates, asset quantities remain fixed and portfolio weights drift as asset prices change.

The baseline portfolio has an initial value of USD 1,000,000 and no cash allocation. Transaction costs are initially assumed to be zero. Portfolio turnover will be retained so transaction costs can be added as a robustness extension.

Monthly rebalancing is intended to provide realistic portfolio dynamics while keeping the portfolio rule independent of the risk models being compared.

## 4. Portfolio P&L and loss

Let $V_{t-1}$ denote the portfolio value immediately before the session-$t$ return is realized.

The one-day portfolio profit and loss is

```math
\mathrm{PnL}_t
=
V_{t-1}R_{p,t}.
```

P&L is positive for a gain and negative for a loss.

Portfolio loss is defined as the negative of P&L:

```math
L_t
=
-\mathrm{PnL}_t
=
-V_{t-1}R_{p,t}.
```

Under this convention:

- a gain produces a negative loss;
- a loss produces a positive loss; and
- VaR and Expected Shortfall are normally reported as positive loss amounts.

A portfolio return will initially be considered unavailable if any return required for a held asset is unavailable. Any later exception must be economically justified and documented.

## 5. Forecast timing and information set

A forecast made at the end of trading session $t$ may use only information available by the end of that session.

The forecast concerns the portfolio loss during the next trading session:

```math
L_{t+1}
=
-V_t R_{p,t+1}.
```

The timing sequence is:

1. Observe market data available through the end of session $t$.
2. Estimate the model using only permitted data through session $t$.
3. Produce VaR and Expected Shortfall forecasts for session $t+1$.
4. Observe the realized portfolio loss $L_{t+1}$.
5. Compare the forecast with the realized loss.

Information from session $t+1$ or later must not influence the forecast produced at time $t$. This restriction applies to return calculation, covariance estimation, volatility estimation, model fitting, and portfolio weights.

In the implementation, a forecast is indexed by the session whose loss will
be observed. Therefore, the forecast stored for session $t$ uses returns only
through session $t-1$ and the portfolio weights known at the beginning of
session $t$.

## 6. Rolling Gaussian estimation convention

The baseline rolling Gaussian model uses the previous 504 trading sessions.
For forecast session $t$, its estimation sample is

```math
\mathbf{R}_{t-504},\ldots,\mathbf{R}_{t-1}.
```

The asset mean vector is the arithmetic sample mean. The covariance estimate
is the ordinary sample covariance matrix with denominator $504-1$:

```math
\widehat{\boldsymbol{\Sigma}}_t
=
\frac{1}{503}
\sum_{s=t-504}^{t-1}
(\mathbf{R}_s-\widehat{\boldsymbol{\mu}}_t)
(\mathbf{R}_s-\widehat{\boldsymbol{\mu}}_t)^\top.
```

No shrinkage or exponential weighting is applied in this reference model.
The matrix must contain finite values, be symmetric, and be positive
semidefinite within a numerical tolerance.

Let $\mathbf{w}_t$ contain the portfolio weights known at the beginning of
forecast session $t$. The portfolio moments are

```math
\widehat{\mu}_{p,t}
=
\mathbf{w}_t^\top\widehat{\boldsymbol{\mu}}_t,

\qquad

\widehat{\sigma}_{p,t}
=
\sqrt{
\mathbf{w}_t^\top
\widehat{\boldsymbol{\Sigma}}_t
\mathbf{w}_t
}.
```

These moments are used to calculate one-session Gaussian VaR and Expected
Shortfall at confidence levels 95%, 97.5%, and 99%.

### 6.1 Gaussian Monte Carlo convention

The baseline rolling simulation uses 100,000 scenarios per forecast date and
the master random seed 20261002. The seed initializes the pseudorandom-number
generator so the same experiment can be reproduced exactly.

One matrix of independent standard-normal shocks is generated from the master
seed and reused across forecast dates. These common random numbers reduce
artificial changes caused only by simulation noise. The simulated asset returns
still change by date because each shock matrix is transformed using that date's
estimated mean vector and covariance matrix.

The 250,000-scenario sample used by the controlled analytic-agreement test is a
test-specific precision setting. It does not replace the 100,000-scenario
baseline used by the rolling experiment.

### 6.2 EWMA Gaussian estimation convention

The EWMA Gaussian model changes the covariance estimator while preserving the
other rolling Gaussian conventions. It uses the same one-session horizon,
forecast dates, 504-session asset mean, beginning-of-session portfolio
weights, Gaussian shocks, confidence levels, realized losses, and strict VaR
exceedance rule. This controlled design is intended to isolate the effect of a
covariance estimate that responds more strongly to recent return innovations.

The baseline decay parameter is

```math
\lambda = 0.94.
```

This value is the conventional daily RiskMetrics choice. It assigns weight
$1-\lambda=0.06$ to the newest outer product and retains weight $\lambda=0.94$
on the preceding covariance estimate. It is a precommitted baseline setting,
not a claim that 0.94 is universally optimal. Alternative decay parameters may
be examined later as robustness checks without replacing the baseline result.

Let $t_0$ denote the first forecast session. The initial EWMA covariance is the
ordinary sample covariance from the same 504-session window used by the
rolling Gaussian model:

```math
\widehat{\boldsymbol{\Sigma}}^{\mathrm{EWMA}}_{t_0}
=
\frac{1}{503}
\sum_{s=t_0-504}^{t_0-1}
(\mathbf{R}_s-\widehat{\boldsymbol{\mu}}_{t_0})
(\mathbf{R}_s-\widehat{\boldsymbol{\mu}}_{t_0})^\top.
```

This initialization provides finite historical variances and correlations on
the first forecast date without a zero-risk warm-up or future information.
After the first forecast, EWMA updates recursively and does not rebuild or
truncate covariance using a rolling 504-session window. The initialization's
remaining weight after $k$ updates is $\lambda^k$.

For every forecast session $t$, the asset mean remains the arithmetic mean of
the 504 returns available before that session:

```math
\widehat{\boldsymbol{\mu}}_t
=
\frac{1}{504}
\sum_{s=t-504}^{t-1}\mathbf{R}_s.
```

The mean is deliberately not set to zero or estimated by a second EWMA
recursion. Retaining the rolling Gaussian mean prevents the model comparison
from changing both the mean and covariance dynamics at once.

After session $t$ is observed, its innovation is the forecast error

```math
\mathbf{u}_t
=
\mathbf{R}_t
-
\widehat{\boldsymbol{\mu}}_t.
```

The mean in this expression was calculated before session $t$ and excludes
$\mathbf{R}_t$. The rolling mean subsequently calculated for session $t+1$
must not be substituted into $\mathbf{u}_t$, because it has already observed
$\mathbf{R}_t$ and would partly allow the return to explain itself.

The innovation from session $t$ updates the covariance for the next trading
session only:

```math
\widehat{\boldsymbol{\Sigma}}^{\mathrm{EWMA}}_{t+1}
=
\lambda
\widehat{\boldsymbol{\Sigma}}^{\mathrm{EWMA}}_t
+
(1-\lambda)
\mathbf{u}_t\mathbf{u}_t^\top.
```

The sequence for each forecast is therefore:

1. Use information through session $t-1$ to calculate
   $\widehat{\boldsymbol{\mu}}_t$ and obtain
   $\widehat{\boldsymbol{\Sigma}}^{\mathrm{EWMA}}_t$.
2. Apply the beginning-of-session weights $\mathbf{w}_t$ and issue the VaR and
   Expected Shortfall forecast for session $t$.
3. Observe $\mathbf{R}_t$ and compare the realized loss with the forecast.
4. Calculate $\mathbf{u}_t$ using the mean that was available before session
   $t$.
5. Update the covariance for the next trading session, $t+1$.

Thus, a session's return may evaluate that session's forecast and update the
next forecast, but it must never influence its own forecast. In particular,
the first EWMA covariance must not be updated with $\mathbf{R}_{t_0-1}$ after
initialization because that return is already included in the initial sample
covariance.

The initial EWMA covariance, rolling mean, portfolio weights, and shared
standard-normal shocks equal their rolling Gaussian counterparts on $t_0$.
The two models must therefore produce identical first-date Monte Carlo
forecasts. They begin to differ only after the first EWMA update. This equality
will be treated as an implementation invariant.

Each covariance matrix must contain finite values, be symmetric, and be
positive semidefinite within a numerical tolerance. The recursion preserves
positive semidefiniteness mathematically because it is a convex combination of
a positive-semidefinite covariance matrix and the positive-semidefinite outer
product $\mathbf{u}_t\mathbf{u}_t^\top$.

## 7. Planned model comparison

The models will be developed and evaluated in the following order:

1. Rolling Gaussian model using a sample covariance matrix.
2. EWMA Gaussian model using a time-varying covariance estimate.
3. Filtered historical simulation using standardized historical residuals.

The rolling Gaussian model provides the benchmark.

Comparing the EWMA Gaussian model with the rolling Gaussian model will help examine the effect of changing volatility while retaining a Gaussian assumption.

Comparing filtered historical simulation with the Gaussian models will help examine the additional effect of non-Gaussian residual behaviour.

Greater complexity will not be treated as evidence of better performance. Calibration must be assessed from the out-of-sample results.

## 8. Evaluation principles

All models will use matching forecast dates and the same realized portfolio losses.

VaR evaluation will consider both:

- whether the number of exceedances is consistent with the selected confidence level; and
- whether exceedances appear independently through time rather than clustering.

A VaR exceedance is recorded when the realized loss is strictly greater than
the corresponding VaR forecast. Equality with the VaR threshold is not
classified as an exceedance.

Fewer exceedances do not automatically indicate a better model. A model can produce too few exceedances because it is excessively conservative.

Expected Shortfall evaluation will examine the severity of losses beyond the VaR threshold rather than relying only on the number of VaR exceedances.

The precise statistical tests and their decision rules will be documented
before the final backtest results are interpreted.

## 9. Decisions still to be finalized

The following choices remain open and must be documented before the relevant experiments begin:

- missing-data exclusion rules;
- statistical-test decision rules.

Material changes to these conventions will be recorded in `docs/decisions.md`.
