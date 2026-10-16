# Methodology

## 1. Purpose

This document records the mathematical definitions, timing conventions, portfolio assumptions, model specifications, and evaluation rules used by the Portfolio Tail-Risk Engine.

The primary research question is:

> Do models that account for changing volatility and non-Gaussian returns produce better-calibrated one-day portfolio VaR and Expected Shortfall forecasts than a rolling Gaussian model?

This is a neutral comparison. No model is assumed to perform better before the out-of-sample evidence is evaluated.

The primary risk horizon is one trading day. All baseline models are compared
using the same portfolio, realized losses, valid forecast dates, and evaluation
rules.

## 2. Price and return convention

The project uses adjusted daily closing-price information according to the
policy in `docs/data-sources.md`.

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

Simple returns are the primary convention because they combine directly across
assets in a linear portfolio.

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

Log returns add across consecutive time periods, which can be useful for
exploratory analysis. However, weighted asset log returns do not directly equal
the return of a rebalanced linear portfolio. Log returns are therefore not the
primary convention for portfolio P&L, VaR, or Expected Shortfall.

### 3.1 Baseline portfolio specification

The baseline experiment uses a fully invested, long-only, equal-weight portfolio containing all 11 assets. Each asset has the target weight

```math
w_i = \frac{1}{11}.
```

The portfolio is rebalanced to equal target weights after the final trading session of each month. The resulting weights apply beginning with the following trading session. Between rebalancing dates, asset quantities remain fixed and portfolio weights drift as asset prices change.

The baseline portfolio has an initial value of USD 1,000,000 and no cash
allocation. Transaction costs are zero in the baseline. Portfolio turnover is
retained so transaction costs can be added as a robustness extension.

Monthly rebalancing is intended to provide realistic portfolio dynamics while keeping the portfolio rule independent of the risk models being compared.

## 4. Portfolio P&L and modeling loss

Let $V_{t-1}$ denote the portfolio value immediately before the session-$t$ return is realized.

The one-day portfolio profit and loss is

```math
\mathrm{PnL}_t
=
V_{t-1}R_{p,t}.
```

P&L is positive for a gain and negative for a loss.

Dollar portfolio loss is the negative of P&L:

```math
\mathcal{L}^{\mathrm{USD}}_t
=
-\mathrm{PnL}_t
=
-V_{t-1}R_{p,t}.
```

The statistical engine works in normalized portfolio-return units. Its
modeled loss variable is

```math
L_t
=
\frac{\mathcal{L}^{\mathrm{USD}}_t}{V_{t-1}}
=
-R_{p,t}.
```

Thus, a stored loss of $0.02$ represents a loss equal to 2% of the portfolio
value immediately before the session. Every realized loss, VaR forecast,
Expected Shortfall forecast, score, and backtest statistic uses this same
unit. If dollar risk is required for reporting, the corresponding forecasts
are obtained without changing the statistical model:

```math
\mathrm{VaR}^{\mathrm{USD}}_{t,\alpha}
=
V_{t-1}q_{t,\alpha},
\qquad
\mathrm{ES}^{\mathrm{USD}}_{t,\alpha}
=
V_{t-1}e_{t,\alpha}.
```

Under this convention:

- a gain produces a negative loss;
- a loss produces a positive loss; and
- VaR and Expected Shortfall are reported as positive loss proportions when
  the forecasted tail lies below zero portfolio return.

A portfolio return is considered unavailable if any return required for a held
asset is unavailable. Any later exception must be economically justified and
documented.

## 5. Forecast timing and information set

A forecast made at the end of trading session $t$ may use only information available by the end of that session.

The forecast concerns the normalized portfolio loss during the next trading
session:

```math
L_{t+1}
=
-R_{p,t+1}.
```

The corresponding dollar loss is

```math
\mathcal{L}^{\mathrm{USD}}_{t+1}
=
V_tL_{t+1}.
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

Under the Gaussian model, the conditional portfolio-return and normalized-loss
distributions are

```math
R_{p,t}\mid\mathcal{F}_{t-1}
\sim
\mathcal{N}
\left(
\widehat{\mu}_{p,t},
\widehat{\sigma}_{p,t}^{2}
\right),
\qquad
L_t\mid\mathcal{F}_{t-1}
\sim
\mathcal{N}
\left(
-\widehat{\mu}_{p,t},
\widehat{\sigma}_{p,t}^{2}
\right).
```

Let $\Phi$ and $\phi$ denote the standard-normal distribution and density
functions, and define

```math
z_{\alpha}=\Phi^{-1}(\alpha).
```

The analytic one-session Gaussian forecasts are then

```math
q_{t,\alpha}^{\mathrm{Gauss}}
=
\mathrm{VaR}_{t,\alpha}^{\mathrm{Gauss}}
=
-\widehat{\mu}_{p,t}
+
\widehat{\sigma}_{p,t}z_{\alpha},
```

and

```math
e_{t,\alpha}^{\mathrm{Gauss}}
=
\mathrm{ES}_{t,\alpha}^{\mathrm{Gauss}}
=
-\widehat{\mu}_{p,t}
+
\widehat{\sigma}_{p,t}
\frac{\phi(z_{\alpha})}{1-\alpha}.
```

For a continuous distribution, the latter equals
$\mathbb{E}[L_t\mid L_t\ge q_{t,\alpha}^{\mathrm{Gauss}}]$. These analytic
values provide the benchmark for the controlled Monte Carlo agreement test;
the official rolling backtests use the empirical simulation estimators below.

### 6.1 Gaussian Monte Carlo convention

The baseline rolling simulation uses 100,000 scenarios per forecast date and
the master random seed 20261002. The seed initializes the pseudorandom-number
generator so the same experiment can be reproduced exactly.

One matrix of independent standard-normal shocks is generated from the master
seed and reused across forecast dates. These common random numbers reduce
artificial changes caused only by simulation noise. The simulated asset returns
still change by date because each shock matrix is transformed using that date's
estimated mean vector and covariance matrix.

Let $S$ be the number of scenarios, let
$\mathbf{z}^{(s)}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_N)$, and let the
forecast-date Cholesky factor $\mathbf{C}_t$ satisfy

```math
\widehat{\boldsymbol{\Sigma}}_t
=
\mathbf{C}_t\mathbf{C}_t^{\top}.
```

For scenario $s$, the simulated asset return and normalized portfolio loss are

```math
\mathbf{R}^{(s)}_t
=
\widehat{\boldsymbol{\mu}}_t
+
\mathbf{C}_t\mathbf{z}^{(s)},
\qquad
L^{(s)}_t
=
-\mathbf{w}_t^{\top}\mathbf{R}^{(s)}_t.
```

The same standard-normal matrix
$\{\mathbf{z}^{(s)}\}_{s=1}^{S}$ is reused at every forecast date, while
$\widehat{\boldsymbol{\mu}}_t$, $\mathbf{C}_t$, and $\mathbf{w}_t$ change
through time.

Write the ordered simulated losses as
$L_{(1),t}\le\cdots\le L_{(S),t}$. The implementation uses NumPy's linear
sample-quantile convention. Define

```math
h=(S-1)\alpha,
\qquad
j=\lfloor h\rfloor,
\qquad
\delta=h-j.
```

The empirical VaR estimator is

```math
\widehat q_{t,\alpha}^{\mathrm{MC}}
=
(1-\delta)L_{(j+1),t}
+
\delta L_{(j+2),t}.
```

Because $0<\alpha<1$, the required order statistics exist. Define the
simulated tail count

```math
M_{t,\alpha}
=
\sum_{s=1}^{S}
\mathbf{1}
\left\{
L_t^{(s)}\ge
\widehat q_{t,\alpha}^{\mathrm{MC}}
\right\}.
```

The empirical Expected Shortfall estimator is

```math
\widehat e_{t,\alpha}^{\mathrm{MC}}
=
\frac{1}{M_{t,\alpha}}
\sum_{s=1}^{S}
L_t^{(s)}
\mathbf{1}
\left\{
L_t^{(s)}\ge
\widehat q_{t,\alpha}^{\mathrm{MC}}
\right\}.
```

The weak inequality is intentional: scenarios exactly equal to the empirical
VaR threshold are included in the simulated ES tail. The rolling Gaussian,
EWMA Gaussian, and filtered historical simulation backtests all use this same
linear-quantile and tail-mean convention.

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

### 6.3 Filtered historical simulation convention

The filtered historical simulation model preserves the EWMA model's
504-session rolling asset mean, EWMA covariance sequence with
$\lambda=0.94$, one-session horizon, beginning-of-session portfolio weights,
forecast dates, confidence levels, and realized losses. It changes the shock
distribution: complete historical asset-level residual vectors are resampled
instead of drawing independent standard-normal shocks.

Let $t_0$ denote the first eligible forecast session. The initial residual
pool uses the same 504-session estimation window as the Gaussian models. Its
sample mean and covariance are

```math
\widehat{\boldsymbol{\mu}}_{t_0}
=
\frac{1}{504}
\sum_{s=t_0-504}^{t_0-1}\mathbf{R}_s,

\qquad

\widehat{\boldsymbol{\Sigma}}_{t_0}
=
\mathbf{C}_{t_0}\mathbf{C}_{t_0}^{\top}.
```

Here $\mathbf{C}_{t_0}$ is the lower-triangular Cholesky factor of the initial
covariance matrix. The symbol $\mathbf{C}$ is kept distinct from the scalar
portfolio-loss variable $L$.

Each return in that initial window is converted into a calibration residual:

```math
\mathbf{z}_s
=
\mathbf{C}_{t_0}^{-1}
\left(
\mathbf{R}_s-\widehat{\boldsymbol{\mu}}_{t_0}
\right),
\qquad
s=t_0-504,\ldots,t_0-1.
```

The implementation solves the corresponding triangular linear system and
does not form a matrix inverse. These initial observations are described as
calibration residuals rather than one-step-ahead forecast errors: their common
mean and covariance use the complete initial estimation window. This creates
no out-of-sample look-ahead because every input predates $t_0$, but it is an
explicit in-sample initialization convention.

The convention preserves the shared first forecast date without requiring a
separate pre-2010 burn-in sample. It also provides the reconstruction identity

```math
\widehat{\boldsymbol{\mu}}_{t_0}
+
\mathbf{C}_{t_0}\mathbf{z}_s
=
\mathbf{R}_s.
```

Consequently, the first FHS return distribution is ordinary historical
simulation over the initial 504 returns, subject only to finite resampling
variation.

After initialization, the residual pool has a fixed length of 504 sessions.
For forecast session $t$, it contains only residuals dated before $t$. After
the session is observed, calculate the genuine one-step-ahead innovation and
standardized residual using the mean and covariance that existed before the
return:

```math
\mathbf{u}_t
=
\mathbf{R}_t-\widehat{\boldsymbol{\mu}}_t,

\qquad

\mathbf{z}_t
=
\mathbf{C}_t^{-1}\mathbf{u}_t.
```

The oldest residual is removed and $\mathbf{z}_t$ is appended only after the
forecast for $t$ has been issued and evaluated. The updated pool is first used
for session $t+1$. After 504 updates, every initial calibration residual has
left the pool and all rows are sequential forecast errors.

For each forecast date, the baseline draws 100,000 complete residual rows with
replacement. All 11 asset components of a selected row remain together, which
preserves their observed joint dependence. Asset columns are never resampled
independently. The master seed is 20261002. Reinitializing the generator with
that seed on each forecast date reuses the same sequence of pool positions,
providing common random numbers while the contents of those positions evolve.

For a sampled residual $\mathbf{z}^{*}$, current conditions are restored with
the forecast-date Cholesky factor:

```math
\mathbf{R}^{*}_t
=
\widehat{\boldsymbol{\mu}}_t
+
\mathbf{C}_t\mathbf{z}^{*}.
```

The corresponding simulated portfolio loss is

```math
L^{*}_t
=
-\mathbf{w}_t^{\top}\mathbf{R}^{*}_t.
```

FHS VaR is the empirical loss quantile at the requested confidence level.
Expected Shortfall is the arithmetic mean of simulated losses greater than or
equal to that VaR threshold. Covariance matrices used for whitening and
rescaling must be finite, symmetric, and numerically positive definite so a
stable Cholesky factor exists.

## 7. Baseline model comparison

The baseline models were developed and evaluated in the following order:

1. Rolling Gaussian model using a sample covariance matrix.
2. EWMA Gaussian model using a time-varying covariance estimate.
3. Filtered historical simulation using standardized historical residuals.

The rolling Gaussian model provides the benchmark.

Comparing the EWMA Gaussian model with the rolling Gaussian model examines the
effect of changing volatility while retaining a Gaussian assumption.

Comparing filtered historical simulation with the Gaussian models examines the
additional effect of non-Gaussian residual behaviour.

Greater complexity is not treated as evidence of better performance.
Calibration is assessed from the out-of-sample results.

## 8. Evaluation principles

All baseline models use matching forecast dates and the same realized portfolio
losses.

VaR evaluation considers both:

- whether the number of exceedances is consistent with the selected confidence level; and
- whether exceedances appear independently through time rather than clustering.

A VaR exceedance is recorded when the realized loss is strictly greater than
the corresponding VaR forecast. Equality with the VaR threshold is not
classified as an exceedance.

For confidence level $\alpha$, define the exceedance indicator and nominal
exceedance probability as

```math
I_t=\mathbf{1}\{L_t>\mathrm{VaR}_{t,\alpha}\},
\qquad p=1-\alpha.
```

Each model and confidence level will first be described by its number of
forecasts, observed exceedance count, expected exceedance count, observed
exceedance rate, and nominal exceedance rate.

### 8.1 Exact uncertainty interval for the exceedance rate

For each model-confidence pair, let

```math
X
=
\sum_{t=1}^{T}I_t
=
x,
\qquad
\widehat p
=
\frac{x}{T},
```

where $T$ is the number of out-of-sample forecasts, $x$ is the observed
number of VaR exceedances, and $\widehat p$ is the observed exceedance
rate.

Under the binomial reference model,

```math
X\sim\mathrm{Binomial}(T,\pi),
```

where $\pi$ is the unknown true exceedance probability. Each observed
exceedance rate will be accompanied by a two-sided 95% exact
Clopper-Pearson confidence interval for $\pi$.

Let $c=0.95$ be the interval confidence level and let

```math
\gamma=1-c.
```

Using $B_q^{-1}(a,b)$ to denote the $q$-quantile of a Beta distribution
with shape parameters $a$ and $b$, the lower bound for $x>0$ is

```math
p_{\mathrm{lower}}
=
B_{\gamma/2}^{-1}
\left(x,T-x+1\right).
```

When $x=0$, the lower bound is defined as

```math
p_{\mathrm{lower}}=0.
```

For $x<T$, the upper bound is

```math
p_{\mathrm{upper}}
=
B_{1-\gamma/2}^{-1}
\left(x+1,T-x\right).
```

When $x=T$, the upper bound is defined as

```math
p_{\mathrm{upper}}=1.
```

The VaR confidence level and the interval confidence level have different
meanings. For example, a 99% VaR implies a nominal exceedance probability
of 1%, while the 95% interval describes uncertainty around the estimated
exceedance probability.

The Clopper-Pearson interval is obtained by inverting equal-tailed exact
binomial tests. Because binomial outcomes are discrete, its true coverage
is generally at least the requested confidence level and may be
conservative.

The interval assumes a binomial sequence with a constant exceedance
probability and independent indicators. Since exceedance clustering can
violate the independence assumption, these bounds are treated as a
nominal uncertainty reference and are reported alongside the
Christoffersen independence test. They do not replace the predeclared
coverage tests or their multiplicity-adjusted conclusions.

### 8.2 Unconditional coverage

The Kupiec likelihood-ratio test will evaluate the null hypothesis

```math
H_0:\Pr(I_t=1)=p
```

against the two-sided alternative

```math
H_1:\Pr(I_t=1)\ne p.
```

If $T$ is the number of forecasts, $x=\sum_t I_t$, and
$\widehat p=x/T$, the statistic is

```math
LR_{\mathrm{uc}}
=
-2\log\left[
\frac{(1-p)^{T-x}p^x}
{(1-\widehat p)^{T-x}\widehat p^x}
\right].
```

Terms of the form $0\log(0)$ are evaluated as zero so samples containing no
exceedances or only exceedances remain defined. Under the null hypothesis,
$LR_{\mathrm{uc}}$ is compared with an asymptotic chi-square distribution
with one degree of freedom.

All tests use a two-sided 5% significance level. The null is rejected only
when the p-value is strictly below 0.05. A non-rejection will be described as
insufficient evidence against the null, not as proof that the model is
correct. The direction of any coverage error will be reported separately:
too few exceedances indicate conservatism and too many indicate
underestimation of risk.

The Kupiec test evaluates only the full-sample exceedance frequency. It does
not establish that exceedances are independent through time.

### 8.3 Independence and conditional coverage

The Christoffersen independence test examines whether a VaR exceedance is
more likely after another exceedance. For consecutive indicators, define the
four transition counts

```math
n_{00},\quad n_{01},\quad n_{10},\quad n_{11},
```

where the first subscript is the preceding session's state and the second is
the current session's state. For example, $n_{11}$ counts an exceedance
followed immediately by another exceedance. The conditional breach-rate
estimates are

```math
\widehat\pi_0
=
\frac{n_{01}}{n_{00}+n_{01}},
\qquad
\widehat\pi_1
=
\frac{n_{11}}{n_{10}+n_{11}},
```

and the common transition breach rate under independence is

```math
\widehat\pi
=
\frac{n_{01}+n_{11}}
{n_{00}+n_{01}+n_{10}+n_{11}}.
```

The independence null states that the next-session breach probability does
not depend on the immediately preceding breach state:

```math
H_0^{\mathrm{ind}}:\pi_0=\pi_1.
```

The restricted and first-order Markov likelihoods are

```math
\mathcal{L}_0
=
(1-\widehat\pi)^{n_{00}+n_{10}}
\widehat\pi^{n_{01}+n_{11}},
```

and

```math
\mathcal{L}_1
=
(1-\widehat\pi_0)^{n_{00}}
\widehat\pi_0^{n_{01}}
\times
(1-\widehat\pi_1)^{n_{10}}
\widehat\pi_1^{n_{11}}.
```

The likelihood-ratio statistic is

```math
LR_{\mathrm{ind}}
=
-2\log\left(\frac{\mathcal{L}_0}{\mathcal{L}_1}\right).
```

Under the independence null, this statistic is compared with an asymptotic
chi-square distribution with one degree of freedom.

Boundary cases with an unobserved preceding state are handled by assigning
zero log-likelihood contribution to that state. This keeps all-zero and
all-one sequences numerically defined. The test concerns first-order
dependence in the exceedance indicators; it does not assert that portfolio
returns or losses themselves are independent.

Christoffersen conditional coverage combines correct unconditional frequency
and first-order independence:

```math
LR_{\mathrm{cc}}
=
LR_{\mathrm{uc}}+LR_{\mathrm{ind}}.
```

Under the joint conditional-coverage null, this statistic is compared with
an asymptotic chi-square distribution with two degrees of freedom.

Rejecting conditional coverage means that the model fails at least one of the
two requirements. The component statistics will always be reported so a
frequency failure can be distinguished from exceedance clustering.

### 8.4 Statistical reporting and multiplicity

Raw statistics and p-values will always be retained. Christoffersen
conditional coverage will be the primary overall VaR adequacy test for each
model-confidence pair, with Kupiec coverage and Christoffersen independence
reported as interpretable component diagnostics. Final simultaneous claims
across the nine model-confidence pairs will additionally report Holm-adjusted
p-values for the primary-test family.

For either nine-test family, let the ordered raw p-values be

```math
p_{(1)}\le p_{(2)}\le\cdots\le p_{(m)},
\qquad m=9.
```

Holm's step-down procedure compares the $j$th ordered p-value with

```math
\frac{s}{m-j+1},
```

where $s$ is the family significance level. Testing proceeds from the
smallest p-value and stops rejecting when the first comparison fails. The
equivalent adjusted p-value for ordered position $j$ is

```math
\widetilde{p}_{(j)}
=
\min\left\{
1,
\max_{1\le k\le j}
\left[(m-k+1)p_{(k)}\right]
\right\}.
```

The running maximum ensures that adjusted p-values cannot decrease as the raw
p-values increase. Adjusted values are then returned to their original
model-confidence rows. This controls the probability of making at least one
false rejection within the declared family. The nine conditional-coverage
tests and the nine joint VaR-ES calibration tests are treated as separate
families because they answer different statistical questions.

Later pairwise model-comparison tests based on VaR quantile loss and joint
VaR-Expected-Shortfall loss will form separate, predeclared multiple-testing
families. Descriptive annual, stress-period, severity, and Basel-style results
will not be presented as independent formal tests unless an inferential rule
has been specified in advance.

Fewer exceedances do not automatically indicate a better model. A model can produce too few exceedances because it is excessively conservative.

Expected Shortfall evaluation will examine the severity of losses beyond the VaR threshold rather than relying only on the number of VaR exceedances.

Exact 95% binomial intervals will accompany exceedance-rate figures to show
sampling uncertainty. Formal conclusions will continue to follow the
predeclared hypothesis tests rather than visual inspection alone.

### 8.5 Expected Shortfall and tail-loss severity

Expected Shortfall evaluates the magnitude of losses in the VaR tail rather
than the frequency of VaR exceedances. For model confidence level $\alpha$,
write the forecast VaR and Expected Shortfall as

```math
q_{t,\alpha}=\mathrm{VaR}_{t,\alpha},
\qquad
e_{t,\alpha}=\mathrm{ES}_{t,\alpha}.
```

For an upper-tail loss distribution, an internally consistent forecast pair
must satisfy

```math
e_{t,\alpha}\ge q_{t,\alpha}.
```

The implementation will validate this ordering using an absolute tolerance of
$10^{-12}$ in portfolio-loss units. This tolerance accommodates insignificant
floating-point differences without accepting a material ES-below-VaR
violation. VaR exceedances continue to use the strict rule

```math
I_{t,\alpha}
=
\mathbf{1}\left\{L_t>q_{t,\alpha}\right\}.
```

Define the set and number of VaR-exceedance sessions as

```math
\mathcal{B}_{\alpha}
=
\left\{t:I_{t,\alpha}=1\right\},
\qquad
x_{\alpha}
=
\sum_{t=1}^{T}I_{t,\alpha}.
```

When $x_{\alpha}>0$, the mean realized loss on exceedance sessions is

```math
\overline{L}_{\mathcal{B},\alpha}
=
\frac{1}{x_{\alpha}}
\sum_{t\in\mathcal{B}_{\alpha}}L_t.
```

The corresponding mean VaR and mean Expected Shortfall forecasts on those
same sessions are

```math
\overline{q}_{\mathcal{B},\alpha}
=
\frac{1}{x_{\alpha}}
\sum_{t\in\mathcal{B}_{\alpha}}q_{t,\alpha},
```

and

```math
\overline{e}_{\mathcal{B},\alpha}
=
\frac{1}{x_{\alpha}}
\sum_{t\in\mathcal{B}_{\alpha}}e_{t,\alpha}.
```

The VaR excess on an exceedance session is

```math
D^{\mathrm{VaR}}_{t,\alpha}
=
L_t-q_{t,\alpha}.
```

It is strictly positive by construction. Its mean is

```math
\overline{D}^{\mathrm{VaR}}_{\alpha}
=
\frac{1}{x_{\alpha}}
\sum_{t\in\mathcal{B}_{\alpha}}
D^{\mathrm{VaR}}_{t,\alpha}.
```

The median VaR excess is

```math
\widetilde{D}^{\mathrm{VaR}}_{\alpha}
=
\mathrm{median}
\left\{
D^{\mathrm{VaR}}_{t,\alpha}:
t\in\mathcal{B}_{\alpha}
\right\},
```

and the maximum VaR excess is

```math
D^{\mathrm{VaR,max}}_{\alpha}
=
\max_{t\in\mathcal{B}_{\alpha}}
D^{\mathrm{VaR}}_{t,\alpha}.
```

The mean describes average breach severity, the median gives a value less
dominated by a few extreme sessions, and the maximum records the worst
observed miss beyond the VaR threshold.

The Expected Shortfall gap on a VaR-exceedance session is

```math
D^{\mathrm{ES}}_{t,\alpha}
=
L_t-e_{t,\alpha},
```

with conditional sample mean

```math
\overline{D}^{\mathrm{ES}}_{\alpha}
=
\frac{1}{x_{\alpha}}
\sum_{t\in\mathcal{B}_{\alpha}}
D^{\mathrm{ES}}_{t,\alpha}.
```

A positive mean Expected Shortfall gap means that realized losses on the VaR
exceedance sessions were, on average, larger than the corresponding ES
forecasts. A negative value means that the ES forecasts were larger on
average. A value near zero indicates close average tail severity, but it does
not by itself establish correct conditional calibration.

If $x_{\alpha}=0$, all statistics conditional on VaR exceedance are undefined
and will be stored as missing values rather than zero. A reported zero would
incorrectly imply that tail severity had been observed and measured.

These summaries are descriptive. Each model's breach set is determined by
its own VaR forecasts, so different models may be summarized over different
dates. Conditional tail averages therefore must not be treated as formal
pairwise model rankings without an appropriate comparative test.

For a diagnostic that uses every forecast session, define the joint VaR-ES
tail contribution

```math
C_{t,\alpha}
=
q_{t,\alpha}
+
\frac{I_{t,\alpha}}{1-\alpha}
\left(L_t-q_{t,\alpha}\right).
```

The corresponding Expected Shortfall identification residual is

```math
Z_{t,\alpha}
=
C_{t,\alpha}-e_{t,\alpha}.
```

When the conditional VaR and Expected Shortfall forecasts are jointly
correct and the loss distribution is continuous at the VaR threshold,

```math
\mathbb{E}\left[
Z_{t,\alpha}
\mid
\mathcal{F}_{t-1}
\right]
=0.
```

The sample mean identification residual is

```math
\overline{Z}_{\alpha}
=
\frac{1}{T}
\sum_{t=1}^{T}Z_{t,\alpha}.
```

A positive value indicates that the joint forecasts understate the realized
tail contribution on average, while a negative value indicates overstatement.
Because this residual depends on both VaR and Expected Shortfall, a nonzero
mean cannot automatically be attributed to the ES component alone when VaR
is miscalibrated.

The formal calibration null and two-sided alternative are

```math
H_0:
\mathbb{E}[Z_{t,\alpha}]=0,
\qquad
H_1:
\mathbb{E}[Z_{t,\alpha}]\ne0.
```

The sample mean need not equal zero exactly even under a correctly calibrated
model. Formal inference must therefore measure how large the observed mean is
relative to its sampling uncertainty. An ordinary independent-observation
standard error is inappropriate when residual volatility changes through time
or residuals remain correlated across nearby sessions. The project uses a
Newey-West heteroskedasticity-and-autocorrelation-consistent, or HAC, estimator
for this uncertainty.

First center the residuals around their sample mean:

```math
\widetilde{Z}_{t,\alpha}
=
Z_{t,\alpha}-\overline{Z}_{\alpha}.
```

For lag $k$, estimate the residual autocovariance using denominator $T$:

```math
\widehat{\gamma}_{k,\alpha}
=
\frac{1}{T}
\sum_{t=k+1}^{T}
\widetilde{Z}_{t,\alpha}
\widetilde{Z}_{t-k,\alpha}.
```

The lag-zero value $\widehat{\gamma}_{0,\alpha}$ measures the residual
variance. Positive values at later lags indicate that residuals of similar
sign or magnitude tend to occur close together. For bandwidth $K$, the
Newey-West estimator applies Bartlett weights

```math
w_k
=
1-\frac{k}{K+1},
\qquad
k=1,\ldots,K,
```

and estimates the long-run variance as

```math
\widehat{\Omega}_{\alpha}
=
\widehat{\gamma}_{0,\alpha}
+
2
\sum_{k=1}^{K}
w_k\widehat{\gamma}_{k,\alpha}.
```

The Bartlett weights decline linearly with the lag, so nearby dependence
receives more weight than more distant dependence. The factor of two includes
the matching positive and negative autocovariance lags. The estimated variance
and standard error of the sample mean are

```math
\widehat{\mathrm{Var}}_{\mathrm{HAC}}
\left(\overline{Z}_{\alpha}\right)
=
\frac{\widehat{\Omega}_{\alpha}}{T},
```

and

```math
\mathrm{SE}_{\mathrm{HAC}}
\left(\overline{Z}_{\alpha}\right)
=
\sqrt{
\frac{\widehat{\Omega}_{\alpha}}{T}
}.
```

Unless an explicit bandwidth is supplied for a controlled test or robustness
analysis, the deterministic bandwidth rule is

```math
K
=
\min\left\{
T-1,
\left\lfloor
4\left(\frac{T}{100}\right)^{2/9}
\right\rfloor
\right\}.
```

Recording $K$ in every result makes the dependence assumption auditable and
reproducible. The HAC test statistic is

```math
T_{\mathrm{HAC},\alpha}
=
\frac{
\overline{Z}_{\alpha}
}{
\mathrm{SE}_{\mathrm{HAC}}
(\overline{Z}_{\alpha})
}.
```

Using its asymptotic standard-normal reference distribution, the two-sided
p-value is

```math
p_{\alpha}
=
2\left[
1-\Phi\left(
\left|T_{\mathrm{HAC},\alpha}\right|
\right)
\right].
```

At significance level $s=0.05$, the individual null is rejected when
$p_{\alpha}<s$. A statistically significant positive mean indicates that the
joint VaR-ES forecasts understate the realized tail contribution on average;
a statistically significant negative mean indicates overstatement. Failure to
reject does not prove that the forecasts are correct: it means the sample does
not provide sufficient evidence against zero mean under this test.

The single-model tables retain raw p-values. When the three models and three
confidence levels are combined, Holm adjustment will be applied across the
nine joint-calibration hypotheses before simultaneous claims are made.
Jointly consistent VaR-ES scoring rules will remain separate and will be used
for formal pairwise model comparison rather than mixed into this calibration
test or the descriptive severity summary.

### 8.6 Strictly consistent forecast scores

Calibration tests ask whether a model satisfies a statistical requirement.
Scoring rules answer a different question: which competing forecast is more
accurate on the common sequence of realized outcomes? A score is called
strictly consistent for a target when its expected value is uniquely minimized
by forecasting the true target. Lower scores are better throughout this
project.

#### VaR quantile loss

In the standard notation for an observation $y$, a proposed probability-$\tau$
quantile $q$, and $0<\tau<1$, the quantile or pinball loss is

```math
Q_{\tau}(y,q)
=
(\tau-\mathbf{1}_{\{y<q\}})(y-q).
```

Equivalently,

```math
Q_{\tau}(y,q)
=
\begin{cases}
\tau(y-q), & y>q,\\
(1-\tau)(q-y), & y\le q.
\end{cases}
```

The score is zero when $y=q$ and nonnegative otherwise. Its asymmetry is what
makes the true $\tau$-quantile, rather than the mean or median, minimize
expected score.

This project models an upper-tail positive portfolio loss $L_t$ and uses the
confidence level $\alpha$ directly. Therefore the project implementation is

```math
Q_{t,\alpha}
=
\begin{cases}
\alpha(L_t-q_{t,\alpha}),
& L_t>q_{t,\alpha},\\
(1-\alpha)(q_{t,\alpha}-L_t),
& L_t\le q_{t,\alpha}.
\end{cases}
```

This is the standard quantile score applied to the project's loss variable;
it is not a different statistical criterion. The strict project breach rule
$L_t>q_{t,\alpha}$ is used in code. At equality, the forecast error and score
are both zero, so choosing a strict or weak indicator does not change the
numerical score.

For model $m$, the descriptive mean score is

```math
\overline{Q}_{m,\alpha}
=
\frac{1}{T}
\sum_{t=1}^{T}Q_{m,t,\alpha}.
```

The corresponding within-confidence descriptive rank is

```math
r^{Q}_{m,\alpha}
=
1
+
\sum_j
\mathbf{1}
\left\{
\overline{Q}_{j,\alpha}
<
\overline{Q}_{m,\alpha}
\right\}.
```

Models are ranked from the smallest to the largest mean score separately at
each confidence level. Rank one denotes the smallest mean quantile loss, and
exactly equal means share the smallest applicable rank. Models must use
identical ordered forecast rows and identical realized losses. Rankings across
different confidence levels are not meaningful because the asymmetric weights
differ. A lower sample mean is only a descriptive advantage until a formal
loss-differential test is applied.

#### Canonical FZ0 score for returns

Expected Shortfall is not elicitable by itself: there is no strictly
consistent one-dimensional score that identifies an ES forecast alone in the
same way quantile loss identifies VaR
([Gneiting, 2011](https://arxiv.org/abs/0912.0902)). Under suitable regularity
conditions, however, the pair consisting of VaR and Expected Shortfall is
jointly elicitable
([Fissler and Ziegel, 2016](https://arxiv.org/abs/1503.08123)). The FZ0 score
is a standard jointly consistent member of the Fissler-Ziegel class
([Patton, Ziegel, and Chen, 2019](https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf)).

The published FZ0 formula is commonly written for a lower return tail. Let
$Y_t$ be a return, let $\beta$ be a small left-tail probability such as 0.05,
let $v_{t,\beta}<0$ be return VaR, and let $e_{t,\beta}<v_{t,\beta}<0$ be
return Expected Shortfall. The canonical score is

```math
S^{\mathrm{FZ0}}
(Y_t,v_{t,\beta},e_{t,\beta};\beta)
=
-\frac{
\mathbf{1}\{Y_t\le v_{t,\beta}\}
(v_{t,\beta}-Y_t)
}{
\beta e_{t,\beta}
}
+
\frac{v_{t,\beta}}{e_{t,\beta}}
+
\log(-e_{t,\beta})
-1.
```

This is the standard literature form. Its negative VaR and ES values reflect
the convention that severe portfolio outcomes are negative returns.

#### Conversion to the project's positive-loss convention

The engine stores severe outcomes as positive losses. Define

```math
L_t=-Y_t,
\qquad
q_{t,\alpha}=-v_{t,\beta},
\qquad
s_{t,\alpha}=-e_{t,\beta},
\qquad
\alpha=1-\beta.
```

Then $s_{t,\alpha}\ge q_{t,\alpha}>0$, and the lower-return-tail event becomes
the upper-loss-tail event:

```math
\mathbf{1}\{Y_t\le v_{t,\beta}\}
=
\mathbf{1}\{L_t\ge q_{t,\alpha}\}.
```

Substituting these identities into the canonical formula gives the project's
positive-loss FZ0 score:

```math
S^{\mathrm{FZ0}}_{t,\alpha}
=
\frac{
\mathbf{1}\{L_t>q_{t,\alpha}\}
(L_t-q_{t,\alpha})
}{
(1-\alpha)s_{t,\alpha}
}
+
\frac{q_{t,\alpha}}{s_{t,\alpha}}
+
\log(s_{t,\alpha})
-1.
```

The implementation uses the existing strict VaR exceedance indicator. The
canonical formula is often written with a weak inequality, but equality makes
$L_t-q_{t,\alpha}=0$, so the two conventions produce the same score.

The four parts have distinct roles:

1. The first term penalizes loss beyond VaR and scales that excess by both the
   nominal tail probability and the ES forecast.
2. The ratio $q_{t,\alpha}/s_{t,\alpha}$ links the VaR and ES forecasts.
3. The logarithm penalizes the overall ES scale and balances the incentive to
   make ES arbitrarily large or small.
4. The constant $-1$ is a normalization and does not affect model rankings.

For example, let $\alpha=0.95$, $q=0.02$, and $s=0.03$. On a non-breach day,

```math
S^{\mathrm{FZ0}}
=
\frac{0.02}{0.03}
+\log(0.03)-1
\approx -3.8399.
```

If the realized loss is $L=0.03$, the additional tail term is

```math
\frac{0.03-0.02}{(1-0.95)0.03}
=
6.6667,
```

so the breach-day score is approximately $2.8268$. FZ0 scores need not be
positive; only their ordering and differences matter, and lower is better.

The logarithm requires a strictly positive loss-domain Expected Shortfall.
The implementation will therefore retain the existing finite and ES-at-least-
VaR checks and additionally require $s_{t,\alpha}>0$ for every scored row.
Changing all losses and forecasts from decimal units to percentage units adds
the same constant to competing scores. Pairwise score differences, rankings,
and Diebold-Mariano conclusions are therefore unchanged as long as every model
uses the same units.

For model $m$, the mean joint score is

```math
\overline{S}^{\mathrm{FZ0}}_{m,\alpha}
=
\frac{1}{T}
\sum_{t=1}^{T}
S^{\mathrm{FZ0}}_{m,t,\alpha}.
```

Within confidence level $\alpha$, the descriptive FZ0 rank is

```math
r^{\mathrm{FZ0}}_{m,\alpha}
=
1
+
\sum_{j}
\mathbf{1}
\left\{
\overline{S}^{\mathrm{FZ0}}_{j,\alpha}
<
\overline{S}^{\mathrm{FZ0}}_{m,\alpha}
\right\}.
```

Rank one therefore denotes the smallest mean FZ0 loss. Models with exactly
equal means share the smallest applicable rank. The implementation calculates
these ranks only after verifying identical ordered forecast indexes and
identical realized losses. As with quantile loss, scores from different
confidence levels are not placed in the same ranking.

The FZ0 score evaluates VaR and Expected Shortfall as a pair. A worse score
cannot be attributed to ES alone when the corresponding VaR is also
misspecified. Formal pairwise comparisons will use the complete daily score
differential rather than only the difference between mean scores.

### 8.7 Pairwise Diebold-Mariano predictive-accuracy tests

Mean-score rankings describe which model performed better in the observed
sample, but a small ranking difference may be ordinary sampling noise. The
Diebold-Mariano, or DM, test asks whether two models have equal expected
predictive accuracy under a chosen loss function. The original framework
allows general loss functions and permits the resulting loss differential to
be serially correlated and non-Gaussian
([Diebold and Mariano, 1995](https://doi.org/10.1080/07350015.1995.10524599)).

Fix one confidence level $\alpha$ and one score family $G$. In this project,
$G$ is either VaR quantile loss or the joint VaR-ES FZ0 loss. For an ordered
pair of models $A$ and $B$, define the daily score differential as

```math
d^{G}_{A,B,t}
=
S^{G}_{A,t,\alpha}
-
S^{G}_{B,t,\alpha}.
```

Because lower scores are better, the ordering has a direct interpretation:

- $d^{G}_{A,B,t}<0$ means model $A$ scored better on date $t$;
- $d^{G}_{A,B,t}>0$ means model $B$ scored better on date $t$;
- reversing the model order reverses the sign but not the two-sided p-value.

The sample mean score differential is

```math
\overline d^{G}_{A,B}
=
\frac{1}{T}
\sum_{t=1}^{T}d^{G}_{A,B,t}.
```

The two-sided hypotheses are

```math
H_0:
\mathbb{E}\!\left[d^{G}_{A,B,t}\right]=0,
\qquad
H_1:
\mathbb{E}\!\left[d^{G}_{A,B,t}\right]\ne0.
```

Thus, the null says that the two models have equal expected predictive
accuracy under the selected score. It does **not** say that either model is
correctly calibrated. Two misspecified models can have equal predictive
accuracy, and one misspecified model can score significantly better than
another.

#### Dependence-robust uncertainty

Nearby daily score differences can be related because market volatility
clusters through time. Treating every $d_t$ as independent can understate the
uncertainty in $\overline d$. The project therefore estimates the long-run
variance with the Newey-West heteroskedasticity-and-autocorrelation-consistent
estimator
([Newey and West, 1987](https://doi.org/10.2307/1913610)). First define

```math
\widetilde d_t
=
d_t-\overline d,
```

and estimate the lag-$k$ autocovariance by

```math
\widehat\gamma_k
=
\frac{1}{T}
\sum_{t=k+1}^{T}
\widetilde d_t\widetilde d_{t-k}.
```

With Bartlett weight

```math
w_k
=
1-\frac{k}{K+1},
```

the estimated long-run variance is

```math
\widehat\Omega_d
=
\widehat\gamma_0
+
2\sum_{k=1}^{K}w_k\widehat\gamma_k.
```

The HAC standard error of the mean and the DM statistic are

```math
\mathrm{SE}_{\mathrm{HAC}}(\overline d)
=
\sqrt{\frac{\widehat\Omega_d}{T}},
\qquad
DM_{A,B}
=
\frac{\overline d_{A,B}}
{\mathrm{SE}_{\mathrm{HAC}}(\overline d_{A,B})}.
```

Under the null and the large-sample approximation, the statistic is compared
with a standard normal distribution. The two-sided p-value is

```math
p_{A,B}
=
2\left[1-\Phi\!\left(\left|DM_{A,B}\right|\right)\right].
```

The HAC calculation changes the estimated uncertainty, not the observed mean
score difference. The implementation reuses the project's tested scalar
Newey-West mean-test engine and its deterministic default bandwidth

```math
K
=
\min\left\{
T-1,
\left\lfloor
4\left(\frac{T}{100}\right)^{2/9}
\right\rfloor
\right\}.
```

The chosen bandwidth is saved with every result. This makes the baseline
reproducible and allows later bandwidth sensitivity checks.

#### Project comparisons and interpretation

At each confidence level, the project compares the three unordered model
pairs:

1. Rolling Gaussian versus EWMA Gaussian;
2. Rolling Gaussian versus filtered historical simulation;
3. EWMA Gaussian versus filtered historical simulation.

For example, suppose the ordered comparison is model $A$ minus model $B$ and

```math
\overline d_{A,B}=-0.00010,
\qquad
\mathrm{SE}_{\mathrm{HAC}}(\overline d_{A,B})=0.00004.
```

Then

```math
DM_{A,B}=-2.5,
```

with a two-sided unadjusted p-value of approximately $0.0124$. The negative
sign says that $A$ has the smaller average score, while the small p-value says
that equal expected predictive accuracy is doubtful before multiple-testing
adjustment. If the same mean difference had a HAC standard error of $0.00015$,
then $DM\approx-0.67$ and $p\approx0.50$: model $A$ would look better in the
sample, but the evidence would not reliably distinguish the two models.

The three model pairs across the three confidence levels produce nine tests
per score family. Holm adjustment is applied separately to:

- the nine VaR quantile-loss DM tests; and
- the nine joint VaR-ES FZ0 DM tests.

Keeping these as two predeclared hypothesis families preserves the distinction
between VaR-only accuracy and joint VaR-ES accuracy. These comparison tests are
also kept separate from the earlier calibration-test family.

The implementation requires identical ordered forecast dates, realized
losses, and confidence levels before forming any differential. It reports the
ordered model names, mean differential, HAC standard error, bandwidth, DM
statistic, raw p-value, Holm-adjusted p-value, and rejection decision. The
sign of the mean and statistic identifies which named model scored better;
the p-value alone does not.

For the baseline, the project uses the large-sample, standard-normal DM test.
The Harvey-Leybourne-Newbold correction is a documented small-sample
alternative ([Harvey, Leybourne, and Newbold, 1997](https://doi.org/10.1016/S0169-2070(96)00719-4)),
but it is not applied here because the baseline contains thousands of
one-session-ahead forecasts and the project already uses a predeclared HAC
procedure. Joint VaR-ES score comparisons by DM tests are supported directly
by the joint-elicitability literature
([Fissler, Ziegel, and Gneiting, 2016](https://arxiv.org/abs/1507.00244)).

### 8.8 Basel-style rolling traffic-light diagnostic

The Basel market-risk backtesting framework classifies a 99% one-day VaR model
from the number of exceptions observed over the most recent 250 trading days.
The current framework calls the three categories green, amber, and red
([Basel Committee on Banking Supervision, MAR32.9](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/32/inforce/2022-01-01/published/2019-12-15)).
Older Basel material often calls the middle category yellow; this project uses
the current term **amber**.

For model $m$, define its 99% VaR exception indicator by

```math
I_{m,t,0.99}
=
\mathbf{1}
\left\{
L_t>q_{m,t,0.99}
\right\}.
```

The strict inequality matches the project's existing VaR exceedance rule. For
each date with 250 available forecast observations, the rolling exception
count is

```math
N_{m,t}^{(250)}
=
\sum_{j=0}^{249}
I_{m,t-j,0.99}.
```

If a model has correct independent 99% coverage, the expected count in a
250-session window is

```math
\mathbb{E}\!\left[N_{m,t}^{(250)}\right]
=
250(1-0.99)
=
2.5.
```

The zone function is

```math
Z\!\left(N_{m,t}^{(250)}\right)
=
\begin{cases}
\text{green}, & 0\le N_{m,t}^{(250)}\le4,\\
\text{amber}, & 5\le N_{m,t}^{(250)}\le9,\\
\text{red}, & 10\le N_{m,t}^{(250)}\le250.
\end{cases}
```

The thresholds are derived for exactly 250 observations at 99% coverage. The
Basel guidance explains that boundaries for other sample sizes must be derived
from the corresponding binomial probabilities rather than copying these
counts unchanged
([Basel Committee on Banking Supervision, MAR99.17](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/99/inforce/2022-01-01/published/2019-12-15)).
Accordingly, the baseline diagnostic will use only the project's 99% VaR rows
and will emit its first classification only after 250 forecast sessions are
available.

This rolling diagnostic answers a different question from the full-sample
tests. Kupiec tests the overall exception frequency, Christoffersen tests
exception independence, and the traffic-light path shows when recent model
performance moves into a supervisory warning region. Overlapping 250-session
windows are strongly dependent, so the sequence of zones is a descriptive
monitoring diagnostic rather than a new sequence of independent hypothesis
tests.

For a compact comparison, let $\mathcal{T}_m$ be model $m$'s eligible rolling
window end dates and let $W_m=|\mathcal{T}_m|$. For zone
$z\in\{\text{green},\text{amber},\text{red}\}$, the number and proportion of
windows in that zone are

```math
C_{m,z}
=
\sum_{t\in\mathcal{T}_m}
\mathbf{1}\{Z(N_{m,t}^{(250)})=z\},
\qquad
R_{m,z}
=
\frac{C_{m,z}}{W_m}.
```

The counts must satisfy

```math
C_{m,\mathrm{green}}
+
C_{m,\mathrm{amber}}
+
C_{m,\mathrm{red}}
=
W_m,
```

and their rates sum to one. The summary also reports the latest count and
zone, the maximum rolling exception count

```math
N_m^{\max}
=
\max_{t\in\mathcal{T}_m}
N_{m,t}^{(250)},
```

and the earliest date attaining that maximum. First amber and first red dates
are retained when those zones occur; otherwise they are recorded as missing.
These quantities describe duration, timing, and worst observed recent
performance without treating overlapping windows as independent evidence.

The project labels the result **Basel-style**, not regulatory compliance. The
Basel rules refer to defined actual and hypothetical trading outcomes within a
bank's regulatory backtesting programme. This research engine instead compares
its documented portfolio-loss series with model VaR forecasts.

#### 8.8.1 Comparative timeline figure

The report visualizes the same tested rolling counts
$N_{m,t}^{(250)}$ in three vertically aligned panels, ordered as rolling
Gaussian, EWMA Gaussian, and filtered historical simulation. Every panel uses
the same calendar axis and exception-count axis. Consequently, differences in
the height and persistence of the paths are genuine model differences rather
than artefacts of panel-specific scaling.

The line is drawn as a step path because the count changes discretely when an
old observation leaves and a new observation enters the 250-session window.
Horizontal boundaries at five and ten exceptions mark entry into the amber and
red zones. Light background bands encode the three zones, while the textual
labels ``GREEN 0–4``, ``AMBER 5–9``, and ``RED 10+`` ensure that interpretation
does not depend on colour alone. Each panel also states its maximum count and
date and its latest count and zone. The raster output is saved at 300 DPI for
documents, and the SVG preserves vector lines and text for publication use.

The figure is descriptive. It does not add observations, alter the exception
definition, or create another hypothesis test. Its source data are the saved
traffic-light path and summary tables, and the report metadata stores the
palette, thresholds, panel order, output filenames, and SHA-256 checksums so
the displayed result remains reproducible.

### 8.9 Predeclared subperiod analysis

The full-sample evaluation remains the project's primary statistical
protocol. Subperiod analysis is a secondary diagnostic of whether calibration,
tail severity, and predictive accuracy appear stable through time. It does not
replace, modify, or selectively reinterpret any completed full-sample test.

For subperiod $s$, let $\mathcal{T}_s$ be its common aligned forecast dates and
let

```math
n_s=|\mathcal{T}_s|.
```

For model $m$ and confidence level $\alpha$, define the subperiod exception
count and rate by

```math
X_{m,s,\alpha}
=
\sum_{t\in\mathcal{T}_s} I_{m,t,\alpha},
\qquad
\widehat p_{m,s,\alpha}
=
\frac{X_{m,s,\alpha}}{n_s},
```

where the unchanged strict exception rule is

```math
I_{m,t,\alpha}
=
\mathbf{1}\{L_t>\mathrm{VaR}_{m,t,\alpha}\}.
```

Under correct unconditional coverage, the nominal exception probability is
$p_\alpha=1-\alpha$ and the expected subperiod count is

```math
E_{s,\alpha}=n_s(1-\alpha).
```

The full-sample rate is a sample-size-weighted average of disjoint subperiod
rates. Consequently, conservative and anti-conservative periods can offset
one another even when the pooled rate is close to nominal. Subperiod tables
are intended to expose that instability without treating a favorable pooled
or local result as proof that the model is correct.

#### 8.9.1 Calendar-year periods

The calendar analysis includes every calendar year from 2012 through 2025,
intersected with the common ordered forecast dates shared by all three models.
The exact observation count is retained for every year; no year is assumed to
contain 250 sessions. A year with fewer than 90% of the largest annual common
observation count is retained and labelled as partial rather than silently
excluded.

Calendar boundaries are economically arbitrary, but they are objective,
exhaustive, familiar to readers, and unaffected by the observed model
results. Every qualifying year must be reported, including quiet years and
years in which the models behave similarly.

#### 8.9.2 Externally defined stress episode

The initial event study uses every U.S. recession in the National Bureau of
Economic Research business-cycle chronology that intersects the common
forecast sample. The NBER chronology defines the only such contraction in the
current sample with a February 2020 peak month and an April 2020 trough month
([NBER Business Cycle Dating](https://www.nber.org/research/business-cycle-dating)).

Because the chronology is monthly, the deterministic date-mapping rule
includes every common forecast session whose calendar month lies from the
published peak month through the published trough month, inclusive. The
current stress episode is therefore the intersection of the common forecast
dates with 2020-02-01 through 2020-04-30. The metadata must record the source,
published peak and trough months, daily mapping convention, realized first
and last forecast dates, and exact observation count.

This episode is an externally dated U.S. economic contraction, not a claim
that every included session experienced equal financial-market stress. Its
short length makes it a descriptive case study. The complete 2020 calendar
year provides a broader companion period. Additional hand-selected event
windows will not be added after inspecting the models' exception paths. A
future market-stress regime based on an external indicator would require a
separate predeclared source, threshold, timing rule, and treatment of
non-contiguous dates before any results are calculated.

#### 8.9.3 Statistics and sparse-tail warnings

Each model-confidence-period row reports:

- period identifier, type, first and last forecast dates, observation count,
  and partial-period flag;
- nominal and observed exception probabilities, expected and observed
  exception counts, and their signed differences;
- the pointwise exact 95% binomial interval for the exception probability;
- mean VaR and Expected Shortfall forecasts;
- VaR-tail observation count and the existing descriptive tail-severity
  statistics;
- the mean all-session joint VaR-Expected-Shortfall identification residual;
- mean VaR quantile loss and mean joint VaR-Expected-Shortfall FZ0 score; and
- descriptive within-period model ranks, retaining ties.

Transition counts may be reported only for a contiguous period and are
descriptive. The subperiod report does not recalculate Basel traffic-light
classifications because their thresholds apply to complete rolling
250-session windows and the existing traffic-light report already preserves
that timeline.

Two reliability flags make sparse tail information explicit:

```math
\texttt{sparse\_expected\_tail}
=
\mathbf{1}\{n_s(1-\alpha)<5\},
```

and

```math
\texttt{sparse\_observed\_tail}
=
\mathbf{1}\{X_{m,s,\alpha}<5\}.
```

If no exception occurs, conditional tail-severity statistics remain missing,
not zero. If one through four exceptions occur, the statistics are retained
but explicitly flagged as unstable. Exact intervals express sampling
uncertainty but do not remove the low power caused by rare events. This is
particularly important for annual 99% VaR: a 250-session year has only 2.5
expected exceptions and, even under correct independent calibration, has
probability $0.99^{250}\approx0.081$ of containing no exception.

#### 8.9.4 Inferential status and multiplicity

The initial subperiod analysis is descriptive. It creates no new reject or
do-not-reject decisions, reports no subperiod Kupiec, Christoffersen,
joint-calibration, or Diebold-Mariano p-values, and therefore creates no new
formal multiple-testing family. Pointwise binomial intervals must be labelled
as non-simultaneous and must not be converted into a visual pass/fail rule.

If formal subperiod inference is proposed later, its hypotheses, eligibility
rules, and multiplicity families must be documented before those p-values are
calculated. At minimum, conditional-coverage, joint-calibration, VaR
quantile-loss comparison, and FZ0 comparison hypotheses would remain separate
families, with Holm adjustment applied across all eligible periods within
each declared family. Full-sample families and conclusions remain unchanged.

#### 8.9.5 Interpretation and reproducible outputs

Interpretation follows a fixed hierarchy:

1. Completed full-sample inference supplies the primary conclusions.
2. Calendar years diagnose temporal stability and locate periods that merit
   explanation.
3. The NBER contraction is a short, externally defined case study.
4. A local adverse result is a warning, not by itself proof of general model
   failure; a favorable local result cannot rescue a model that fails the
   primary protocol.
5. Too few exceptions may indicate excessive conservatism rather than better
   forecasting.
6. Changes in descriptive model ranks do not establish statistically
   different predictive accuracy.

The versioned report saves `subperiod_definitions.csv`,
`subperiod_var_coverage.csv`, `subperiod_es_tail_severity.csv`,
`subperiod_joint_calibration.csv`, and
`subperiod_predictive_accuracy.csv`.

The interval figure adds a binomial reference band to help interpret the
observed rates. For nominal probability $p_\alpha=1-\alpha$ and reference
sample size $n_{\mathrm{ref}}$, define

```math
X_{\mathrm{ref},\alpha}
\sim
\mathrm{Binomial}
\left(n_{\mathrm{ref}},p_\alpha\right).
```

Let $F^{-1}_{X}(u)=\min\{k:F_X(k)\ge u\}$ be the discrete binomial quantile.
The displayed central 95% reference band is

```math
\mathcal{R}_{\alpha}
=
\left[
\frac{F^{-1}_{X_{\mathrm{ref},\alpha}}(0.025)}
     {n_{\mathrm{ref}}},
\frac{F^{-1}_{X_{\mathrm{ref},\alpha}}(0.975)}
     {n_{\mathrm{ref}}}
\right].
```

The calendar panels use $n_{\mathrm{ref}}=252$, while the NBER stress panels
use their actual length, $n_{\mathrm{ref}}=62$. This construction conditions
on the nominal probability and asks what rates it would commonly generate. It
is therefore conceptually different from the Clopper-Pearson interval, which
starts from an observed count and describes uncertainty about the unknown
probability $\pi$. The reference band is descriptive and does not create an
additional acceptance test.

The first implemented figure is a
confidence-level-faceted heatmap of observed-minus-nominal exception rates.
It uses percentage-point cell labels and one shared symmetric scale centered
at zero, so the sign and color retain the same meaning across every model and
confidence level. Calendar years 2012–2025 occupy the main panels. The
February–April 2020 NBER contraction is shown in a separate strip because it
overlaps calendar 2020 and contains only 62 sessions. Positive red cells mean
more exceptions than nominal and therefore tail-risk underestimation;
negative blue cells mean fewer exceptions and possible conservatism. A note
explicitly warns that the plot is descriptive and that sparse tail counts can
produce large deviations, especially at 99% VaR.

The second implemented figure shows observed exception rates with pointwise
95% Clopper-Pearson exact binomial intervals. The three upper panels contain
calendar years at 95%, 97.5%, and 99% VaR. A separate lower row contains the
overlapping NBER contraction because its 62-session sample creates wider
intervals and requires a different horizontal scale. The 95%, 97.5%, and 99%
calendar panels use fitted upper limits of 20%, 16%, and 12.5%, respectively,
so rare-event estimates are not compressed. All recession panels share a
0%-40% axis, preserving direct comparability within the stress row. A small
unlabelled negative margin prevents zero-exception markers from being clipped;
the first displayed tick remains 0%. Model identity is encoded by a fixed
colorblind-safe color and marker combination, while a dashed vertical line
marks the relevant nominal exception probability. Alternating annual-row
shading supports horizontal tracking. Every panel also includes the central
95% binomial reference range implied by its nominal probability: annual panels
use 252 reference sessions and stress panels use 62. These pale-green regions
are visual benchmarks only; the plotted Clopper-Pearson interval for each
estimate uses its actual period sample size. Recession estimates are labelled
with observed-count/sample-size pairs, positioned beyond the upper interval
bound.

The figure states that the intervals are non-simultaneous and
non-multiplicity-adjusted; it therefore supplies uncertainty context without
creating an additional visual pass-or-fail rule.

The predictive-score results are separated into a main-text summary and a
supporting annual grid. The main figure contains side-by-side QL and FZ0 panels.
Each panel plots the number of calendar years in which Rolling Gaussian, EWMA,
or FHS achieved the lowest period-average score at 95%, 97.5%, and 99%. The
February-April 2020 NBER period is excluded from those 14-year totals and its
all-level FHS result is stated separately. A faint line at (14/3) shows the
count implied by an even three-model split. It is a visual reference, not an
inferential threshold or null expectation.

The supporting grid groups results by score family first: all three QL
confidence levels appear on the left and all three FZ0 levels on the right.
Calendar years 2012-2025 occupy the main rows, while the overlapping
February-April 2020 NBER period remains a separate stress strip. Every cell
prints its saved minimum rank from one through three. Rank-one cells use the
winning model's fixed house color; ranks two and three remain gray text on
white. This preserves annual transparency while making the main EWMA-versus-FHS
pattern easier to see.

A generated companion CSV reports the period-average QL and FZ0 values and
ranks for every period-model-confidence combination. Neither figure encodes a
HAC-standardized statistic, p-value, or statistical significance. This avoids
introducing unplanned subperiod Diebold-Mariano comparisons after observing the
results. Formal inference remains the completed full-sample, Holm-adjusted
Diebold-Mariano analysis.

The subperiod figures use the shared definitions in `tailrisk.style` for model
identity, publication typography, semantic colors, confidence-level ordering,
and reusable plotting behavior. They retain the same calendar range and
separate NBER stress strip. Helvetica is preferred for report figures, with
Arial and DejaVu Sans as deterministic fallbacks. Raster copies are saved at
300 DPI and SVG copies preserve the vector text and geometry.

Metadata identifies the report as `descriptive_secondary_analysis`, retains
the source-backtest snapshot identifiers and checksums, records the complete
period-definition rules and sparse-tail flags, and stores a SHA-256 checksum
for every table and figure. New runs create new versioned directories without
overwriting earlier subperiod snapshots.

## 9. Baseline status and future protocol decisions

The baseline protocol described above is complete and frozen for the reported
2012–2025 comparison. Raw missing prices are preserved, one-session returns are
never implicitly filled, and portfolio construction requires a complete return
vector for every held asset. The retained baseline retrieval contained no
missing adjusted-close observations, so no imputation or partial-portfolio rule
affected the reported baseline results.

The Dynamic Quantile test is not part of the completed baseline statistical
protocol. If it is added as a later research extension, its regressors, lag
count, eligibility rules, and multiple-testing family must be declared before
its results are calculated.

Material changes to these conventions will be recorded in `docs/decisions.md`.
