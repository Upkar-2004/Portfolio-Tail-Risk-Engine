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
\mathbf{L}_{t_0}\mathbf{L}_{t_0}^{\top}.
```

Each return in that initial window is converted into a calibration residual:

```math
\mathbf{z}_s
=
\mathbf{L}_{t_0}^{-1}
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
\mathbf{L}_{t_0}\mathbf{z}_s
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
\mathbf{L}_t^{-1}\mathbf{u}_t.
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
\mathbf{L}_t\mathbf{z}^{*}.
```

The corresponding simulated portfolio loss is

```math
L^{*}_{p,t}
=
-\mathbf{w}_t^{\top}\mathbf{R}^{*}_t.
```

FHS VaR is the empirical loss quantile at the requested confidence level.
Expected Shortfall is the arithmetic mean of simulated losses greater than or
equal to that VaR threshold. Covariance matrices used for whitening and
rescaling must be finite, symmetric, and numerically positive definite so a
stable Cholesky factor exists.

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

For confidence level $\alpha$, define the exceedance indicator and nominal
exceedance probability as

```math
I_t=\mathbf{1}\{L_t>\operatorname{VaR}_{t,\alpha}\},
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
X\sim\operatorname{Binomial}(T,\pi),
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
q_{t,\alpha}=\operatorname{VaR}_{t,\alpha},
\qquad
e_{t,\alpha}=\operatorname{ES}_{t,\alpha}.
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
\operatorname{median}
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
\widehat{\operatorname{Var}}_{\mathrm{HAC}}
\left(\overline{Z}_{\alpha}\right)
=
\frac{\widehat{\Omega}_{\alpha}}{T},
```

and

```math
\operatorname{SE}_{\mathrm{HAC}}
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
\operatorname{SE}_{\mathrm{HAC}}
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

## 9. Decisions still to be finalized

The following choices remain open and must be documented before the relevant experiments begin:

- missing-data exclusion rules;
- the detailed DQ-test regressors and lag count;
- the HAC bandwidth rule for pairwise forecast-comparison tests.

Material changes to these conventions will be recorded in `docs/decisions.md`.
