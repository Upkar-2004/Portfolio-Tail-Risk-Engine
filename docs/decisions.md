# Decision Log

## 2026-10-05: Fixed the EWMA Gaussian estimation conventions

**Status:** Accepted

### Context

The EWMA Gaussian model is intended to test whether a covariance estimate that
responds more strongly to recent returns improves one-day portfolio VaR and
Expected Shortfall forecasts. Changing the forecast mean at the same time
would confound that comparison, while an arbitrary zero or diagonal covariance
initialization would distort the first forecasts.

### Decision

- Use a baseline EWMA decay parameter of $\lambda=0.94$ and require it to be a
  finite real number strictly between zero and one. Boolean values are invalid
  even though Python treats them as integers.
- Initialize the first EWMA forecast with the ordinary sample covariance from
  its scheduled 504-session estimation window, using denominator 503.
- Retain the rolling Gaussian model's 504-session arithmetic asset mean for
  both the Gaussian forecast location and the innovation definition.
- For forecast session $t$, calculate the mean using returns through session
  $t-1$. After observing $\mathbf{R}_t$, define the innovation as
  $\mathbf{u}_t=\mathbf{R}_t-\widehat{\boldsymbol{\mu}}_t$.
- Use that innovation only to update the next trading session's covariance:
  $\widehat{\boldsymbol{\Sigma}}^{\mathrm{EWMA}}_{t+1}
  =\lambda\widehat{\boldsymbol{\Sigma}}^{\mathrm{EWMA}}_t
  +(1-\lambda)\mathbf{u}_t\mathbf{u}_t^\top$.
- After initialization, evolve covariance recursively without a rolling cutoff.
- Keep the forecast schedule, beginning-of-session weights, confidence levels,
  scenario count, random seed, shared standard-normal shocks, realized losses,
  and strict VaR exceedance rule identical to the rolling Gaussian baseline.

### Consequences

- The model comparison changes covariance dynamics while holding the Gaussian
  mean and simulation design constant.
- The first EWMA and rolling Gaussian Monte Carlo forecasts must be identical;
  divergence begins only after the first EWMA update.
- Session $t$ can evaluate its own forecast and update the forecast for the
  next trading session, but it cannot influence its own forecast.
- The primary specification is intentionally different from classic zero-mean
  RiskMetrics. A zero-mean or alternative-decay specification may be examined
  later as a documented robustness check.

## 2026-10-11: Fixed the filtered historical simulation conventions

**Status:** Accepted

### Context

The FHS model must share the 2012-01-04 first forecast date and the 504-session
information set used by the completed Gaussian baselines. The available
baseline data contain exactly 504 returns before that date, so a separate
sequence of 504 strictly one-step-ahead residuals cannot be constructed
without adding a pre-2010 burn-in dataset or delaying the comparison period.

### Decision

- Construct a 504-row initial calibration pool by demeaning and whitening the
  first forecast's 504 estimation returns with that window's sample mean and
  sample covariance.
- Describe these initial rows as in-sample calibration residuals, not as
  historical one-step-ahead forecast errors. All inputs still predate the
  first forecast, so the initialization introduces no out-of-sample
  look-ahead.
- Retain a fixed 504-row rolling pool. After observing session $t$, calculate
  its innovation with the pre-session 504-day mean and standardize it with the
  covariance used to forecast $t$. Drop the oldest row and append the new row
  for use beginning with session $t+1$.
- Use the EWMA covariance sequence with $\lambda=0.94$ for both residual
  standardization and forecast-date rescaling.
- Resample complete multivariate residual rows with replacement. Never sample
  asset columns independently.
- Draw 100,000 residual rows per forecast date with master seed 20261002 and
  reuse the same random pool positions across dates.
- Transform each sampled residual with the forecast-date rolling mean and
  EWMA covariance, apply beginning-of-session portfolio weights, and define
  loss as the negative simulated portfolio return.
- Calculate empirical VaR from the simulated-loss quantile and Expected
  Shortfall from losses at or above the VaR threshold.

### Consequences

- All three models retain identical forecast dates, portfolio weights, and
  realized losses.
- The first FHS forecast reduces to ordinary historical simulation over the
  initial estimation window because whitening and rescaling use the same mean
  and covariance.
- Initial calibration rows leave the pool one at a time and are completely
  replaced by sequential forecast errors after 504 forecast sessions.
- The model isolates empirical non-Gaussian joint shocks while retaining the
  EWMA volatility dynamics already used by the Gaussian comparison model.
- A separate pre-sample burn-in period remains available as a later robustness
  check rather than part of the baseline specification.

## 2026-10-12: Fixed the core VaR statistical-testing protocol

**Status:** Accepted

### Context

The three baseline models now cover identical forecast dates and realized
losses. Statistical decision rules must be fixed before comparative results
are interpreted so thresholds are not selected in response to observed
p-values.

### Decision

- Define an exceedance by the strict rule
  $I_t=\mathbf{1}\{L_t>\operatorname{VaR}_{t,\alpha}\}$. Equality is not an
  exceedance.
- Use nominal exceedance probability $p=1-\alpha$ and state the Kupiec null as
  $H_0:\Pr(I_t=1)=p$, which concerns the unknown breach probability rather
  than the observed rate.
- Use two-sided tests with a 5% significance level and reject only when the
  raw p-value is strictly below 0.05.
- Report observed and expected rates and counts alongside every test so the
  direction and magnitude of miscalibration remain visible.
- Use the asymptotic chi-square reference distributions for Kupiec and
  Christoffersen likelihood-ratio statistics, with boundary likelihood terms
  evaluated using the convention $0\log(0)=0$.
- Treat Christoffersen conditional coverage as the primary overall VaR
  adequacy test. Report Kupiec unconditional coverage and Christoffersen
  independence as component diagnostics.
- Retain all raw p-values and additionally apply Holm adjustment across the
  nine primary model-confidence hypotheses when making simultaneous adequacy
  claims.
- Use the language "reject" or "do not reject" rather than treating a
  non-rejection as proof that a model is correct.
- Keep later VaR-score and joint VaR-Expected-Shortfall model comparisons in
  separate, predeclared multiple-testing families.

### Consequences

- Too few exceedances are interpreted as evidence of conservatism, not
  automatically as superior risk performance.
- Kupiec results cannot establish that exceedances are independent; the
  Christoffersen layer remains necessary.
- A raw p-value below 0.05 may not support a simultaneous final claim after
  multiplicity adjustment, so both values will be reported.
- Descriptive subperiod, severity, and traffic-light results will not be
  presented as formal hypothesis tests without separately documented rules.

## 2026-10-13: Fixed the initial Expected Shortfall severity protocol

**Status:** Accepted

### Context

Expected Shortfall describes average loss severity beyond VaR rather than a
second quantile threshold. It therefore must not be evaluated by inventing an
"ES exceedance rate" with a nominal probability. The initial ES layer should
remain interpretable while preserving a clear boundary between descriptive
tail summaries and later formal joint VaR-ES inference.

### Decision

- Define the tail sample for each model-confidence pair using the existing
  strict VaR exceedance rule
  $I_{t,\alpha}=\mathbf{1}\{L_t>\operatorname{VaR}_{t,\alpha}\}$.
- Require every Expected Shortfall forecast to be finite and at least as large
  as its corresponding VaR forecast. Allow an absolute numerical tolerance of
  $10^{-12}$ in portfolio-loss units for insignificant floating-point error.
- Report the number of forecasts and VaR exceedances together with mean
  realized exceedance loss, mean VaR and ES on exceedance dates, and the mean,
  median, and maximum loss excess over VaR.
- Define the ES gap on a VaR-exceedance date as realized loss minus forecast
  ES and report its conditional sample mean.
- Store conditional severity statistics as missing when no VaR exceedance is
  observed. Zero would incorrectly imply that tail severity was observed and
  measured as zero.
- Treat these outputs as descriptive diagnostics. Each model defines its own
  exceedance-date set, so conditional tail summaries are not formal pairwise
  rankings.
- Implement the all-session joint VaR-ES identification residual separately
  from the descriptive severity summary. Dependence-robust uncertainty and
  jointly consistent scoring rules will be specified before formal inference
  or pairwise model comparison.

### Consequences

- Tail-severity tables will explain the magnitude of VaR failures without
  misrepresenting ES as a threshold with a fixed exception probability.
- The exceedance count will remain visible beside every conditional statistic
  so estimates based on sparse 99% tails are not presented without context.
- A positive mean ES gap will indicate that realized losses on breach dates
  exceeded forecast ES on average; a negative value will indicate the reverse.
- Descriptive ES results alone will not establish joint calibration or model
  superiority.

## 2026-10-13: Fixed HAC inference for joint VaR-ES calibration

**Status:** Accepted

### Context

The all-session VaR-ES identification residual has expected value zero when
the conditional VaR and Expected Shortfall forecasts are jointly calibrated.
Its sample mean is nevertheless exposed to changing volatility and serial
dependence, so an independent-observation standard error could overstate the
effective amount of information and produce misleading rejection decisions.

### Decision

- Test the two-sided null that the population mean identification residual is
  zero separately for each model-confidence pair.
- Estimate uncertainty with the scalar Newey-West HAC long-run variance using
  Bartlett weights and autocovariances divided by the observation count.
- Use the deterministic automatic bandwidth
  $\min\{T-1,\lfloor4(T/100)^{2/9}\rfloor\}$ unless an explicit bandwidth is
  supplied for a controlled test or a documented robustness analysis.
- Record the selected bandwidth, long-run variance, standard error, test
  statistic, raw p-value, significance level, and rejection decision.
- Use a two-sided asymptotic standard-normal p-value and a default individual
  significance level of 5%.
- Retain raw p-values in single-model results. Apply Holm adjustment across the
  nine model-confidence joint-calibration hypotheses before making simultaneous
  conclusions in the combined report.

### Consequences

- The test accounts for heteroskedasticity and short-range serial dependence
  without changing the underlying sample mean residual.
- A positive significant mean indicates joint tail-risk understatement, while
  a negative significant mean indicates overstatement.
- A failure to reject is reported as insufficient evidence against calibration,
  not proof that the model is correct.
- The bandwidth remains visible and reproducible, and alternative bandwidths
  can be examined later without changing the baseline rule.

## 2026-10-13: Fixed the VaR and joint VaR-ES scoring conventions

**Status:** Accepted

### Context

Formal model ranking requires scoring rules that are consistent for the
forecasted statistical targets. VaR can be scored with quantile loss, whereas
Expected Shortfall must be scored jointly with VaR. Published FZ0 formulas are
usually written for negative returns and a small lower-tail probability, while
this project stores positive portfolio losses and high confidence levels.

### Decision

- Use standard quantile loss for VaR and rank models by mean score separately
  at each confidence level, with lower scores preferred.
- Use the canonical return-domain FZ0 definition as the literature reference
  and explicitly transform it with $L=-Y$, $q=-v$, $s=-e$, and
  $\alpha=1-\beta$ before implementation.
- In the project's positive-loss convention, calculate
  $\mathbf{1}\{L>q\}(L-q)/[(1-\alpha)s]+q/s+\log(s)-1$.
- Reuse the strict project VaR exceedance rule. Equality produces zero tail
  excess, so it is numerically equivalent to the weak inequality in the
  canonical score.
- Require finite forecasts, $s\ge q$, and strictly positive Expected Shortfall
  for every FZ0-scored row.
- Compare models only on identical ordered forecast rows, identical realized
  losses, the same confidence level, and the same loss units.
- Treat sample mean-score rankings as descriptive until pairwise
  Diebold-Mariano tests with dependence-robust uncertainty are applied.

### Consequences

- The implementation remains mathematically equivalent to the published FZ0
  score while matching the engine's existing loss-sign convention.
- FZ0 values may be negative; only lower-versus-higher scores and score
  differences carry ranking information.
- Changing common units adds a common constant to FZ0 scores but leaves model
  rankings and pairwise score differences unchanged.
- A joint-score difference cannot be attributed to Expected Shortfall alone
  when VaR is also misspecified.

## 2026-10-13: Fixed the pairwise Diebold-Mariano comparison protocol

**Status:** Accepted

### Context

Mean quantile-loss and FZ0 rankings show which model scored best in the
observed sample but do not establish whether the difference is larger than
sampling variation. Daily score differences can also be serially dependent
because market conditions and forecast errors cluster through time.

### Decision

- Define every ordered loss differential as model A's daily score minus model
  B's daily score, so a negative mean favors model A and a positive mean favors
  model B.
- Test equal expected predictive accuracy with a two-sided, asymptotic
  Diebold-Mariano statistic.
- Estimate the mean differential's uncertainty with the existing scalar
  Newey-West HAC engine and its deterministic default bandwidth, recording
  that bandwidth in every result.
- Require identical ordered dates, confidence levels, and realized losses
  before comparing two forecasts.
- Compare all three model pairs at each of the three confidence levels.
- Apply Holm adjustment separately to the nine VaR quantile-loss comparisons
  and the nine joint VaR-ES FZ0 comparisons. Keep both families separate from
  calibration tests.
- Use the large-sample normal reference distribution for the baseline rather
  than the Harvey-Leybourne-Newbold small-sample correction.

### Consequences

- A descriptive ranking becomes a formal statement about whether a model's
  expected score is detectably lower than another model's expected score.
- The p-value accounts for heteroskedasticity and short-range serial dependence
  in the score differential.
- Statistical significance does not establish correct calibration; it only
  distinguishes the two models under the chosen scoring rule.
- Direction remains auditable because every result records the ordered model
  names and the signed mean differential.
