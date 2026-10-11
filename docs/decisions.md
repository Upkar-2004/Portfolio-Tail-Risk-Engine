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
