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
