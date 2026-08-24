# Methodology

# Methodology

## 1. Purpose

This document records the mathematical definitions, timing conventions, portfolio assumptions, model specifications, and evaluation rules used by the Portfolio Tail-Risk Engine.

The primary research question is:

> Do models that account for changing volatility and non-Gaussian returns produce better-calibrated one-day portfolio VaR and Expected Shortfall forecasts than a rolling Gaussian model?

This is a neutral comparison. No model is assumed to perform better before the out-of-sample evidence is evaluated.

The primary risk horizon is one trading day. All models will ultimately be compared using the same portfolio, realized losses, valid forecast dates, and evaluation rules.

## 2. Price and return convention

The project will use adjusted daily closing-price information according to the policy in `docs/data-sources.md`.

Let \(P^{\mathrm{adj}}_{i,t}\) denote the adjusted closing price of asset \(i\) for trading session \(t\). Its one-day simple return is

$$
R_{i,t}
=
\frac{P^{\mathrm{adj}}_{i,t}}
     {P^{\mathrm{adj}}_{i,t-1}}
-1.
$$

A positive return represents an increase in economic value, while a negative return represents a decrease.

The two prices must correspond to consecutive valid trading sessions. If either required price is missing, the one-day return will initially be treated as unavailable. A return spanning multiple sessions must not be labelled as a one-day return.

## 3. Why simple returns are primary

Simple returns will be the primary convention because they combine directly across assets in a linear portfolio.

Let \(w_{i,t-1}\) denote the weight of asset \(i\) established before the return for session \(t\) is realized. The portfolio return is

$$
R_{p,t}
=
\sum_{i=1}^{N}
w_{i,t-1}R_{i,t}.
$$

The use of \(t-1\) weights prevents the calculation from using portfolio positions chosen after observing the session-\(t\) returns.

Logarithmic returns are defined by

$$
r_{i,t}
=
\ln\left(
\frac{P^{\mathrm{adj}}_{i,t}}
     {P^{\mathrm{adj}}_{i,t-1}}
\right)
=
\ln(1+R_{i,t}).
$$

Log returns add across consecutive time periods, which can be useful for exploratory analysis. However, weighted asset log returns do not directly equal the return of a rebalanced linear portfolio. Log returns will therefore not be the primary convention for portfolio P&L, VaR, or Expected Shortfall.

## 4. Portfolio P&L and loss

Let \(V_{t-1}\) denote the portfolio value immediately before the session-\(t\) return is realized.

The one-day portfolio profit and loss is

$$
\mathrm{PnL}_t
=
V_{t-1}R_{p,t}.
$$

P&L is positive for a gain and negative for a loss.

Portfolio loss is defined as the negative of P&L:

$$
L_t
=
-\mathrm{PnL}_t
=
-V_{t-1}R_{p,t}.
$$

Under this convention:

- a gain produces a negative loss;
- a loss produces a positive loss; and
- VaR and Expected Shortfall are normally reported as positive loss amounts.

A portfolio return will initially be considered unavailable if any return required for a held asset is unavailable. Any later exception must be economically justified and documented.

## 5. Forecast timing and information set

A forecast made at the end of trading session \(t\) may use only information available by the end of that session.

The forecast concerns the portfolio loss during the next trading session:

$$
L_{t+1}
=
-V_t R_{p,t+1}.
$$

The timing sequence is:

1. Observe market data available through the end of session \(t\).
2. Estimate the model using only permitted data through session \(t\).
3. Produce VaR and Expected Shortfall forecasts for session \(t+1\).
4. Observe the realized portfolio loss \(L_{t+1}\).
5. Compare the forecast with the realized loss.

Information from session \(t+1\) or later must not influence the forecast produced at time \(t\). This restriction applies to return calculation, covariance estimation, volatility estimation, model fitting, and portfolio weights.

## 6. Planned model comparison

The models will be developed and evaluated in the following order:

1. Rolling Gaussian model using a sample covariance matrix.
2. EWMA Gaussian model using a time-varying covariance estimate.
3. Filtered historical simulation using standardized historical residuals.

The rolling Gaussian model provides the benchmark.

Comparing the EWMA Gaussian model with the rolling Gaussian model will help examine the effect of changing volatility while retaining a Gaussian assumption.

Comparing filtered historical simulation with the Gaussian models will help examine the additional effect of non-Gaussian residual behaviour.

Greater complexity will not be treated as evidence of better performance. Calibration must be assessed from the out-of-sample results.

## 7. Evaluation principles

All models will use matching forecast dates and the same realized portfolio losses.

VaR evaluation will consider both:

- whether the number of exceedances is consistent with the selected confidence level; and
- whether exceedances appear independently through time rather than clustering.

Fewer exceedances do not automatically indicate a better model. A model can produce too few exceedances because it is excessively conservative.

Expected Shortfall evaluation will examine the severity of losses beyond the VaR threshold rather than relying only on the number of VaR exceedances.

The precise statistical tests, confidence levels, estimation windows, and decision rules will be documented before the final backtest results are interpreted.

## 8. Decisions still to be finalized

The following choices remain open and must be documented before the relevant experiments begin:

- asset universe and selection rule;
- historical sample period;
- portfolio weights;
- rebalancing frequency;
- treatment of cash;
- portfolio base currency;
- confidence levels for VaR and Expected Shortfall;
- estimation-window lengths;
- covariance-estimation details;
- missing-data exclusion rules;
- transaction-cost assumptions;
- number of simulation scenarios;
- random seeds; and
- statistical-test decision rules.

Material changes to these conventions will be recorded in `docs/decisions.md`.