# Results and Interpretation

This document records the main findings from the verified project outputs in
plain language. It separates descriptive evidence from formal statistical
conclusions and will be updated as new figures are completed.

## Subperiod VaR exception-rate heatmap

### Question

Did each VaR model produce approximately the expected number of exceptions in
each year, and how did that change during the February-April 2020 recession?

A VaR exception occurs when the realized portfolio loss is larger than the
model's VaR forecast. The expected exception rates are:

- 5% for 95% VaR;
- 2.5% for 97.5% VaR; and
- 1% for 99% VaR.

### How to read the figure

Each heatmap cell shows the observed exception rate minus the expected
exception rate, measured in percentage points.

- Red means there were more exceptions than expected. The model
  underestimated tail risk during that period.
- Blue means there were fewer exceptions than expected. The model may have
  been conservative during that period.
- A value close to zero means the observed exception rate was close to its
  nominal target.

Fewer exceptions are not automatically better. A well-calibrated 99% VaR
model should still be exceeded approximately 1% of the time.

### Main finding

Descriptively, VaR calibration varied materially through time. The rolling
Gaussian model exhibited the largest regime-dependent deviations, while EWMA
Gaussian and filtered historical simulation generally remained closer to
nominal coverage. Nevertheless, all three models underestimated tail risk
during the February-April 2020 contraction.

### Examples

In 2018, the rolling Gaussian 95% model produced 37 exceptions from 251
forecasts. Its observed exception rate was 14.7%, compared with the expected
5%, giving a deviation of approximately +9.7 percentage points. EWMA Gaussian
and filtered historical simulation were much closer to the 5% target that
year.

During the 62-session February-April 2020 recession, the rolling Gaussian 99%
model produced 12 exceptions even though only 0.62 were expected. EWMA
Gaussian produced five and filtered historical simulation produced four.
Therefore, all three models underestimated risk during the sudden contraction,
but the rolling Gaussian model showed the largest miss.

### Important limitation

The heatmap is descriptive. It shows where observed exception rates were above
or below their targets, but its colors are not formal pass-or-fail decisions.
Annual 99% VaR results and the short recession period contain very small
expected exception counts, so individual observations can create large-looking
changes. The project's full-sample, multiplicity-adjusted tests remain the
primary formal evidence.

## Observed exception rates with exact intervals

This figure shows each observed exception rate as a dot. The horizontal line
through the dot is its pointwise 95% Clopper-Pearson exact binomial interval.
The dashed vertical line is the nominal exception rate: 5% for 95% VaR, 2.5%
for 97.5% VaR, and 1% for 99% VaR.

The upper row shows calendar-year estimates. To keep the rarer 97.5% and 99%
results readable, the three panels use fitted upper limits of 20%, 16%, and
12.5%, respectively. The lower row shows the separate 62-session
February-April 2020 NBER contraction on a common 0%-40% scale because its
rates and uncertainty intervals are much larger than most annual values.
Alternating, very light row shading helps keep the three model estimates
aligned with the correct year.

The pale-green band in each panel is the central 95% range of exception rates
that a binomial model would generate if the nominal exception probability were
correct. The annual panels use a 252-session reference year; the stress panels
use their actual 62-session length. These are visual benchmarks, not new
formal acceptance rules. The exact horizontal interval around each point
continues to use that period's actual number of observations.

### How to read the figure

- If an interval crosses the dashed nominal line, the observed result remains
  compatible with the target in this individual pointwise comparison.
- If an interval lies entirely to the right of the nominal line, the data
  indicate more exceptions than the target and therefore tail-risk
  underestimation for that period.
- If an interval lies entirely to the left, the data indicate fewer exceptions
  than the target and possible excessive conservatism for that period.

The interval crossing a target does not prove that the model is correctly
calibrated. It means only that this period does not provide clear evidence of
a difference at the pointwise 5% level under the binomial assumptions.

For example, observing zero 99% exceptions in one year does not prove that the
true exception probability is zero. There may simply have been too few days to
observe a rare exception. The interval shows this uncertainty.

The phrase "pointwise 95% exact binomial interval" means:

- **Binomial:** each forecast day has two possible outcomes for this purpose:
  exception or no exception.
- **95% interval:** the interval is constructed by a method that captures the
  true exception probability at least 95% of the time over repeated samples,
  under the binomial model.
- **Exact:** the calculation uses the binomial distribution directly instead
  of a rough large-sample approximation.
- **Pointwise:** every period-model-confidence interval is calculated
  separately. The intervals are not adjusted to make a simultaneous statement
  about all 135 comparisons.

### Main finding

The exact intervals confirm that many annual deviations are uncertain because
tail exceptions are rare. The clearest recurring weakness is the rolling
Gaussian model in 2018: its observed exception rate lies above the nominal
target at all three confidence levels, with none of the three intervals
crossing its target. EWMA Gaussian and filtered historical simulation are much
closer to their nominal rates in that year.

The February-April 2020 contraction shows a broad stress failure rather than
only a rolling-Gaussian problem. At 99% VaR, the intervals for all three
models lie entirely above the 1% nominal rate. The rolling Gaussian estimate
is still the most extreme: 12 exceptions in 62 sessions, or 19.4%, compared
with a 1% target.

### Examples

In 2012, the rolling Gaussian 99% model produced zero exceptions in 249
sessions. Its exact interval runs from 0% to approximately 1.47%, so it still
contains the 1% target. Zero observed exceptions therefore do not establish a
true exception probability of zero or prove that the model was superior.

In 2018, the rolling Gaussian 95% model produced 37 exceptions in 251
sessions: 14.7%, with an exact interval of approximately 10.6% to 19.7%. The
entire interval is above the 5% target, making this a much clearer calibration
miss than a dot alone would show.

During the 2020 contraction, the 95% EWMA Gaussian and filtered historical
simulation estimates were each 7 exceptions in 62 sessions, or 11.3%. Their
intervals are wide - approximately 4.7% to 21.9% - and still cross the 5% target.
At 99% VaR, however, every model's interval is above the 1% target.

### Important limitation

These intervals are pointwise, not simultaneous, and are not adjusted for the
135 period-model-confidence comparisons in the figure. The binomial model also
treats the exception probability as constant within a period and does not
model serial dependence. Accordingly, the figure is a research-quality
descriptive diagnostic, not a replacement for the project's primary
full-sample calibration tests and multiplicity-adjusted conclusions.

## Subperiod predictive-score summary and annual ranks

### Question

Which model produced the lowest period-average VaR quantile loss, and which
produced the lowest joint VaR-Expected-Shortfall FZ0 loss? Did the answer
change through time or across confidence levels?

Exception counts alone cannot answer this question. Daily forecast errors can
offset one another: a model may be too conservative on some sessions and too
aggressive on others while still producing an annual exception count close to
its target. The two scoring rules use every forecast session and reward the
forecast functional they were designed to evaluate.

### How to read the figures and companion table

The main-text figure plots the number of calendar years in which each model
ranked first at 95%, 97.5%, and 99%. Its left panel covers QL and its right
panel covers FZ0. The dashed line at (14/3) represents an even split of 14
calendar years across three models and is a visual reference only. The
overlapping 62-session February-April 2020 NBER result is stated separately and
is excluded from the annual counts.

The supporting grid supplies the year-by-year detail. Its QL half contains the
95%, 97.5%, and 99% blocks on the left, and its FZ0 half contains the same
sequence on the right. Rank 1 denotes the lowest mean score, rank 2 the middle
score, and rank 3 the highest score. Rank-one cells use the winning model's
house color; ranks two and three remain gray text on white. The generated
companion CSV reports the corresponding period-average score magnitudes and
ranks. RG denotes Rolling Gaussian, EWMA denotes EWMA Gaussian, and FHS denotes
Filtered Historical Simulation.

The FZ0 calculation uses the project's positive-loss convention: Expected
Shortfall is at least as large as VaR. FZ0 scores may be negative, but lower
scores remain better when models are evaluated on the same sessions,
confidence level, and units.

### Main finding

The descriptive ranking changes with the forecast target. EWMA Gaussian most
frequently ranked first for VaR quantile loss at 95% and 97.5%, attaining the
lowest annual mean in 10 of 14 years at each level. At 99%, FHS led in 7 years,
compared with 5 for EWMA and 2 for Rolling Gaussian.

For the joint VaR-Expected-Shortfall FZ0 score, FHS led more frequently as the
confidence level increased: 7 of 14 years at 95%, 8 at 97.5%, and 10 at 99%.
Rolling Gaussian ranked last in 12 of 14 years for quantile loss at every
confidence level. It also ranked last in 12 years for FZ0 at 95% and 97.5% and
in 9 years at 99%.

During the February-April 2020 contraction, FHS ranked first, EWMA second, and
Rolling Gaussian third for both score families at all three confidence levels.
This is a consistent stress-period pattern, but it is based on only 62
sessions.

### Connection to the formal evidence

The annual rank counts do not establish that EWMA and FHS have different
expected predictive accuracy. In the full-sample Holm-adjusted
Diebold-Mariano analysis, EWMA and FHS were not significantly different for
either score family at any confidence level. The formal results more clearly
support the conclusion that the adaptive models outperformed Rolling Gaussian
for selected score-confidence combinations.

One possible explanation for the descriptive FHS pattern is that its filtered
empirical residual distribution preserves non-Gaussian tail features that the
Gaussian innovation assumptions do not. However, FZ0 evaluates VaR and
Expected Shortfall jointly, so the ranking cannot identify Expected Shortfall
or distributional tail shape as the cause. That mechanism remains a research
hypothesis rather than a demonstrated result.

### Important limitation

The figures encode ordinal ranks or first-place counts, not confidence
intervals or p-values. The companion table supplies score magnitudes, but a
change from rank 1 to rank 2 may still reflect a very small difference. No
subperiod Diebold-Mariano statistics were calculated, and the line positions
and cell colors must not be interpreted as local significance decisions. Formal
inference remains the completed full-sample, multiplicity-adjusted comparison.
