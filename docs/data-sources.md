# Data Sources

## 1. Purpose and Scope

This project uses daily market data to calculate one-day returns for a portfolio of liquid assets. These returns will provide the historical observations used to estimate volatility and covariance and to produce one-day Value at Risk (VaR) and Expected Shortfall (ES) forecasts.

The required data includes daily closing-price information and corporate actions, particularly stock splits and cash dividends. Corporate-action information is necessary because a change in the unadjusted closing price does not always represent an economic gain or loss for the investor. The project will therefore evaluate adjusted price data when defining the return series used by the risk models.

The initial study is limited to end-of-day data and a primary risk horizon of one trading day. It does not require intraday prices, real-time market feeds, or a live trading system. After the core models, backtests, and validation workflow are complete, the project may include a terminal-based interface for configuring experiments, running the risk engine, and viewing results. This interface will use the validated research pipeline rather than introduce a separate modelling or data-processing system. The asset universe, sample period, portfolio weights, and rebalancing convention will be documented before the empirical analysis begins.


## 2. Candidate data source

The initial candidate data source is Yahoo Finance, accessed programmatically through the Python `yfinance` library. Yahoo Finance provides historical daily prices and corporate-action information, including cash dividends and stock splits, for many liquid securities.

Yahoo Finance is the data provider, while `yfinance` is an independent open-source access tool. The `yfinance` project is not affiliated with, endorsed by, or vetted by Yahoo. Its documentation describes the retrieved Yahoo Finance data as intended for research and educational purposes and directs users to Yahoo’s terms of use for applicable usage restrictions.

This source is suitable for the project’s initial research and learning stages because it is accessible, supports reproducible Python-based retrieval, and provides the price and corporate-action fields needed to investigate daily returns. It is not assumed to be an institutional-quality or point-in-time market-data service. Its observations and historical adjustments must be validated before they are used by the risk models.

The project will record the retrieval date, ticker symbols, requested date range, data frequency, `yfinance` version, and relevant download settings. This metadata will help identify changes caused by package updates, vendor corrections, or differences between repeated downloads.

### Sources

- [yfinance documentation](https://ranaroussi.github.io/yfinance/index.html), accessed 2026-08-24.
- [Yahoo Finance: Download historical data](https://in.help.yahoo.com/kb/finance/download-historical-data-yahoo-finance-sln2311.html), accessed 2026-08-24.


## 3. Required data fields

The project requires enough information to calculate daily returns, identify corporate actions, validate the observations, and reproduce the dataset. These are logical data requirements; the exact column names returned by a data provider or library may differ.

| Field | Purpose |
|---|---|
| Trading date | Identifies the market session associated with each observation. |
| Ticker | Identifies the asset to which the observation belongs. |
| Raw closing price | Records the unadjusted end-of-day market price. |
| Adjusted price | Accounts for applicable stock splits and cash dividends. |
| Cash dividends | Identifies distributions that affect shareholder returns. |
| Stock splits | Identifies changes in share count and quoted price per share. |
| Currency | Identifies the currency in which the asset is quoted. |
| Exchange | Helps identify trading calendars, holidays, and time zones. |
| Retrieval metadata | Records when and how the data was downloaded. |

The raw closing price and adjusted price information serve different purposes. The raw close is an observed market price, while the adjusted series is a derived historical series intended to make prices on different dates economically comparable. Both should initially be retained: the adjusted information can support return calculation, while the raw close, dividends, and split records can be used to audit unexpected returns.

The project must preserve missing observations as missing during ingestion. A missing price is not a price of zero and does not imply a zero return. Rules for validating, aligning, or excluding missing observations will be defined separately before returns are calculated.

Trading dates must also be treated as market-session identifiers rather than as an assumption that every asset trades on every calendar date. When several assets are combined, differences in holidays, suspensions, listing histories, or missing data can produce different sets of valid trading dates.


## 4. Price and return-input policy

The primary return series will be calculated from adjusted closing-price information rather than directly from raw closing prices. Adjusted prices are preferred because stock splits and cash dividends can change the raw price without representing an equivalent economic gain or loss for the shareholder. Using unadjusted prices could therefore introduce artificial extreme returns and distort estimated volatility, covariance, VaR, and Expected Shortfall.

Raw closing prices will also be retained. They will be used with the reported dividend and stock-split fields to investigate corporate-action dates, verify large price adjustments, and diagnose suspicious returns. The raw close will not automatically be treated as the investor’s complete return.

For data retrieval, automatic adjustment of all price fields will initially be disabled. This is intended to preserve the distinction between the raw closing price and the adjusted closing-price information. Corporate-action data will be requested separately in the same retrieval process. The returned columns will be inspected and validated because their exact structure may depend on the installed `yfinance` version and download settings.

The use of adjusted closing prices does not imply that the data is error-free. Adjusted values are derived by the data provider and may be affected by missing corporate actions, corrections, or later revisions. Unusually large returns will therefore be checked against the raw close, dividend records, split records, and, when necessary, an independent source.

The exact mathematical definition of the daily return—including the choice between simple and logarithmic returns—will be recorded in `docs/methodology.md`. The data-source policy determines which price information enters that calculation; it does not by itself define the return formula.


## 5. Proposed yfinance retrieval settings

The project will specify important `yfinance` download arguments explicitly rather than relying on library defaults. This makes the intended dataset easier to understand and helps protect the workflow from changes to default behaviour in later package versions.

| Setting | Proposed value | Reason |
|---|---:|---|
| `interval` | `"1d"` | The primary risk horizon is one trading day, so the project requires daily observations. |
| `auto_adjust` | `False` | Preserves the distinction between raw closing prices and adjusted price information. |
| `actions` | `True` | Requests dividend and stock-split information for validation and return interpretation. |
| `keepna` | `True` | Retains rows containing missing values so they can be inspected rather than silently removed during retrieval. |
| `repair` | `False` initially | Avoids silently modifying downloaded observations before the original data has been inspected. |
| `prepost` | `False` | Excludes pre-market and after-hours observations; they are outside the daily close-to-close study. |
| `rounding` | `False` | Avoids unnecessary loss of numerical precision during retrieval. |
| `threads` | `False` initially | Makes early downloads and error messages easier to inspect while the data workflow is being validated. |
| `progress` | `False` | Prevents progress output from interfering with logs and automated runs. |

The requested `start` date is inclusive: an observation on that date may be returned if it is a valid trading session. The requested `end` date is exclusive: observations on that date are not returned. For example, an end date of `2024-01-01` requests data only up to the last available trading session before `2024-01-01`.

The date range will be specified explicitly rather than through a relative period such as `"5y"`. A relative period would produce a different sample depending on the retrieval date, whereas fixed boundaries make an experiment easier to reproduce.

The installed `yfinance` version will be recorded with each dataset. After retrieval, the program will validate the returned column names and structure instead of assuming that a particular package version always returns the same schema. In particular, it will confirm that raw close, adjusted price information, dividends, and stock splits are available as expected.

These settings form an initial retrieval policy and may be revised after inspecting a small sample. Any material change that affects the meaning of the return series or the reproducibility of the study will be documented.

- [yfinance download API reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), accessed 2026-08-24.


## 6. Data-source limitations

Yahoo Finance and `yfinance` are convenient for an educational research project, but they are not assumed to provide institutional-quality or point-in-time market data. The following limitations must be considered when interpreting the results.

### 6.1 Unofficial programmatic access

`yfinance` is an independent open-source library and is not affiliated with, endorsed by, or vetted by Yahoo. Its programmatic access may be affected by changes to Yahoo Finance’s interfaces, rate limits, network availability, or changes to the library itself. A successful download is therefore not sufficient evidence that the returned dataset is complete or correct.

### 6.2 Historical revisions

Historical prices and corporate-action adjustments may be corrected after their original publication. Downloading the same ticker and date range at different times may therefore produce different results. The project will record retrieval metadata and retain reproducible data snapshots where permitted.

### 6.3 Corporate-action errors

Adjusted prices depend on accurate dividend and stock-split information. A missing, duplicated, or incorrectly recorded corporate action could create a false return or conceal a genuine market movement. Large returns will be compared with the raw close, dividend records, and stock-split records before being accepted.

### 6.4 Missing observations and unequal calendars

Assets may have missing observations because of exchange holidays, trading suspensions, listing dates, delistings, or data-provider errors. Assets from different exchanges may also follow different trading calendars. Missing prices will not automatically be converted into zero returns or filled without an explicit and tested rule.

### 6.5 Ticker and instrument changes

Ticker symbols are not permanent identifiers. Companies can change symbols, move exchanges, merge, split into new entities, or cease trading. Reusing a symbol does not necessarily mean that it represents the same economic instrument throughout the sample period. Instrument identity and relevant corporate events must therefore be checked.

### 6.6 Survivorship and selection bias

Selecting assets that are liquid and actively traded today may exclude companies that failed, were acquired, or were delisted during the historical period. This can make the historical portfolio appear less risky than a portfolio selected using information available at the time. The project must define and document its asset-selection rule before interpreting model performance.

### 6.7 Currency and market differences

Prices from different currencies cannot be combined directly without defining an exchange-rate and portfolio-currency policy. Exchanges can also differ in time zones, closing times, holidays, and settlement conventions. The initial asset universe should avoid unnecessary cross-market complexity unless these differences are handled explicitly.

### 6.8 Licensing and intended use

The availability of data through a website or Python library does not remove its terms of use or licensing restrictions. The data is being considered for personal, educational research. Redistribution of raw vendor data will be avoided unless the applicable terms clearly permit it.

These limitations do not automatically make the source unsuitable. They define the validation, documentation, and caution required before the data can support conclusions about VaR and Expected Shortfall calibration.


## 7. Multi-asset date alignment and missing-data policy

A portfolio return combines asset returns referring to the same holding period. Prices must therefore be aligned by market session before the portfolio return is calculated. Matching observations only by row position is not valid because different assets may have different missing dates.

The initial asset universe should, where practical, consist of liquid instruments following the same primary trading calendar. This reduces complications caused by different holidays, time zones, and market-closing times. Even within a common market, individual observations may still be missing because of trading suspensions, listing changes, or data-provider errors.

The project will construct or obtain an appropriate trading-session calendar and align each asset’s prices to that calendar. A missing observation will remain explicitly missing during ingestion and validation. It will not automatically be:

- replaced with zero;
- interpreted as a zero return;
- forward-filled from the previous session; or
- removed before examining the resulting return horizon.

A one-day asset return for session \(t\) requires valid, comparable prices for both session \(t-1\) and session \(t\). If either price is unavailable, the one-day return will initially be marked as missing. A return calculated from the last available price several sessions earlier must not be labelled as a one-day return.

A portfolio return for session \(t\) requires a valid one-day return for every asset held by the portfolio, unless a separate and economically justified missing-data rule has been defined. If any required component return is unavailable, the portfolio return will initially be treated as unavailable rather than silently calculated from an incomplete set of assets.

Missing observations will be investigated before exclusion. The investigation should distinguish among:

- normal exchange holidays;
- asset-specific trading suspensions;
- listing or delisting boundaries;
- corporate actions;
- ticker changes;
- data-provider failures; and
- instruments following different market calendars.

Any observation removed from the final dataset will be recorded with its date, affected instrument, reason, and applied rule. All models will use the same valid forecast and evaluation dates so that model comparisons remain fair.


## 8. Data-validation policy

Downloaded data will be validated before prices are transformed into returns. Validation and cleaning are separate stages: validation identifies possible problems, while cleaning applies explicitly documented decisions. A suspicious observation will not be deleted or modified merely because it looks unusual.

Validation results will be divided into errors and review warnings. Errors prevent the affected data from entering the model. Warnings identify observations that require investigation but may represent genuine market events.

### 8.1 Structural checks

The dataset will be checked for:

- required fields;
- expected column names and structure;
- valid ticker identifiers;
- parseable trading dates;
- duplicate ticker-date combinations;
- dates in chronological order;
- observations outside the requested date range; and
- empty or failed ticker downloads.

Each ticker and trading date should identify at most one daily observation. Duplicate records must be investigated rather than arbitrarily keeping the first or last value.

### 8.2 Price checks

Raw and adjusted prices used by the project must be numeric, finite, and strictly positive. Zero, negative, infinite, or non-numeric prices are invalid for the planned return calculations.

Missing prices will remain marked as missing until their cause and treatment have been determined. They will not automatically be replaced with zero or with the preceding price.

### 8.3 Corporate-action checks

Dividend and stock-split fields must be numeric and finite. A nonzero stock-split entry must represent a positive split ratio. Corporate-action dates will be compared with large differences between raw and adjusted price movements.

A large price change on a dividend or split date is not automatically an error. Conversely, the presence of an adjusted price does not prove that the corporate-action information is correct.

### 8.4 Return-based diagnostic checks

Preliminary returns may be calculated solely for validation after the price fields pass structural checks. Unusually large absolute returns will be flagged for review rather than automatically removed or winsorized.

For each flagged observation, the investigation should consider:

- whether a dividend or split occurred;
- whether the ticker or listing changed;
- whether the price currency or units changed;
- whether an observation is missing or duplicated;
- whether the movement appears in an independent source; and
- whether the movement is a genuine market event.

Genuine extreme returns must remain in the dataset because tail observations are directly relevant to VaR and Expected Shortfall. Removing them simply because they are extreme would bias the research.

### 8.5 Cross-asset and coverage checks

For every asset, the validation process will report:

- first and last available dates;
- number and proportion of missing sessions;
- duplicate count;
- corporate-action count;
- number of flagged returns; and
- currency and exchange information where available.

The common period in which the portfolio can be evaluated will be determined only after inspecting these coverage results. All three risk models must ultimately be compared using the same valid forecast dates and realized portfolio losses.

### 8.6 Validation records

The project will retain a validation summary and a record of material exclusions or corrections. Each manual change should identify the affected ticker and date, the original value, the applied treatment, the reason, and the evidence supporting the decision.

Automatic repair features will not replace this record. If an automated repair method is later used, the original and repaired values will be compared and the method will be documented.


## 9. Data storage and reproducibility

The project will separate downloaded source data from cleaned and model-ready datasets. This prevents later processing steps from overwriting the original observations and makes data-quality decisions easier to audit.

The intended data stages are:

1. **Raw data:** the original response obtained from the data source, preserved without manual modification where storage and licensing terms permit.
2. **Validated data:** the raw observations together with validation results and identified warnings or errors.
3. **Processed data:** aligned prices and calculated returns produced according to documented rules.
4. **Model inputs:** the final matrices and portfolio information supplied to a specific experiment.

Each dataset or retrieval run should be accompanied by metadata containing:

- data provider;
- retrieval tool and version;
- retrieval timestamp and time zone;
- requested ticker symbols;
- requested start and exclusive end dates;
- observation frequency;
- relevant download settings;
- returned fields;
- currency and exchange information where available;
- validation results;
- applied exclusions or corrections; and
- a stable file checksum where practical.

Raw downloaded data will not be silently overwritten by a later retrieval. If the source revises its history, the new download should be stored or identified as a separate version so that differences can be investigated.

Processed datasets must be reproducible from the retained raw inputs and documented transformation rules. Manual edits to price files will be avoided. When a correction is necessary, it will be represented as an explicit, reviewable transformation that preserves the original value and records the supporting evidence.

Raw vendor data will not be committed to the public repository unless its licensing terms clearly permit redistribution. Code, configuration, metadata schemas, and documentation needed to reproduce the retrieval and processing workflow may still be version-controlled.



### Initial portfolio size and future expansion

The baseline empirical study will use exactly 11 large-cap US-listed companies, with one company selected from each major equity-market sector. This portfolio size will remain fixed while the rolling Gaussian, EWMA Gaussian, and filtered historical simulation models are developed and compared.

Using the same 11 assets for every baseline model ensures that differences in VaR and Expected Shortfall results are attributable to the models rather than to changes in portfolio composition. The baseline portfolio will not be expanded or replaced after model results have been examined.

The choice of 11 assets is a research-design convention, not a software limitation. Data structures, return calculations, covariance estimation, simulation, and backtesting will be designed for a general number of assets \(N\). No implementation should hard-code an assumption that \(N=11\).

After the baseline study is complete and validated, the project may examine larger portfolios as separate experiments. Possible extensions include:

- two or more companies from each sector;
- a broader large-cap equity portfolio;
- alternative company-selection rules;
- alternative portfolio-weighting rules; and
- larger synthetic inputs for scalability and performance testing.

These extensions will be labelled as robustness or scalability studies. They will not replace the original 11-asset baseline, and their results will be reported separately.

Expansion will begin only after:

1. the 11-asset data has passed validation;
2. the three baseline models have been implemented;
3. look-ahead and alignment tests pass;
4. the rolling backtest is operational;
5. covariance and simulation diagnostics have been reviewed; and
6. baseline results can be reproduced from recorded configuration.