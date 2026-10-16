# Research and Engineering Roadmap

## Current foundation

Release `v0.1.0` is the completed Python research baseline. It establishes the
portfolio, three-model comparison, statistical protocol, verified result
tables, publication figures, documentation conventions, and automated test
suite. Future work extends this foundation and does not replace its preserved
configuration or findings.

## Release principles

1. `main` remains tested, documented, and suitable for public presentation.
2. Material research changes receive a new configuration or predeclared
   protocol instead of silently changing the baseline.
3. Generated results remain non-overwriting and checksum traced.
4. Reusable engine improvements return to `main` through short-lived feature
   branches.
5. Paper-specific files use dedicated `papers/` subdirectories and reference a
   tagged engine release instead of maintaining divergent core implementations.
6. Statistical claims remain separated into primary inference, robustness, and
   descriptive diagnostics.

## Near-term portfolio release

- [x] Complete the three-model Python baseline.
- [x] Complete full-sample calibration and predictive-accuracy evaluation.
- [x] Complete the initial calendar-year and 2020 stress-period analysis.
- [x] Establish a shared publication figure style and tracked output gallery.
- [x] Separate the public landing page from the detailed reproduction guide.
- [x] Install the package and pass the complete test suite in a clean
  environment.
- [x] Verify public Markdown links and tracked publication assets.
- [ ] Recheck the public repository from a fresh clone after the release commit
  is pushed.
- [x] Record explicit all-rights-reserved terms for the initial public release.
- [ ] Tag the verified portfolio baseline as `v0.1.0`.

## Research extensions

Candidate research releases may add:

- GARCH-family volatility filters and alternative EWMA decay specifications;
- extreme-value tail models;
- copula or multivariate dependence extensions;
- alternative portfolio construction and rebalancing rules;
- additional externally dated stress periods;
- formal regime-comparison procedures;
- transaction-cost and liquidity-risk sensitivity; and
- model-confidence-set or forecast-combination analysis.

Each extension requires its own motivation, assumptions, tests, configuration,
inferential family, and result lineage.

## Paper tracks

The first paper can use the three-model baseline as a focused empirical study
of volatility adaptation, non-Gaussian filtered residuals, and joint VaR-ES
forecast evaluation. Later papers can introduce one coherent methodological
extension at a time.

A future structure may use:

```text
papers/
  paper-01-baseline-comparison/
  paper-02-tail-model-extension/
  paper-03-portfolio-robustness/
```

Manuscripts, paper-specific configurations, tables, and supplementary material
belong in those directories. Shared models, evaluation methods, and plotting
infrastructure remain in the core package.

## Engineering extensions

The planned C++ phase will port selected, already validated numerical routines
behind a small interface. It begins only when its scope, correctness oracles,
and benchmark protocol are fixed. It is an acceleration and systems-engineering
track, not a missing dependency of the Python research baseline.

Later engineering releases may also add:

- a unified experiment command;
- a broader continuous-integration matrix across supported Python versions;
- packaged example or synthetic data for fast end-to-end demonstrations;
- schema-version migrations for long-lived artifacts; and
- measured performance benchmarks after correctness is preserved.

## Version outline

| Release | Meaning |
| --- | --- |
| `v0.1.0` | Completed three-model Python research baseline |
| `v0.2.0` | First material model or robustness extension |
| `v0.3.0` | Paper-specific or portfolio-level extension |
| `v1.0.0` | Mature research engine with the validated native acceleration layer |
