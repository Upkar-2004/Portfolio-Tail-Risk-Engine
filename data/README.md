# Generated Data and Result Snapshots

This directory contains local, reproducible artifacts created by the project
commands. Snapshot contents are not edited manually and are not committed to
Git because market data, forecast tables, and repeated evaluation runs can be
large or machine-specific.

The pipeline uses the following stages:

- `raw/`: downloaded market-data snapshots and retrieval metadata;
- `processed/`: validated asset-return matrices;
- `portfolio/`: portfolio paths and weight histories;
- `backtests/`: model forecasts, realized losses, and exceedances;
- `evaluations/`: statistical tables, figure snapshots, metadata, and checksums;
- `synthetic/`: optional generated inputs for controlled scalability work.

Every completed snapshot receives a new identifier and directory. Commands
must not overwrite an existing snapshot. Each snapshot's `metadata.json`
records its source lineage, configuration, and SHA-256 checksums.

Canonical baseline snapshot identifiers and checksums are recorded in
[`docs/project-checkpoints.md`](../docs/project-checkpoints.md). Selected
publication-facing figures and compact tables are copied to `reports/` and
committed there so the public repository can display the results without
including the underlying generated datasets.
