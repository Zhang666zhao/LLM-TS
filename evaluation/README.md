# Evaluation contract

Evaluation is method-agnostic. It consumes completed result files and never
imports method implementation modules.

## Variable result schema

`variable_metrics.csv` uses one row per method, dataset, variable, and seed:

```text
benchmark_id,method,dataset,frequency,feature_id,variable,seed,
context_length,prediction_length,test_windows,mse,mae,source
```

MSE and MAE must be finite and non-negative. Dataset IDs and counts must match
the versioned data manifest. Dataset summaries are arithmetic means across
variables; benchmark summaries are macro means across datasets. This matches
the existing workbook because every variable within a dataset has the same
number of test windows and horizon values.

Training regime is method metadata rather than a metric column. Reports must
distinguish zero-shot TS-RAG/Chronos-Bolt from supervised per-dataset DLinear;
their scores share an evaluation protocol but not a training protocol.

## Run result schema

New inference adapters should also write `metrics.json` with:

- `schema_version`, `benchmark_id`, `run_id`, `method`, and `dataset`;
- `seed`, `context_length`, and `prediction_length`;
- `metrics.mse` and `metrics.mae`;
- `sample_count`, `value_count`, `code_commit`, and
  `data_manifest_sha256`;
- `status: complete` only after the full dataset succeeds.

The benchmark_v1 files were imported from a completed workbook. Run
`scripts/import_existing_results.py` to reproduce the committed CSV and JSON;
this performs no model inference.
