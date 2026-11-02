# Server result evidence exported 2026-11-02

Read-only export from GPU_Foundation, preserving original file bytes and run
directories. `manifest.json` lists every source path, SHA256 and byte count.
No new training, inference, API requests or prediction-array recomputation.

## Coverage

- Five datasets ETTh1/ETTh2/ETTm1/ETTm2/weather:175 jobs, including25 baselines,
  100 A/B×L1/L2 oracle jobs and50 matched-zero controls. Reused109 jobs remain
  under their original six-suite run paths;66 new jobs under continuation.
  Directory symlinks are resolved and duplicate source jobs exported only once.
- Exchange:35 jobs from its original independent oracle suite.
- September original backbones:28 evaluations, four configurations
  PatchTST/iTransformer/FreTS/iTransformer+FreDF × seven datasets.
  DLinear's original per-dataset configs/metrics are already archived at
  `results/benchmark_v1/source/dlinear/` in the repository.
- Suite-level budget/status/comparison/paired checks and available provenance.
  Six-suite timeout records remain historical evidence, not a claim that all
  210 tasks completed. Electricity oracle coverage is still incomplete; this
  export includes its suite summary but does not complete its per-job archive.

## Files and verification

2,193 copied files,8,737,809 bytes,238 unique job directories (manifest/README
excluded from copied-file count). Allowlist: metrics/config/status/data_version,
initial hashes, validation metrics, config alignment, provenance, training
summary, variable/horizon metrics, git revision and epoch training JSONL.
Optional files missing on the source are not fabricated; see manifest for
exact inventory. Environment dumps and arbitrary console logs are excluded.

Verified SHA256 of all transferred files against server-generated hashes;
175 existing published summary rows agree on MSE/MAE/OT metrics; all50 saved
initialization/common-epoch pairing checks pass.28 original backbone MSE/MAE
pairs also agree with the published September summary. These checks verify
archival consistency, not a fresh computation from prediction arrays.

PatchTST/Electricity uses attempt02 (evaluation of the prior best checkpoint),
not a fresh training run. Its attempt01 interrupted training record remains on
the server; do not interpret attempt02 as a new training budget.

No datasets, label caches, model weights, prediction/target arrays or secrets
are included. Absolute server paths in historical evidence are provenance,
not portable configuration. Original backbone source dependencies and their
version/license mapping are still not a complete self-contained replay package.
Oracle results use true future labels and are not deployable forecast gains.

Exporter: `scripts/archive_research_results.py` (refuses archive overwrite).
