# LLM-TS

## 未来方向标签辅助预测研究分支

本分支用于从oracle方向标签走向历史输入可预测标签。先读
[简版计划](research/direction_conditioning/PLAN.md)、
[历史口径](benchmarks/oracle_v1/protocol.md)与
[归档状态](benchmarks/oracle_v1/provenance/README.md)。
固定输入为历史时间序列，输出为逐变量-1/0/+1；新标签阈值可配置，
历史±0.5结果不改写。本次未启动训练或API调用，未完成远端完整归档。
已有[MD报告](reports/oracle_v1/report.md)与[单文件HTML报告](reports/oracle_v1/report.html)。


LLM-TS is a shared benchmark repository for comparing time-series forecasting
methods under one data and evaluation contract. Method implementations live in
`methods/`; datasets, checkpoints, large artifacts, and full run directories do
not belong in Git.

The first benchmark snapshot compares **TS-RAG**, **Chronos-Bolt**, and
supervised **DLinear** using
context length 512, prediction length 64, and variable-level MSE/MAE. The
published result is imported from an existing completed experiment. Importing
or validating it does not rerun inference.

## Repository layout

```text
configs/                 benchmark, experiment, and local-path templates
data/manifests/          versioned dataset contracts (metadata only)
evaluation/              result schema, import, validation, and comparison
methods/                 isolated method implementations and adapters
results/benchmark_v1/    committed, compact benchmark results
scripts/                 common entry points
tests/                   contract and regression tests
```

## Existing benchmark result

The original workbook is preserved at
`results/benchmark_v1/source/TSRAG_ChronosBolt_variable_metrics.xlsx`.
Machine-readable outputs are generated from its `Long data` sheet:

```bash
python scripts/import_existing_results.py
python scripts/compare_all.py
python -m unittest discover -s tests -v
```

## Running a method

Copy the local path template and point it at shared NFS storage:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
python scripts/run_method.py --method tsrag --dataset ETTh1 --dry-run
python scripts/run_method.py --method chronos_bolt --dataset ETTh1 --dry-run
python scripts/run_method.py --method dlinear --dataset ETTh1 --dry-run
```

Remove `--dry-run` only when an inference run is intentionally requested.
Every new method must satisfy the contract in `methods/README.md` and emit the
schema documented in `evaluation/README.md`.

## Collaboration

Create one feature branch per method or evaluation change. Keep run IDs
immutable, record the Git commit and data-manifest hash, and merge through
review instead of overwriting another contributor's result directory.
