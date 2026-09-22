# DLinear shared baseline

This method trains the official shared DLinear model separately on each
benchmark dataset with context length 512 and prediction length 64. It is a
supervised per-dataset baseline, unlike the zero-shot TS-RAG and Chronos-Bolt
runs, so reports must label the training regime.

The exact upstream `models/DLinear.py` from `cure-lab/LTSF-Linear` commit
`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` is vendored with its Apache-2.0
license. Its expected SHA-256 is
`0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc`.

`source/train_dlinear_shared.py` produces checkpoints, predictions, and native
metrics on shared storage. The adapter converts the completed native metrics to
the common LLM-TS run schema. Large runtime files must not be committed.
The imported trainer was adapted only so the exact vendored model file can be
used without embedding an upstream `.git` directory; both the upstream commit
and model checksum remain enforced.

Use the common launcher:

```bash
python scripts/run_method.py --method dlinear --dataset ETTh1 --dry-run
```
