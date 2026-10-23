# Direction-conditioning: obtaining data and reproducing results

Git contains code, configurations, prompts, reports and lightweight aggregate
metrics. Do not commit raw/standardized series, window caches, LLM raw requests
or responses, per-window prediction arrays, label NPZs, checkpoints or secrets.
The Git HTML report retains the prompt and seven-variable label example but
omits historical values. The full private report stays in ignored `runs/`.

## 1. Public dataset for new experiments

Source: [ETDataset (dataset authors)](https://github.com/zhouhaoyi/ETDataset).
Follow its license and citation requirements. Download explicitly, not during
training import or tests:

```bash
python3 scripts/download_etth1.py
```

Only Python standard library is required. The script pins upstream commit
`1d16c8f4f943005d613b5bc962e9eeb06058cf07`, verifies SHA256 before writing,
refuses overwrite, and saves provenance beside `data/raw/ETTh1.csv` (ignored).
The public file has 17,420 rows and columns date, HUFL, HULL, MUFL, MULL, LUFL,
LULL, OT. Verified public SHA256:
`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`.

**This public file is not byte-identical to our frozen research CSV.**
Whether the difference is only serialization or includes values has not been
audited here. Do not substitute it silently or bypass historical hash checks.
Use a new data/run version for new experiments until semantic equivalence is
separately established. This script does not claim exact historical replay.

## 2. Exact historical replay for authorized collaborators

Ask the experiment maintainer for the frozen `baseline_raw_v1/ETTh1.csv` and
its manifest through approved shared storage, not Git. Verify locally:

```bash
python3 scripts/download_etth1.py --verify-frozen /path/to/ETTh1.csv
```

Required historical SHA256:
`3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d`.
Historical server runners expect the CSV under
`<project>/artifacts/baseline_raw_v1/ETTh1.csv` plus its original manifest.
Keep ordering, hourly timestamps, seven variables, context512/horizon64 and
train-only numerical scaler. Target starts (inclusive, zero-based): train
512..8576, val8640..11456, test11520..14336. Counts8065/2817/2817.
Do not use the older benchmark_v1 manifest's counts for this experiment.

Exact G1/G2 replay also requires approved access to the frozen LLM label
NPZs/manifest from `20261019_dao_etth1_allsplits_v1`, and G1/G3/B require old
baseline/Oracle checkpoints and original trainer dependencies. These are not
included in Git. Fresh API calls may produce different labels and cost money;
they are NOT a substitute for replaying the frozen cache.

## 3. Portability and credentials

This branch archives the executed server research workflow, not a fully
self-contained pip training package. Some historical runners retain fixed
server paths and pinned original trainer/checkpoint hashes. Read their PLANs
before execution; adapt paths in a new version instead of rewriting evidence.
The root pyproject does not declare all GPU research dependencies. Provision
the approved original environment or separately validate a new environment.

DAO requires authorized network access and `TOKEN_SERVICE_API_KEY` from the
environment. Never put credentials in Git or example commands. The Flash
config is an earlier smoke-test configuration, not the final reported setup;
the final run uses Pro/concise, reasoning enabled, stride32. Offline tests do
not call the API. No automatic downloads, API calls or training are authorized
merely by cloning this branch.

Reports can be read without any dataset. Rebuilding the HTML currently requires
private metric files and read-only access to the archived server example; the
default builder removes historical numbers from its Git output. An explicit
`--private-example` output is restricted to ignored `runs/`.
