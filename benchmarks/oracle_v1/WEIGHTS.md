# Historical forecasting checkpoints (2026-11-02)

Weights are distributed as GitHub Release assets, not ordinary Git blobs:

https://github.com/Zhang666zhao/LLM-TS/releases/tag/forecast-weights-20261102

## Scope

- 175 five-dataset baseline/oracle/zero checkpoints: ETTh1, ETTh2, ETTm1,
  ETTm2, weather; five methods and A/B × L1/L2/zero, plus base_zero.
- 35 Exchange baseline/oracle/zero checkpoints.
- 28 original September backbone checkpoints: PatchTST, iTransformer, FreTS,
  iTransformer+FreDF across seven datasets.
- Seven original September DLinear checkpoints.

Total245 checkpoint references, grouped into seven dataset-specific TAR assets.
This release does not include the later ETTh1 G2 LLM-conditioned checkpoints,
and does not claim the incomplete Electricity oracle suite is finished.
Electricity contains only the original five backbone checkpoints.

`weights_manifest.json` maps each checkpoint to its result job, repository
config path, SHA256, byte count and tensor count. `bundles.json` and SHA256SUMS
describe the archive files. All checkpoints were inspected read-only on CPU
with torch.load(weights_only=True): nonempty string-to-tensor state dictionaries
only; all tensors finite. No training or GPU evaluation was performed during
packaging. Each transferred archive member was verified against its source hash.

## Download and extract

Install/authenticate GitHub CLI if necessary. From the repository root, for example:

```bash
mkdir -p artifacts/weights-release-download
gh release download forecast-weights-20261102 --repo Zhang666zhao/LLM-TS \
  --dir artifacts/weights-release-download \
  --pattern 'weights-ETTh2-20261102.tar' \
  --pattern weights_manifest.json --pattern bundles.json --pattern SHA256SUMS
python3 scripts/extract_research_weights.py \
  --bundle artifacts/weights-release-download/weights-ETTh2-20261102.tar \
  --bundles artifacts/weights-release-download/bundles.json \
  --manifest artifacts/weights-release-download/weights_manifest.json \
  --output artifacts/restored-runs
```

The verifier checks archive/member SHA256, blocks traversal and links, refuses
existing output files, and recreates the original relative run paths. Run it
for each desired dataset archive. No automatic data downloads or API calls.

## Load and reproduce

Instantiate the exact original architecture/config, then load the checkpoint
on CPU with `torch.load(path, map_location='cpu', weights_only=True)` and use
`model.load_state_dict(state, strict=True)`. Only load trusted files after hash
verification. Do not use `strict=False` to disguise an architecture mismatch.

Oracle conditioned checkpoints contain the trained conditioning module and
backbone; B freezes its backbone during training, but its saved state_dict
contains that backbone as well. State dictionaries include model buffers but
not datasets, label caches, prediction arrays or optimizer/sampler/RNG state.
They support restoring prediction weights, NOT exact interrupted-training resume.

PatchTST/Electricity attempt02 evaluated attempt01/best.pth. The manifest keeps
the result job at attempt02 and checkpoint source at attempt01, verifies the
SHA256 in attempt02/config.json, and never invents an attempt02 weight file.

Match numerical scaler, variable order,512/64 windows and original checkpoint
selection. A/B/L1/L2 are separate models. Oracle labels require true future
information; these weights do not make oracle performance deployable. Dataset
access is separate: see data/RESEARCH_DATA.md. Original backbone dependencies,
source versions and upstream licenses still need to be supplied and respected;
this weights release does not by itself make the repository a portable training
package. Result evidence corresponds to repository commit52ac699.

No original datasets, normalized historical series, API keys, labels, targets,
or predictions are included in the assets. The binary files remain ignored by Git.
