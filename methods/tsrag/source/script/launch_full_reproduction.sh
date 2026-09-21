#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CODE_ROOT="$PROJECT_ROOT/TS-RAG"
PYTHON_BIN=${PYTHON_BIN:-/nfs/dataset-ofs-661-142656/qiminhao/Env/dynamo_env/bin/python}
TSRAG_RUN=${TSRAG_RUN:-$PROJECT_ROOT/runs/20260817_TSRAG_official_inference}
BASELINE_RUN=${BASELINE_RUN:-$PROJECT_ROOT/runs/20260817_ChronosBolt_baseline_inference}
SMOKE_RUN=${SMOKE_RUN:-$PROJECT_ROOT/runs/20260817_TSRAG_ETTh1_smoke_mse-mae}
SUMMARY_ROOT=${SUMMARY_ROOT:-$PROJECT_ROOT/runs/20260817_TSRAG_reproduction_summary}
BASELINE_ROOT=${BASELINE_ROOT:-$PROJECT_ROOT/artifacts/baseline_raw_v1}

snapshot_run() {
  local run_root=$1
  mkdir -p "$run_root/logs" "$run_root/diagnostics"
  cp "$CODE_ROOT/configs/inference_reproduction.yaml" "$run_root/config.yaml"
  cp "$CODE_ROOT/script/launch_full_reproduction.sh" "$run_root/run.sh"
  git -C "$PROJECT_ROOT" rev-parse HEAD > "$run_root/git_commit.txt"
  {
    echo "manifest_sha256: $(sha256sum "$PROJECT_ROOT/Data/download_manifest.json" | awk '{print $1}')"
    echo "manifest_path: $PROJECT_ROOT/Data/download_manifest.json"
    echo "download_complete: $(test -f "$PROJECT_ROOT/Data/DOWNLOAD_COMPLETE" && echo true || echo false)"
    echo "split_version: official_repository_v1"
    echo "label_version: context512_horizon64"
  } > "$run_root/data_version.txt"
  {
    date -Is
    "$PYTHON_BIN" -V
    "$PYTHON_BIN" -m pip freeze
    nvidia-smi
  } > "$run_root/env_snapshot.txt" 2>&1
}

snapshot_run "$TSRAG_RUN"
snapshot_run "$BASELINE_RUN"
snapshot_run "$SMOKE_RUN"
mkdir -p "$SUMMARY_ROOT"

echo "[$(date -Is)] ETTh1 smoke test"
BATCH_SIZE_CANDIDATES="8 4 1" "$CODE_ROOT/script/run_inference_reproduction.sh" \
  tsrag ETTh1 0 "$SMOKE_RUN" 2

run_gpu_stream() {
  local model=$1
  local gpu=$2
  local output=$3
  shift 3
  local dataset
  for dataset in "$@"; do
    "$CODE_ROOT/script/run_inference_reproduction.sh" "$model" "$dataset" "$gpu" "$output"
  done
}

wait_both() {
  local first=$1
  local second=$2
  local first_status=0
  local second_status=0
  wait "$first" || first_status=$?
  wait "$second" || second_status=$?
  if [[ "$first_status" -ne 0 || "$second_status" -ne 0 ]]; then
    echo "GPU streams failed: gpu0_status=$first_status gpu1_status=$second_status" >&2
    return 1
  fi
}

echo "[$(date -Is)] full TS-RAG evaluation"
run_gpu_stream tsrag 0 "$TSRAG_RUN" ETTh1 ETTm1 weather electricity &
pid0=$!
run_gpu_stream tsrag 1 "$TSRAG_RUN" ETTh2 ETTm2 exchange_rate &
pid1=$!
wait_both "$pid0" "$pid1"

echo "[$(date -Is)] derive immutable baseline views"
"$PYTHON_BIN" "$CODE_ROOT/prepare_baseline_views.py" \
  --data-root "$PROJECT_ROOT/Data" \
  --output-root "$BASELINE_ROOT" \
  > "$BASELINE_RUN/logs/prepare_baseline_views.log" 2>&1
cp "$BASELINE_ROOT/manifest.json" "$BASELINE_RUN/baseline_view_manifest.json"

echo "[$(date -Is)] full Chronos-Bolt baseline evaluation"
run_gpu_stream chronos_bolt 0 "$BASELINE_RUN" ETTh1 ETTm1 weather electricity &
pid0=$!
run_gpu_stream chronos_bolt 1 "$BASELINE_RUN" ETTh2 ETTm2 exchange_rate &
pid1=$!
wait_both "$pid0" "$pid1"

"$PYTHON_BIN" "$CODE_ROOT/summarize_reproduction.py" \
  --tsrag-run "$TSRAG_RUN" \
  --baseline-run "$BASELINE_RUN" \
  --output "$SUMMARY_ROOT" \
  > "$SUMMARY_ROOT/summary.log" 2>&1

cp "$SUMMARY_ROOT/summary.md" "$TSRAG_RUN/summary.md"
cp "$SUMMARY_ROOT/summary.md" "$BASELINE_RUN/summary.md"
echo "[$(date -Is)] reproduction complete"
