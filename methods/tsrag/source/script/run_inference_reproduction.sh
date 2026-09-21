#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model is required: tsrag or chronos_bolt}
DATASET=${2:?dataset is required}
GPU=${3:?gpu index is required}
OUTPUT=${4:?output directory is required}
MAX_BATCHES=${5:-}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CODE_ROOT="$PROJECT_ROOT/TS-RAG"
PYTHON_BIN=${PYTHON_BIN:-/nfs/dataset-ofs-661-142656/qiminhao/Env/dynamo_env/bin/python}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/Data}
BASELINE_ROOT=${BASELINE_ROOT:-$PROJECT_ROOT/artifacts/baseline_raw_v1}
NUM_WORKERS=${NUM_WORKERS:-4}
export USE_TF=0
export TRANSFORMERS_NO_TF=1

run_at_batch_size() {
  local batch_size=$1
  local -a command=(
    "$PYTHON_BIN" "$CODE_ROOT/inference_reproduce.py"
    --model "$MODEL"
    --dataset "$DATASET"
    --gpu 0
    --batch-size "$batch_size"
    --num-workers "$NUM_WORKERS"
    --data-root "$DATA_ROOT"
    --baseline-root "$BASELINE_ROOT"
    --output "$OUTPUT/$DATASET"
  )
  if [[ "$MODEL" == "tsrag" && -n "${CHECKPOINT_OVERRIDE:-}" ]]; then
    command+=(--checkpoint "$CHECKPOINT_OVERRIDE")
  fi
  if [[ -n "$MAX_BATCHES" ]]; then
    command+=(--max-batches "$MAX_BATCHES")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}"
}

mkdir -p "$OUTPUT/$DATASET/logs"
if [[ -f "$OUTPUT/$DATASET/metrics.json" ]] && "$PYTHON_BIN" - "$OUTPUT/$DATASET/metrics.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)
PY
then
  echo "[$(date -Is)] already complete: model=$MODEL dataset=$DATASET"
  exit 0
fi

for batch_size in ${BATCH_SIZE_CANDIDATES:-256 128 64 32 16 8 4 1}; do
  echo "[$(date -Is)] model=$MODEL dataset=$DATASET gpu=$GPU batch_size=$batch_size"
  if run_at_batch_size "$batch_size" 2>&1 | tee "$OUTPUT/$DATASET/logs/batch_${batch_size}.log"; then
    exit 0
  fi
  if ! grep -Eqi "out of memory|CUDA error: out of memory" "$OUTPUT/$DATASET/logs/batch_${batch_size}.log"; then
    echo "Non-OOM failure; refusing to change benchmark execution." >&2
    exit 1
  fi
  "$PYTHON_BIN" - <<'PY'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY
done

echo "All batch sizes failed with OOM" >&2
exit 1
