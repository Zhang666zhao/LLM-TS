#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CODE_ROOT="$PROJECT_ROOT/TS-RAG"
PYTHON_BIN=${PYTHON_BIN:-/nfs/dataset-ofs-661-142656/qiminhao/Env/dynamo_env/bin/python}
CONFIG=${CONFIG:-$CODE_ROOT/configs/train_arm_official.yaml}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/training_v1}
SMOKE_RUN=${SMOKE_RUN:-$PROJECT_ROOT/runs/20260817_TSRAG_ARM_smoke20_loss}
TRAIN_RUN=${TRAIN_RUN:-$PROJECT_ROOT/runs/20260817_TSRAG_ARM_official10k_mse-mae}
MAX_WALL_SECONDS=${MAX_WALL_SECONDS:-21600}
export USE_TF=0
export TRANSFORMERS_NO_TF=1

snapshot_run() {
  local run_root=$1
  mkdir -p "$run_root/logs" "$run_root/diagnostics" "$run_root/checkpoints"
  cp "$CONFIG" "$run_root/config.yaml"
  cp "$CODE_ROOT/script/launch_arm_training.sh" "$run_root/run.sh"
  git -C "$PROJECT_ROOT" rev-parse HEAD > "$run_root/git_commit.txt"
  {
    echo "download_manifest_sha256: $(sha256sum "$PROJECT_ROOT/Data/download_manifest.json" | awk '{print $1}')"
    echo "training_manifest: $ARTIFACT_ROOT/training_data_manifest.json"
    echo "training_data_rows: 28013980"
    echo "retrieval_rows: 2792864"
    echo "benchmark_split_version: frozen_inference_v1"
  } > "$run_root/data_version.txt"
  {
    date -Is
    "$PYTHON_BIN" -V
    "$PYTHON_BIN" -m pip freeze
    nvidia-smi
  } > "$run_root/env_snapshot.txt" 2>&1
}

echo "[$(date -Is)] validate data and build retrieval store"
mkdir -p "$ARTIFACT_ROOT"
"$PYTHON_BIN" "$CODE_ROOT/build_training_artifacts.py" \
  --data-root "$PROJECT_ROOT/Data" \
  --output-root "$ARTIFACT_ROOT" \
  > "$ARTIFACT_ROOT/build.log" 2>&1

snapshot_run "$SMOKE_RUN"
echo "[$(date -Is)] start 20-update smoke"
set +e
"$PYTHON_BIN" "$CODE_ROOT/train_arm_reproduce.py" \
  --config "$CONFIG" \
  --project-root "$PROJECT_ROOT" \
  --run-dir "$SMOKE_RUN" \
  --max-updates 20 \
  --draft \
  > "$SMOKE_RUN/logs/console.log" 2>&1
smoke_status=$?
set -e
if [[ "$smoke_status" -ne 0 ]]; then
  if grep -Eqi "out of memory|CUDA error: out of memory" "$SMOKE_RUN/logs/console.log"; then
    fallback="$PROJECT_ROOT/runs/20260817_TSRAG_ARM_smoke20_accum2_loss"
    snapshot_run "$fallback"
    "$PYTHON_BIN" "$CODE_ROOT/train_arm_reproduce.py" \
      --config "$CONFIG" \
      --project-root "$PROJECT_ROOT" \
      --run-dir "$fallback" \
      --max-updates 20 \
      --batch-size 128 \
      --gradient-accumulation 2 \
      --draft \
      > "$fallback/logs/console.log" 2>&1
    SMOKE_RUN="$fallback"
  else
    echo "Smoke failed for a non-OOM reason" >&2
    exit "$smoke_status"
  fi
fi

"$PYTHON_BIN" - "$SMOKE_RUN/metrics.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d["status"] == "draft_complete", d
assert d["completed_updates"] == 20, d
assert d["trainable_parameters"] == 4775425, d
assert d["frozen_unchanged"] and d["trainable_changed"], d
assert d["state_key_count"] == 287, d
PY

echo "[$(date -Is)] validate smoke checkpoint with frozen inference"
smoke_checkpoint=$("$PYTHON_BIN" - "$SMOKE_RUN/metrics.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["post_checkpoint"])
PY
)
CHECKPOINT_OVERRIDE="$smoke_checkpoint" BATCH_SIZE_CANDIDATES="8 4 1" \
  "$CODE_ROOT/script/run_inference_reproduction.sh" \
  tsrag ETTh1 0 "$SMOKE_RUN/inference_validation" 1 \
  > "$SMOKE_RUN/logs/inference_validation.log" 2>&1

snapshot_run "$TRAIN_RUN"
echo "[$(date -Is)] start official 10000-update training"
set +e
timeout --signal=TERM --kill-after=120 "$MAX_WALL_SECONDS" \
  "$PYTHON_BIN" "$CODE_ROOT/train_arm_reproduce.py" \
    --config "$CONFIG" \
    --project-root "$PROJECT_ROOT" \
    --run-dir "$TRAIN_RUN" \
    > "$TRAIN_RUN/logs/console.log" 2>&1
train_status=$?
set -e
if [[ "$train_status" -ne 0 ]]; then
  echo "Official training failed or exceeded the wall-time budget: status=$train_status" >&2
  exit "$train_status"
fi

official_checkpoint=$("$PYTHON_BIN" - "$TRAIN_RUN/metrics.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d["status"] == "complete", d
assert d["completed_updates"] == 10000, d
assert d["frozen_unchanged"] and d["trainable_changed"], d
assert d["state_key_count"] == 287, d
print(d["official_checkpoint"])
PY
)

echo "[$(date -Is)] evaluate fixed official-semantics checkpoint"
EVAL_ROOT="$TRAIN_RUN/evaluation"
run_stream() {
  local gpu=$1
  shift
  local dataset
  for dataset in "$@"; do
    CHECKPOINT_OVERRIDE="$official_checkpoint" \
      "$CODE_ROOT/script/run_inference_reproduction.sh" tsrag "$dataset" "$gpu" "$EVAL_ROOT"
  done
}
run_stream 0 ETTh1 ETTm1 weather electricity &
pid0=$!
run_stream 1 ETTh2 ETTm2 exchange_rate &
pid1=$!
status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
  echo "Evaluation streams failed: gpu0=$status0 gpu1=$status1" >&2
  exit 1
fi

"$PYTHON_BIN" "$CODE_ROOT/summarize_reproduction.py" \
  --tsrag-run "$EVAL_ROOT" \
  --baseline-run "$PROJECT_ROOT/runs/20260817_ChronosBolt_baseline_inference" \
  --output "$TRAIN_RUN/evaluation_summary" \
  > "$TRAIN_RUN/logs/evaluation_summary.log" 2>&1

cp "$TRAIN_RUN/evaluation_summary/summary.md" "$TRAIN_RUN/summary.md"
"$PYTHON_BIN" - "$TRAIN_RUN/metrics.json" "$EVAL_ROOT/metrics.json" <<'PY'
import json, sys
train_path, eval_path = sys.argv[1:]
train = json.load(open(train_path))
evaluation = json.load(open(eval_path))
train["evaluation"] = {
    "average_mse": evaluation["metric_value"],
    "average_mae": evaluation["secondary_metrics"]["mae"],
    "all_within_tolerance": evaluation["all_within_tolerance"],
    "datasets": evaluation["datasets"],
}
train["status"] = "keep" if evaluation["all_within_tolerance"] else "failed_acceptance"
open(train_path, "w").write(json.dumps(train, indent=2, sort_keys=True) + "\n")
PY
echo "[$(date -Is)] ARM training reproduction complete"
