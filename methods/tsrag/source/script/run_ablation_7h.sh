#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CODE_ROOT="$PROJECT_ROOT/TS-RAG"
PYTHON_BIN=${PYTHON_BIN:-/nfs/dataset-ofs-661-142656/qiminhao/Env/dynamo_env/bin/python}
CONFIG=${CONFIG:-$CODE_ROOT/configs/ablation_7h.yaml}
FULL_ROOT=${FULL_ROOT:-$PROJECT_ROOT/runs/20260826_TSRAG_ablation7h}
SMOKE_ROOT=${SMOKE_ROOT:-$PROJECT_ROOT/runs/20260826_TSRAG_ablation7h_smoke}
RUN_ROOT=${RUN_ROOT:-$FULL_ROOT}
TOTAL_WALL_SECONDS=${TOTAL_WALL_SECONDS:-25200}
BODY_WALL_SECONDS=${BODY_WALL_SECONDS:-24000}
NO_NEW_ROUND_SECONDS=${NO_NEW_ROUND_SECONDS:-21600}
MONITOR_SECONDS=${MONITOR_SECONDS:-600}
MIN_FREE_KB=${MIN_FREE_KB:-524288000}
ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/training_v1"
OFFICIAL_CHECKPOINT="$PROJECT_ROOT/Data/TS-RAG-ChronosBolt/pytorch_model.bin"
BASELINE_ROOT="$PROJECT_ROOT/artifacts/baseline_raw_v1"
export USE_TF=0 TRANSFORMERS_NO_TF=1 PYTHONUNBUFFERED=1

usage() {
  echo "usage: $0 smoke|start|status|tail|stop" >&2
}

now_epoch() { date +%s; }
now_iso() { date -Is; }

write_status() {
  local directory=$1 status=$2 message=${3:-}
  mkdir -p "$directory"
  "$PYTHON_BIN" - "$directory/status.json" "$status" "$message" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, status, message = sys.argv[1:]
payload = {"status": status, "message": message, "updated_at": datetime.now(timezone.utc).astimezone().isoformat()}
temporary = path + ".tmp"
open(temporary, "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

preflight() {
  cd "$PROJECT_ROOT"
  [[ -z "$(git status --short)" ]] || { echo "dirty worktree" >&2; git status --short; return 2; }
  [[ -f Data/DOWNLOAD_COMPLETE && -f Data/download_manifest.json ]]
  [[ -f "$OFFICIAL_CHECKPOINT" ]]
  [[ -f "$ARTIFACT_ROOT/training_data_manifest.json" ]]
  [[ -f "$ARTIFACT_ROOT/retrieval_sequences_f32.npy" ]]
  [[ $(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "NVIDIA RTX A6000") -ge 2 ]]
  local free_kb
  free_kb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
  (( free_kb >= MIN_FREE_KB )) || { echo "insufficient disk: ${free_kb}KB" >&2; return 3; }
  "$PYTHON_BIN" "$CODE_ROOT/test_ablation_logic.py"
}

snapshot_suite() {
  mkdir -p "$RUN_ROOT"/{logs,experiments,anchors,diagnostics}
  cp "$CONFIG" "$RUN_ROOT/config.yaml"
  cp "$0" "$RUN_ROOT/run.sh"
  git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_ROOT/git_commit.txt"
  git -C "$PROJECT_ROOT" status --short > "$RUN_ROOT/git_status.txt"
  sha256sum "$PROJECT_ROOT/Data/download_manifest.json" "$OFFICIAL_CHECKPOINT" > "$RUN_ROOT/data_version.txt"
  {
    now_iso
    "$PYTHON_BIN" -V
    "$PYTHON_BIN" -m pip freeze
    nvidia-smi
  } > "$RUN_ROOT/env_snapshot.txt" 2>&1
  cat > "$RUN_ROOT/experiment_registry.tsv" <<'EOF'
experiment	kind	mixer	intervention	top_k	seed	updates
causal_correct	inference	official	correct	10	2021	0
causal_zero	inference	official	zero	10	2021	0
causal_shuffle	inference	official	shuffle_future	10	2021	0
causal_repeat_top1	inference	official	repeat_top1	10	2021	0
causal_topk1	inference	official	correct	1	2021	0
uniform_s2021_5k	training	uniform	correct	10	2021	5000
raw_softmax_s2021_5k	training	raw_softmax	correct	10	2021	5000
full_arm_s2022_10k	training	official	correct	10	2022	10000
distance_aware_s2021_10k	training	distance_aware	correct	10	2021	10000
distance_aware_s2022_10k	training	distance_aware	correct	10	2022	10000
distance_aware_s2023_10k	training	distance_aware	correct	10	2023	10000
EOF
}

is_safety_failure() {
  local log=$1
  grep -Eqi 'Non-finite|indices out of range|Frozen backbone parameters changed|checkpoint mismatch|state keys|No space left|insufficient disk' "$log"
}

run_logged() {
  local gpu=$1 label=$2 log=$3
  shift 3
  mkdir -p "$(dirname "$log")"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" stdbuf -oL -eL "$@" 2>&1 \
    | sed -u "s|^|[GPU${gpu}][${label}] |" \
    | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

evaluate_dataset() {
  local experiment=$1 model=$2 mixer=$3 intervention=$4 top_k=$5 checkpoint=$6 gpu=$7 dataset=$8 max_batches=${9:-} seed=${10:-2021}
  local output="$RUN_ROOT/experiments/$experiment/evaluation/$dataset"
  local expected=complete
  [[ -n "$max_batches" ]] && expected=smoke_complete
  if [[ -f "$output/metrics.json" ]] && grep -q "\"status\": \"$expected\"" "$output/metrics.json"; then
    echo "[$(now_iso)] [GPU${gpu}][${experiment}] skip completed dataset=$dataset"
    return 0
  fi
  mkdir -p "$output/logs"
  local batch status=1
  for batch in 256 128 64 32 16 8 4 1; do
    local command=("$PYTHON_BIN" "$CODE_ROOT/inference_reproduce.py"
      --model "$model" --dataset "$dataset" --gpu 0 --batch-size "$batch"
      --output "$output" --mixer-variant "$mixer"
      --retrieval-intervention "$intervention" --effective-top-k "$top_k"
      --seed "$seed" --save-sample-errors)
    [[ "$model" == tsrag ]] && command+=(--checkpoint "$checkpoint")
    [[ -n "$max_batches" ]] && command+=(--max-batches "$max_batches")
    if run_logged "$gpu" "$experiment/$dataset" "$output/logs/batch_${batch}.log" "${command[@]}"; then
      status=0
      break
    fi
    if ! grep -Eqi 'out of memory|CUDA error: out of memory' "$output/logs/batch_${batch}.log"; then
      break
    fi
  done
  if (( status != 0 )) && grep -Eqi 'out of memory|CUDA error: out of memory' "$output/logs/batch_${batch}.log"; then
    touch "$RUN_ROOT/SAFETY_STOP"
  fi
  return "$status"
}

evaluate_experiment() {
  local experiment=$1 model=$2 mixer=$3 intervention=$4 top_k=$5 checkpoint=$6 gpu=$7 max_batches=${8:-} seed=${9:-2021}
  local exp_root="$RUN_ROOT/experiments/$experiment"
  mkdir -p "$exp_root/evaluation"
  write_status "$exp_root" evaluating "gpu=$gpu"
  local failed=0 dataset
  local datasets=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity exchange_rate)
  [[ -n "$max_batches" ]] && datasets=(ETTh1)
  for dataset in "${datasets[@]}"; do
    if ! evaluate_dataset "$experiment" "$model" "$mixer" "$intervention" "$top_k" "$checkpoint" "$gpu" "$dataset" "$max_batches" "$seed"; then
      failed=1
      local log="$exp_root/evaluation/$dataset/logs"
      if find "$log" -type f -maxdepth 1 -print0 2>/dev/null | xargs -0 grep -Eqi 'Non-finite|indices out of range|checkpoint mismatch|state keys'; then
        touch "$RUN_ROOT/SAFETY_STOP"
      fi
      break
    fi
  done
  if (( failed )); then
    write_status "$exp_root" failed "evaluation failure"
    return 1
  fi
  [[ -n "$max_batches" ]] || write_status "$exp_root" complete "evaluation complete"
}

train_experiment() {
  local experiment=$1 mixer=$2 seed=$3 updates=$4 gpu=$5 screening=$6 max_batches=${7:-}
  local exp_root="$RUN_ROOT/experiments/$experiment"
  if [[ -f "$exp_root/status.json" ]] && grep -q '"status": "complete"' "$exp_root/status.json"; then
    echo "[$(now_iso)] [GPU${gpu}][${experiment}] skip completed experiment"
    return 0
  fi
  if [[ -f "$exp_root/status.json" ]] && grep -q '"status": "failed"' "$exp_root/status.json" \
      && [[ "${RETRY_FAILED:-0}" != 1 ]]; then
    echo "[$(now_iso)] [GPU${gpu}][${experiment}] skip failed experiment (set RETRY_FAILED=1 to retry)"
    return 1
  fi
  mkdir -p "$exp_root"
  cp "$CONFIG" "$exp_root/config.yaml"
  printf '%q ' "$PYTHON_BIN" "$CODE_ROOT/train_arm_reproduce.py" --mixer-variant "$mixer" --seed "$seed" --max-updates "$updates" > "$exp_root/command.txt"
  printf '\n' >> "$exp_root/command.txt"
  write_status "$exp_root" training "gpu=$gpu attempt=1"
  local attempt=1 train_root="$exp_root/train_attempt1" status=1
  if [[ -e "$train_root" ]]; then
    train_root="$exp_root/train_restart_$(date +%Y%m%dT%H%M%S)_attempt1"
  fi
  mkdir -p "$train_root/logs"
  local command=("$PYTHON_BIN" "$CODE_ROOT/train_arm_reproduce.py"
    --config "$CONFIG" --project-root "$PROJECT_ROOT" --run-dir "$train_root"
    --max-updates "$updates" --gpu 0 --seed "$seed" --mixer-variant "$mixer")
  [[ "$screening" == 1 ]] && command+=(--draft)
  if run_logged "$gpu" "$experiment/train" "$train_root/logs/console.log" "${command[@]}"; then
    status=0
  elif grep -Eqi 'out of memory|CUDA error: out of memory' "$train_root/logs/console.log"; then
    attempt=2
    train_root="${train_root%_attempt1}_attempt2_accum2"
    mkdir -p "$train_root/logs"
    write_status "$exp_root" training "gpu=$gpu attempt=2 batch=128 accumulation=2"
    command=("$PYTHON_BIN" "$CODE_ROOT/train_arm_reproduce.py"
      --config "$CONFIG" --project-root "$PROJECT_ROOT" --run-dir "$train_root"
      --max-updates "$updates" --batch-size 128 --gradient-accumulation 2
      --gpu 0 --seed "$seed" --mixer-variant "$mixer")
    [[ "$screening" == 1 ]] && command+=(--draft)
    if run_logged "$gpu" "$experiment/train-retry" "$train_root/logs/console.log" "${command[@]}"; then
      status=0
    elif grep -Eqi 'out of memory|CUDA error: out of memory' "$train_root/logs/console.log"; then
      touch "$RUN_ROOT/SAFETY_STOP"
    fi
  fi
  if (( status != 0 )); then
    is_safety_failure "$train_root/logs/console.log" && touch "$RUN_ROOT/SAFETY_STOP"
    write_status "$exp_root" failed "training failed attempt=$attempt"
    return 1
  fi
  echo "$train_root" > "$exp_root/selected_train_dir.txt"
  local checkpoint
  checkpoint=$("$PYTHON_BIN" - "$train_root/metrics.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d["official_checkpoint"] or d["post_checkpoint"])
PY
)
  sha256sum "$checkpoint" > "$exp_root/checkpoint_sha256.txt"
  evaluate_experiment "$experiment" tsrag "$mixer" correct 10 "$checkpoint" "$gpu" "$max_batches" "$seed"
}

run_two_commands() {
  local round=$1 command0=$2 command1=$3
  echo "[$(now_iso)] ROUND_START $round"
  write_status "$RUN_ROOT" running "$round"
  ( eval "$command0" ) & local pid0=$!
  ( eval "$command1" ) & local pid1=$!
  echo "$pid0 $pid1" > "$RUN_ROOT/active_pids.txt"
  wait "$pid0" || true
  wait "$pid1" || true
  : > "$RUN_ROOT/active_pids.txt"
  echo "[$(now_iso)] ROUND_END $round"
  [[ ! -f "$RUN_ROOT/SAFETY_STOP" ]]
}

monitor_loop() {
  while true; do
    local now start elapsed remaining
    now=$(now_epoch); start=$(cat "$RUN_ROOT/start_epoch.txt"); elapsed=$((now-start)); remaining=$((TOTAL_WALL_SECONDS-elapsed))
    echo "[$(now_iso)] HEARTBEAT elapsed=${elapsed}s remaining=${remaining}s"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || true
    df -h "$PROJECT_ROOT" | tail -n 1 || true
    [[ -f "$RUN_ROOT/active_pids.txt" ]] && echo "active_pids=$(cat "$RUN_ROOT/active_pids.txt")"
    find "$RUN_ROOT/experiments" -path '*/logs/train.jsonl' -type f -print0 2>/dev/null \
      | while IFS= read -r -d '' file; do echo "latest[$file]=$(tail -n 1 "$file")"; done
    sleep "$MONITOR_SECONDS"
  done
}

elapsed_seconds() { echo $(( $(now_epoch) - $(cat "$RUN_ROOT/start_epoch.txt") )); }
can_start_round() { (( $(elapsed_seconds) < NO_NEW_ROUND_SECONDS )); }

round1_gpu0() {
  local anchor="$RUN_ROOT/anchors/chronos_bolt_sample_errors"
  local dataset
  for dataset in ETTh1 ETTh2 ETTm1 ETTm2 weather electricity exchange_rate; do
    local output="$anchor/$dataset"
    if [[ ! -f "$output/metrics.json" ]] || ! grep -q '"status": "complete"' "$output/metrics.json"; then
      mkdir -p "$output/logs"
      run_logged 0 "anchor/chronos/$dataset" "$output/logs/console.log" \
        "$PYTHON_BIN" "$CODE_ROOT/inference_reproduce.py" --model chronos_bolt --dataset "$dataset" --gpu 0 \
        --batch-size 256 --output "$output" --save-sample-errors || return 1
    fi
  done
  evaluate_experiment causal_correct tsrag official correct 10 "$OFFICIAL_CHECKPOINT" 0
  evaluate_experiment causal_zero tsrag official zero 10 "$OFFICIAL_CHECKPOINT" 0
}

round1_gpu1() {
  evaluate_experiment causal_shuffle tsrag official shuffle_future 10 "$OFFICIAL_CHECKPOINT" 1
  evaluate_experiment causal_repeat_top1 tsrag official repeat_top1 10 "$OFFICIAL_CHECKPOINT" 1
  evaluate_experiment causal_topk1 tsrag official correct 1 "$OFFICIAL_CHECKPOINT" 1
}

suite_body() {
  monitor_loop & local monitor_pid=$!
  trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT TERM INT

  run_two_commands round1_causal "round1_gpu0" "round1_gpu1" || return 90
  can_start_round || return 0
  run_two_commands round2_screening \
    "train_experiment uniform_s2021_5k uniform 2021 5000 0 1" \
    "train_experiment raw_softmax_s2021_5k raw_softmax 2021 5000 1 1" || return 90
  can_start_round || return 0
  run_two_commands round3_matched \
    "train_experiment full_arm_s2022_10k official 2022 10000 0 0" \
    "train_experiment distance_aware_s2021_10k distance_aware 2021 10000 1 0" || return 90
  can_start_round || return 0
  run_two_commands round4_distance_seeds \
    "train_experiment distance_aware_s2022_10k distance_aware 2022 10000 0 0" \
    "train_experiment distance_aware_s2023_10k distance_aware 2023 10000 1 0" || return 90
}

run_suite() {
  echo "$$" > "$RUN_ROOT/orchestrator.pid"
  now_epoch > "$RUN_ROOT/start_epoch.txt"
  snapshot_suite
  write_status "$RUN_ROOT" preflight "starting"
  preflight
  set +e
  timeout --signal=TERM --kill-after=600 "$BODY_WALL_SECONDS" "$0" _body
  local body_status=$?
  set -e
  write_status "$RUN_ROOT" summarizing "body_status=$body_status"
  "$PYTHON_BIN" "$CODE_ROOT/analyze_ablation.py" --suite-root "$RUN_ROOT" \
    > "$RUN_ROOT/logs/analysis.log" 2>&1 || true
  if [[ -f "$RUN_ROOT/SAFETY_STOP" ]]; then
    write_status "$RUN_ROOT" safety_stopped "body_status=$body_status"
  elif [[ "$body_status" -eq 0 ]]; then
    write_status "$RUN_ROOT" complete "suite finished"
  else
    write_status "$RUN_ROOT" partial "body_status=$body_status"
  fi
  echo "[$(now_iso)] suite finished body_status=$body_status"
}

smoke_suite() {
  RUN_ROOT="$SMOKE_ROOT"
  if [[ -e "$RUN_ROOT" ]]; then
    echo "Smoke directory already exists; set SMOKE_ROOT to a new path: $RUN_ROOT" >&2
    return 2
  fi
  snapshot_suite
  now_epoch > "$RUN_ROOT/start_epoch.txt"
  preflight
  export RUN_ROOT
  run_two_commands smoke_train_a \
    "train_experiment smoke_official official 2021 20 0 1 2" \
    "train_experiment smoke_distance distance_aware 2021 20 1 1 2"
  run_two_commands smoke_train_b \
    "train_experiment smoke_uniform uniform 2021 20 0 1 2" \
    "train_experiment smoke_raw raw_softmax 2021 20 1 1 2"
  run_two_commands smoke_inference_a \
    "evaluate_experiment smoke_zero tsrag official zero 10 '$OFFICIAL_CHECKPOINT' 0 2" \
    "evaluate_experiment smoke_shuffle tsrag official shuffle_future 10 '$OFFICIAL_CHECKPOINT' 1 2"
  run_two_commands smoke_inference_b \
    "evaluate_experiment smoke_repeat tsrag official repeat_top1 10 '$OFFICIAL_CHECKPOINT' 0 2" \
    "evaluate_experiment smoke_topk1 tsrag official correct 1 '$OFFICIAL_CHECKPOINT' 1 2"
  "$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
training = ("smoke_official", "smoke_distance", "smoke_uniform", "smoke_raw")
inference = training + ("smoke_zero", "smoke_shuffle", "smoke_repeat", "smoke_topk1")
for name in training:
    selected = (root / "experiments" / name / "selected_train_dir.txt").read_text().strip()
    metrics = json.load(open(Path(selected) / "metrics.json"))
    assert metrics["status"] == "draft_complete", (name, metrics["status"])
    assert metrics["completed_updates"] == 20, (name, metrics["completed_updates"])
    assert metrics["frozen_unchanged"] and metrics["trainable_changed"], name
for name in inference:
    metrics = json.load(open(root / "experiments" / name / "evaluation" / "ETTh1" / "metrics.json"))
    assert metrics["status"] == "smoke_complete", (name, metrics["status"])
    assert metrics["sample_count"] > 0 and math.isfinite(metrics["mse"]) and math.isfinite(metrics["mae"]), name
print("smoke artifact contracts passed")
PY
  write_status "$RUN_ROOT" complete "all smoke checks passed"
  echo "[$(now_iso)] smoke complete: $RUN_ROOT"
}

action=${1:-}
case "$action" in
  smoke)
    smoke_suite
    ;;
  start)
    mkdir -p "$FULL_ROOT/logs"
    if [[ -f "$FULL_ROOT/orchestrator.pid" ]] && kill -0 "$(cat "$FULL_ROOT/orchestrator.pid")" 2>/dev/null; then
      echo "suite already running pid=$(cat "$FULL_ROOT/orchestrator.pid")" >&2
      exit 2
    fi
    if [[ -f "$FULL_ROOT/status.json" ]] && grep -q '"status": "complete"' "$FULL_ROOT/status.json"; then
      echo "suite already complete: $FULL_ROOT" >&2
      exit 2
    fi
    RUN_ROOT="$FULL_ROOT" nohup setsid timeout --signal=TERM --kill-after=30 "$TOTAL_WALL_SECONDS" \
      "$0" _run > "$FULL_ROOT/logs/master.log" 2>&1 < /dev/null &
    echo $! > "$FULL_ROOT/launcher.pid"
    echo "started pid=$!"
    echo "follow with: $0 tail"
    ;;
  _run)
    RUN_ROOT=${RUN_ROOT:-$FULL_ROOT}
    run_suite
    ;;
  _body)
    RUN_ROOT=${RUN_ROOT:-$FULL_ROOT}
    suite_body
    ;;
  status)
    echo "run_root=$RUN_ROOT"
    [[ -f "$RUN_ROOT/status.json" ]] && cat "$RUN_ROOT/status.json" || echo "status=not_started"
    for file in "$RUN_ROOT"/launcher.pid "$RUN_ROOT"/orchestrator.pid; do
      if [[ -f "$file" ]]; then
        pid=$(cat "$file"); kill -0 "$pid" 2>/dev/null && alive=alive || alive=exited
        echo "$(basename "$file")=$pid $alive"
      fi
    done
    [[ -f "$RUN_ROOT/active_pids.txt" ]] && echo "active_pids=$(cat "$RUN_ROOT/active_pids.txt")"
    if [[ -d "$RUN_ROOT/experiments" ]]; then
      for state in complete draft_complete failed training evaluating; do
        count=$({ find "$RUN_ROOT/experiments" -name status.json -type f -exec grep -l "\"status\": \"$state\"" {} + 2>/dev/null || true; } | wc -l | tr -d ' ')
        echo "$state=$count"
      done
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || true
    [[ -f "$RUN_ROOT/logs/master.log" ]] && tail -n 20 "$RUN_ROOT/logs/master.log"
    ;;
  tail)
    mkdir -p "$RUN_ROOT/logs"
    touch "$RUN_ROOT/logs/master.log"
    tail -n 100 -f "$RUN_ROOT/logs/master.log"
    ;;
  stop)
    [[ -f "$RUN_ROOT/launcher.pid" ]] || { echo "no launcher pid" >&2; exit 2; }
    pid=$(cat "$RUN_ROOT/launcher.pid")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid"
      echo "TERM sent to process group $pid"
    else
      echo "process already exited"
    fi
    ;;
  *) usage; exit 2 ;;
esac
