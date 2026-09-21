#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CODE_ROOT="$PROJECT_ROOT/TS-RAG"
TRAIN_PY=${TRAIN_PY:-/nfs/dataset-ofs-661-142656/qiminhao/Env/dynamo_env/bin/python}
QWEN_BASE_PY=${QWEN_BASE_PY:-/nfs/dataset-ofs-661-142656/qiminhao/Env/qwen_inference/bin/python}
ENV_ROOT=${ENV_ROOT:-/nfs/dataset-ofs-661-142656/qiminhao/Env}
FAISS_ENV=${FAISS_ENV:-$ENV_ROOT/tsrag_faiss_env}
QWEN_FAISS_ENV=${QWEN_FAISS_ENV:-$ENV_ROOT/tsrag_qwen_faiss_env_v2}
FAISS_PY="$FAISS_ENV/bin/python"
QWEN_PY="$QWEN_FAISS_ENV/bin/python"
CONFIG=${CONFIG:-$CODE_ROOT/configs/retrieval_ablation_offline.yaml}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/runs/20260827_TSRAG_retrieval_ablation_offline}
TOPK_RUN_ROOT=${TOPK_RUN_ROOT:-$PROJECT_ROOT/runs/20260901_TSRAG_topk_ablation_official10k}
SMOKE_ROOT=${SMOKE_ROOT:-$PROJECT_ROOT/runs/20260827_TSRAG_retrieval_ablation_smoke}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/retrieval_ablation_v2}
SMOKE_ARTIFACT_ROOT=${SMOKE_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/retrieval_ablation_v2_smoke}
TRAINING_ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/training_v1"
RETRIEVAL_STORE="$TRAINING_ARTIFACT_ROOT/retrieval_sequences_f32.npy"
TRAINING_ROOT="$PROJECT_ROOT/Data/TS-RAG-Data/pretrain_pairs_ctx512"
BASE_MODEL="$PROJECT_ROOT/Data/chronos-bolt-base"
QWEN_MODEL="$ARTIFACT_ROOT/model_cache/Qwen3-Embedding-0.6B"
RUNTIME_TMP=${RUNTIME_TMP:-$PROJECT_ROOT/artifacts/retrieval_ablation_runtime_tmp}
SOCKET_TMP=${SOCKET_TMP:-/dev/shm/tsrag_retrieval_${UID}}
ARTIFACT_SCRIPT="$CODE_ROOT/retrieval_ablation_artifacts.py"
TRAIN_SCRIPT="$CODE_ROOT/train_arm_reproduce.py"
INFERENCE_SCRIPT="$CODE_ROOT/inference_reproduce.py"
REPORT_SCRIPT="$CODE_ROOT/summarize_retrieval_ablation.py"
TOTAL_WALL_SECONDS=${TOTAL_WALL_SECONDS:-259200}
NO_NEW_EXPERIMENT_SECONDS=${NO_NEW_EXPERIMENT_SECONDS:-244800}
TERM_SECONDS=${TERM_SECONDS:-257400}
KILL_SECONDS=${KILL_SECONDS:-258600}
MONITOR_SECONDS=${MONITOR_SECONDS:-1800}
MIN_FREE_KB=${MIN_FREE_KB:-524288000}
NUM_SHARDS=${NUM_SHARDS:-64}
SCHEDULE_ROWS=${SCHEDULE_ROWS:-2560000}
EXPECTED_EXPERIMENTS=${EXPECTED_EXPERIMENTS:-9}
export USE_TF=0 TRANSFORMERS_NO_TF=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$SOCKET_TMP" PIP_CACHE_DIR="$RUNTIME_TMP/pip_cache" HF_HOME="$RUNTIME_TMP/huggingface"

usage() { echo "usage: $0 smoke|start|start-topk|status|status-topk|tail|tail-topk|stop|stop-topk|report|report-topk" >&2; }
now_iso() { date -Is; }
now_epoch() { date +%s; }

write_status() {
  local root=$1 status=$2 stage=$3 message=${4:-}
  mkdir -p "$root"
  "$TRAIN_PY" - "$root/status.json" "$status" "$stage" "$message" <<'PY'
import json,os,sys
from datetime import datetime,timezone
path,status,stage,message=sys.argv[1:]
data={"status":status,"stage":stage,"message":message,"updated_at":datetime.now(timezone.utc).astimezone().isoformat()}
tmp=path+".tmp"; open(tmp,"w").write(json.dumps(data,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}

run_logged() {
  local gpu=$1 label=$2 log=$3
  shift 3
  mkdir -p "$(dirname "$log")"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" stdbuf -oL -eL "$@" 2>&1 | sed -u "s|^|[GPU${gpu}][${label}] |" | tee -a "$log"
  local result=${PIPESTATUS[0]}
  set -e
  return "$result"
}

bootstrap_envs() {
  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"
  [[ -x "$FAISS_PY" ]] || "$TRAIN_PY" -m venv --system-site-packages "$FAISS_ENV"
  if [[ ! -x "$QWEN_PY" ]]; then
    "$QWEN_BASE_PY" -m venv "$QWEN_FAISS_ENV"
    qwen_parent_site=$($QWEN_BASE_PY -c 'import site; print(site.getsitepackages()[0])')
    qwen_child_site=$($QWEN_PY -c 'import site; print(site.getsitepackages()[0])')
    printf '%s\n' "$qwen_parent_site" > "$qwen_child_site/qwen_inference_parent.pth"
  fi
  "$FAISS_PY" -c 'import faiss,numpy; assert hasattr(faiss,"StandardGpuResources"); assert numpy.__version__ == "1.26.4"' 2>/dev/null || \
    "$FAISS_PY" -m pip install --no-input --no-deps numpy==1.26.4 faiss-gpu-cu12==1.12.0
  "$QWEN_PY" -c 'import faiss,gluonts,numpy; assert hasattr(faiss,"StandardGpuResources"); assert numpy.__version__ == "1.26.4"' 2>/dev/null || \
    "$QWEN_PY" -m pip install --no-input --no-deps numpy==1.26.4 faiss-gpu-cu12==1.12.0 gluonts==0.14.4
}

download_qwen() {
  [[ -f "$QWEN_MODEL/model.safetensors" ]] && return 0
  mkdir -p "$(dirname "$QWEN_MODEL")"
  "$QWEN_PY" - "$QWEN_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-Embedding-0.6B",local_dir=sys.argv[1])
PY
}

preflight() {
  cd "$PROJECT_ROOT"
  [[ -z "$(git status --short)" ]] || { echo "dirty worktree" >&2; git status --short; return 2; }
  [[ -f Data/DOWNLOAD_COMPLETE && -f Data/download_manifest.json && -f "$RETRIEVAL_STORE" ]]
  [[ -f "$BASE_MODEL/model.safetensors" ]]
  [[ $(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'NVIDIA RTX A6000') -ge 2 ]]
  local free_kb; free_kb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
  (( free_kb >= MIN_FREE_KB )) || { echo "insufficient disk: ${free_kb}KB" >&2; return 3; }
  "$TRAIN_PY" "$CODE_ROOT/test_retrieval_ablation.py"
}

snapshot_suite() {
  mkdir -p "$RUN_ROOT"/{logs,experiments,diagnostics}
  cp "$CONFIG" "$RUN_ROOT/config.yaml"; cp "$0" "$RUN_ROOT/run.sh"
  git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_ROOT/git_commit.txt"
  git -C "$PROJECT_ROOT" status --short > "$RUN_ROOT/git_status.txt"
  sha256sum "$PROJECT_ROOT/Data/download_manifest.json" "$RETRIEVAL_STORE" > "$RUN_ROOT/data_version.txt"
  { now_iso; "$TRAIN_PY" -V; "$TRAIN_PY" -m pip freeze; "$FAISS_PY" -m pip freeze; "$QWEN_PY" -m pip freeze; nvidia-smi; } \
    > "$RUN_ROOT/env_snapshot.txt" 2>&1
  cat > "$RUN_ROOT/experiment_registry.tsv" <<'EOF'
experiment	gpu	top_k	retrieval	updates	seed
chronos_t5_k1_s2021	0	1	official	10000	2021
chronos_t5_k5_s2021	1	5	official	10000	2021
chronos_t5_k10_s2021	0	10	official	10000	2021
chronos_t5_k15_s2021	1	15	official	10000	2021
chronos_t5_k20_s2021	0	20	official	10000	2021
random_k10_s2021	1	10	random_k10	10000	2021
abs_pearson_k10_s2021	0	10	abs_pearson_k10	10000	2021
chronos_bolt_embed_k10_s2021	1	10	chronos_bolt_k10	10000	2021
qwen3_text_embed_k10_s2021	0	10	qwen3_text_k10	10000	2021
EOF
}

snapshot_topk_suite() {
  mkdir -p "$RUN_ROOT"/{logs,experiments,diagnostics}
  cp "$CONFIG" "$RUN_ROOT/config.yaml"; cp "$0" "$RUN_ROOT/run.sh"
  git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_ROOT/git_commit.txt"
  git -C "$PROJECT_ROOT" status --short > "$RUN_ROOT/git_status.txt"
  sha256sum "$PROJECT_ROOT/Data/download_manifest.json" "$RETRIEVAL_STORE" \
    "$ARTIFACT_ROOT/query_schedule/manifest.json" "$ARTIFACT_ROOT/arm_init_seed2021.pth" > "$RUN_ROOT/data_version.txt"
  { now_iso; "$TRAIN_PY" -V; "$TRAIN_PY" -m pip freeze; nvidia-smi; } > "$RUN_ROOT/env_snapshot.txt" 2>&1
  cat > "$RUN_ROOT/experiment_registry.tsv" <<'EOF'
experiment	gpu	top_k	retrieval	updates	seed
chronos_t5_k1_s2021	0	1	official	10000	2021
chronos_t5_k5_s2021	1	5	official	10000	2021
chronos_t5_k10_s2021	0	10	official	10000	2021
chronos_t5_k15_s2021	1	15	official	10000	2021
chronos_t5_k20_s2021	0	20	official	10000	2021
EOF
}

freeze_core_artifacts() {
  local schedule=$1 rows=$2 root=$3
  mkdir -p "$root"
  "$TRAIN_PY" "$ARTIFACT_SCRIPT" freeze-schedule --training-root "$TRAINING_ROOT" --output "$schedule" \
    --rows "$rows" --batch-size 4096 --shuffle-buffer 10000
  "$TRAIN_PY" "$ARTIFACT_SCRIPT" arm-init --base-model "$BASE_MODEL" --output "$root/arm_init_seed2021.pth"
}

encode_worker() {
  local encoder=$1 source=$2 gpu=$3 parity=$4 py=$5 model=$6 batch=$7
  local shard
  for ((shard=parity; shard<NUM_SHARDS; shard+=2)); do
    local cmd=("$py" "$ARTIFACT_SCRIPT" encode --encoder "$encoder" --source "$source" --retrieval-store "$RETRIEVAL_STORE"
      --schedule "$ARTIFACT_ROOT/query_schedule" --output "$ARTIFACT_ROOT/embeddings" --gpu 0
      --num-shards "$NUM_SHARDS" --shard-index "$shard" --batch-size "$batch")
    [[ -n "$model" ]] && cmd+=(--model-path "$model")
    run_logged "$gpu" "encode/${encoder}/${source}/${shard}" "$RUN_ROOT/logs/encode_${encoder}_${source}_gpu${gpu}.log" "${cmd[@]}" || return 1
  done
}

encode_all() {
  local encoder=$1 py=$2 model=$3 batch=$4 source
  for source in database queries; do
    encode_worker "$encoder" "$source" 0 0 "$py" "$model" "$batch" & local p0=$!
    encode_worker "$encoder" "$source" 1 1 "$py" "$model" "$batch" & local p1=$!
    echo "$p0 $p1" > "$RUN_ROOT/active_pids.txt"; wait "$p0"; wait "$p1"; : > "$RUN_ROOT/active_pids.txt"
  done
}

search_worker() {
  local encoder=$1 artifact=$2 gpu=$3 parity=$4 py=$5 shard
  for ((shard=parity; shard<NUM_SHARDS; shard+=2)); do
    run_logged "$gpu" "search/${encoder}/${shard}" "$RUN_ROOT/logs/search_${encoder}_gpu${gpu}.log" \
      "$py" "$ARTIFACT_SCRIPT" search --encoder "$encoder" --embeddings "$ARTIFACT_ROOT/embeddings" \
      --index "$ARTIFACT_ROOT/indexes/${encoder}.faiss" --audit "$ARTIFACT_ROOT/indexes/${encoder}_audit.json" \
      --query-hashes "$ARTIFACT_ROOT/query_schedule/query_hashes_u64.npy" --database-hashes "$ARTIFACT_ROOT/database_hashes_u64.npy" \
      --output "$ARTIFACT_ROOT/$artifact" --gpu 0 --num-shards "$NUM_SHARDS" --shard-index "$shard" \
      --total-queries "$SCHEDULE_ROWS" || return 1
  done
}

build_method() {
  local encoder=$1 artifact=$2 py=$3 model=$4 batch=$5
  write_status "$RUN_ROOT" running "artifact_$encoder" encoding
  encode_all "$encoder" "$py" "$model" "$batch"
  mkdir -p "$ARTIFACT_ROOT/indexes"
  run_logged 0 "index/$encoder" "$RUN_ROOT/logs/index_${encoder}.log" "$py" "$ARTIFACT_SCRIPT" build-index \
    --encoder "$encoder" --embeddings "$ARTIFACT_ROOT/embeddings" --output "$ARTIFACT_ROOT/indexes/${encoder}.faiss" \
    --gpu 0 --num-shards "$NUM_SHARDS"
  run_logged 0 "audit/$encoder" "$RUN_ROOT/logs/audit_${encoder}.log" "$py" "$ARTIFACT_SCRIPT" audit-index \
    --encoder "$encoder" --embeddings "$ARTIFACT_ROOT/embeddings" --index "$ARTIFACT_ROOT/indexes/${encoder}.faiss" \
    --output "$ARTIFACT_ROOT/indexes/${encoder}_audit.json" --gpu 0 --num-shards "$NUM_SHARDS" --minimum-recall 0.95
  search_worker "$encoder" "$artifact" 0 0 "$py" & local p0=$!
  search_worker "$encoder" "$artifact" 1 1 "$py" & local p1=$!
  echo "$p0 $p1" > "$RUN_ROOT/active_pids.txt"; wait "$p0"; wait "$p1"; : > "$RUN_ROOT/active_pids.txt"
  "$py" "$ARTIFACT_SCRIPT" merge-search --method "$artifact" --output "$ARTIFACT_ROOT/$artifact" \
    --num-shards "$NUM_SHARDS" --total-queries "$SCHEDULE_ROWS"
  "$TRAIN_PY" "$ARTIFACT_SCRIPT" validate --root "$ARTIFACT_ROOT/$artifact"
}

test_view_worker() {
  local method=$1 gpu=$2 py=$3 model=$4; shift 4
  local dataset
  for dataset in "$@"; do
    local cmd=("$py" "$ARTIFACT_SCRIPT" test-view --project-root "$PROJECT_ROOT" --method "$method" --dataset "$dataset"
      --output "$ARTIFACT_ROOT/test_views" --gpu 0 --batch-size 128)
    [[ -n "$model" ]] && cmd+=(--model-path "$model")
    run_logged "$gpu" "test-view/$method/$dataset" "$RUN_ROOT/logs/test_view_${method}_gpu${gpu}.log" "${cmd[@]}" || return 1
  done
}

build_test_views() {
  local method=$1 py=$2 model=$3
  test_view_worker "$method" 0 "$py" "$model" ETTh1 ETTm1 weather electricity & local p0=$!
  test_view_worker "$method" 1 "$py" "$model" ETTh2 ETTm2 exchange_rate & local p1=$!
  echo "$p0 $p1" > "$RUN_ROOT/active_pids.txt"; wait "$p0"; wait "$p1"; : > "$RUN_ROOT/active_pids.txt"
}

prepare_full_artifacts() {
  write_status "$RUN_ROOT" running freeze_schedule "2.56m fixed queries"
  freeze_core_artifacts "$ARTIFACT_ROOT/query_schedule" "$SCHEDULE_ROWS" "$ARTIFACT_ROOT"
  [[ -f "$ARTIFACT_ROOT/database_hashes_u64.npy" ]] || "$TRAIN_PY" "$ARTIFACT_SCRIPT" database-hashes \
    --retrieval-store "$RETRIEVAL_STORE" --output "$ARTIFACT_ROOT/database_hashes_u64.npy"
  "$TRAIN_PY" "$ARTIFACT_SCRIPT" random --schedule "$ARTIFACT_ROOT/query_schedule" \
    --database-hashes "$ARTIFACT_ROOT/database_hashes_u64.npy" --output "$ARTIFACT_ROOT/random_k10"
  build_method pearson abs_pearson_k10 "$FAISS_PY" "" 4096
  build_method bolt chronos_bolt_k10 "$FAISS_PY" "$BASE_MODEL" 512
  build_method qwen qwen3_text_k10 "$QWEN_PY" "$QWEN_MODEL" 32
  build_test_views official "$FAISS_PY" ""
  build_test_views random "$FAISS_PY" ""
  build_test_views pearson "$FAISS_PY" ""
  build_test_views bolt "$FAISS_PY" "$BASE_MODEL"
  build_test_views qwen "$QWEN_PY" "$QWEN_MODEL"
}

evaluate_dataset() {
  local experiment=$1 checkpoint=$2 top_k=$3 artifact=$4 gpu=$5 dataset=$6 max_batches=${7:-}
  local output="$RUN_ROOT/experiments/$experiment/evaluation/$dataset" expected=complete
  [[ -n "$max_batches" ]] && expected=smoke_complete
  [[ -f "$output/metrics.json" ]] && grep -q "\"status\": \"$expected\"" "$output/metrics.json" && return 0
  mkdir -p "$output/logs"
  local batch status=1
  for batch in 256 128 64 32 16 8 4 1; do
    local cmd=("$TRAIN_PY" "$INFERENCE_SCRIPT" --model tsrag --dataset "$dataset" --gpu 0 --batch-size "$batch"
      --output "$output" --checkpoint "$checkpoint" --effective-top-k "$top_k" --seed 2021 --save-sample-errors)
    if [[ "$artifact" != official ]]; then
      local method=${artifact%_k10}; [[ "$method" == abs_pearson ]] && method=pearson
      [[ "$method" == chronos_bolt ]] && method=bolt; [[ "$method" == qwen3_text ]] && method=qwen
      local view_root="$ARTIFACT_ROOT/test_views/$method/$dataset"
      [[ -n "$max_batches" ]] && view_root="$SMOKE_ARTIFACT_ROOT/test_views/$method/$dataset"
      cmd+=(--retrieval-artifact "$view_root")
    fi
    [[ -n "$max_batches" ]] && cmd+=(--max-batches "$max_batches")
    if run_logged "$gpu" "$experiment/eval/$dataset" "$output/logs/batch_${batch}.log" "${cmd[@]}"; then status=0; break; fi
    grep -Eqi 'out of memory|CUDA error: out of memory' "$output/logs/batch_${batch}.log" || break
  done
  return "$status"
}

train_experiment() {
  local experiment=$1 top_k=$2 artifact=$3 gpu=$4 updates=${5:-10000} max_batches=${6:-}
  local exp_root="$RUN_ROOT/experiments/$experiment"
  [[ -f "$exp_root/status.json" ]] && grep -q '"status": "complete"' "$exp_root/status.json" && return 0
  mkdir -p "$exp_root"; write_status "$exp_root" running training "gpu=$gpu top_k=$top_k retrieval=$artifact"
  local train_root="$exp_root/train_attempt1" schedule="$ARTIFACT_ROOT/query_schedule" init="$ARTIFACT_ROOT/arm_init_seed2021.pth"
  [[ ! -e "$train_root" ]] || train_root="$exp_root/train_restart_$(date +%Y%m%dT%H%M%S)_attempt1"
  [[ -z "$max_batches" ]] || { schedule="$SMOKE_ARTIFACT_ROOT/query_schedule"; init="$SMOKE_ARTIFACT_ROOT/arm_init_seed2021.pth"; }
  mkdir -p "$train_root/logs"
  local cmd=("$TRAIN_PY" "$TRAIN_SCRIPT" --config "$CONFIG" --project-root "$PROJECT_ROOT" --run-dir "$train_root"
    --max-updates "$updates" --gpu 0 --seed 2021 --mixer-variant official --top-k "$top_k"
    --query-schedule "$schedule" --arm-init-checkpoint "$init")
  if [[ "$artifact" != official ]]; then
    local side_root="$ARTIFACT_ROOT"; [[ -z "$max_batches" ]] || side_root="$SMOKE_ARTIFACT_ROOT"
    cmd+=(--retrieval-artifact "$side_root/$artifact")
  fi
  local status=1
  if run_logged "$gpu" "$experiment/train" "$train_root/logs/console.log" "${cmd[@]}"; then status=0
  elif grep -Eqi 'out of memory|CUDA error: out of memory' "$train_root/logs/console.log"; then
    train_root="$exp_root/train_attempt2_accum2"; mkdir -p "$train_root/logs"
    cmd=("$TRAIN_PY" "$TRAIN_SCRIPT" --config "$CONFIG" --project-root "$PROJECT_ROOT" --run-dir "$train_root"
      --max-updates "$updates" --batch-size 128 --gradient-accumulation 2 --gpu 0 --seed 2021 --mixer-variant official
      --top-k "$top_k" --query-schedule "$schedule" --arm-init-checkpoint "$init")
    [[ "$artifact" == official ]] || cmd+=(--retrieval-artifact "$side_root/$artifact")
    run_logged "$gpu" "$experiment/train-retry" "$train_root/logs/console.log" "${cmd[@]}" && status=0
  fi
  if (( status != 0 )); then
    write_status "$exp_root" failed training "nonzero exit"
    if grep -Eqi 'Non-finite|indices out of range|Frozen backbone parameters changed|checkpoint mismatch|state keys|Retrieval signs' "$train_root/logs/console.log"; then
      touch "$RUN_ROOT/SAFETY_STOP"
    fi
    return 1
  fi
  echo "$train_root" > "$exp_root/selected_train_dir.txt"
  local checkpoint
  checkpoint=$($TRAIN_PY - "$train_root/metrics.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(d.get("official_checkpoint") or d["post_checkpoint"])
PY
)
  sha256sum "$checkpoint" > "$exp_root/checkpoint_sha256.txt"
  local datasets=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity exchange_rate); [[ -z "$max_batches" ]] || datasets=(ETTh1)
  local dataset
  for dataset in "${datasets[@]}"; do
    evaluate_dataset "$experiment" "$checkpoint" "$top_k" "$artifact" "$gpu" "$dataset" "$max_batches" || \
      { write_status "$exp_root" failed evaluation "$dataset"; return 1; }
  done
  write_status "$exp_root" complete complete "training and evaluation complete"
}

elapsed_seconds() { echo $(( $(now_epoch) - $(cat "$RUN_ROOT/start_epoch.txt") )); }
can_start_experiment() { [[ ! -f "$RUN_ROOT/SAFETY_STOP" ]] && (( $(elapsed_seconds) < NO_NEW_EXPERIMENT_SECONDS )); }
gpu0_queue() {
  train_experiment chronos_t5_k1_s2021 1 official 0 || true
  can_start_experiment && train_experiment chronos_t5_k10_s2021 10 official 0 || true
  can_start_experiment && train_experiment chronos_t5_k20_s2021 20 official 0 || true
  can_start_experiment && train_experiment abs_pearson_k10_s2021 10 abs_pearson_k10 0 || true
  can_start_experiment && train_experiment qwen3_text_embed_k10_s2021 10 qwen3_text_k10 0 || true
}
gpu1_queue() {
  train_experiment chronos_t5_k5_s2021 5 official 1 || true
  can_start_experiment && train_experiment chronos_t5_k15_s2021 15 official 1 || true
  can_start_experiment && train_experiment random_k10_s2021 10 random_k10 1 || true
  can_start_experiment && train_experiment chronos_bolt_embed_k10_s2021 10 chronos_bolt_k10 1 || true
}

monitor_loop() {
  while true; do
    local elapsed remaining; elapsed=$(elapsed_seconds); remaining=$((TOTAL_WALL_SECONDS-elapsed))
    echo "[$(now_iso)][MONITOR] elapsed=${elapsed}s remaining=${remaining}s"
    [[ -f "$RUN_ROOT/status.json" ]] && echo "suite=$(tr -d '\n' < "$RUN_ROOT/status.json")"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || true
    free -h || true; df -h "$PROJECT_ROOT" | tail -n 1 || true
    [[ -f "$RUN_ROOT/active_pids.txt" ]] && echo "active_pids=$(cat "$RUN_ROOT/active_pids.txt")"
    find "$RUN_ROOT/experiments" -path '*/logs/train.jsonl' -type f -print0 2>/dev/null | \
      while IFS= read -r -d '' f; do echo "latest[$f]=$(tail -n 1 "$f")"; done
    { find "$RUN_ROOT/logs" "$RUN_ROOT/experiments" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | \
      sort -nr | head -n 6 | cut -d' ' -f2- | while IFS= read -r f; do echo "tail[$f]=$(tail -n 1 "$f")"; done; } || true
    local completed failed pending
    completed=$({ grep -Rl '"status": "complete"' "$RUN_ROOT/experiments"/*/status.json 2>/dev/null || true; } | wc -l)
    failed=$({ grep -Rl '"status": "failed"' "$RUN_ROOT/experiments"/*/status.json 2>/dev/null || true; } | wc -l)
    pending=$((EXPECTED_EXPERIMENTS-completed-failed)); if (( pending < 0 )); then pending=0; fi
    echo "experiments completed=$completed failed=$failed pending=$pending"
    local free_kb; free_kb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if (( free_kb < MIN_FREE_KB )) || grep -ERqi 'Non-finite|indices out of range|Frozen backbone parameters changed|checkpoint mismatch' "$RUN_ROOT/logs" "$RUN_ROOT/experiments" 2>/dev/null; then
      touch "$RUN_ROOT/SAFETY_STOP"
      [[ -f "$RUN_ROOT/active_pids.txt" ]] && xargs -r kill -TERM < "$RUN_ROOT/active_pids.txt" 2>/dev/null || true
    fi
    date -Is > "$RUN_ROOT/last_monitor.txt"; sleep "$MONITOR_SECONDS"
  done
}

suite_body() {
  monitor_loop & local monitor=$!; trap 'kill "$monitor" 2>/dev/null || true' EXIT TERM INT
  prepare_full_artifacts
  write_status "$RUN_ROOT" running training "two GPU queues"
  gpu0_queue & local p0=$!; gpu1_queue & local p1=$!; echo "$p0 $p1" > "$RUN_ROOT/active_pids.txt"
  wait "$p0" || true; wait "$p1" || true; : > "$RUN_ROOT/active_pids.txt"
}

topk_gpu0_queue() {
  train_experiment chronos_t5_k1_s2021 1 official 0 || true
  can_start_experiment && train_experiment chronos_t5_k10_s2021 10 official 0 || true
  can_start_experiment && train_experiment chronos_t5_k20_s2021 20 official 0 || true
}

topk_gpu1_queue() {
  train_experiment chronos_t5_k5_s2021 5 official 1 || true
  can_start_experiment && train_experiment chronos_t5_k15_s2021 15 official 1 || true
}

topk_suite_body() {
  monitor_loop & local monitor=$!; trap 'kill "$monitor" 2>/dev/null || true' EXIT TERM INT
  write_status "$RUN_ROOT" running topk_training "K=1,5,10,15,20; two GPU queues"
  topk_gpu0_queue & local p0=$!; topk_gpu1_queue & local p1=$!
  echo "$p0 $p1" > "$RUN_ROOT/active_pids.txt"
  wait "$p0" || true; wait "$p1" || true; : > "$RUN_ROOT/active_pids.txt"
}

run_suite() {
  echo "$$" > "$RUN_ROOT/orchestrator.pid"; now_epoch > "$RUN_ROOT/start_epoch.txt"
  snapshot_suite; write_status "$RUN_ROOT" running preflight starting; preflight
  set +e; timeout --signal=TERM --kill-after=$((KILL_SECONDS-TERM_SECONDS)) "$TERM_SECONDS" "$0" _body; local result=$?; set -e
  write_status "$RUN_ROOT" running report "body_status=$result"
  "$TRAIN_PY" "$REPORT_SCRIPT" --suite-root "$RUN_ROOT" > "$RUN_ROOT/logs/report.log" 2>&1 || true
  if [[ "$result" -eq 0 ]] && grep -q '"status": "complete"' "$RUN_ROOT/metrics.json" 2>/dev/null; then
    write_status "$RUN_ROOT" complete complete "all experiments complete"
  else write_status "$RUN_ROOT" partial complete "body_status=$result"; fi
}

run_topk_suite() {
  echo "$$" > "$RUN_ROOT/orchestrator.pid"; now_epoch > "$RUN_ROOT/start_epoch.txt"
  snapshot_topk_suite; write_status "$RUN_ROOT" running preflight starting; preflight
  [[ -f "$ARTIFACT_ROOT/query_schedule/ARTIFACT_COMPLETE" && -f "$ARTIFACT_ROOT/arm_init_seed2021.pth" ]] || {
    echo "missing frozen query schedule or ARM initialization" >&2; exit 3;
  }
  set +e
  timeout --signal=TERM --kill-after=$((KILL_SECONDS-TERM_SECONDS)) "$TERM_SECONDS" "$0" _body_topk
  local result=$?
  set -e
  write_status "$RUN_ROOT" running report "body_status=$result"
  "$TRAIN_PY" "$REPORT_SCRIPT" --suite-root "$RUN_ROOT" --scope topk > "$RUN_ROOT/logs/report.log" 2>&1 || true
  if [[ "$result" -eq 0 ]] && grep -q '"status": "complete"' "$RUN_ROOT/metrics.json" 2>/dev/null; then
    write_status "$RUN_ROOT" complete complete "all five Top-K experiments complete"
  else
    write_status "$RUN_ROOT" partial complete "body_status=$result"
  fi
}

smoke_suite() {
  bootstrap_envs; download_qwen; preflight
  [[ ! -e "$SMOKE_ROOT" ]] || { echo "Smoke root already exists: $SMOKE_ROOT" >&2; return 2; }
  mkdir -p "$SMOKE_ROOT"/{logs,experiments}; RUN_ROOT="$SMOKE_ROOT"; ARTIFACT_ROOT="$SMOKE_ARTIFACT_ROOT"
  run_logged 0 smoke/faiss-train "$SMOKE_ROOT/logs/faiss_train.log" env CUDA_VISIBLE_DEVICES=0 "$FAISS_PY" "$ARTIFACT_SCRIPT" smoke-faiss-gpu
  run_logged 1 smoke/faiss-qwen "$SMOKE_ROOT/logs/faiss_qwen.log" env CUDA_VISIBLE_DEVICES=1 "$QWEN_PY" "$ARTIFACT_SCRIPT" smoke-faiss-gpu
  freeze_core_artifacts "$SMOKE_ARTIFACT_ROOT/query_schedule" 5120 "$SMOKE_ARTIFACT_ROOT"
  "$TRAIN_PY" "$ARTIFACT_SCRIPT" smoke-sidecars --schedule "$SMOKE_ARTIFACT_ROOT/query_schedule" --output "$SMOKE_ARTIFACT_ROOT"
  run_logged 0 smoke/bolt "$SMOKE_ROOT/logs/bolt.log" "$FAISS_PY" "$ARTIFACT_SCRIPT" encode --encoder bolt --source database \
    --retrieval-store "$RETRIEVAL_STORE" --schedule "$SMOKE_ARTIFACT_ROOT/query_schedule" --model-path "$BASE_MODEL" \
    --output "$SMOKE_ARTIFACT_ROOT/smoke_embeddings" --gpu 0 --num-shards 1 --shard-index 0 --max-rows 2 --batch-size 2
  run_logged 1 smoke/qwen "$SMOKE_ROOT/logs/qwen.log" "$QWEN_PY" "$ARTIFACT_SCRIPT" encode --encoder qwen --source database \
    --retrieval-store "$RETRIEVAL_STORE" --schedule "$SMOKE_ARTIFACT_ROOT/query_schedule" --model-path "$QWEN_MODEL" \
    --output "$SMOKE_ARTIFACT_ROOT/smoke_embeddings" --gpu 0 --num-shards 1 --shard-index 0 --max-rows 2 --batch-size 2
  run_logged 0 smoke/test-view "$SMOKE_ROOT/logs/test_view.log" "$FAISS_PY" "$ARTIFACT_SCRIPT" test-view \
    --project-root "$PROJECT_ROOT" --method random --dataset ETTh1 --output "$SMOKE_ARTIFACT_ROOT/test_views" --gpu 0 --max-samples 512
  run_logged 0 smoke/test-view-official "$SMOKE_ROOT/logs/test_view_official.log" "$FAISS_PY" "$ARTIFACT_SCRIPT" test-view \
    --project-root "$PROJECT_ROOT" --method official --dataset ETTh1 --output "$SMOKE_ARTIFACT_ROOT/test_views" --gpu 0 --max-samples 512
  local method; for method in pearson bolt qwen; do mkdir -p "$SMOKE_ARTIFACT_ROOT/test_views/$method"; \
    cp -a "$SMOKE_ARTIFACT_ROOT/test_views/random/ETTh1" "$SMOKE_ARTIFACT_ROOT/test_views/$method/ETTh1"; done
  train_experiment chronos_t5_k1_s2021 1 official 0 20 2 & local p0=$!
  train_experiment chronos_t5_k5_s2021 5 official 1 20 2 & local p1=$!; wait "$p0"; wait "$p1"
  train_experiment chronos_t5_k10_s2021 10 official 0 20 2 & p0=$!
  train_experiment chronos_t5_k15_s2021 15 official 1 20 2 & p1=$!; wait "$p0"; wait "$p1"
  train_experiment chronos_t5_k20_s2021 20 official 0 20 2 & p0=$!
  train_experiment random_k10_s2021 10 random_k10 1 20 2 & p1=$!; wait "$p0"; wait "$p1"
  train_experiment abs_pearson_k10_s2021 10 abs_pearson_k10 0 20 2 & p0=$!
  train_experiment chronos_bolt_embed_k10_s2021 10 chronos_bolt_k10 1 20 2 & p1=$!; wait "$p0"; wait "$p1"
  train_experiment qwen3_text_embed_k10_s2021 10 qwen3_text_k10 0 20 2
  touch "$SMOKE_ROOT/SMOKE_PASS"; write_status "$SMOKE_ROOT" complete smoke "all smoke checks passed"
  echo "SMOKE PASS: $SMOKE_ROOT"
}

show_status() {
  [[ -f "$RUN_ROOT/status.json" ]] && cat "$RUN_ROOT/status.json" || echo "no formal suite status"
  [[ -f "$RUN_ROOT/orchestrator.pid" ]] && ps -p "$(cat "$RUN_ROOT/orchestrator.pid")" -o pid,etime,stat,cmd || true
  [[ -f "$RUN_ROOT/last_monitor.txt" ]] && echo "last_monitor=$(cat "$RUN_ROOT/last_monitor.txt")"
  if [[ -f "$RUN_ROOT/start_epoch.txt" ]]; then
    local elapsed remaining completed failed pending
    elapsed=$(( $(now_epoch) - $(cat "$RUN_ROOT/start_epoch.txt") )); remaining=$((TOTAL_WALL_SECONDS-elapsed))
    if (( remaining < 0 )); then remaining=0; fi
    completed=$({ grep -Rl '"status": "complete"' "$RUN_ROOT/experiments"/*/status.json 2>/dev/null || true; } | wc -l)
    failed=$({ grep -Rl '"status": "failed"' "$RUN_ROOT/experiments"/*/status.json 2>/dev/null || true; } | wc -l)
    pending=$((EXPECTED_EXPERIMENTS-completed-failed)); if (( pending < 0 )); then pending=0; fi
    echo "elapsed_seconds=$elapsed remaining_seconds=$remaining completed=$completed failed=$failed pending=$pending"
    { find "$RUN_ROOT/logs" "$RUN_ROOT/experiments" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | \
      sort -nr | head -n 4 | cut -d' ' -f2- | while IFS= read -r f; do echo "latest[$f]=$(tail -n 1 "$f")"; done; } || true
  fi
  find "$RUN_ROOT/experiments" -maxdepth 2 -name status.json -print -exec cat {} \; 2>/dev/null || true
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || true
  df -h "$PROJECT_ROOT" | tail -n 1 || true
}

stop_suite() {
  [[ -f "$RUN_ROOT/orchestrator.pid" ]] || { echo "no orchestrator pid"; return 0; }
  local pid; pid=$(cat "$RUN_ROOT/orchestrator.pid")
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local waited=0; while kill -0 "$pid" 2>/dev/null && (( waited < 600 )); do sleep 5; waited=$((waited+5)); done
  kill -KILL -- "-$pid" 2>/dev/null || true; write_status "$RUN_ROOT" stopped stopped "user requested stop"
}

case "${1:-}" in
  smoke) smoke_suite ;;
  start)
    [[ -f "$SMOKE_ROOT/SMOKE_PASS" ]] || { echo "Run smoke first: $0 smoke" >&2; exit 2; }
    bootstrap_envs; download_qwen; preflight
    [[ ! -e "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT" >&2; exit 2; }
    mkdir -p "$RUN_ROOT/logs"; nohup setsid "$0" _run > "$RUN_ROOT/logs/orchestrator.log" 2>&1 < /dev/null &
    echo $! > "$RUN_ROOT/orchestrator.pid"; echo "orchestrator_pid=$!"; echo "log=$RUN_ROOT/logs/orchestrator.log"
    echo "status: bash $0 status"; echo "tail: bash $0 tail"
    ;;
  _run) run_suite ;;
  _body) suite_body ;;
  start-topk)
    [[ -f "$SMOKE_ROOT/SMOKE_PASS" ]] || { echo "Run smoke first: $0 smoke" >&2; exit 2; }
    bootstrap_envs; preflight
    [[ ! -e "$TOPK_RUN_ROOT" ]] || { echo "Top-K run root already exists: $TOPK_RUN_ROOT" >&2; exit 2; }
    mkdir -p "$TOPK_RUN_ROOT/logs"
    nohup env RUN_ROOT="$TOPK_RUN_ROOT" EXPECTED_EXPERIMENTS=5 setsid "$0" _run_topk \
      > "$TOPK_RUN_ROOT/logs/orchestrator.log" 2>&1 < /dev/null &
    echo $! > "$TOPK_RUN_ROOT/orchestrator.pid"
    echo "orchestrator_pid=$!"; echo "log=$TOPK_RUN_ROOT/logs/orchestrator.log"
    echo "status: bash $0 status-topk"; echo "tail: bash $0 tail-topk"
    ;;
  _run_topk) run_topk_suite ;;
  _body_topk) topk_suite_body ;;
  status) show_status ;;
  status-topk) RUN_ROOT="$TOPK_RUN_ROOT"; EXPECTED_EXPERIMENTS=5; show_status ;;
  tail) tail -n 200 -f "$RUN_ROOT/logs/orchestrator.log" ;;
  tail-topk) tail -n 200 -f "$TOPK_RUN_ROOT/logs/orchestrator.log" ;;
  stop) stop_suite ;;
  stop-topk) RUN_ROOT="$TOPK_RUN_ROOT"; stop_suite ;;
  report) "$TRAIN_PY" "$REPORT_SCRIPT" --suite-root "$RUN_ROOT" ;;
  report-topk) "$TRAIN_PY" "$REPORT_SCRIPT" --suite-root "$TOPK_RUN_ROOT" --scope topk ;;
  *) usage; exit 2 ;;
esac
