set -euo pipefail

ROOT=$GEORX_ROOT/main
PY=python3
export SPHERE_CAL_WINDOWS=32
export SPHERE_HOLDOUT_WINDOWS=256
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p runs/logs

GATE_LOG="runs/logs/sphere_gate_cal32_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date -Is)] sphere gate start calibration=$SPHERE_CAL_WINDOWS holdout=$SPHERE_HOLDOUT_WINDOWS" | tee -a "$GATE_LOG"
set +e
"$PY" -u code/run_pipeline.py --stages diag_recall --seeds 1 --device cuda --batch-q 128 >>"$GATE_LOG" 2>&1
gate_rc=$?
set -e
echo "[$(date -Is)] sphere gate rc=$gate_rc" | tee -a "$GATE_LOG"
if [[ "$gate_rc" != "0" ]]; then
  echo "BLOCKED: sphere gate process failed; native was not started" | tee -a "$GATE_LOG"
  exit "$gate_rc"
fi

gate_pass="$($PY -c 'import json; print(json.load(open("runs/gate/GATE.json"))["gate_pass"])')"
if [[ "$gate_pass" != "True" ]]; then
  echo "BLOCKED: sphere gate_pass=$gate_pass; native was not started" | tee -a "$GATE_LOG"
  exit 2
fi

"$PY" -u prepare_native_shards.py | tee -a "$GATE_LOG"

start_one() {
  local name="$1"; shift
  local log="runs/logs/native_${name}_$(date +%Y%m%d_%H%M%S).log"
  nohup "$@" >"$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"runs/native_${name}.pid"
  printf 'pid=%s\nlog=%s\ncommand=%q\nstarted=%s\n' "$pid" "$log" "$*" "$(date -Is)" >"runs/native_${name}.meta"
  echo "[$(date -Is)] $name pid=$pid log=$log" | tee -a "$GATE_LOG"
}

start_one gpu0 env CUDA_VISIBLE_DEVICES=0 "$PY" -u native_shard.py \
  --jobs-file runs/native_shard_plan.json --worker gpu0 --device cuda --batch-q 128
start_one gpu1 env CUDA_VISIBLE_DEVICES=1 "$PY" -u native_shard.py \
  --jobs-file runs/native_shard_plan.json --worker gpu1 --device cuda --batch-q 128
start_one gpu2 env CUDA_VISIBLE_DEVICES=2 "$PY" -u native_shard.py \
  --jobs-file runs/native_shard_plan.json --worker gpu2 --device cuda --batch-q 128
start_one gpu3 env CUDA_VISIBLE_DEVICES=3 "$PY" -u native_shard.py \
  --jobs-file runs/native_shard_plan.json --worker gpu3 --device cuda --batch-q 128
start_one cpu_seed3_light env OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 "$PY" -u native_shard.py \
  --jobs-file runs/native_shard_plan.json --worker cpu_seed3_light --device cpu --batch-q 32

echo "[$(date -Is)] native main experiment started; merge after all five workers finish" | tee -a "$GATE_LOG"
