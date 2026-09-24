set -euo pipefail
ROOT=$GEORX_ROOT/main
PY=python3
LOGDIR="$ROOT/runs/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/wait_h4_then_joint_${STAMP}.out"
cd "$ROOT"
nohup env PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com \
  "$PY" -u code/wait_h4_then_joint.py >> "$LOG" 2>&1 &
echo "pid=$! log=$LOG"
echo "tail -f $LOG"
