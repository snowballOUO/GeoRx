set -euo pipefail
ROOT=$GEORX_ROOT/injection
PY=python3
cd "$ROOT"
PHASE="${1:-full}"
case "$PHASE" in smoke|full) ;; *) echo "phase must be smoke or full"; exit 2;; esac
mkdir -p runs/logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/logs/supervisor_${PHASE}_${STAMP}.out"
nohup "$PY" -u code/supervise.py --phase "$PHASE" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "runs/${PHASE}_nohup.pid"
echo "supervisor_pid=$PID log=$ROOT/$LOG"
