
set -euo pipefail

ROOT="$GEORX_ROOT/main"
PY="python3"
cd "$ROOT"
mkdir -p runs/logs

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="runs/logs/run_${STAMP}.out"
PIDFILE="runs/pipeline.pid"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "pipeline already running pid=$(cat "$PIDFILE"); refuse to start a second." >&2
  exit 1
fi

setsid nohup "$PY" -u code/run_pipeline.py "$@" >"$OUT" 2>&1 &
echo $! > "$PIDFILE"
disown || true
echo "launched pid=$(cat "$PIDFILE")  log=$OUT"
echo "follow: tail -f $ROOT/runs/logs/pipeline.log"
