set -euo pipefail
ROOT=$GEORX_ROOT/main
PY=python3
cd "$ROOT"
mkdir -p runs/logs runs/multi_inject
exec "$PY" -u code/launch_multi_inject.py "$@"
