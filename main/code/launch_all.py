

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("$GEORX_ROOT/main")
PY = "python3"
LOGDIR = ROOT / "runs" / "logs"
REPAIR_PID = ROOT / "runs" / "repair.pid.json"
INJECT_PID = ROOT / "runs" / "inject.pid.json"
CE_PID = ROOT / "runs" / "ce.pid.json"
WEIGHTS = ROOT / "runs" / "ce_training" / "weights"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ce-waiter", action="store_true")
    args = ap.parse_args(argv)
    if args.ce_waiter:
        return _ce_waiter()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    _print_resources()
    if args.dry_run:
        print("dry-run: would launch repair-4gpu, inject-9cpu, ce-waiter")
        return 0

    repair = _run([
        PY, "-u", str(ROOT / "code" / "launch_shards.py"),
        "--stages", "repair", "--seeds", "1", "2", "3",
        "--device", "cuda", "--gpus", "0,1,2,3",
        "--pidfile", str(REPAIR_PID),
    ])
    print("repair launcher rc", repair)

    inject = _run([
        PY, "-u", str(ROOT / "code" / "launch_shards.py"),
        "--stages", "inject", "--seeds", "1", "2", "3",
        "--device", "cpu", "--cpu-workers", "9", "--cpu-omp", "8",
        "--pidfile", str(INJECT_PID),
        "--batch-q", "32",
    ])
    print("inject launcher rc", inject)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = LOGDIR / f"ce_waiter_{stamp}.out"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["CUDA_VISIBLE_DEVICES"] = ""  
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [PY, "-u", str(ROOT / "code" / "launch_all.py"), "--ce-waiter"],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    CE_PID.write_text(json.dumps(
        {"name": "ce_waiter", "pid": proc.pid, "log": str(log)}, indent=2))
    print(f"ce waiter pid={proc.pid}  log={log}")
    print("follow:")
    print(f"  tail -f {LOGDIR}/shard_gpu0_edis_*.out")
    print(f"  tail -f {LOGDIR}/shard_cpu0_encoders_*.out")
    print(f"  tail -f {log}")
    return 0


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _print_resources() -> None:
    subprocess.call(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                     "--format=csv"])
    subprocess.call(["bash", "-lc", "free -h | head -2; echo CPUs=$(nproc); uptime"])


def _ce_waiter() -> int:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print("ce-waiter: download", flush=True)
    rc = subprocess.call([PY, "-u", str(ROOT / "code" / "ce_train.py"), "--download"],
                         cwd=str(ROOT))
    if rc != 0:
        print(f"ce-waiter: download failed rc={rc}", flush=True)
        return rc
    print("ce-waiter: waiting for GPU3 repair worker to exit", flush=True)
    while _repair_gpu_live("3"):
        time.sleep(30)
    print("ce-waiter: GPU3 free, training local CE", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    rc = subprocess.call(
        [PY, "-u", str(ROOT / "code" / "ce_train.py"), "--train-local", "--device", "cuda"],
        cwd=str(ROOT), env=env)
    print(f"ce-waiter: train-local rc={rc}", flush=True)
    return rc


def _repair_gpu_live(gpu: str) -> bool:
    if not REPAIR_PID.is_file():
        return False
    try:
        workers = json.loads(REPAIR_PID.read_text())
    except Exception:
        return False
    for w in workers:
        if str(w.get("gpu")) != str(gpu):
            continue
        pid = int(w["pid"])
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    return False


if __name__ == "__main__":
    raise SystemExit(main())
