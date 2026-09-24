

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
CPUSET = "0-39"
SMALLPOOL = [
    "mscoco_task0", "nights_task4", "fashion200k_task0",
    "fashioniq_task7", "cirr_task7",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu3-waiter", action="store_true")
    args = ap.parse_args(argv)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    if args.gpu3_waiter:
        return _gpu3_after_train()
    if args.dry_run:
        print("would: train GPU3 + fill-ce GPU0-2 + waiter GPU3 fill")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    train_log = LOGDIR / f"ce_train_{stamp}.out"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    with train_log.open("w") as fh:
        proc = subprocess.Popen(
            ["taskset", "-c", CPUSET, PY, "-u", str(ROOT / "code" / "ce_train.py"),
             "--train-local", "--device", "cuda"],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (ROOT / "runs" / "ce_train.pid").write_text(str(proc.pid))
    print(f"ce-train gpu3 pid={proc.pid} log={train_log}", flush=True)

    rc = subprocess.call([
        PY, "-u", str(ROOT / "code" / "launch_shards.py"),
        "--stages", "repair", "--fill-ce",
        "--seeds", "1", "2", "3",
        "--device", "cuda", "--gpus", "0,1,2",
        "--cpuset", CPUSET,
        "--pidfile", str(ROOT / "runs" / "ce_fill.pid.json"),
    ], cwd=str(ROOT))
    print(f"fill-ce gpu0-2 launcher rc={rc}", flush=True)

    wait_log = LOGDIR / f"ce_gpu3_waiter_{stamp}.out"
    with wait_log.open("w") as fh:
        w = subprocess.Popen(
            ["taskset", "-c", CPUSET, PY, "-u", str(ROOT / "code" / "launch_ce.py"),
             "--gpu3-waiter"],
            cwd=str(ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1",
                                "HF_ENDPOINT": "https://hf-mirror.com"},
            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
    (ROOT / "runs" / "ce_gpu3_waiter.pid").write_text(str(w.pid))
    print(f"gpu3 waiter pid={w.pid} log={wait_log}", flush=True)
    print("follow:")
    print(f"  tail -f {train_log}")
    print(f"  tail -f {LOGDIR}/shard_gpu0_edis_*.out")
    print(f"  tail -f {wait_log}")
    return 0


def _gpu3_after_train() -> int:
    pid_path = ROOT / "runs" / "ce_train.pid"
    print("gpu3-waiter: waiting for ce-train", flush=True)
    if pid_path.is_file():
        pid = int(pid_path.read_text().strip())
        while _alive(pid):
            time.sleep(30)
    else:
        print("gpu3-waiter: no ce_train.pid; assume train already finished", flush=True)
    print("gpu3-waiter: starting smallpool CE fill", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["SPHERE_FILL_CE"] = "1"
    env["SPHERE_SKIP_CE"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = LOGDIR / f"ce_fill_gpu3_{stamp}.out"
    with log.open("w") as fh:
        rc = subprocess.call(
            ["taskset", "-c", CPUSET, PY, "-u", str(ROOT / "code" / "run_pipeline.py"),
             "--stages", "repair", "--seeds", "1", "2", "3",
             "--device", "cuda", "--batch-q", "32",
             "--datasets", *SMALLPOOL],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
        )
    print(f"gpu3-waiter: fill rc={rc} log={log}", flush=True)
    (ROOT / "runs" / "ce_fill_gpu3.pid").write_text(json.dumps(
        {"pid": os.getpid(), "log": str(log), "rc": rc}, indent=2))
    return rc


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
