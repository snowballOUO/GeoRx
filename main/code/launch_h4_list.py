

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("$GEORX_ROOT/main")
PY = "python3"
LOGDIR = ROOT / "runs" / "logs"

SHARDS = [
    ("0", "40-52", ["mscoco_task0", "fashion200k_task0", "visualnews_task0"]),
    ("1", "53-65", ["webqa_task1", "edis_task2", "oven_task6", "infoseek_task6"]),
    ("2", "66-79", ["nights_task4", "fashioniq_task7", "cirr_task7"]),
]


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pids = []
    for gpu, cpus, loads in SHARDS:
        log = LOGDIR / f"h4_list_gpu{gpu}_{stamp}.out"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        cmd = [
            "taskset", "-c", cpus, PY, "-u", str(ROOT / "code" / "ce_list_train.py"),
            "--device", "cuda", "--steps", "4000", "--batch", "16",
            "--overwrite", "--loads", *loads,
        ]
        with log.open("w") as fh:
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pids.append((gpu, proc.pid, log, loads))
        print(f"gpu{gpu} pid={proc.pid} cpus={cpus} loads={loads} log={log}", flush=True)
    (ROOT / "runs" / "h4_list.pid.json").write_text(
        __import__("json").dumps(
            [{"gpu": g, "pid": p, "log": str(lg), "loads": ld} for g, p, lg, ld in pids],
            indent=2) + "\n")
    print("follow:")
    for _, _, log, _ in pids:
        print(f"  tail -f {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
