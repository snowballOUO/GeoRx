

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "python3"
WORKER = ROOT / "code" / "repair_worker_v2.py"
RUNS = ROOT / "runs"
LOGS = RUNS / "logs"
PIDFILE = RUNS / "workers.json"





SHARDS = [
    ("gpu0_edis_mscoco", "cuda", "0", "0-9", [
        "edis_task2:1,2,3",
        "mscoco_task0:1,2,3",
        "webqa_task1:1,2,3",
        "infoseek_task6:1,2,3",
    ]),
    ("gpu1_visualnews_nights1", "cuda", "1", "10-19", [
        "visualnews_task0:1,2,3",
        "nights_task4:1",
    ]),
    ("gpu2_fashion200k_cirr12_nights2", "cuda", "2", "20-29", [
        "fashion200k_task0:1,2,3",
        "cirr_task7:1,2",
        "nights_task4:2",
    ]),
    ("gpu3_fashioniq_cirr3_nights3", "cuda", "3", "30-39", [
        "fashioniq_task7:1,2,3",
        "cirr_task7:3",
        "nights_task4:3",
    ]),
    ("cpu40_oven", "cpu", "", "40-79", [
        "oven_task6:1,2,3",
    ]),
]


def merge() -> dict:
    rows = []
    for path in sorted(RUNS.glob("S*/repair/*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("state") == "complete":
            rows.extend(payload.get("rows") or [])
    if not rows:
        return {"n_rows": 0, "n_ok": 0, "n_failed": 0}
    fields = sorted({key for row in rows for key in row})
    out = RUNS / "h4_listwise_repair_retrain_delta.csv"
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)
    summary = {
        "stage": "h4_true_listwise_repair_retrained_weights",
        "n_rows": len(rows),
        "n_ok": sum(row.get("status") == "ok" for row in rows),
        "n_failed": sum(row.get("status") != "ok" for row in rows),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (RUNS / "h4_listwise_repair_retrain_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    workers = []
    processes = []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name, device, gpu, affinity, jobs in SHARDS:
        log = LOGS / f"{name}_{stamp}.out"
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "H4_WORKER_NAME": name})
        if device == "cuda":
            env.update({
                "CUDA_VISIBLE_DEVICES": gpu,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            })
            visible_device = "cuda"
        else:
            env.update({
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "40",
                "MKL_NUM_THREADS": "40",
                "OPENBLAS_NUM_THREADS": "40",
                "NUMEXPR_NUM_THREADS": "40",
            })
            visible_device = "cpu"
        cmd = ["taskset", "-c", affinity, PY, "-u", str(WORKER),
               "--device", visible_device, "--worker-name", name,
               "--batch-q", "32"]
        for job in jobs:
            cmd.extend(["--job", job])
        with log.open("w") as fh:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        workers.append({"name": name, "device": device, "gpu": gpu,
                        "affinity": affinity, "pid": proc.pid,
                        "jobs": jobs, "log": str(log), "cmd": cmd})
        print(f"launched {name} pid={proc.pid} device={device} log={log}", flush=True)
    PIDFILE.write_text(json.dumps(workers, indent=2) + "\n")

    if args.wait:
        rc = 0
        for proc in processes:
            rc = max(rc, int(proc.wait() != 0))
        result = merge()
        print(f"finished rc={rc} summary={result}", flush=True)
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
