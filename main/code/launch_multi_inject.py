

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("$GEORX_ROOT/main")
PY = "python3"
PIDFILE = ROOT / "runs" / "multi_inject" / "shards.pid.json"
LOGDIR = ROOT / "runs" / "logs"
WORKER = ROOT / "code" / "stage_multi_inject.py"


DATASET_SHARDS = [
    {"name": "gpu0_edis", "gpu": "0", "datasets": ["edis_task2"], "batch_q": 8},
    {"name": "gpu1_oven_infoseek", "gpu": "1",
     "datasets": ["oven_task6", "infoseek_task6"], "batch_q": 16},
    {"name": "gpu2_news_webqa", "gpu": "2",
     "datasets": ["visualnews_task0", "webqa_task1"], "batch_q": 16},
    {"name": "gpu3_smallpool", "gpu": "3",
     "datasets": ["mscoco_task0", "nights_task4", "fashion200k_task0",
                  "fashioniq_task7", "cirr_task7"], "batch_q": 32},
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--cpuset", default="40-79",
                    help="taskset -c list; empty string disables")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--kill", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args(argv)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    if args.kill:
        return _kill()
    if args.status:
        return _status()

    live = _live_workers()
    if live:
        names = ", ".join(f"{w['name']}:pid={w['pid']}" for w in live)
        raise SystemExit(f"multi_inject already running ({names}); "
                         f"python code/launch_multi_inject.py --kill first")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    specs = []
    for i, spec in enumerate(DATASET_SHARDS):
        if i >= len(gpus):
            break
        s = dict(spec)
        s["gpu"] = gpus[i]
        s["name"] = spec["name"].replace(f"gpu{spec['gpu']}", f"gpu{gpus[i]}", 1)
        specs.append(s)

    print("90 cells × 3 seeds × 26 multi-subsets (|S|=2..5) = 7020")
    print("3 = independent run-seeds {1,2,3}, each with its own sealed split.")
    print("Not a vote, not a union of Φ. Real embedding space, no sphere plant.")
    if args.dry_run:
        for s in specs:
            print(f"  {s['name']:24s} gpu={s['gpu']} datasets={s['datasets']} "
                  f"batch_q={s['batch_q']}")
        return 0

    workers = [_spawn(spec, args) for spec in specs]
    PIDFILE.write_text(json.dumps(workers, indent=2))
    print(f"launched {len(workers)} shards  pidfile={PIDFILE}")
    for w in workers:
        print(f"  {w['name']}  gpu={w['gpu']}  pid={w['pid']}  log={w['log']}")
    if args.wait:
        return _wait(workers)
    return 0


def _spawn(spec: dict, args) -> dict:
    cmd = [
        PY, "-u", str(WORKER),
        "--seeds", *[str(s) for s in args.seeds],
        "--device", args.device,
        "--datasets", *spec["datasets"],
        "--batch-q", str(spec["batch_q"]),
    ]
    if args.cpuset:
        cmd = ["taskset", "-c", args.cpuset] + cmd

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = LOGDIR / f"multi_inject_{spec['name']}_{stamp}.out"
    log_f = log.open("w")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "name": spec["name"], "gpu": spec["gpu"], "pid": proc.pid,
        "log": str(log), "cmd": cmd, "datasets": spec["datasets"],
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _live_workers() -> list[dict]:
    if not PIDFILE.is_file():
        return []
    live = []
    for w in json.loads(PIDFILE.read_text()):
        if _pid_alive(int(w["pid"])):
            live.append(w)
    return live


def _kill() -> int:
    if not PIDFILE.is_file():
        print(f"no pidfile {PIDFILE}")
        return 0
    workers = json.loads(PIDFILE.read_text())
    for w in workers:
        pid = int(w["pid"])
        if not _pid_alive(pid):
            print(f"  already dead {w['name']} pid={pid}")
            continue
        os.killpg(pid, signal.SIGTERM)
        print(f"  SIGTERM {w['name']} pid={pid}")
    time.sleep(2)
    for w in workers:
        pid = int(w["pid"])
        if _pid_alive(pid):
            os.killpg(pid, signal.SIGKILL)
            print(f"  SIGKILL {w['name']} pid={pid}")
    PIDFILE.unlink(missing_ok=True)
    return 0


def _status() -> int:
    prog = ROOT / "runs" / "multi_inject" / "progress.json"
    n = 0
    root = ROOT / "runs" / "multi_inject"
    if root.is_dir():
        for p in root.glob("S*/*/*.json"):
            if p.name.endswith("error.json") or p.name == "cell_error.json":
                continue
            n += 1
    print(f"complete json ≈ {n} / 7020")
    if prog.is_file():
        print(prog.read_text())
    live = _live_workers()
    if live:
        for w in live:
            print(f"live {w['name']} pid={w['pid']} log={w['log']}")
    else:
        print("no live shards")
    return 0


def _wait(workers: list[dict]) -> int:
    pending = {int(w["pid"]): w for w in workers}
    while pending:
        for pid in list(pending):
            if not _pid_alive(pid):
                print(f"shard exit {pending[pid]['name']} pid={pid}")
                del pending[pid]
        if pending:
            time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
