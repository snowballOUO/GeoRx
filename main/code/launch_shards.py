

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
PIDFILE = ROOT / "runs" / "shards.pid.json"
LOGDIR = ROOT / "runs" / "logs"


ENCODER_SHARDS = [
    {"name": "gpu0_e5v_blipff", "gpu": "0",
     "encoders": ["e5v_llava_next", "blip_ff_large"]},
    {"name": "gpu1_vlm_clipsf", "gpu": "1",
     "encoders": ["vlm2vec_phi3", "clip_sf_large"]},
    {"name": "gpu2_gme_openclip", "gpu": "2",
     "encoders": ["gme_qwen2vl_2b", "openclip_fft"]},
    {"name": "gpu3_small", "gpu": "3",
     "encoders": ["clip_vitb32", "siglip_base", "blip2_vitL"]},
]


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

REPAIR_STAGES = {"repair"}
ENCODER_STAGES = {"native", "diag_recall", "inject", "eval"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["native"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="physical GPU ids, comma-separated")
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--cpu-omp", type=int, default=8,
                    help="OMP/MKL threads per CPU worker")
    ap.add_argument("--batch-q", type=int, default=0,
                    help="0 = use the shard default (repair) or 128 (native)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--kill", action="store_true")
    ap.add_argument("--pidfile", default=str(PIDFILE),
                    help="json pid list so inject-CPU and repair-GPU can coexist")
    ap.add_argument("--fill-ce", action="store_true",
                    help="resume repair only to fill CE rows; never redo analytic")
    ap.add_argument("--cpuset", default="",
                    help="taskset -c list, e.g. 0-39")
    ap.add_argument("--wait", action="store_true",
                    help="block until all shard workers exit")
    args = ap.parse_args(argv)
    pidfile = Path(args.pidfile)

    if args.kill:
        return _kill(pidfile)

    LOGDIR.mkdir(parents=True, exist_ok=True)
    if pidfile.is_file():
        live = _live_workers(pidfile)
        if live:
            names = ", ".join(f"{w['name']}:pid={w['pid']}" for w in live)
            raise SystemExit(f"shards already running ({names}); "
                             f"python code/launch_shards.py --kill --pidfile {pidfile} first")

    specs = _specs(args)
    if args.dry_run:
        for s in specs:
            print(_fmt_spec(s, args))
        return 0

    workers = []
    for spec in specs:
        workers.append(_spawn(spec, args))
    pidfile.write_text(json.dumps(workers, indent=2))
    print(f"launched {len(workers)} shards  pidfile={pidfile}")
    for w in workers:
        print(f"  {w['name']}  gpu={w.get('gpu','cpu')}  pid={w['pid']}  log={w['log']}")
    if args.wait:
        return _wait(workers)
    return 0


def _specs(args) -> list[dict]:
    stages = set(args.stages)
    use_dataset = bool(stages & REPAIR_STAGES) and not (stages & ENCODER_STAGES)
    if args.device == "cpu":
        base = DATASET_SHARDS if use_dataset else ENCODER_SHARDS
        
        return _cpu_chunks(base, args.cpu_workers, by_dataset=use_dataset)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    src = DATASET_SHARDS if use_dataset else ENCODER_SHARDS
    out = []
    for i, spec in enumerate(src):
        if i >= len(gpus):
            break
        s = dict(spec)
        s["gpu"] = gpus[i]
        s["name"] = spec["name"].replace(f"gpu{spec['gpu']}", f"gpu{gpus[i]}", 1)
        out.append(s)
    return out


def _cpu_chunks(base: list[dict], n_workers: int, *, by_dataset: bool) -> list[dict]:
    key = "datasets" if by_dataset else "encoders"
    items = []
    for spec in base:
        items.extend(spec[key])
    n_workers = max(1, min(int(n_workers), len(items)))
    chunks = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        chunks[i % n_workers].append(item)
    out = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        spec = {"name": f"cpu{i}_{key}", "gpu": "", key: chunk, "batch_q": 32}
        out.append(spec)
    return out


def _spawn(spec: dict, args) -> dict:
    cmd = [
        PY, "-u", str(ROOT / "code" / "run_pipeline.py"),
        "--stages", *args.stages,
        "--seeds", *[str(s) for s in args.seeds],
        "--device", args.device,
    ]
    if spec.get("encoders"):
        cmd += ["--encoders", *spec["encoders"]]
    if spec.get("datasets"):
        cmd += ["--datasets", *spec["datasets"]]
    bq = args.batch_q or spec.get("batch_q") or 128
    cmd += ["--batch-q", str(bq)]
    if getattr(args, "cpuset", ""):
        cmd = ["taskset", "-c", args.cpuset] + cmd

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT") or "https://hf-mirror.com"
    if getattr(args, "fill_ce", False):
        env["SPHERE_FILL_CE"] = "1"
        env["SPHERE_SKIP_CE"] = "0"
    elif "repair" in args.stages and "eval" not in args.stages:
        env.setdefault("SPHERE_SKIP_CE", "1")
    if args.device == "cuda" and spec.get("gpu") != "":
        env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        omp = str(args.cpu_omp)
        env["OMP_NUM_THREADS"] = omp
        env["MKL_NUM_THREADS"] = omp
        env["OPENBLAS_NUM_THREADS"] = omp
        env["NUMEXPR_NUM_THREADS"] = omp

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = LOGDIR / f"shard_{spec['name']}_{stamp}.out"
    log_f = log.open("w")
    
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "name": spec["name"],
        "gpu": spec.get("gpu", ""),
        "pid": proc.pid,
        "log": str(log),
        "cmd": cmd,
        "encoders": spec.get("encoders"),
        "datasets": spec.get("datasets"),
    }


def _fmt_spec(spec: dict, args) -> str:
    who = spec.get("encoders") or spec.get("datasets")
    return (f"{spec['name']:24s}  gpu={spec.get('gpu') or 'cpu':4s}  "
            f"device={args.device}  {who}")


def _live_workers(pidfile: Path | None = None) -> list[dict]:
    pidfile = pidfile or PIDFILE
    if not pidfile.is_file():
        return []
    workers = json.loads(pidfile.read_text())
    live = []
    for w in workers:
        pid = int(w["pid"])
        if _pid_alive(pid):
            live.append(w)
    return live


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill(pidfile: Path | None = None) -> int:
    pidfile = pidfile or PIDFILE
    if not pidfile.is_file():
        print(f"no shard pidfile {pidfile}")
        return 0
    workers = json.loads(pidfile.read_text())
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
    pidfile.unlink(missing_ok=True)
    return 0


def _wait(workers: list[dict]) -> int:
    rc = 0
    pending = {int(w["pid"]): w for w in workers}
    while pending:
        for pid in list(pending):
            if not _pid_alive(pid):
                print(f"shard exit {pending[pid]['name']} pid={pid}")
                del pending[pid]
        if pending:
            time.sleep(15)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
