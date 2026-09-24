

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path("$GEORX_ROOT/main")
PY = "python3"
H4_DIR = ROOT / "runs" / "ce_training" / "h4_listwise"
OUT = ROOT / "runs" / "joint_phi_sweep"
LOGDIR = ROOT / "runs" / "logs"
LOADS = [
    "mscoco_task0", "visualnews_task0", "fashion200k_task0", "webqa_task1",
    "edis_task2", "nights_task4", "oven_task6", "infoseek_task6",
    "fashioniq_task7", "cirr_task7",
]
SHARDS = [
    ("0", "40-52"),
    ("1", "53-65"),
    ("2", "66-79"),
]


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


def _trainers_alive() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except subprocess.CalledProcessError:
        return []
    hits = []
    for line in out.splitlines():
        if "ce_list_train.py" in line and "grep" not in line:
            hits.append(line.strip())
    return hits


def _h4_ready() -> tuple[bool, str]:
    missing = []
    for load in LOADS:
        pt = H4_DIR / f"ce_list_local__{load}.pt"
        tmp = H4_DIR / f"ce_list_local__{load}.pt.tmp"
        if tmp.is_file():
            return False, f"{load} still writing tmp"
        if not pt.is_file():
            missing.append(load)
            continue
        try:
            try:
                blob = torch.load(pt, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(pt, map_location="cpu")
        except Exception as exc:
            return False, f"{load} load failed: {exc!r}"
        if blob.get("kind") != "list_mixer":
            return False, f"{load} kind={blob.get('kind')!r} (refuse PairHead/ITM)"
    if missing:
        return False, "missing " + ",".join(missing)
    trainers = _trainers_alive()
    if trainers:
        return False, f"ce_list_train still running ({len(trainers)})"
    return True, "10 list_mixer checkpoints; trainers idle"


def wait_h4(poll_s: int = 60) -> None:
    log("[wait] H4 list_mixer weights (10 loads), then joint Φ sweep on GPU0-2")
    while True:
        ok, why = _h4_ready()
        log(f"[wait] {why}")
        if ok:
            return
        time.sleep(poll_s)


def launch_workers() -> list[subprocess.Popen]:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    procs = []
    for gpu, cpus in SHARDS:
        logp = LOGDIR / f"joint_phi_gpu{gpu}_{stamp}.out"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        cmd = [
            "taskset", "-c", cpus, PY, "-u", str(ROOT / "code" / "joint_phi_sweep.py"),
            "--device", "cuda", "--batch-q", "64",
            "--shard", gpu, "--n-shards", "3",
            "--seeds", "1", "2", "3",
        ]
        fh = logp.open("w")
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append(proc)
        log(f"[launch] gpu{gpu} pid={proc.pid} cpus={cpus} log={logp}")
    (OUT / "workers.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "workers.json").write_text(json.dumps(
        [{"gpu": g, "pid": p.pid} for (g, _), p in zip(SHARDS, procs)], indent=2) + "\n")
    return procs


def summarize() -> None:
    rows = []
    cell_dir = OUT / "cells"
    if not cell_dir.is_dir():
        log("[summary] no cell csv yet")
        return
    for p in sorted(cell_dir.glob("S*_*.csv")):
        with p.open() as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        log("[summary] empty")
        return
    fields = list(rows[0].keys())
    all_p = OUT / "all.csv"
    with all_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    beats = [r for r in rows if r.get("beats_best_single") == "1"]
    beats_exact = [r for r in beats if r.get("combo", r.get("phi_set")) == r["phi_set"]]
    extra_2way = [r for r in beats
                  if r.get("phi_set") == "h1;h2;h4" and r.get("combo") in ("h1;h2", "h1;h4", "h2;h4")]
    by_case = defaultdict(list)
    for r in beats_exact:
        by_case[r["phi_set"]].append(r)
    n_cells = len({(r["seed"], r["cell_id"]) for r in beats_exact})
    lines = [
        "# Joint Φ sweep: joints that beat the best single method",
        "",
        "Exploratory. H2=α-QE. H4=`ce_list_local` list_mixer (staging).",
        "Never ITM / PairHead / `ce_pair_*`. Not channel-hit, not locked compose.",
        "A joint beats the best **single method in that combo** (not the cosine baseline).",
        "",
        f"Total variant rows: {len(rows)}. "
        f"Exact-Φ joints beating their singles: {len(beats_exact)} rows / {n_cells} cells.",
        "",
    ]
    for case in ("h1;h2", "h1;h4", "h2;h4", "h1;h2;h4"):
        recs = by_case.get(case, [])
        n_cell = len({(r["seed"], r["cell_id"]) for r in recs})
        lines.append(f"## {case} ({n_cell} cells, {len(recs)} beating rows)")
        lines.append("")
        seen = {}
        for r in recs:
            key = (r["seed"], r["cell_id"])
            seen.setdefault(key, [])
            seen[key].append(r)
        if not seen:
            lines.append("none")
            lines.append("")
            continue
        for (seed, cid), vs in sorted(seen.items()):
            vs = sorted(vs, key=lambda x: -float(x["dR@10"]))
            best = vs[0]
            methods = ", ".join(f"{v['variant']}({v['params']})" for v in vs[:8])
            lines.append(
                f"- S{seed} `{cid}` best joint `{best['variant']}` `{best['params']}` "
                f"ΔR@10={float(best['dR@10']):+.4f} vs single {float(best['dR@10_best_single']):+.4f} "
                f"(also: {methods})"
            )
        lines.append("")
    if extra_2way:
        lines.append(f"## 2-way joints on Φ={{h1,h2,h4}} cells ({len(extra_2way)} rows)")
        lines.append("")
        seen = {}
        for r in extra_2way:
            key = (r["seed"], r["cell_id"], r.get("combo"))
            seen.setdefault(key, [])
            seen[key].append(r)
        for (seed, cid, combo), vs in sorted(seen.items()):
            vs = sorted(vs, key=lambda x: -float(x["dR@10"]))
            best = vs[0]
            lines.append(
                f"- S{seed} `{cid}` combo `{combo}` `{best['variant']}` `{best['params']}` "
                f"ΔR@10={float(best['dR@10']):+.4f} vs single {float(best['dR@10_best_single']):+.4f}"
            )
        lines.append("")
    fail = list((OUT / "cells").glob("*.err.txt"))
    if fail:
        lines += ["## Failures", ""]
        for p in fail:
            lines.append(f"- `{p.name}`: {p.read_text().strip()[:300]}")
        lines.append("")
    (OUT / "BEATS_SINGLE.md").write_text("\n".join(lines) + "\n")
    log(f"[summary] wrote {all_p} and {OUT / 'BEATS_SINGLE.md'} "
        f"exact_beats={len(beats_exact)} extra_2way={len(extra_2way)}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wait_h4()
    procs = launch_workers()
    rc = 0
    for p in procs:
        c = p.wait()
        log(f"[wait] worker pid={p.pid} rc={c}")
        if c:
            rc = c
    summarize()
    log(f"[done] rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
