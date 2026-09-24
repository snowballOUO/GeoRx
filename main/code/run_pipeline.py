

from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path

import common as C

STAGE_ORDER = ["diag_recall", "native", "inject", "repair", "eval", "compose"]
STAGE_MODULES = {
    "diag_recall": "stage_diag_recall",
    "native": "stage_native",
    "inject": "stage_inject",
    "repair": "stage_repair",
    "eval": "stage_eval",
    "compose": "stage_compose",
}


def make_logger(logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    logf = logs_dir / "pipeline.log"

    def log(msg: str) -> None:
        line = f"{C.now()} {msg}"
        print(line, flush=True)
        with logf.open("a") as fh:
            fh.write(line + "\n")

    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["native"],
                    choices=STAGE_ORDER, help="which stages to run, in fixed order")
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--encoders", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--cells", nargs="+",
                    help="exact encoder__dataset ids (avoids encoder×dataset cartesian product)")
    ap.add_argument("--batch-q", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fresh", action="store_true", help="ignore prior per-cell outputs")
    args = ap.parse_args(argv)

    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)
    log = make_logger(ctx.logs)
    if args.cells:
        pairs = []
        for cid in args.cells:
            if cid not in ctx.cells:
                raise SystemExit(f"unknown cell {cid}")
            encoder, dataset = cid.split("__", 1)
            pairs.append((encoder, dataset))
    else:
        pairs = C.all_pairs(args.encoders, args.datasets)
    stages = [s for s in STAGE_ORDER if s in set(args.stages)]  

    log(f"=== pipeline start stages={stages} seeds={args.seeds} pairs={len(pairs)} "
        f"device={args.device} fresh={args.fresh} ===")
    overall = {"started": C.now(), "stages": {}}
    t_all = time.perf_counter()
    for stage in stages:
        mod = importlib.import_module(STAGE_MODULES[stage])
        for seed in args.seeds:
            if args.fresh:
                todo = pairs
            else:
                todo = [(e, d) for (e, d) in pairs if not mod.cell_done(ctx, seed, f"{e}__{d}")]
            log(f"--- stage={stage} seed={seed} todo={len(todo)}/{len(pairs)} ---")
            t0 = time.perf_counter()
            status = mod.run(ctx, seed, pairs, log)  
            status["wall_s"] = round(time.perf_counter() - t0, 1)
            overall["stages"].setdefault(stage, {})[str(seed)] = status
            if status.get("n_fail", 0):
                log(f"!!! stage={stage} seed={seed} had {status['n_fail']} failures; "
                    f"continuing (resume will retry).")
    overall["finished"] = C.now()
    overall["wall_s"] = round(time.perf_counter() - t_all, 1)
    
    
    
    
    C.write_json(ctx.logs / f"pipeline_status_{os.getpid()}.json", overall)
    log(f"=== pipeline done {overall['wall_s']}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
