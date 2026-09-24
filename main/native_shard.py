
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import common as C
import stage_native as N


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-file", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--batch-q", type=int, default=128)
    args = ap.parse_args()

    plan = json.loads(Path(args.jobs_file).read_text())
    worker = plan["workers"][args.worker]
    jobs = worker["jobs"]
    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)
    if not C.gate_passed(ctx):
        raise SystemExit("sphere gate is not passed; native shard remains blocked")

    out = ctx.runs / "native_shards" / f"{args.worker}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    started = C.now()
    for job in jobs:
        seed = int(job["seed"])
        cid = str(job["cell_id"])
        cell_dir = ctx.runs / f"S{seed}" / "evaluations" / cid
        t0 = time.perf_counter()
        rec = {"seed": seed, "cell_id": cid, "worker": args.worker}
        try:
            if N.cell_done(ctx, seed, cid):
                rec["state"] = "skip"
            else:
                cell = ctx.cells[cid]
                N._run_cell(ctx, seed, cell, cell_dir,
                            lambda msg: print(f"[{args.worker}] {msg}", flush=True))
                rec["state"] = "ok"
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            print(f"[{args.worker}] {rec['state']} S{seed} {cid} {rec['wall_s']}s", flush=True)
        except Exception as exc:  
            rec.update({"state": "fail", "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "wall_s": round(time.perf_counter() - t0, 1)})
            print(f"[{args.worker}] FAIL S{seed} {cid}: {exc!r}", flush=True)
        records.append(rec)

    C.write_json(out, {"worker": args.worker, "device": args.device,
                       "batch_q": args.batch_q, "started": started,
                       "finished": C.now(), "n_jobs": len(jobs),
                       "records": records})
    return 0 if all(r["state"] != "fail" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
