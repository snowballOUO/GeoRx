

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import common_ablation as A

VARIANTS = ["R-full", "R-dim", "R-pool", "R-universal", "R-self"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=A.SEEDS)
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the three fixed synthetic W0/W1 geometries")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    out = root / "runs"
    out.mkdir(parents=True, exist_ok=True)
    ctx = A.setup_context()
    rows = []
    started = A.now()

    for variant in VARIANTS:
        for seed in args.seeds:
            for encoder, dataset in A.pairs():
                cid = f"{encoder}__{dataset}"
                cell = ctx.cells[cid]
                t0 = time.perf_counter()
                seg, inp = A.load_pool_and_segments(cell, seed)
                batch_q = A.C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
                if variant == "R-self":
                    profile, cfg = A.fit_native_profile(
                        inp, ctx.device, batch_q, seed, mode="robust_joint")
                    source = "native_train_cal"
                else:
                    rd, rn = A.variant_dims(variant, cell.dim, cell.n_pool)
                    profile, _ = A.fit_sphere_profile(
                        rd, rn, ctx.device, batch_q, A.C.n1_null_seed(rd, rn))
                    cfg = A.diag_cfg(ctx.device, batch_q, inp["pool"])
                    source = f"sphere_d{rd}_n{rn}"
                diag = A.diagnose_variant(
                    profile, inp["fault_q"], inp["pool"], cfg,
                    seed + 4, n_windows=A.C.FAULT_WINDOWS,
                )
                rows.append({
                    "variant": variant, "seed": seed, "encoder": encoder,
                    "dataset": dataset, "dim": cell.dim,
                    "n_pool": cell.n_pool, "reference": source,
                    "tau": diag["tau"], "phi": ";".join(diag["phi"]),
                    "z_h1": diag["per_type"]["h1"]["z"],
                    "z_h2": diag["per_type"]["h2"]["z"],
                    "z_h3": diag["per_type"]["h3"]["z"],
                    "z_h4": diag["per_type"]["h4"]["z"],
                    "z_h5": diag["per_type"]["h5"]["z"],
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                })
                del seg, inp, profile
                gc.collect()
                A.C._free()

        
        if not args.no_gate and variant != "R-self":
            def builder(d, n, device, bq, s):
                rd, rn = A.variant_dims(variant, d, n)
                return A.fit_sphere_profile(rd, rn, device, bq,
                                            A.C.n1_null_seed(rd, rn))
            gate_rows = A.gate_records(builder, ctx.device, ctx.batch_q, 1)
            for r in gate_rows:
                r.update({"variant": variant, "seed": 1})
            A.write_json(out / f"gate_{variant}.json", gate_rows)

    fields = list(rows[0]) if rows else []
    if fields:
        with (out / "reference_model_ablation.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    A.write_json(out / "status.json", {
        "experiment": "20260919_ablation_reference_model",
        "state": "complete", "started": started, "finished": A.now(),
        "variants": VARIANTS, "seeds": args.seeds,
        "cells": [f"{e}__{d}" for e, d in A.pairs()],
        "qrels_opened": False, "corrections_run": False,
    })


if __name__ == "__main__":
    main()
