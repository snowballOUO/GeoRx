

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import common_ablation as A

VARIANTS = ["C-joint", "C-independent", "C-mom", "C-noH3guard"]
MODE = {
    "C-joint": "robust_joint",
    "C-independent": "independent",
    "C-mom": "mom_joint",
    "C-noH3guard": "robust_joint",
}


def family_fp(profile, pool, cfg, dim, null_seed, guard):
    q = A.n1_queries(32 * A.WINDOW_SIZE, dim, null_seed + 50)
    fam = 0
    per_type = {t: 0 for t in A.C.TYPE_METRICS}
    for w in range(32):
        sub = q[w * A.WINDOW_SIZE:(w + 1) * A.WINDOW_SIZE]
        out = A.diagnose_variant(profile, sub, pool, cfg,
                                 null_seed + 60 + w,
                                 n_windows=1, h3_guard=guard)
        if out["phi"]:
            fam += 1
        for t in per_type:
            if t in out["phi"]:
                per_type[t] += 1
    return {"family_fp_windows": fam, "n_windows": 32,
            "family_fp_rate": fam / 32.0,
            "per_type_windows": per_type}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=A.SEEDS)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    out = root / "runs"
    out.mkdir(parents=True, exist_ok=True)
    ctx = A.setup_context()
    rows = []
    fp_rows = []
    started = A.now()

    for variant in VARIANTS:
        mode = MODE[variant]
        guard = variant != "C-noH3guard"
        for seed in args.seeds:
            for encoder, dataset in A.pairs():
                cid = f"{encoder}__{dataset}"
                cell = ctx.cells[cid]
                t0 = time.perf_counter()
                _, inp = A.load_pool_and_segments(cell, seed)
                bq = A.C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
                profile, cfg = A.fit_native_profile(
                    inp, ctx.device, bq, seed, mode=mode)
                out_d = A.diagnose_variant(
                    profile, inp["fault_q"], inp["pool"], cfg,
                    seed + 4, n_windows=A.C.FAULT_WINDOWS,
                    h3_guard=guard,
                )
                rows.append({
                    "variant": variant, "seed": seed, "encoder": encoder,
                    "dataset": dataset, "dim": cell.dim,
                    "n_pool": cell.n_pool, "phi": ";".join(out_d["phi"]),
                    "tau": out_d["tau"],
                    "z_h1": out_d["per_type"]["h1"]["z"],
                    "z_h2": out_d["per_type"]["h2"]["z"],
                    "z_h3": out_d["per_type"]["h3"]["z"],
                    "z_h4": out_d["per_type"]["h4"]["z"],
                    "z_h5": out_d["per_type"]["h5"]["z"],
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                })
                del inp, profile
                gc.collect()
                A.C._free()

        if not args.no_gate:
            def builder(d, n, device, bq, s, mode=mode):
                return A.fit_sphere_profile(
                    d, n, device, bq, A.C.n1_null_seed(d, n), mode=mode)
            gate = A.gate_records(builder, ctx.device, ctx.batch_q, 1,
                                  h3_guard=guard)
            A.write_json(out / f"gate_{variant}.json", gate)
            
            fp_for_variant = []
            for geom in A.GEOMS:
                d, n = geom["d"], geom["n_pool"]
                ns = A.C.n1_null_seed(d, n)
                profile, pool = builder(d, n, ctx.device, geom["batch_q"], 1)
                cfg = A.diag_cfg(ctx.device, geom["batch_q"], pool)
                rec = family_fp(profile, pool, cfg, d, ns, guard)
                rec.update({"variant": variant, "geom": geom["name"]})
                fp_for_variant.append(rec)
                del pool, profile
                A.C._free()
            fp_rows.extend(fp_for_variant)

    if rows:
        with (out / "conformal_ablation.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if fp_rows:
        with (out / "sphere_false_positive.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fp_rows[0]))
            writer.writeheader()
            writer.writerows(fp_rows)
    A.write_json(out / "status.json", {
        "experiment": "20260919_ablation_conformal",
        "state": "complete", "started": started, "finished": A.now(),
        "variants": VARIANTS, "seeds": args.seeds,
        "qrels_opened": False, "corrections_run": False,
    })


if __name__ == "__main__":
    main()

