

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

import common as C
import corrections

SMOKE_CELLS = [
    "clip_vitb32__mscoco_task0",
    "clip_sf_large__cirr_task7",
    "blip2_vitL__nights_task4",
]
KS = (1, 5, 10)

ALPHAS = [0.20, 0.10, 0.05, 0.02, 0.0, -0.05, -0.10, -0.20, -0.30, -0.50]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cells", nargs="+", default=SMOKE_CELLS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-q", type=int, default=64)
    args = ap.parse_args(argv)
    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)

    def log(msg: str) -> None:
        print(f"{C.now()} {msg}", flush=True)

    rows = []
    for cid in args.cells:
        cell = ctx.cells[cid]
        diag = C.load_sealed_diagnosis(ctx, args.seed, cid)
        phi = list(diag.get("phi") or [])
        seg = C.repartition(cell, args.seed)
        labeled = C.load_eval_labeled(cell, seg)
        query, pool, positives, ref_q = (
            labeled["eval_q"], labeled["pool"], labeled["positives"], labeled["ref_q"])
        bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
        k_out = max(KS + (C.RERANK_N,))
        params = {
            "h1_k": 10, "csls_k": 10, "qe_k": 10, "qe_power": 3.0,
            "csls_reference_queries": ref_q,
        }
        _, cos_i, st = corrections.search(
            "cosine", query, pool, k_out, params=params, device=ctx.device, batch_q=bq)
        if st != "ok":
            raise RuntimeError(st)
        base10 = float(C.recall_hits(cos_i, positives, 10).mean())
        log(f"[h2grid] {cid} phi={';'.join(phi)} baseR10={base10:.3f}")
        for alpha in ALPHAS:
            t0 = time.perf_counter()
            p = dict(params)
            p["qe_alpha"] = float(alpha)
            if abs(alpha) < 1e-15:
                top_i, status = cos_i, "ok"
            else:
                _, top_i, status = corrections.search(
                    "alpha_qe", query, pool, k_out, params=p,
                    device=ctx.device, batch_q=bq)
            if status != "ok":
                raise RuntimeError(status)
            r10 = float(C.recall_hits(top_i, positives, 10).mean())
            d10 = r10 - base10
            row = {
                "seed": int(args.seed), "encoder": cell.encoder, "dataset": cell.dataset,
                "phi_set": ";".join(phi) if phi else "empty",
                "h2_in_phi": int("h2" in phi),
                "method_family": "alpha_qe",
                "qe_alpha": float(alpha),
                "qe_k": 10, "qe_power": 3.0,
                "R@10_base": base10, "R@10": r10, "dR@10": d10,
                "useful": int(d10 >= C.USEFUL_DR10),
                "variant_wall_s": round(time.perf_counter() - t0, 2),
                "scope": "exploratory_h2_alpha_grid_smoke",
                "locked_appendix": 0,
            }
            rows.append(row)
            log(f"  alpha={alpha:+.2f} dR@10={d10:+.4f} useful={row['useful']}")
        gc.collect()
        if C.torch is not None and C.torch.cuda.is_available():
            C.torch.cuda.empty_cache()
    out = ctx.runs / f"S{args.seed}" / "h2_alpha_grid_smoke.csv"
    C.locked_csv(out, rows, fieldnames=list(rows[0].keys()))
    log(f"[h2grid] wrote {out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
