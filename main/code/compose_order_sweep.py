

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

import common as C
import compose as CMP
import corrections

SMOKE_CELLS = [
    "clip_vitb32__mscoco_task0",
    "clip_sf_large__cirr_task7",
    "blip2_vitL__nights_task4",
]
KS = (1, 5, 10)


VARIANTS = [
    ("csls", ["csls_p050"], "single"),
    ("qe", ["alpha_qe_a010"], "single"),
    ("csls_then_qe", ["csls_p050", "alpha_qe_a010"], "joint"),
    ("qe_then_csls", ["alpha_qe_a010", "csls_p050"], "joint"),
    ("csls_then_qe_then_csls", ["csls_p050", "alpha_qe_a010", "csls_p050"], "joint_keep_penalty"),
]


def _params(ref_q):
    return {
        "h1_k": 10, "csls_k": 10, "qe_k": 10, "qe_alpha": 0.10, "qe_power": 3.0,
        "diffusion_knn": 10, "diffusion_steps": 1, "diffusion_beta": 0.10,
        "diffusion_max_pool": 2_000_000,
        "csls_reference_queries": ref_q,
    }


def _recall(top_i, positives):
    return {k: float(C.recall_hits(top_i, positives, k).mean()) for k in KS}


def run_cell(ctx, seed, cell, log):
    diag = C.load_sealed_diagnosis(ctx, seed, cell.cell_id)
    phi = list(diag.get("phi") or [])
    P = C.predicted_P(phi, cell.dataset)
    repair = json.loads((ctx.runs / f"S{seed}" / "repair" / f"{cell.cell_id}.json").read_text())
    if not repair.get("analytic_complete"):
        raise RuntimeError(f"{cell.cell_id}: analytic repair incomplete")
    seg = C.repartition(cell, seed)
    if diag["split_manifest"].get("q_eval_indices_sha256") != C.hash_idx(seg["eval"]):
        raise RuntimeError("Q_eval hash drifted vs diagnosis")
    labeled = C.load_eval_labeled(cell, seg)
    query, pool, positives, ref_q = (
        labeled["eval_q"], labeled["pool"], labeled["positives"], labeled["ref_q"])
    bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
    k_out = max(KS + (C.RERANK_N,))
    params = _params(ref_q)

    t0 = time.perf_counter()
    cos_v, cos_i, st = corrections.search(
        "cosine", query, pool, k_out, params=params, device=ctx.device, batch_q=bq)
    if st != "ok":
        raise RuntimeError(st)
    base = _recall(cos_i, positives)
    cosine_wall = round(time.perf_counter() - t0, 2)

    indep = {r["method"]: r for r in (repair.get("rows") or []) if r.get("status") == "ok"}
    best_single = max(
        float(indep[m]["dR@10"]) for m in ("csls_p050", "alpha_qe_a010") if m in indep)
    rows = []
    for name, chain, kind in VARIANTS:
        t1 = time.perf_counter()
        packed = CMP.run_chain(
            query, pool, chain, k=k_out, params=params, device=ctx.device,
            batch_q=bq, cosine_top_idx=cos_i, cosine_top_val=cos_v)
        if packed["status"] != "ok":
            raise RuntimeError(f"{name} failed: {packed['status']}")
        rec = _recall(packed["top_idx"], positives)
        d10 = rec[10] - base[10]
        delta = d10 - best_single
        qe_src = None
        for step in packed["steps"]:
            if step["method"] == "alpha_qe_a010":
                qe_src = step.get("neighbor_source")
        match = None
        if name == "csls" and "csls_p050" in indep:
            match = abs(d10 - float(indep["csls_p050"]["dR@10"])) < 1e-12
        if name == "qe" and "alpha_qe_a010" in indep:
            match = abs(d10 - float(indep["alpha_qe_a010"]["dR@10"])) < 1e-12
        row = {
            "seed": int(seed), "encoder": cell.encoder, "dataset": cell.dataset,
            "phi_set": ";".join(phi) if phi else "empty",
            "P": ";".join(P) if P else "empty",
            "variant": name, "kind": kind, "chain": ">".join(chain),
            "neighbor_source_qe": qe_src,
            "R@10_base": base[10], "R@10": rec[10], "dR@10": d10,
            "dR@10_best_single": best_single,
            "delta_vs_best_single": delta,
            "beats_best_single": int(d10 > best_single + 1e-15),
            "beats_best_by_useful": int(delta >= C.USEFUL_DR10),
            "indep_match": match,
            "baseline_match_repair": abs(base[10] - float(repair["baseline"]["R@10"])) < 1e-12,
            "cosine_wall_s": cosine_wall,
            "variant_wall_s": round(time.perf_counter() - t1, 2),
            "scope": "exploratory_smoke_order_sweep",
            "locked_appendix": 0,
        }
        rows.append(row)
        log(f"[sweep s{seed}] {cell.cell_id} {name} d10={d10:+.4f} "
            f"vs_best={delta:+.4f} beats={row['beats_best_single']} match={match}")
    return rows


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

    all_rows = []
    for cid in args.cells:
        cell = ctx.cells[cid]
        t0 = time.perf_counter()
        try:
            all_rows.extend(run_cell(ctx, args.seed, cell, log))
            log(f"[sweep] done {cid} {time.perf_counter()-t0:.1f}s")
        except Exception as exc:
            log(f"[sweep] FAIL {cid} {exc!r}")
            raise
        finally:
            gc.collect()
            if C.torch is not None and C.torch.cuda.is_available():
                C.torch.cuda.empty_cache()
    out = ctx.runs / f"S{args.seed}" / "compose_order_sweep.csv"
    fields = list(all_rows[0].keys()) if all_rows else []
    C.locked_csv(out, all_rows, fieldnames=fields)
    log(f"[sweep] wrote {out} n={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
