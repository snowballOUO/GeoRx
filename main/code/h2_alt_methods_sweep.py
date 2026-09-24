

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


def _recall(top_i, positives):
    return {k: float(C.recall_hits(top_i, positives, k).mean()) for k in KS}


def mmr_rerank(sim_q: np.ndarray, sim_cc: np.ndarray, lam: float) -> np.ndarray:
    n = sim_q.shape[0]
    if lam >= 1.0 - 1e-12:
        return np.argsort(-sim_q, kind="stable")
    max_red = np.zeros(n, dtype=np.float64)
    taken = np.zeros(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int64)
    for t in range(n):
        sc = lam * sim_q - (1.0 - lam) * max_red
        sc = np.where(taken, -1e18, sc)
        j = int(np.argmax(sc))
        order[t] = j
        taken[j] = True
        max_red = np.maximum(max_red, sim_cc[j])
    return order


def whiten_scores(q: np.ndarray, vecs: np.ndarray, eps: float) -> np.ndarray:
    n, d = vecs.shape
    if n < 2:
        return vecs @ q
    mu = vecs.mean(axis=0)
    x = vecs - mu
    q0 = q - mu
    
    try:
        _, s, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return vecs @ q
    r = s.size
    var = (s * s) / max(n - 1, 1)
    scale = np.sqrt(var + float(eps))
    scale = np.maximum(scale, 1e-8)
    q_w = (vt @ q0) / scale
    c_w = (x @ vt.T) / scale
    return c_w @ q_w


def apply_order(short_idx: np.ndarray, order: np.ndarray, n_pool: int, k_out: int) -> np.ndarray:
    ranked = short_idx[order]
    out = np.full(k_out, 0, dtype=np.int64)
    m = min(k_out, ranked.size)
    out[:m] = ranked[:m]
    if m < k_out:
        
        out[m:] = short_idx[min(m, short_idx.size - 1)]
    return out


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
        query, pool, positives = labeled["eval_q"], labeled["pool"], labeled["positives"]
        bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
        k_out = max(KS + (C.RERANK_N,))
        t0 = time.perf_counter()
        cos_v, cos_i, st = corrections.search(
            "cosine", query, pool, k_out, device=ctx.device, batch_q=bq)
        if st != "ok":
            raise RuntimeError(st)
        base = _recall(cos_i, positives)
        log(f"[h2alt] {cid} phi={';'.join(phi)} baseR10={base[10]:.3f} "
            f"cosine {time.perf_counter()-t0:.1f}s")
        nq = query.shape[0]

        def emit(method, n_short, params, top_i, wall, note=""):
            rec = _recall(top_i, positives)
            d = {k: rec[k] - base[k] for k in KS}
            row = {
                "seed": int(args.seed), "encoder": cell.encoder, "dataset": cell.dataset,
                "phi_set": ";".join(phi) if phi else "empty",
                "method": method, "n_short": int(n_short),
                "params": params, "note": note,
                "R@10_base": base[10], "R@10": rec[10],
                "dR@1": d[1], "dR@5": d[5], "dR@10": d[10],
                "useful": int(d[10] >= C.USEFUL_DR10),
                "wall_s": round(wall, 3),
                "scope": "exploratory_h2_alt_smoke",
            }
            rows.append(row)
            log(f"  {method} N={n_short} {params} dR@10={d[10]:+.4f} useful={row['useful']}")

        
        for n_short in (50, 100):
            short = cos_i[:, :n_short]
            pack = []
            for qi in range(nq):
                sl = short[qi]
                v = pool[sl]
                sim_q = (v @ query[qi]).astype(np.float64, copy=False)
                sim_cc = (v @ v.T).astype(np.float64, copy=False)
                pack.append((sl, sim_q, sim_cc))
            for lam in (0.3, 0.5, 0.7, 0.9, 1.0):
                t2 = time.perf_counter()
                new_i = np.empty((nq, k_out), dtype=np.int64)
                for qi, (sl, sim_q, sim_cc) in enumerate(pack):
                    order = mmr_rerank(sim_q, sim_cc, lam)
                    new_i[qi] = apply_order(sl, order, pool.shape[0], k_out)
                emit("mmr", n_short, f"lambda={lam:.1f}", new_i, time.perf_counter() - t2)

        
        for n_short in (50, 100):
            short = cos_i[:, :n_short]
            for eps in (1e-4, 1e-2, 0.1, 1.0):
                t2 = time.perf_counter()
                new_i = np.empty((nq, k_out), dtype=np.int64)
                for qi in range(nq):
                    sl = short[qi]
                    v = pool[sl]
                    sc = whiten_scores(query[qi], v, eps)
                    order = np.argsort(-sc, kind="stable")
                    new_i[qi] = apply_order(sl, order, pool.shape[0], k_out)
                emit("local_whiten", n_short, f"eps={eps:g}", new_i, time.perf_counter() - t2)

        
        n_short = 100
        short = cos_i[:, :n_short]
        for temp in (0.05, 0.1, 0.3, 0.5, 1.0):
            t2 = time.perf_counter()
            new_i = np.empty((nq, k_out), dtype=np.int64)
            for qi in range(nq):
                sl = short[qi]
                v = pool[sl]
                sim_q = v @ query[qi]
                order = np.argsort(-(sim_q / float(temp)), kind="stable")
                new_i[qi] = apply_order(sl, order, pool.shape[0], k_out)
            emit("local_temp", n_short, f"T={temp:g}", new_i, time.perf_counter() - t2,
                 note="strictly_increasing_in_cosine_expect_dR0")

        gc.collect()
        if C.torch is not None and C.torch.cuda.is_available():
            C.torch.cuda.empty_cache()

    out = ctx.runs / f"S{args.seed}" / "h2_alt_methods_smoke.csv"
    C.locked_csv(out, rows, fieldnames=list(rows[0].keys()))
    log(f"[h2alt] wrote {out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
