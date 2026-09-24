

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch

import common as C
import corrections

SMOKE_CELLS = [
    "clip_vitb32__mscoco_task0",
    "clip_sf_large__cirr_task7",
    "blip2_vitL__nights_task4",
]
KS = (1, 5, 10)
N_SHORT = 100
K_LIST = (10, 20, 50)
BETA_LIST = (0.25, 0.5, 1.0, 2.0)


def pool_knn_idx(pool: np.ndarray, k: int, device, batch: int = 256) -> np.ndarray:
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    n = p_t.shape[0]
    k = min(k, n - 1)
    idx = torch.empty((n, k), dtype=torch.int64, device=device)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sim = p_t[s:e] @ p_t.T
        for j, gi in enumerate(range(s, e)):
            sim[j, gi] = -1e9
        _, i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        idx[s:e] = i
    return idx.cpu().numpy()


def reciprocal_sets(knn: np.ndarray) -> list:
    n, k = knn.shape
    member = [set(row.tolist()) for row in knn]
    out = []
    for v in range(n):
        rec = [int(z) for z in knn[v] if v in member[z]]
        out.append(set(rec))
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def rerank_shortlist(cos_v, cos_i, q_nn, item_sets, beta, k_out):
    nq, n_short = cos_i.shape
    new_i = np.empty((nq, k_out), dtype=np.int64)
    new_v = np.empty((nq, k_out), dtype=np.float32)
    for qi in range(nq):
        nqset = q_nn[qi]
        sl = cos_i[qi]
        sc = np.empty(n_short, dtype=np.float64)
        for j, vid in enumerate(sl.tolist()):
            sc[j] = float(cos_v[qi, j]) + beta * jaccard(nqset, item_sets[int(vid)])
        order = np.argsort(-sc, kind="stable")
        ranked = sl[order]
        m = min(k_out, ranked.size)
        new_i[qi, :m] = ranked[:m]
        new_v[qi, :m] = sc[order][:m]
        if m < k_out:
            new_i[qi, m:] = ranked[m - 1]
            new_v[qi, m:] = sc[order][m - 1]
    return new_v, new_i


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
        base = {k: float(C.recall_hits(cos_i, positives, k).mean()) for k in KS}
        log(f"[schemeB] {cid} phi={';'.join(phi)} baseR10={base[10]:.3f} "
            f"pool={pool.shape[0]} cosine {time.perf_counter()-t0:.1f}s")

        t_knn = time.perf_counter()
        knn50 = pool_knn_idx(pool, max(K_LIST), ctx.device)
        log(f"  pool kNN k={max(K_LIST)} {time.perf_counter()-t_knn:.1f}s")
        recip50 = None

        short = cos_i[:, :N_SHORT]
        short_v = cos_v[:, :N_SHORT]

        def emit(method, k_nn, beta, top_i, wall):
            rec = {k: float(C.recall_hits(top_i, positives, k).mean()) for k in KS}
            d10 = rec[10] - base[10]
            row = {
                "seed": int(args.seed), "encoder": cell.encoder, "dataset": cell.dataset,
                "phi_set": ";".join(phi) if phi else "empty",
                "method": method, "k_nn": int(k_nn), "beta": float(beta),
                "n_short": N_SHORT,
                "R@10_base": base[10], "R@10": rec[10], "dR@10": d10,
                "useful": int(d10 >= C.USEFUL_DR10),
                "wall_s": round(wall, 3),
                "scope": "exploratory_h2_schemeB_smoke",
            }
            rows.append(row)
            log(f"  {method} k={k_nn} beta={beta:g} dR@10={d10:+.4f} useful={row['useful']}")

        for k_nn in K_LIST:
            knn = knn50[:, :k_nn]
            q_nn = [set(row[:k_nn].tolist()) for row in cos_i]
            item_sets = [set(knn[v].tolist()) for v in range(knn.shape[0])]
            for beta in BETA_LIST:
                t2 = time.perf_counter()
                _, top_i = rerank_shortlist(short_v, short, q_nn, item_sets, beta, k_out)
                emit("shared_nn", k_nn, beta, top_i, time.perf_counter() - t2)

            t1 = time.perf_counter()
            rec_sets = reciprocal_sets(knn)
            log(f"  reciprocal sets k={k_nn} {time.perf_counter()-t1:.1f}s")
            q_rec = []
            for qi in range(query.shape[0]):
                nqset = []
                for z in cos_i[qi, :k_nn].tolist():
                    z = int(z)
                    zk = int(knn[z, -1])
                    thr = float(np.dot(pool[z], pool[zk]))
                    if float(np.dot(query[qi], pool[z])) >= thr:
                        nqset.append(z)
                q_rec.append(set(nqset))
            for beta in BETA_LIST:
                t2 = time.perf_counter()
                _, top_i = rerank_shortlist(short_v, short, q_rec, rec_sets, beta, k_out)
                emit("k_reciprocal", k_nn, beta, top_i, time.perf_counter() - t2)

        gc.collect()
        if C.torch is not None and C.torch.cuda.is_available():
            C.torch.cuda.empty_cache()

    out = ctx.runs / f"S{args.seed}" / "h2_schemeB_smoke.csv"
    C.locked_csv(out, rows, fieldnames=list(rows[0].keys()))
    log(f"[schemeB] wrote {out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
