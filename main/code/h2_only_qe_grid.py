

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch

import common as C


H2_ONLY = [
    (1, "blip_ff_large__visualnews_task0"),
    (2, "blip_ff_large__webqa_task1"),
    (3, "blip_ff_large__fashioniq_task7"),
    (3, "gme_qwen2vl_2b__nights_task4"),
]
KS = (1, 5, 10)
ALPHAS = (0.20, 0.10, 0.05, 0.02, 0.0, -0.05, -0.10, -0.20, -0.30)
QE_KS = (5, 10, 20, 50)
POWERS = (1.0, 3.0)
LOCKED = (0.10, 10, 3.0)  


def _recall(top_i, positives):
    return {k: float(C.recall_hits(top_i, positives, k).mean()) for k in KS}


def _topk_idx(sim: torch.Tensor, k: int) -> np.ndarray:
    k = min(k, sim.shape[1])
    return torch.topk(sim, k, dim=1, largest=True, sorted=True).indices.cpu().numpy()


def _mix(query: torch.Tensor, pool: torch.Tensor, nn_idx: torch.Tensor, nn_val: torch.Tensor,
         qe_k: int, power: float) -> torch.Tensor:
    v = nn_val[:, :qe_k].clamp(min=0).pow(power)
    w = v / (v.sum(dim=1, keepdim=True) + 1e-8)
    gathered = pool[nn_idx[:, :qe_k]]
    return torch.bmm(w.unsqueeze(1), gathered).squeeze(1)


def run_cell(ctx, seed: int, cid: str, device: torch.device, batch_q: int, log) -> list:
    cell = ctx.cells[cid]
    diag = C.load_sealed_diagnosis(ctx, seed, cid)
    phi = list(diag.get("phi") or [])
    if phi != ["h2"]:
        log(f"[h2only] SKIP {cid} s{seed}: sealed phi={phi} (want [h2])")
        return []
    seg = C.repartition(cell, seed)
    labeled = C.load_eval_labeled(cell, seg)
    query_np, pool_np, positives = labeled["eval_q"], labeled["pool"], labeled["positives"]
    k_out = max(KS + (C.RERANK_N,))
    q = torch.from_numpy(np.ascontiguousarray(query_np)).to(device)
    p = torch.from_numpy(np.ascontiguousarray(pool_np)).to(device)
    nq = q.shape[0]
    max_k = min(max(QE_KS), p.shape[0])

    t0 = time.perf_counter()
    nn_val = torch.empty((nq, max_k), device=device)
    nn_idx = torch.empty((nq, max_k), dtype=torch.int64, device=device)
    cos_top = np.empty((nq, k_out), dtype=np.int64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        sim = q[s:e] @ p.T
        v, i = torch.topk(sim, max_k, dim=1, largest=True, sorted=True)
        nn_val[s:e], nn_idx[s:e] = v, i
        cos_top[s:e] = _topk_idx(sim, k_out)
        del sim
    base = _recall(cos_top, positives)
    log(f"[h2only] {cid} s{seed} phi=h2 nq={nq} n_pool={cell.n_pool} dim={cell.dim} "
        f"base R1/5/10={base[1]:.3f}/{base[5]:.3f}/{base[10]:.3f} "
        f"neighbors {time.perf_counter()-t0:.1f}s")

    rows = []
    for qe_k in QE_KS:
        for power in POWERS:
            mix = _mix(q, p, nn_idx, nn_val, qe_k, power)
            for alpha in ALPHAS:
                t1 = time.perf_counter()
                if abs(alpha) < 1e-15:
                    rec = dict(base)
                    status = "cosine_identity"
                else:
                    q2 = torch.nn.functional.normalize((1.0 - alpha) * q + alpha * mix, dim=-1)
                    top = np.empty((nq, k_out), dtype=np.int64)
                    for s in range(0, nq, batch_q):
                        e = min(s + batch_q, nq)
                        sim = q2[s:e] @ p.T
                        top[s:e] = _topk_idx(sim, k_out)
                        del sim
                    rec = _recall(top, positives)
                    status = "ok"
                    del q2
                d = {k: rec[k] - base[k] for k in KS}
                locked = int((abs(alpha - LOCKED[0]) < 1e-12) and qe_k == LOCKED[1]
                             and abs(power - LOCKED[2]) < 1e-12)
                row = {
                    "seed": seed, "encoder": cell.encoder, "dataset": cell.dataset,
                    "cell_id": cid, "phi_set": "h2",
                    "method_family": "alpha_qe",
                    "qe_alpha": float(alpha), "qe_k": int(qe_k), "qe_power": float(power),
                    "locked_setting": locked,
                    "R@1_base": base[1], "R@5_base": base[5], "R@10_base": base[10],
                    "R@1": rec[1], "R@5": rec[5], "R@10": rec[10],
                    "dR@1": d[1], "dR@5": d[5], "dR@10": d[10],
                    "useful": int(d[10] >= C.USEFUL_DR10),
                    "status": status,
                    "variant_wall_s": round(time.perf_counter() - t1, 3),
                    "scope": "exploratory_h2_only_qe_grid",
                }
                rows.append(row)
                mark = " LOCKED" if locked else ""
                log(f"  a={alpha:+.2f} k={qe_k} p={power:g} "
                    f"dR1/5/10={d[1]:+.4f}/{d[5]:+.4f}/{d[10]:+.4f} "
                    f"useful={row['useful']}{mark}")
            del mix
    del q, p, nn_val, nn_idx
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-q", type=int, default=64)
    args = ap.parse_args(argv)
    ctx = C.build_context(batch_q=args.batch_q, device_str="cpu")
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    def log(msg: str) -> None:
        print(f"{C.now()} {msg}", flush=True)

    log(f"[h2only] device={device} n_cells={len(H2_ONLY)} "
        f"alphas={list(ALPHAS)} qe_k={list(QE_KS)} powers={list(POWERS)}")
    rows = []
    for seed, cid in H2_ONLY:
        rows.extend(run_cell(ctx, seed, cid, device, args.batch_q, log))
    out = ctx.runs / "h2_only_qe_grid.csv"
    C.locked_csv(out, rows, fieldnames=list(rows[0].keys()) if rows else None)
    log(f"[h2only] wrote {out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
