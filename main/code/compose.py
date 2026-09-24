
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import common as C
import corrections


def analytic_chain(P: List[str]) -> List[str]:
    pset = set(P)
    return [m for m in C.COMPOSE_ORDER if m in pset]


def expanded_queries(
    query: np.ndarray,
    pool: np.ndarray,
    neighbor_idx: np.ndarray,
    params: Dict,
    device,
    batch_q: int,
) -> np.ndarray:
    qe_k = int(params.get("qe_k", 10))
    alpha = float(params.get("qe_alpha", 0.10))
    power = float(params.get("qe_power", 3.0))
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    nq, dim = query.shape
    qe_k = min(qe_k, neighbor_idx.shape[1], pool.shape[0])
    out = torch.empty((nq, dim), dtype=torch.float32)
    neigh = torch.from_numpy(np.ascontiguousarray(neighbor_idx[:, :qe_k])).to(device)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        i0 = neigh[s:e]
        gathered = p_t[i0]
        v0 = (q_t[s:e].unsqueeze(1) * gathered).sum(dim=-1)
        w = v0.clamp(min=0).pow(power)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        mix = torch.bmm(w.unsqueeze(1), gathered).squeeze(1)
        q2 = (1.0 - alpha) * q_t[s:e] + alpha * mix
        out[s:e] = torch.nn.functional.normalize(q2, dim=-1).cpu()
    return out.numpy()


def alpha_qe_from_neighbors(
    query: np.ndarray,
    pool: np.ndarray,
    k: int,
    neighbor_idx: np.ndarray,
    params: Dict,
    device,
    batch_q: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q2 = expanded_queries(query, pool, neighbor_idx, params, device, batch_q)
    top_v, top_i, status = corrections.search(
        "cosine", q2, pool, k, params=params, device=device, batch_q=batch_q)
    if status != "ok":
        raise RuntimeError(f"α-QE cosine on q' failed: {status}")
    return top_v, top_i, q2


def run_chain(
    query: np.ndarray,
    pool: np.ndarray,
    chain: List[str],
    *,
    k: int,
    params: Dict,
    device,
    batch_q: int,
    cosine_top_idx: Optional[np.ndarray] = None,
    cosine_top_val: Optional[np.ndarray] = None,
) -> Dict:
    if not chain:
        raise ValueError("compose chain is empty")
    prev_idx = cosine_top_idx
    prev_val = cosine_top_val
    query_cur = query
    query_is_original = True
    neighbor_source = "cosine"
    steps = []
    for method in chain:
        if method == "csls_p050":
            top_v, top_i, status = corrections.search(
                method, query_cur, pool, k,
                cosine_top_idx=cosine_top_idx if query_is_original else None,
                cosine_top_val=cosine_top_val if query_is_original else None,
                params=params, device=device, batch_q=batch_q)
            src = "cosine" if query_is_original else "expanded_query"
        elif method == "alpha_qe_a010":
            if prev_idx is None:
                raise RuntimeError("α-QE compose needs a previous ranking")
            qe_k = min(int(params.get("qe_k", 10)), prev_idx.shape[1])
            top_v, top_i, q2 = alpha_qe_from_neighbors(
                query, pool, k, prev_idx[:, :qe_k], params, device, batch_q)
            query_cur = q2
            query_is_original = False
            status = "ok"
            src = neighbor_source
        elif method == "diffusion":
            
            
            if method == chain[0]:
                top_v, top_i, status = corrections.search(
                    method, query, pool, k, params=params,
                    device=device, batch_q=batch_q)
                src = "cosine"
            else:
                status = "skipped_compose_needs_dense_prev_scores"
                top_v = top_i = None
                src = neighbor_source
        else:
            raise KeyError(method)
        steps.append({
            "method": method, "status": status, "neighbor_source": src,
        })
        if status != "ok" or top_i is None:
            return {"status": status, "steps": steps, "top_idx": None, "top_val": None,
                    "neighbor_source": src}
        prev_idx, prev_val = top_i, top_v
        neighbor_source = method
        steps[-1]["top10_head"] = top_i[0, :10].tolist()
        steps[-1]["top_idx"] = top_i
        steps[-1]["top_val"] = top_v
    return {
        "status": "ok",
        "steps": steps,
        "top_idx": prev_idx,
        "top_val": prev_val,
        "neighbor_source": steps[-1]["neighbor_source"] if chain[-1] == "alpha_qe_a010"
        else neighbor_source,
        "chain": chain,
    }
