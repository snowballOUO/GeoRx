
from __future__ import annotations

from typing import Optional, Tuple

import os
import numpy as np
import torch

_n_threads = os.environ.get("OMP_NUM_THREADS")
if _n_threads:
    try:
        torch.set_num_threads(int(_n_threads))
    except (TypeError, ValueError, RuntimeError):
        pass


def get_device(name: str = "cpu") -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def brute_topk(
    queries: np.ndarray,
    pool: np.ndarray,
    k: int,
    *,
    device: Optional[torch.device] = None,
    batch_q: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    device = device or get_device("cpu")
    nq, npool = queries.shape[0], pool.shape[0]
    k = min(int(k), npool)
    q_t = torch.from_numpy(np.ascontiguousarray(queries)).to(device)
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    vals = torch.empty((nq, k), dtype=torch.float32, device="cpu")
    idxs = torch.empty((nq, k), dtype=torch.int64, device="cpu")
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        sim = q_t[s:e] @ p_t.T
        v, i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        vals[s:e] = v.cpu()
        idxs[s:e] = i.cpu()
        del sim
    return vals.numpy(), idxs.numpy()


def recall_at_k(
    top_idx: np.ndarray,
    positives: list,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> dict:
    n = top_idx.shape[0]
    hits = {k: 0 for k in ks}
    n_eval = 0
    for qi in range(n):
        gold = {g for g in positives[qi] if g >= 0}
        if not gold:
            continue
        n_eval += 1
        pred = top_idx[qi].tolist()
        for k in ks:
            if gold.intersection(pred[:k]):
                hits[k] += 1
    denom = max(n_eval, 1)
    out = {f"recall@{k}": hits[k] / denom for k in ks}
    out["n_eval"] = n_eval
    return out
