
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from retrieve import brute_topk, get_device


def available_methods() -> List[str]:
    return [
        "hubness",
        "hub_z_l04",
        "hub_z_l01",
        "hub_rel",
        "csls",
        "csls_p010",
        "csls_p020",
        "csls_p030",
        "csls_p040",
        "csls_p050",
        "csls_n010",
        "csls_n020",
        "csls_n030",
        "csls_pool",
        "alpha_qe",
        "alpha_qe_a002",
        "alpha_qe_a005",
        "alpha_qe_a010",
        "alpha_qe_a020",
        "alpha_qe_n005",
        "alpha_qe_n010",
        "diffusion",
        "listwise_ce",
        "cross_encoder",
    ]


def search(
    method: str,
    query: np.ndarray,
    pool: np.ndarray,
    k: int,
    *,
    cosine_top_idx: Optional[np.ndarray] = None,
    cosine_top_val: Optional[np.ndarray] = None,
    h1_counts: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    device=None,
    batch_q: int = 128,
) -> Tuple[np.ndarray, np.ndarray, str]:
    params = params or {}
    device = device or get_device("cpu")
    if method == "cosine":
        v, i = brute_topk(query, pool, k, device=device, batch_q=batch_q)
        return v, i, "ok"
    if method == "hubness":
        return _hubness(query, pool, k, h1_counts, params, device, batch_q)
    if method == "hub_z_l04":
        p = {**params, "hub_lambda": float(params.get("hub_z_l04_lambda", 0.4))}
        return _hubness(query, pool, k, h1_counts, p, device, batch_q)
    if method == "hub_z_l01":
        p = {**params, "hub_lambda": float(params.get("hub_z_l01_lambda", 0.01))}
        return _hubness(query, pool, k, h1_counts, p, device, batch_q)
    if method == "hub_rel":
        return _hub_rel(query, pool, k, h1_counts, params, device, batch_q)
    if method == "csls" or method.startswith("csls_p") or method.startswith("csls_n"):
        penalties = {
            "csls": 0.5,
            "csls_p010": 0.1,
            "csls_p020": 0.2,
            "csls_p030": 0.3,
            "csls_p040": 0.4,
            "csls_p050": 0.5,
            "csls_n010": -0.1,
            "csls_n020": -0.2,
            "csls_n030": -0.3,
        }
        if method not in penalties:
            raise KeyError(f"Unknown CSLS penalty variant {method}")
        return _csls(
            query, pool, k, params, device, batch_q,
            cosine_top_val=cosine_top_val, r_c_side="query",
            penalty_lambda=penalties[method],
        )
    if method == "csls_pool":
        return _csls(
            query, pool, k, params, device, batch_q,
            cosine_top_val=cosine_top_val, r_c_side="pool",
        )
    if method == "alpha_qe" or method.startswith("alpha_qe_"):
        alphas = {
            "alpha_qe_a002": 0.02,
            "alpha_qe_a005": 0.05,
            "alpha_qe_a010": 0.1,
            "alpha_qe_a020": 0.2,
            "alpha_qe_n005": -0.05,
            "alpha_qe_n010": -0.1,
        }
        if method == "alpha_qe":
            p = params
        elif method in alphas:
            p = {**params, "qe_alpha": alphas[method]}
        else:
            raise KeyError(f"Unknown alpha-QE variant {method}")
        return _alpha_qe(query, pool, k, p, device, batch_q)
    if method == "diffusion":
        return _diffusion(query, pool, k, params, device, batch_q)
    if method in ("listwise_ce", "cross_encoder"):
        ckpt = params.get("ce_ckpt")
        if not ckpt:
            return _empty_topk(query.shape[0], k), _empty_idx(query.shape[0], k), "skipped_no_ckpt"
        return _empty_topk(query.shape[0], k), _empty_idx(query.shape[0], k), "skipped_ckpt_not_wired"
    raise KeyError(f"Unknown method {method}")


def _empty_topk(nq, k):
    return np.zeros((nq, k), dtype=np.float32)


def _empty_idx(nq, k):
    return np.full((nq, k), -1, dtype=np.int64)


def _hubness(query, pool, k, counts, params, device, batch_q):
    lam = float(params.get("hub_lambda", 0.4))
    if counts is None:
        _, idx = brute_topk(query, pool, int(params.get("h1_k", 10)), device=device, batch_q=batch_q)
        counts = np.bincount(idx.reshape(-1), minlength=pool.shape[0]).astype(np.float64)
    r = counts.astype(np.float32)
    std = float(r.std())
    if std > 1e-8:
        r = (r - float(r.mean())) / std
    _note_scale(params, "hub_z", r, lam)
    r_t = torch.from_numpy(r).to(device)
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    nq, npool = query.shape[0], pool.shape[0]
    k = min(k, npool)
    vals = torch.empty((nq, k), dtype=torch.float32)
    idxs = torch.empty((nq, k), dtype=torch.int64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        sim = q_t[s:e] @ p_t.T
        sim = sim - lam * r_t.unsqueeze(0)
        v, i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        vals[s:e] = v.cpu()
        idxs[s:e] = i.cpu()
    return vals.numpy(), idxs.numpy(), "ok"


def _hub_rel(query, pool, k, counts, params, device, batch_q):
    lam = float(params.get("hub_rel_lambda", 0.1))
    if counts is None:
        _, idx = brute_topk(query, pool, int(params.get("h1_k", 10)), device=device, batch_q=batch_q)
        counts = np.bincount(idx.reshape(-1), minlength=pool.shape[0]).astype(np.float64)
    mean = float(np.mean(counts)) + 1e-8
    penalty = (counts.astype(np.float32) / mean) - 1.0
    _note_scale(params, "hub_rel", penalty, lam)
    return _subtract_item_term(query, pool, k, lam * penalty, device, batch_q)


def _csls(query, pool, k, params, device, batch_q, cosine_top_val=None, r_c_side="query", penalty_lambda=0.5):
    kk = int(params.get("csls_k", params.get("h1_k", 10)))
    reference = params.get("csls_reference_queries")
    if reference is None:
        reference = query
        reference_tag = "action_queries"
    else:
        reference_tag = f"independent_ref_{int(np.asarray(reference).shape[0])}"
    ref_n = int(np.asarray(reference).shape[0])
    kk = max(1, min(kk, pool.shape[0] - 1, ref_n))
    if cosine_top_val is not None and cosine_top_val.shape[1] >= kk:
        r_q = cosine_top_val[:, :kk].mean(axis=1).astype(np.float32)
    else:
        v, _ = brute_topk(query, pool, kk, device=device, batch_q=batch_q)
        r_q = v.mean(axis=1).astype(np.float32)
    cache = params.setdefault("_csls_r_c_cache", {})
    cache_key = (
        r_c_side,
        kk,
        int(np.asarray(reference).shape[0]),
        int(pool.shape[0]),
        reference_tag,
    )
    r_c = cache.get(cache_key)
    if r_c is None:
        if r_c_side == "pool":
            r_c = _mean_topk_sim(pool, pool, kk, device, batch=min(256, pool.shape[0]), drop_self=True)
        else:
            r_c = _mean_topk_sim(
                pool,
                np.asarray(reference, dtype=np.float32),
                kk,
                device,
                batch=min(256, pool.shape[0]),
                drop_self=False,
            )
        cache[cache_key] = r_c
    _note_scale(
        params,
        f"csls_{r_c_side}_p{int(round(100 * penalty_lambda)):03d}",
        r_c,
        penalty_lambda,
        extra={"r_q_mean": float(r_q.mean()), "r_c_mean": float(r_c.mean())},
    )
    r_c_t = torch.from_numpy(np.ascontiguousarray(r_c)).to(device)
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    nq, npool = query.shape[0], pool.shape[0]
    k = min(k, npool)
    vals = torch.empty((nq, k), dtype=torch.float32)
    idxs = torch.empty((nq, k), dtype=torch.int64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        cos = q_t[s:e] @ p_t.T
        sim = cos - float(penalty_lambda) * r_c_t.unsqueeze(0)
        v, i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        vals[s:e] = v.cpu()
        idxs[s:e] = i.cpu()
    return vals.numpy(), idxs.numpy(), "ok"


def _subtract_item_term(query, pool, k, penalty_c, device, batch_q):
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    r_t = torch.from_numpy(np.ascontiguousarray(penalty_c.astype(np.float32))).to(device)
    nq, npool = query.shape[0], pool.shape[0]
    k = min(k, npool)
    vals = torch.empty((nq, k), dtype=torch.float32)
    idxs = torch.empty((nq, k), dtype=torch.int64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        sim = q_t[s:e] @ p_t.T - r_t.unsqueeze(0)
        v, i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        vals[s:e] = v.cpu()
        idxs[s:e] = i.cpu()
    return vals.numpy(), idxs.numpy(), "ok"


def _mean_topk_sim(src, dst, k, device, batch, drop_self=False):
    k = min(int(k), dst.shape[0] - (1 if drop_self and src.shape[0] == dst.shape[0] else 0))
    k = max(1, k)
    s_t = torch.from_numpy(np.ascontiguousarray(src)).to(device)
    d_t = torch.from_numpy(np.ascontiguousarray(dst)).to(device)
    n = src.shape[0]
    out = torch.empty(n, dtype=torch.float32)
    same = drop_self and src.shape[0] == dst.shape[0]
    for i in range(0, n, batch):
        e = min(i + batch, n)
        sim = s_t[i:e] @ d_t.T
        if same:
            for j, gi in enumerate(range(i, e)):
                sim[j, gi] = -1e9
        else:
            sim = torch.where(sim >= 0.999, torch.full_like(sim, -1e9), sim)
        v, _ = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        out[i:e] = v.mean(dim=1)
    return out.cpu().numpy()


def _note_scale(params, name, term, lam, extra=None):
    sink = params.get("_scale")
    if sink is None:
        return
    t = np.asarray(term, dtype=np.float64)
    rec = {
        "name": name,
        "lam": float(lam),
        "term_mean": float(t.mean()),
        "term_std": float(t.std()),
        "penalty_std": float(abs(lam) * t.std()),
        "penalty_min": float(lam * t.min()),
        "penalty_max": float(lam * t.max()),
    }
    if extra:
        rec.update(extra)
    sink[name] = rec


def _alpha_qe(query, pool, k, params, device, batch_q):
    qe_k = int(params.get("qe_k", 10))
    alpha = float(params.get("qe_alpha", 0.5))
    power = float(params.get("qe_power", 3.0))
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    nq, npool = query.shape[0], pool.shape[0]
    k = min(k, npool)
    qe_k = min(qe_k, npool)
    vals = torch.empty((nq, k), dtype=torch.float32)
    idxs = torch.empty((nq, k), dtype=torch.int64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        cos = q_t[s:e] @ p_t.T
        v0, i0 = torch.topk(cos, qe_k, dim=1, largest=True, sorted=True)
        w = v0.clamp(min=0).pow(power)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        
        gathered = p_t[i0]  
        mix = torch.bmm(w.unsqueeze(1), gathered).squeeze(1)
        q2 = (1.0 - alpha) * q_t[s:e] + alpha * mix
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        sim = q2 @ p_t.T
        v, i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        vals[s:e] = v.cpu()
        idxs[s:e] = i.cpu()
    return vals.numpy(), idxs.numpy(), "ok"


def _diffusion(query, pool, k, params, device, batch_q):
    max_pool = int(params.get("diffusion_max_pool", 80000))
    if pool.shape[0] > max_pool:
        return (
            _empty_topk(query.shape[0], k),
            _empty_idx(query.shape[0], k),
            f"skipped_pool_{pool.shape[0]}_gt_{max_pool}",
        )
    knn = int(params.get("diffusion_knn", 10))
    steps = int(params.get("diffusion_steps", 5))
    beta = float(params.get("diffusion_beta", 0.5))
    knn = min(knn, pool.shape[0] - 1)
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    idx_g, w_g = _pool_knn(p_t, knn, device, batch=min(256, pool.shape[0]))
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    nq, npool = query.shape[0], pool.shape[0]
    k = min(k, npool)
    vals = torch.empty((nq, k), dtype=torch.float32)
    idxs = torch.empty((nq, k), dtype=torch.int64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        s0 = q_t[s:e] @ p_t.T  
        s_cur = s0
        for _ in range(steps):
            
            neigh = s_cur[:, idx_g]  
            agg = (neigh * w_g.unsqueeze(0)).sum(dim=-1)
            s_cur = (1.0 - beta) * s0 + beta * agg
        v, i = torch.topk(s_cur, k, dim=1, largest=True, sorted=True)
        vals[s:e] = v.cpu()
        idxs[s:e] = i.cpu()
    return vals.numpy(), idxs.numpy(), "ok"


def _pool_knn(p_t: torch.Tensor, knn: int, device, batch: int = 256):
    n = p_t.shape[0]
    idx = torch.empty((n, knn), dtype=torch.int64, device=device)
    val = torch.empty((n, knn), dtype=torch.float32, device=device)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sim = p_t[s:e] @ p_t.T
        for j, gi in enumerate(range(s, e)):
            sim[j, gi] = -1e9
        v, i = torch.topk(sim, knn, dim=1, largest=True, sorted=True)
        idx[s:e] = i
        val[s:e] = v
    w = val.clamp(min=0)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    return idx, w
