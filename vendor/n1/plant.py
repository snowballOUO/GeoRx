
from __future__ import annotations

import numpy as np

from n1 import l2_normalize
from retrieve import brute_topk, get_device
from unified import inject_h1, inject_h3, pool_axis



LADDER = {
    "h1": [
        {"level": 0, "mode": "shift", "beta": 1.0},
        {"level": 1, "mode": "shift", "beta": 4.0},
        {"level": 2, "mode": "shift", "beta": 16.0},
        {"level": 3, "mode": "unique_hub", "t": 0.50},
        {"level": 4, "mode": "unique_hub", "t": 1.00},
        
        {"level": 5, "mode": "query_hub", "t": 0.35},
        {"level": 6, "mode": "query_hub", "t": 0.75},
        {"level": 7, "mode": "query_hub", "t": 1.00},
    ],
    "h2": [
        {"level": 0, "mode": "blocks", "t": 0.85, "blocks": 32},
        {"level": 1, "mode": "blocks", "t": 1.00, "blocks": 8},
        {"level": 2, "mode": "blocks", "t": 1.00, "blocks": 256},
        {"level": 3, "mode": "near_dup", "n_hubs": 16, "jitter": 1e-3, "frac": 0.50},
        {"level": 4, "mode": "near_dup", "n_hubs": 8, "jitter": 1e-4, "frac": 1.00},
        
        {"level": 5, "mode": "query_focus", "t": 0.50, "k": 5},
        {"level": 6, "mode": "query_focus", "t": 1.00, "k": 5},
        
        {"level": 7, "mode": "collapsed_poles", "tightness": 1.00, "residual": 0.05},
        {"level": 8, "mode": "local_shrink", "t": 0.50, "k": 16},
        
        
        {"level": 9, "mode": "sign_aware", "t_focus": 1.00, "k_focus": 5, "t_blocks": 1.00, "blocks": 32},
        {"level": 10, "mode": "sign_aware_exact", "t_focus": 1.00, "k_focus": 5, "t_blocks": 1.00, "blocks": 32},
        
        {"level": 11, "mode": "dim_gated", "d_cut": 2048, "t_focus": 1.00, "k_focus": 5, "t_blocks": 1.00, "blocks": 32},
    ],
    "h3": [
        {"level": 0, "mode": "islands", "residual": 0.05, "tightness": 1.00, "bridge_k": 64},
        {"level": 1, "mode": "islands", "residual": 0.02, "tightness": 1.00, "bridge_k": 64},
        {"level": 2, "mode": "islands", "residual": 0.00, "tightness": 1.00, "bridge_k": 64, "q_jitter": 1e-3},
        {"level": 3, "mode": "islands", "residual": 0.00, "tightness": 1.00, "bridge_k": 128, "q_jitter": 1e-3},
        {"level": 4, "mode": "islands", "residual": 0.00, "tightness": 1.00, "bridge_k": 256, "q_jitter": 1e-3},
        
        {"level": 9, "mode": "islands", "residual": 0.00, "tightness": 0.85, "bridge_k": 0},
        {"level": 10, "mode": "islands", "residual": 0.00, "tightness": 0.80, "bridge_k": 0},
        {"level": 11, "mode": "islands", "residual": 0.00, "tightness": 0.70, "bridge_k": 32},
        {"level": 12, "mode": "islands", "residual": 0.02, "tightness": 0.85, "bridge_k": 32, "bridge_delta": 0.02},
    ],
    "h4": [
        {"level": 0, "mode": "mean", "t": 0.85},
        {"level": 1, "mode": "mean", "t": 0.97},
        {"level": 2, "mode": "twins", "pair_cos": 0.98},
        {"level": 3, "mode": "twins", "pair_cos": 0.999},
        {"level": 4, "mode": "twins", "pair_cos": 0.9999},
        {"level": 5, "mode": "mean", "t": 1.00},
    ],
    "h5": [
        {"level": 0, "mode": "zero_peak", "q": 0.90},
        {"level": 1, "mode": "zero_peak", "q": 0.50},
        {"level": 2, "mode": "boost_peak", "q": 0.90, "gamma": 4.0},
        {"level": 3, "mode": "boost_peak", "q": 0.80, "gamma": 16.0},
        {"level": 4, "mode": "boost_peak", "q": 0.50, "gamma": 64.0},
        
        {"level": 5, "mode": "equal_coord", "t": 0.50},
        {"level": 6, "mode": "equal_coord", "t": 1.00},
    ],
}


def plant_world(kind: str, query: np.ndarray, pool: np.ndarray, seed: int, spec: dict | None = None, device=None):
    spec = dict(spec or LADDER[kind][0])
    if kind == "h1":
        return plant_h1(query, pool, seed, spec)
    if kind == "h2":
        return plant_h2(query, pool, seed, spec, device=device)
    if kind == "h3":
        return plant_h3(query, pool, seed, spec)
    if kind == "h4":
        return np.array(query, dtype=np.float32, copy=True), plant_h4(pool, seed, spec)
    if kind == "h5":
        return plant_h5(query, pool, seed, spec)
    raise ValueError(kind)


def plant_h1(query: np.ndarray, pool: np.ndarray, seed: int, spec: dict):
    rng = np.random.default_rng(int(seed))
    mode = spec.get("mode", "shift")
    if mode == "shift":
        u = rng.standard_normal(pool.shape[1]).astype(np.float32)
        u = u / max(float(np.linalg.norm(u)), 1e-8)
        return np.array(query, dtype=np.float32, copy=True), l2_normalize(pool + float(spec["beta"]) * u)
    if mode == "unique_hub":
        t = float(spec["t"])
        center = l2_normalize(pool.mean(axis=0, keepdims=True))[0]
        hub_idx = int(np.argmax(pool @ center))
        hub = pool[hub_idx] / max(float(np.linalg.norm(pool[hub_idx])), 1e-8)
        out = np.array(pool, dtype=np.float32, copy=True)
        mask = np.ones(pool.shape[0], dtype=bool)
        mask[hub_idx] = False
        dots = out[mask] @ hub
        ortho = out[mask] - dots[:, None] * hub
        out[mask] = (1.0 - t) * out[mask] + t * ortho
        out[hub_idx] = hub
        return np.array(query, dtype=np.float32, copy=True), l2_normalize(out)
    if mode == "query_hub":
        return inject_h1(query, pool, strength=float(spec["t"]))
    raise ValueError(mode)


def plant_h2(query: np.ndarray, pool: np.ndarray, seed: int, spec: dict, device=None):
    rng = np.random.default_rng(int(seed))
    mode = spec.get("mode", "blocks")
    if mode == "blocks":
        k = min(int(spec["blocks"]), int(pool.shape[0]))
        cents = l2_normalize(rng.standard_normal((k, pool.shape[1])).astype(np.float32))
        assign = np.argmax(pool @ cents.T, axis=1)
        out = np.array(pool, dtype=np.float32, copy=True)
        t = float(spec["t"])
        for b in range(k):
            hit = assign == b
            if not np.any(hit):
                continue
            c = pool[hit].mean(axis=0)
            out[hit] = (1.0 - t) * pool[hit] + t * c
        return np.array(query, dtype=np.float32, copy=True), l2_normalize(out)
    if mode == "near_dup":
        n_hubs = min(int(spec["n_hubs"]), int(pool.shape[0]))
        frac = float(spec["frac"])
        jitter = float(spec["jitter"])
        hubs = l2_normalize(rng.standard_normal((n_hubs, pool.shape[1])).astype(np.float32))
        n_take = max(n_hubs, int(round(frac * pool.shape[0])))
        take = rng.choice(pool.shape[0], size=n_take, replace=False)
        assign = np.argmax(pool[take] @ hubs.T, axis=1)
        out = np.array(pool, dtype=np.float32, copy=True)
        out[take] = hubs[assign]
        if jitter > 0:
            out[take] = out[take] + jitter * rng.standard_normal(out[take].shape).astype(np.float32)
        return np.array(query, dtype=np.float32, copy=True), l2_normalize(out)
    if mode == "query_focus":
        t = float(spec["t"])
        k = min(int(spec.get("k", 5)), int(pool.shape[0]))
        device = device or get_device("cuda")
        batch = 32 if pool.shape[1] >= 2048 else 128
        _, idx = brute_topk(query, pool, k, device=device, batch_q=batch)
        nbr = pool[idx].mean(axis=1)
        q2 = l2_normalize((1.0 - t) * query + t * nbr)
        return q2, np.array(pool, dtype=np.float32, copy=True)
    if mode == "collapsed_poles":
        axis = pool_axis(pool, seed)
        return inject_h3(
            query, pool,
            strength=1.00, axis=axis, tightness=float(spec.get("tightness", 1.0)),
            residual=float(spec.get("residual", 0.05)), seed=seed,
            bridge_k=0, bridge_delta=0.02, bulk_noise=1e-4,
        )
    if mode == "local_shrink":
        t = float(spec["t"])
        k = min(int(spec.get("k", 16)), int(pool.shape[0]) - 1)
        device = device or get_device("cuda")
        batch = 16 if pool.shape[0] > 20000 else 64
        _, idx = brute_topk(pool, pool, k + 1, device=device, batch_q=batch)
        means = pool[idx[:, 1:]].mean(axis=1)
        out = l2_normalize((1.0 - t) * pool + t * means)
        return np.array(query, dtype=np.float32, copy=True), out
    if mode == "sign_aware":
        from geometry import pool_pairwise_mean
        g = float(pool_pairwise_mean(pool))
        if g < 0.0:
            nested = {"mode": "blocks", "t": float(spec.get("t_blocks", 1.0)), "blocks": int(spec.get("blocks", 32))}
        else:
            nested = {"mode": "query_focus", "t": float(spec.get("t_focus", 1.0)), "k": int(spec.get("k_focus", 5))}
        return plant_h2(query, pool, seed, nested, device=device)
    if mode == "sign_aware_exact":
        g = _exact_pairwise_mean(pool)
        if g < 0.0:
            nested = {"mode": "blocks", "t": float(spec.get("t_blocks", 1.0)), "blocks": int(spec.get("blocks", 32))}
        else:
            nested = {"mode": "query_focus", "t": float(spec.get("t_focus", 1.0)), "k": int(spec.get("k_focus", 5))}
        return plant_h2(query, pool, seed, nested, device=device)
    if mode == "dim_gated":
        if int(pool.shape[1]) >= int(spec.get("d_cut", 2048)):
            nested = {"mode": "blocks", "t": float(spec.get("t_blocks", 1.0)), "blocks": int(spec.get("blocks", 32))}
        else:
            nested = {"mode": "query_focus", "t": float(spec.get("t_focus", 1.0)), "k": int(spec.get("k_focus", 5))}
        return plant_h2(query, pool, seed, nested, device=device)
    raise ValueError(mode)


def _exact_pairwise_mean(pool: np.ndarray) -> float:
    x = np.asarray(pool, dtype=np.float64)
    n = int(x.shape[0])
    total = x.sum(axis=0)
    return float((float(total @ total) - n) / max(n * (n - 1), 1))


def plant_h3(query: np.ndarray, pool: np.ndarray, seed: int, spec: dict):
    axis = pool_axis(pool, seed)
    q2, c2 = inject_h3(
        query, pool,
        strength=1.00, axis=axis, tightness=float(spec.get("tightness", 1.0)),
        residual=float(spec.get("residual", 0.05)), seed=seed,
        bridge_k=int(spec.get("bridge_k", 64)),
        bridge_delta=float(spec.get("bridge_delta", 0.02)),
        bulk_noise=1e-4,
    )
    jitter = float(spec.get("q_jitter", 0.0))
    if jitter > 0:
        rng = np.random.default_rng(int(seed) + 7)
        q2 = l2_normalize(q2 + jitter * rng.standard_normal(q2.shape).astype(np.float32))
    return q2, c2


def plant_h4(pool: np.ndarray, seed: int, spec: dict):
    mode = spec.get("mode", "mean")
    if mode == "mean":
        mean = pool.mean(axis=0)
        t = float(spec["t"])
        return l2_normalize((1.0 - t) * pool + t * mean)
    if mode == "twins":
        rng = np.random.default_rng(int(seed))
        pair_cos = float(spec["pair_cos"])
        out = np.array(pool, dtype=np.float32, copy=True)
        n = int(out.shape[0]) // 2 * 2
        perm = rng.permutation(n)
        a, b = perm[0::2], perm[1::2]
        mid = l2_normalize(0.5 * (out[a] + out[b]))
        noise = rng.standard_normal(out[a].shape).astype(np.float32)
        noise = noise - (noise * mid).sum(axis=1, keepdims=True) * mid
        noise = l2_normalize(noise)
        alpha = float(np.sqrt(max(1.0 - pair_cos, 0.0)))
        out[a] = l2_normalize(mid + alpha * noise)
        out[b] = l2_normalize(mid - alpha * noise)
        return out
    raise ValueError(mode)


def plant_h5(query: np.ndarray, pool: np.ndarray, seed: int, spec: dict):
    del seed
    mode = spec.get("mode", "zero_peak")
    if mode == "equal_coord":
        t = float(spec["t"])
        return _equal_coord(query, t), _equal_coord(pool, t)
    mean = np.abs(pool.mean(axis=0))
    q = float(spec["q"])
    dims = mean >= float(np.quantile(mean, q))
    p2 = np.array(pool, dtype=np.float32, copy=True)
    q2 = np.array(query, dtype=np.float32, copy=True)
    if mode == "zero_peak":
        p2[:, dims] = 0.0
        q2[:, dims] = 0.0
        return l2_normalize(q2), l2_normalize(p2)
    if mode == "boost_peak":
        gamma = float(spec["gamma"])
        p2[:, dims] *= gamma
        q2[:, dims] *= gamma
        return l2_normalize(q2), l2_normalize(p2)
    raise ValueError(mode)


def _equal_coord(x: np.ndarray, t: float) -> np.ndarray:
    mag = 1.0 / np.sqrt(float(x.shape[1]))
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    target = (signs * mag).astype(np.float32)
    return l2_normalize((1.0 - t) * x + t * target)
