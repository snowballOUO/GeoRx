
from __future__ import annotations

import numpy as np

from retrieve import brute_topk


STRENGTH = 0.75
H3_TIGHT_PRIMARY = 0.90
H3_TIGHT_BACKUP = 1.00
H3_QUERY_RESIDUAL = 0.05
H3_BRIDGE_K = 64
H3_BRIDGE_DELTA = 0.02
H3_BULK_NOISE = 1e-4


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def pool_axis(pool: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = min(2048, int(pool.shape[0]))
    idx = rng.choice(pool.shape[0], size=n, replace=False)
    x = pool[idx] - pool[idx].mean(axis=0, keepdims=True)
    u = rng.standard_normal(pool.shape[1]).astype(np.float32)
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    for _ in range(8):
        u = x.T @ (x @ u)
        u = u / max(float(np.linalg.norm(u)), 1e-8)
    center = normalize(pool.mean(axis=0, keepdims=True))[0]
    hub = pool[int(np.argmax(pool @ center))]
    hub = hub / max(float(np.linalg.norm(hub)), 1e-8)
    u = u - float(u @ hub) * hub
    return normalize(u[None])[0]


def inject_h1(query, pool, strength: float = STRENGTH, **_):
    center = normalize(pool.mean(axis=0, keepdims=True))[0]
    hub_idx = int(np.argmax(pool @ center))
    hub = pool[hub_idx] / max(float(np.linalg.norm(pool[hub_idx])), 1e-8)
    t = min(1.0, max(0.0, float(strength)))
    q2 = normalize((1.0 - t) * query + t * hub)
    out = np.array(pool, dtype=np.float32, copy=True)
    mask = np.ones(pool.shape[0], dtype=bool)
    mask[hub_idx] = False
    dots = out[mask] @ hub
    out[mask] = out[mask] - dots[:, None] * hub
    out[hub_idx] = hub
    return q2, normalize(out)


def inject_h2(query, pool, strength: float = STRENGTH, *, device=None, batch_q: int = 256, **_):
    k = min(5, pool.shape[0])
    t = min(1.0, max(0.0, float(strength)))
    _, idx = brute_topk(query, pool, k, device=device, batch_q=batch_q)
    means = pool[idx].mean(axis=1)
    disp = np.zeros_like(pool, dtype=np.float32)
    weight = np.zeros(pool.shape[0], dtype=np.float32)
    for qi in range(idx.shape[0]):
        rows = idx[qi]
        disp[rows] += means[qi] - pool[rows]
        weight[rows] += 1.0
    out = np.array(pool, dtype=np.float32, copy=True)
    hit = weight > 0
    out[hit] = pool[hit] + t * (disp[hit] / weight[hit, None])
    nbr = out[idx].mean(axis=1)
    q2 = (1.0 - t) * query + t * nbr
    return normalize(q2), normalize(out)


def _at_query_cosine(centroid: np.ndarray, query: np.ndarray, target: float) -> np.ndarray:
    c = np.asarray(centroid, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    c0 = float(np.clip(c @ q, -1.0 + 1e-6, 1.0 - 1e-6))
    tgt = float(min(max(target, c0 + 1e-5), 0.999))
    denom = max(1.0 - c0 * c0, 1e-12)
    a = float(np.sqrt(max((1.0 - tgt * tgt) / denom, 0.0)))
    b = tgt - a * c0
    return normalize((a * c + b * q)[None])[0]


def inject_h3(
    query,
    pool,
    strength: float = STRENGTH,
    *,
    axis,
    tightness: float = H3_TIGHT_PRIMARY,
    residual: float = H3_QUERY_RESIDUAL,
    noise: float = 0.0,
    seed: int = 20260904,
    bridge_k: int = H3_BRIDGE_K,
    bridge_delta: float = H3_BRIDGE_DELTA,
    bulk_noise: float = H3_BULK_NOISE,
    **_,
):
    del noise
    alpha = float(strength)
    t = float(tightness)
    u = np.asarray(axis, dtype=np.float32)
    s = pool @ u
    sign = np.sign(s)
    sign[sign == 0] = 1.0
    out = pool + alpha * sign[:, None] * u
    thin = np.abs(s) < 0.15
    if np.any(thin):
        out[thin] = out[thin] + alpha * sign[thin][:, None] * u
    out = normalize(out)
    s2 = out @ u
    pos, neg = s2 > 0, s2 < 0
    if not np.any(pos) or not np.any(neg):
        raise RuntimeError("H3 warp produced one island")
    c_pos = out[pos].mean(axis=0)
    c_neg = out[neg].mean(axis=0)
    out[pos] = (1.0 - t) * out[pos] + t * c_pos
    out[neg] = (1.0 - t) * out[neg] + t * c_neg
    out = normalize(out)
    c_pos = normalize(out[pos].mean(axis=0, keepdims=True))[0]
    c_neg = normalize(out[neg].mean(axis=0, keepdims=True))[0]
    proj = query - (query @ u)[:, None] * u
    q2 = normalize(c_pos + c_neg + float(residual) * proj)
    mid = normalize((c_pos + c_neg)[None])[0]
    c0 = float(mid @ c_pos)
    target = c0 + float(bridge_delta)
    keep_n = max(int(bridge_k), 0)
    rng = np.random.default_rng(int(seed))
    if keep_n > 0:
        pos_ids = np.flatnonzero(pos)
        neg_ids = np.flatnonzero(neg)
        n_pos = min(keep_n, int(pos_ids.size))
        n_neg = min(keep_n, int(neg_ids.size))
        pos_br = rng.choice(pos_ids, size=n_pos, replace=False)
        neg_br = rng.choice(neg_ids, size=n_neg, replace=False)
        out[pos_br] = _at_query_cosine(c_pos, mid, target)
        out[neg_br] = _at_query_cosine(c_neg, mid, target)
    jitter = float(bulk_noise)
    if jitter > 0:
        out = normalize(out + jitter * rng.standard_normal(out.shape).astype(np.float32))
    if int(out.shape[0]) != int(pool.shape[0]):
        raise RuntimeError(f"H3 dropped corpus ids: {pool.shape[0]} -> {out.shape[0]}")
    return q2, out


def inject_h4(query, pool, strength: float = STRENGTH, *, device=None, batch_q: int = 256, **_):
    t = min(1.0, max(0.0, float(strength)))
    k = min(2, pool.shape[0])
    _, idx = brute_topk(query, pool, k, device=device, batch_q=batch_q)
    a = idx[:, 0]
    b = idx[:, 1] if idx.shape[1] > 1 else idx[:, 0]
    mid = 0.5 * (pool[a] + pool[b])
    disp = np.zeros_like(pool, dtype=np.float32)
    weight = np.zeros(pool.shape[0], dtype=np.float32)
    for qi in range(idx.shape[0]):
        rows = np.array([a[qi], b[qi]], dtype=np.int64)
        disp[rows] += mid[qi] - pool[rows]
        weight[rows] += 1.0
    out = np.array(pool, dtype=np.float32, copy=True)
    hit = weight > 0
    out[hit] = pool[hit] + t * (disp[hit] / weight[hit, None])
    out = normalize(out)
    mid_q = 0.5 * (out[a] + out[b])
    q2 = (1.0 - t) * query + t * mid_q
    return normalize(q2), normalize(out)


def inject_h5(
    query,
    pool,
    strength: float = STRENGTH,
    *,
    device=None,
    batch_q: int = 256,
    h5_mode: str = "nbhd",
    **_,
):
    g = min(1.0, max(0.0, float(strength)))
    k_nbhd = 10 if h5_mode == "nbhd10" else 5
    k = min(k_nbhd, pool.shape[0])
    _, idx = brute_topk(query, pool, k, device=device, batch_q=batch_q)
    top1 = idx[:, 0]
    if h5_mode in ("nbhd", "nbhd10"):
        scale = np.ones_like(pool, dtype=np.float32)
        q2 = np.array(query, dtype=np.float32, copy=True)
        for i in range(query.shape[0]):
            had = np.abs(query[i] * pool[top1[i]])
            dims = had >= float(np.quantile(had, 0.90))
            q2[i, dims] *= (1.0 - g)
            factor = np.ones(pool.shape[1], dtype=np.float32)
            factor[dims] = 1.0 - g
            scale[idx[i]] = np.minimum(scale[idx[i]], factor)
        return normalize(q2), normalize(pool * scale)

    acc = np.zeros_like(pool, dtype=np.float32)
    weight = np.zeros(pool.shape[0], dtype=np.float32)
    q2 = np.array(query, dtype=np.float32, copy=True)
    for i in range(query.shape[0]):
        q = query[i]
        had = np.abs(q * pool[top1[i]])
        dims = had >= float(np.quantile(had, 0.90))
        if h5_mode == "nbhd_q_restore":
            q2[i, dims] *= (1.0 - g)
        for j in idx[i]:
            v2 = np.array(pool[int(j)], dtype=np.float32, copy=True)
            if np.any(dims):
                v2[dims] *= (1.0 - g)
                if int(j) == int(top1[i]):
                    npk = ~dims
                    if np.any(npk):
                        v2[npk] = v2[npk] + g * q[npk]
            acc[int(j)] += normalize(v2[None])[0]
            weight[int(j)] += 1.0
    out = np.array(pool, dtype=np.float32, copy=True)
    hit = weight > 0
    out[hit] = acc[hit] / weight[hit, None]
    return normalize(q2), normalize(out)


INJECTORS = {
    "h1": ("hubness", inject_h1),
    "h2": ("neighborhood_overconcentration", inject_h2),
    "h3": ("manifold_fragmentation", inject_h3),
    "h4": ("score_ambiguity", inject_h4),
    "h5": ("weak_interaction", inject_h5),
}
