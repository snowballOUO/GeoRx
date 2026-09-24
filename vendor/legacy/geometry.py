
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from h1_variants import h1_variants
from retrieve import brute_topk


def l0_gold_pair_cosine(
    query: np.ndarray,
    pool: np.ndarray,
    positives: List[List[int]],
    query_train: Optional[np.ndarray] = None,
    positives_train: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    test_mean = _mean_gold_cos(query, pool, positives)
    out = {"l0_gold_pair_cosine_test": test_mean}
    if query_train is not None and positives_train:
        train_mean = _mean_gold_cos(query_train, pool, positives_train)
        out["l0_gold_pair_cosine_train"] = train_mean
        out["l0_train_test_ratio"] = (
            train_mean / test_mean if abs(test_mean) > 1e-8 else float("nan")
        )
    else:
        out["l0_gold_pair_cosine_train"] = float("nan")
        out["l0_train_test_ratio"] = float("nan")
    return out


def run_diagnostics(
    query: np.ndarray,
    pool: np.ndarray,
    *,
    h1_k: int = 10,
    h2_n: int = 50,
    h3_knn: int = 10,
    h3_list_n: Optional[int] = None,
    h1_ks=(1, 5, 10, 50),
    h2_sample_queries: int = 256,
    h3_sample_queries: int = 64,
    h5_sample_queries: int = 128,
    pool_pairwise_mean_value: Optional[float] = None,
    device=None,
    batch_q: int = 256,
) -> Dict:
    list_n = int(h2_n if h3_list_n is None else h3_list_n)
    k_need = max(h1_k, h2_n, list_n, h3_knn + 1, 2, max(h1_ks))
    k_need = min(k_need, pool.shape[0])
    vals, idxs = brute_topk(query, pool, k_need, device=device, batch_q=batch_q)
    h1 = h1_koccurrence(idxs, n_pool=pool.shape[0], ks=h1_ks, primary_k=h1_k)
    counts = h1.pop("k_occurrence_counts", None)
    h2 = h2_neighbor_ratio(
        query, pool, idxs, vals, n=min(h2_n, k_need),
        n_sample_queries=h2_sample_queries, global_mean=pool_pairwise_mean_value,
    )
    h3 = h3_nn_graph(
        pool, idxs, n=min(list_n, k_need), knn=min(h3_knn, list_n - 1),
        n_sample_queries=h3_sample_queries,
    )
    h4 = h4_score_shape(vals)
    h5 = h5_interaction_gap(
        query, pool, idxs, n=min(h2_n, k_need), n_sample_queries=h5_sample_queries
    )
    return {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "h5": h5,
        "cosine_topk_k": int(k_need),
        "h1_counts": counts,
        "cosine_top_idx": idxs,
        "cosine_top_val": vals,
    }


def h1_koccurrence(
    top_idx: np.ndarray,
    n_pool: int,
    ks=(1, 5, 10, 50),
    primary_k: int = 10,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    kmax = top_idx.shape[1]
    for k in ks:
        kk = min(k, kmax)
        counts = np.bincount(top_idx[:, :kk].reshape(-1), minlength=n_pool).astype(np.float64)
        fam = h1_variants(counts, n_query=int(top_idx.shape[0]), k=kk)
        out[f"skewness_k{k}"] = fam["skew_raw"]
        out[f"gini_k{k}"] = fam["gini"]
        out[f"max_k{k}"] = fam["max"]
        out[f"mean_k{k}"] = fam["mean"]
        out[f"frac_never_k{k}"] = fam["frac_never"]
        out[f"skew_excess_k{k}"] = fam["skew_excess"]
        out[f"skew_over_null_k{k}"] = fam["skew_over_null"]
        out[f"cv_over_null_k{k}"] = fam["cv_over_null"]
        if k == primary_k:
            out["k_occurrence_skewness"] = fam["skew_raw"]
            out["skew_excess"] = fam["skew_excess"]
            out["skew_over_null"] = fam["skew_over_null"]
            out["cv_over_null"] = fam["cv_over_null"]
            out["lambda_uniform"] = fam["lambda_uniform"]
            out["k_occurrence_counts"] = counts
    return out


def h2_neighbor_ratio(
    query: np.ndarray,
    pool: np.ndarray,
    top_idx: np.ndarray,
    top_val: np.ndarray,
    n: int,
    n_sample_queries: int = 256,
    seed: int = 42,
    global_mean: Optional[float] = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    nq = top_idx.shape[0]
    take = min(n_sample_queries, nq)
    qis = rng.choice(nq, size=take, replace=False) if nq > take else np.arange(nq)
    rows = top_idx[qis, :n]
    vecs = pool[rows]
    sums = vecs.sum(axis=1, dtype=np.float64)
    diag = np.sum(vecs.astype(np.float64) ** 2, axis=(1, 2))
    denom = max(n * (n - 1), 1)
    intra = (np.sum(sums ** 2, axis=1) - diag) / denom
    if global_mean is None:
        global_mean = pool_pairwise_mean(pool, seed=seed)
    ratios = intra / (float(global_mean) + 1e-8)
    return {
        "neighbor_sim_ratio": float(np.mean(ratios)),
        "topn_pairwise_mean": float(np.mean(intra)),
        "pool_pairwise_mean": float(global_mean),
    }


def pool_pairwise_mean(pool: np.ndarray, n_sample: int = 2048, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    ns = min(int(n_sample), pool.shape[0])
    samp = rng.choice(pool.shape[0], size=ns, replace=False)
    vecs = pool[samp].astype(np.float64, copy=False)
    total = vecs.sum(axis=0)
    diag = float(np.sum(vecs ** 2))
    denom = ns * (ns - 1)
    return float((float(total @ total) - diag) / denom) if denom else 1e-8


def h3_nn_graph(
    pool: np.ndarray,
    top_idx: np.ndarray,
    n: int,
    knn: int,
    n_sample_queries: int = 64,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    nq = top_idx.shape[0]
    take = min(n_sample_queries, nq)
    qis = rng.choice(nq, size=take, replace=False) if nq > take else np.arange(nq)
    knn = max(1, min(knn, n - 1))
    ncomp, ccoef, frag, largest, second = [], [], [], [], []
    for qi in qis:
        rows = top_idx[qi, :n]
        vecs = pool[rows]
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, -1e9)
        nn = np.argpartition(-sim, kth=knn, axis=1)[:, :knn]
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            adj[i, nn[i]] = 1
        und = np.maximum(adj, adj.T)
        sizes = _component_sizes(und)
        n_pts = max(int(und.shape[0]), 1)
        largest_frac = sizes[0] / n_pts if sizes else 1.0
        second_frac = sizes[1] / n_pts if len(sizes) > 1 else 0.0
        ncomp.append(len(sizes))
        ccoef.append(_clustering_coef(und))
        largest.append(largest_frac)
        second.append(second_frac)
        frag.append(1.0 - largest_frac)
    return {
        "n_components_mean": float(np.mean(ncomp)),
        "clustering_coef_mean": float(np.mean(ccoef)),
        "fragmentation": float(np.mean(frag)),
        "largest_component_frac_mean": float(np.mean(largest)),
        "second_frac_mean": float(np.mean(second)),
    }


def h4_score_shape(top_val: np.ndarray) -> Dict[str, float]:
    if top_val.shape[1] < 2:
        return {"top1_top2_gap": float("nan"), "topn_score_std": float("nan")}
    gap = top_val[:, 0] - top_val[:, 1]
    return {
        "top1_top2_gap": float(gap.mean()),
        "topn_score_std": float(top_val.std(axis=1).mean()),
    }


def h5_interaction_gap(
    query: np.ndarray,
    pool: np.ndarray,
    top_idx: np.ndarray,
    n: int,
    n_sample_queries: int = 128,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    nq = top_idx.shape[0]
    take = min(n_sample_queries, nq)
    qis = rng.choice(nq, size=take, replace=False) if nq > take else np.arange(nq)
    extras, gaps_abs, pair_left = [], [], []
    for qi in qis:
        q = query[qi]
        rows = top_idx[qi, :n]
        v = pool[rows]
        hadamard = q * v
        peak = hadamard.max(axis=1)
        meanp = hadamard.mean(axis=1)
        ad = np.mean(np.abs(q - v), axis=1)
        pair_left.append(float(peak[0] - meanp[0]))
        if n > 1:
            extras.append(
                float((peak[0] - peak[1:].mean()) - (meanp[0] - meanp[1:].mean()))
            )
            gaps_abs.append(float(ad[1:].mean() - ad[0]))
        else:
            extras.append(0.0)
            gaps_abs.append(0.0)
    return {
        "interaction_gap": float(np.mean(extras)),
        "pair_peak_leftover_mean": float(np.mean(pair_left)),
        "l1_gap": float(np.mean(gaps_abs)),
        "note": "unlabeled peak-minus-mean Hadamard leftover; original H5 used label MI (forbidden)",
    }


def _mean_gold_cos(query, pool, positives) -> float:
    vals = []
    for qi, golds in enumerate(positives):
        if not golds:
            continue
        g = [x for x in golds if 0 <= x < pool.shape[0]]
        if not g:
            continue
        vals.append(float((query[qi] @ pool[g].T).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def _component_sizes(und: np.ndarray) -> list:
    n = und.shape[0]
    seen = np.zeros(n, dtype=bool)
    sizes = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in np.flatnonzero(und[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def _n_components(und: np.ndarray) -> int:
    return len(_component_sizes(und))


def _largest_component_frac(und: np.ndarray) -> float:
    n = und.shape[0]
    if n <= 0:
        return 1.0
    seen = np.zeros(n, dtype=bool)
    best = 0
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in np.flatnonzero(und[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        best = max(best, size)
    return float(best) / float(n)


def _clustering_coef(und: np.ndarray) -> float:
    n = und.shape[0]
    coefs = []
    for i in range(n):
        nbr = np.flatnonzero(und[i])
        k = nbr.size
        if k < 2:
            continue
        sub = und[np.ix_(nbr, nbr)]
        e = np.triu(sub, k=1).sum()
        coefs.append(2.0 * e / (k * (k - 1)))
    return float(np.mean(coefs)) if coefs else 0.0
