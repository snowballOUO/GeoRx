
from __future__ import annotations

from typing import Dict

import numpy as np


def h1_variants(counts: np.ndarray, n_query: int, k: int) -> Dict[str, float]:
    counts = np.asarray(counts, dtype=np.float64)
    n_pool = int(counts.size)
    k = int(k)
    n_query = int(n_query)
    total = float(n_query * k)
    lam = total / max(n_pool, 1)  
    p = k / max(n_pool, 1)
    
    var_bin = n_query * p * (1.0 - p)
    skew_null = float((1.0 - 2.0 * p) / np.sqrt(var_bin)) if var_bin > 1e-12 else float("nan")
    skew_raw = _sample_skew(counts)
    pos = counts[counts > 0]
    skew_pos = (
        _sample_skew(pos) if pos.size >= 8 else float("nan")
    )
    mean = float(counts.mean())
    std = float(counts.std())
    cv = std / mean if mean > 1e-12 else float("nan")
    cv_null = float(np.sqrt((1.0 - p) / (n_query * p))) if n_query * p > 1e-12 else float("nan")
    shares = counts / max(total, 1e-12)
    hhi = float((shares ** 2).sum())
    hhi_uniform = 1.0 / max(n_pool, 1)
    return {
        "k": k,
        "n_query": n_query,
        "n_pool": n_pool,
        "lambda_uniform": lam,
        "frac_never": float((counts == 0).mean()),
        "mean": mean,
        "std": std,
        "max": float(counts.max()),
        "skew_raw": skew_raw,
        "skew_null_binomial": skew_null,
        "skew_excess": skew_raw - skew_null if np.isfinite(skew_null) else float("nan"),
        "skew_over_null": skew_raw / skew_null if abs(skew_null) > 1e-8 else float("nan"),
        "skew_positive": skew_pos,
        "n_positive": int(pos.size),
        "gini": _gini(counts),
        "gini_positive": _gini(pos) if pos.size else float("nan"),
        "max_over_lambda": float(counts.max() / lam) if lam > 1e-12 else float("nan"),
        "cv": cv,
        "cv_over_null": cv / cv_null if cv_null and cv_null > 1e-12 else float("nan"),
        "hhi": hhi,
        "hhi_over_uniform": hhi / hhi_uniform if hhi_uniform > 0 else float("nan"),
    }


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n = x.size
    return float((2.0 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def _sample_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return float("nan")
    centered = x - float(x.mean())
    m2 = float(np.mean(centered ** 2))
    if m2 <= 1e-30:
        return float("nan")
    g1 = float(np.mean(centered ** 3) / (m2 ** 1.5))
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)
