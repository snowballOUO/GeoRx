
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
LEGACY = Path(__file__).resolve().parents[2] / "legacy"
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(CODE))

import profile as profile_mod
from diagnosis import compute_pool_context, configure_h5_primary, diagnose_readings
from profile import collect_window_readings, fit_profile

profile_mod.METRIC_SCALE_FLOOR["h3_second_frac"] = 0.02
profile_mod.METRIC_SCALE_FLOOR["h5_pair_peak_leftover"] = 1e-5

SEED_NULL = 20260908
WINDOW_SIZE = 256
N1_TRAIN_WINDOWS = 32
N1_CAL_WINDOWS = 20
N1_EVAL_WINDOWS = 2
ALPHA = 0.05


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def sphere_bank(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return l2_normalize(rng.standard_normal((int(n), int(dim))).astype(np.float32))


def n1_corpus(n_pool: int, dim: int, seed: int = SEED_NULL) -> np.ndarray:
    return sphere_bank(n_pool, dim, seed)


def n1_queries(n_query: int, dim: int, seed: int) -> np.ndarray:
    return sphere_bank(n_query, dim, seed)


def diag_cfg(device, batch_q: int, pool: np.ndarray) -> dict:
    cfg = {
        "h1_k": 10,
        "h2_n": 50,
        "h3_knn": 10,
        "batch_q": int(batch_q),
        "device": device,
        "h2_sample_queries": 256,
        "h3_sample_queries": 256,
        "h5_sample_queries": 128,
    }
    cfg.update(compute_pool_context(pool))
    return cfg


def fit_n1_profile(pool, device, batch_q: int, seed: int = SEED_NULL):
    configure_h5_primary("pair_peak")
    dim = int(pool.shape[1])
    n_train_q = N1_TRAIN_WINDOWS * WINDOW_SIZE
    n_cal_q = N1_CAL_WINDOWS * WINDOW_SIZE
    train_q = n1_queries(n_train_q, dim, seed + 11)
    cal_q = n1_queries(n_cal_q, dim, seed + 12)
    cfg = diag_cfg(device, batch_q, pool)
    train_rows = collect_window_readings(
        train_q, pool, n_windows=N1_TRAIN_WINDOWS, window_size=WINDOW_SIZE,
        seed=seed + 21, diagnosis_config=cfg,
    )
    cal_rows = collect_window_readings(
        cal_q, pool, n_windows=N1_CAL_WINDOWS, window_size=WINDOW_SIZE,
        seed=seed + 22, diagnosis_config=cfg,
    )
    profile = fit_profile(
        train_rows, cal_rows, alpha=ALPHA,
        split_manifest={
            "null": "isotropic_gaussian_l2",
            "n_pool": int(pool.shape[0]),
            "dim": dim,
            "seed": int(seed),
            "train_windows": N1_TRAIN_WINDOWS,
            "cal_windows": N1_CAL_WINDOWS,
        },
        config_snapshot={"h5_primary": "pair_peak", "window_size": WINDOW_SIZE},
    )
    return profile, cfg


def eval_against_profile(profile, query, pool, cfg, seed: int, n_windows: int = N1_EVAL_WINDOWS):
    from profile import diagnose_query_windows
    return diagnose_query_windows(
        profile, query, pool,
        n_windows=n_windows, window_size=WINDOW_SIZE,
        seed=seed, diagnosis_config=cfg,
    )


def lamp_pack(diagnosis: dict) -> dict:
    flags = sorted(diagnosis.get("anomalies") or {})
    evidence = diagnosis.get("evidence") or {}
    scores = {k: float(v["robust_score"]) for k, v in evidence.items()}
    tau = None
    if evidence:
        tau = float(next(iter(evidence.values()))["threshold_score"])
    return {
        "flags": flags,
        "tau": tau,
        "scores": scores,
        "readings": {k: float(v) for k, v in (diagnosis.get("readings") or {}).items()},
        "any_lamp": bool(flags),
    }
