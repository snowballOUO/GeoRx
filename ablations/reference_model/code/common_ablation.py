
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
BASE = PACKAGE_ROOT / "main"
BASE_CODE = BASE / "code"
N1_CODE = PACKAGE_ROOT / "vendor" / "n1"
LEGACY_CODE = PACKAGE_ROOT / "vendor" / "legacy"
for p in (str(BASE_CODE), str(N1_CODE), str(LEGACY_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C  
from n1 import (  
    ALPHA,
    N1_CAL_WINDOWS,
    N1_TRAIN_WINDOWS,
    WINDOW_SIZE,
    diag_cfg,
    eval_against_profile,
    lamp_pack,
    n1_corpus,
    n1_queries,
)
from plant import plant_world  
from profile import (  
    METRIC_SPECS,
    METRIC_SCALE_FLOOR,
    _score,
    collect_window_readings,
    diagnose_query_windows,
    fit_profile,
)

ENCODERS = ["clip_sf_large", "gme_qwen2vl_2b"]
DATASETS = ["mscoco_task0", "nights_task4", "edis_task2"]
PROTOCOL_DATASETS = ["mscoco_task0", "edis_task2"]
SEEDS = [1, 2]
N_STAR = 40038
D_STAR = 768
H3_MIN = 0.15

GEOMS = [
    {"name": "mscoco_768", "d": 768, "n_pool": 5000, "batch_q": 128},
    {"name": "nights_768", "d": 768, "n_pool": 40038, "batch_q": 128},
    {"name": "cirr_4096", "d": 4096, "n_pool": 21551, "batch_q": 32},
]

FROZEN_PATH = Path(os.environ.get("GEORX_FROZEN_STRENGTH", str(PACKAGE_ROOT / "reference_strength.json")))
FROZEN = json.loads(FROZEN_PATH.read_text())["frozen_strength"]


def setup_context(batch_q: int = 128):
    C.configure_new_metrics()
    return C.build_context(batch_q=batch_q, device_str="cuda")


def pairs(protocol: bool = False):
    ds = PROTOCOL_DATASETS if protocol else DATASETS
    return [(e, d) for e in ENCODERS for d in ds]


def variant_dims(name: str, dim: int, n_pool: int) -> Tuple[int, int]:
    if name == "R-full":
        return int(dim), int(n_pool)
    if name == "R-dim":
        return int(dim), N_STAR
    if name == "R-pool":
        return D_STAR, int(n_pool)
    if name == "R-universal":
        return D_STAR, N_STAR
    raise KeyError(name)


def read_rows(dataset, idx: np.ndarray) -> np.ndarray:
    return C._read_rows(dataset, np.asarray(idx, dtype=np.int64))


def load_pool_and_segments(cell, seed: int):
    seg = C.repartition(cell, seed)
    inp = C.load_detector_inputs(cell, seg)
    return seg, inp


def fit_sphere_profile(dim: int, n_pool: int, device, batch_q: int, seed: int,
                       mode: str = "robust_joint"):
    null_seed = C.n1_null_seed(dim, n_pool)
    pool = n1_corpus(n_pool, dim, null_seed)
    bq = C._batch_for(dim, n_pool, batch_q)
    cfg = diag_cfg(device, bq, pool)
    train_q = n1_queries(N1_TRAIN_WINDOWS * WINDOW_SIZE, dim, null_seed + 11)
    cal_q = n1_queries(N1_CAL_WINDOWS * WINDOW_SIZE, dim, null_seed + 12)
    train_rows = collect_window_readings(
        train_q, pool, n_windows=N1_TRAIN_WINDOWS,
        window_size=WINDOW_SIZE, seed=null_seed + 21, diagnosis_config=cfg,
    )
    cal_rows = collect_window_readings(
        cal_q, pool, n_windows=N1_CAL_WINDOWS,
        window_size=WINDOW_SIZE, seed=null_seed + 22, diagnosis_config=cfg,
    )
    profile = fit_custom_profile(train_rows, cal_rows, mode=mode)
    profile["split_manifest"] = {
        "null": "isotropic_gaussian_l2", "dim": dim,
        "n_pool": n_pool, "seed": null_seed,
        "train_windows": N1_TRAIN_WINDOWS,
        "cal_windows": N1_CAL_WINDOWS,
    }
    return profile, pool


def fit_custom_profile(train_rows, cal_rows, mode: str = "robust_joint"):
    if mode == "robust_joint":
        return fit_profile(
            train_rows, cal_rows, alpha=ALPHA,
            split_manifest={"source": "ablation"},
            config_snapshot={"mode": mode, "window_size": WINDOW_SIZE},
        )

    out = fit_profile(
        train_rows, cal_rows, alpha=ALPHA,
        split_manifest={"source": "ablation"},
        config_snapshot={"mode": mode, "window_size": WINDOW_SIZE},
    )
    for name, spec in METRIC_SPECS.items():
        values = np.asarray([r[name] for r in train_rows], dtype=np.float64)
        center = float(np.mean(values)) if mode == "mom_joint" else float(np.median(values))
        if mode == "mom_joint":
            scale = float(np.std(values, ddof=1))
            scale = max(scale, float(METRIC_SCALE_FLOOR.get(name, 1e-8)), 1e-8)
        else:
            mad = float(np.median(np.abs(values - center)))
            iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
            scale = max(1.4826 * mad, iqr / 1.349,
                        float(METRIC_SCALE_FLOOR.get(name, 1e-8)), 1e-8)
        draft = {"center": center, "scale": scale,
                 "direction": spec["direction"]}
        scores = np.asarray([_score(float(r[name]), draft) for r in cal_rows])
        rec = out["metrics"][name]
        rec.update({"center": center, "scale": scale,
                    "train_values": values.tolist(),
                    "calibration_scores": scores.tolist()})

    if mode == "independent":
        for name in METRIC_SPECS:
            scores = np.asarray(out["metrics"][name]["calibration_scores"])
            threshold = float(np.quantile(scores, 0.95, method="higher"))
            out["metrics"][name]["threshold_score"] = max(threshold, 1e-8)
        out["familywise_control"]["mode"] = "independent_95th_quantile"
    else:
        max_scores = []
        for i in range(len(cal_rows)):
            max_scores.append(max(0.0, *(float(out["metrics"][n]["calibration_scores"][i])
                                         for n in METRIC_SPECS)))
        order = int(np.ceil((len(max_scores) + 1) * (1.0 - ALPHA)))
        threshold = max(float(np.sort(max_scores)[order - 1]), 1e-8)
        for name in METRIC_SPECS:
            out["metrics"][name]["threshold_score"] = threshold
        out["familywise_control"]["mode"] = mode
    return out


def fit_native_profile(inp, device, batch_q: int, seed: int, mode="robust_joint"):
    cfg = diag_cfg(device, batch_q, inp["pool"])
    train_rows = collect_window_readings(
        inp["train_q"], inp["pool"], n_windows=N1_TRAIN_WINDOWS,
        window_size=WINDOW_SIZE, seed=seed + 21, diagnosis_config=cfg,
    )
    cal_rows = collect_window_readings(
        inp["cal_q"], inp["pool"], n_windows=N1_CAL_WINDOWS,
        window_size=WINDOW_SIZE, seed=seed + 22, diagnosis_config=cfg,
    )
    return fit_custom_profile(train_rows, cal_rows, mode=mode), cfg


def diagnose_variant(profile, query, pool, cfg, seed: int,
                     window_size: int = WINDOW_SIZE,
                     n_windows: int = 2, h3_guard: bool = True):
    diagnosis = diagnose_query_windows(
        profile, query, pool, n_windows=n_windows,
        window_size=window_size, seed=seed, diagnosis_config=cfg,
    )
    pack = lamp_pack(diagnosis)
    flags = set(pack.get("flags") or {})
    readings = pack.get("readings") or {}
    scores = pack.get("scores") or {}
    phi = []
    per_type = {}
    for typ, (metric, anomaly) in C.TYPE_METRICS.items():
        z = scores.get(metric)
        obs = readings.get(metric)
        lamp = anomaly in flags
        if typ == "h3" and lamp and h3_guard and (obs is None or obs < H3_MIN):
            lamp = False
        if lamp:
            phi.append(typ)
        per_type[typ] = {"metric": metric, "observed": obs,
                         "z": z, "lamp": bool(lamp),
                         "tau": pack.get("tau")}
    tau = pack.get("tau")
    return {"phi": phi, "flags": sorted(flags), "tau": tau,
            "readings": readings, "per_type": per_type}


def fit_window_profile(query, pool, cfg, window_size: int, n_train: int,
                       n_cal: int, seed: int, mode="robust_joint"):
    train_rows = collect_window_readings(
        query[:n_train * window_size], pool, n_windows=n_train,
        window_size=window_size, seed=seed + 21, diagnosis_config=cfg,
    )
    start = n_train * window_size
    cal_rows = collect_window_readings(
        query[start:start + n_cal * window_size], pool, n_windows=n_cal,
        window_size=window_size, seed=seed + 22, diagnosis_config=cfg,
    )
    return fit_custom_profile(train_rows, cal_rows, mode=mode)


def read_valid_queries(cell, seed: int, need: int):
    valid = C.valid_query_indices(cell)
    if len(valid) < need:
        raise RuntimeError(f"{cell.cell_id}: need {need}, have {len(valid)} valid queries")
    rng = np.random.default_rng(C.bc.cell_seed(seed, cell) + 190919)
    idx = rng.permutation(valid)[:need]
    with h5py.File(cell.path, "r") as f:
        q = read_rows(f["query/emb"], idx)
        pool = read_rows(f["pool/emb"], np.arange(cell.n_pool))
    return q, pool, idx


def gate_records(profile_builder, device, batch_q: int, seed: int,
                 h3_guard: bool = True):
    rows = []
    for geom in GEOMS:
        d, n_pool = geom["d"], geom["n_pool"]
        null_seed = C.n1_null_seed(d, n_pool)
        pool0 = n1_corpus(n_pool, d, null_seed)
        cfg0 = diag_cfg(device, geom["batch_q"], pool0)
        profile, _ = profile_builder(d, n_pool, device, geom["batch_q"], seed)
        q0 = n1_queries(2 * WINDOW_SIZE, d, null_seed + 30)
        w0 = diagnose_variant(profile, q0, pool0, cfg0, null_seed + 40,
                              h3_guard=h3_guard)
        rows.append({"geom": geom["name"], "kind": "W0",
                     "phi": w0["phi"], "n_target": None,
                     "target_hit": not bool(w0["phi"])})
        for kind in ("h1", "h2", "h3", "h4", "h5"):
            spec = dict(FROZEN[kind])
            salt = null_seed + 100 + ord(kind[1]) + 17 * int(spec.get("level", 0))
            q_tr = n1_queries(N1_TRAIN_WINDOWS * WINDOW_SIZE, d, salt + 1)
            q_ca = n1_queries(N1_CAL_WINDOWS * WINDOW_SIZE, d, salt + 2)
            q_ev = n1_queries(2 * WINDOW_SIZE, d, salt + 3)
            q_p, pool_p = plant_world(kind, q_ev, pool0, salt, spec, device=device)
            cfg_p = diag_cfg(device, geom["batch_q"], pool_p)
            out = diagnose_variant(profile, q_p, pool_p, cfg_p, salt + 40,
                                  h3_guard=h3_guard)
            rows.append({"geom": geom["name"], "kind": kind,
                         "phi": out["phi"], "n_target": kind,
                         "target_hit": kind in out["phi"]})
        C._free(pool0)
        del pool0
    return rows


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
