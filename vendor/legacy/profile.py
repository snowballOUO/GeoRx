
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Dict, Iterable, Tuple

import numpy as np

from diagnosis import METRIC_SPECS, compute_readings, diagnose_readings



METRIC_SCALE_FLOOR = {
    "h1_skew_over_null": 1e-3,
    "h1_cv_over_null": 1e-3,
    "h2_neighbor_sim_ratio": 1e-3,
    "h3_fragmentation": 0.02,
    "h3_clustering": 0.02,
    "h4_top1_top2_gap": 1e-4,
    "h5_interaction_gap": 1e-5,
}


def split_train_queries(
    query_train: np.ndarray,
    *,
    train_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    if query_train is None or len(query_train) < 12:
        raise ValueError("query/emb_train with at least 12 rows is required")
    if train_fraction <= 0 or calibration_fraction <= 0 or train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must be positive and sum to < 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(query_train))
    n_train = int(len(order) * train_fraction)
    n_cal = int(len(order) * calibration_fraction)
    a, b = order[:n_train], order[n_train : n_train + n_cal]
    c = order[n_train + n_cal :]
    manifest = {
        "seed": int(seed),
        "source": "query/emb_train",
        "train_indices_sha256": _hash_indices(a),
        "calibration_indices_sha256": _hash_indices(b),
        "fault_eval_indices_sha256": _hash_indices(c),
        "n_train": int(len(a)),
        "n_calibration": int(len(b)),
        "n_fault_eval": int(len(c)),
        "test_source": "query/emb (strictly separate; not used here)",
    }
    return query_train[a], query_train[b], query_train[c], manifest


def collect_window_readings(
    query: np.ndarray,
    pool: np.ndarray,
    *,
    n_windows: int,
    window_size: int,
    seed: int,
    diagnosis_config: Dict,
) -> list[Dict[str, float]]:
    if len(query) == 0:
        raise ValueError("empty query split")
    rng = np.random.default_rng(seed)
    size = min(int(window_size), len(query))
    rows = []
    for _ in range(int(n_windows)):
        idx = rng.choice(len(query), size=size, replace=False)
        rows.append(compute_readings(query[idx], pool, diagnosis_config))
    return rows


def fit_profile(
    train_readings: Iterable[Dict[str, float]],
    calibration_readings: Iterable[Dict[str, float]],
    *,
    alpha: float,
    split_manifest: Dict,
    config_snapshot: Dict,
) -> Dict:
    train_rows = list(train_readings)
    cal_rows = list(calibration_readings)
    if not train_rows or not cal_rows:
        raise ValueError("both train and calibration windows are required")
    metrics = {}
    calibration_scores_by_metric = {}
    for name, spec in METRIC_SPECS.items():
        values = np.asarray([r[name] for r in train_rows], dtype=np.float64)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        scale = max(
            1.4826 * mad,
            iqr / 1.349,
            abs(center) * 1e-6,
            float(METRIC_SCALE_FLOOR.get(name, 1e-8)),
            1e-8,
        )
        draft = {"center": center, "scale": scale, "direction": spec["direction"]}
        scores = np.asarray([_score(float(r[name]), draft) for r in cal_rows], dtype=np.float64)
        calibration_scores_by_metric[name] = scores
        metrics[name] = {
            "anomaly": spec["anomaly"],
            "direction": spec["direction"],
            "center": center,
            "scale": scale,
            "scale_floor": float(METRIC_SCALE_FLOOR.get(name, 1e-8)),
            "train_values": values.tolist(),
            "calibration_scores": scores.tolist(),
        }
    max_scores = np.asarray(
        [
            max(0.0, *(float(calibration_scores_by_metric[name][i]) for name in METRIC_SPECS))
            for i in range(len(cal_rows))
        ],
        dtype=np.float64,
    )
    order_k = min(len(max_scores), math.ceil((len(max_scores) + 1) * (1.0 - alpha)))
    global_threshold = max(float(np.sort(max_scores)[order_k - 1]), 1e-8)
    for record in metrics.values():
        record["threshold_score"] = global_threshold
        lower, upper = _normal_bounds(
            float(record["center"]), float(record["scale"]), global_threshold, record["direction"]
        )
        record["normal_lower"] = lower
        record["normal_upper"] = upper
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_source": "train embeddings only",
        "threshold_source": "clean calibration embeddings only",
        "test_used_for_fit": False,
        "repair_outcome_used": False,
        "alpha": float(alpha),
        "familywise_control": {
            "method": "max_directional_nonconformity_split_conformal",
            "role": "controlled_fault_detection_only",
            "calibration_windows": len(cal_rows),
            "order_statistic_k": int(order_k),
            "global_threshold_score": global_threshold,
            "finite_sample_miscoverage_upper_bound": float(
                (len(max_scores) + 1 - order_k) / (len(max_scores) + 1)
            ),
            "calibration_max_scores": max_scores.tolist(),
        },
        "split_manifest": split_manifest,
        "config_snapshot": config_snapshot,
        "metrics": metrics,
    }
    payload["profile_id"] = _hash_json(payload)
    return payload


def diagnose_query_windows(
    profile: Dict,
    query: np.ndarray,
    pool: np.ndarray,
    *,
    n_windows: int,
    window_size: int,
    seed: int,
    diagnosis_config: Dict,
) -> Dict:
    windows = collect_window_readings(
        query,
        pool,
        n_windows=n_windows,
        window_size=window_size,
        seed=seed,
        diagnosis_config=diagnosis_config,
    )
    aggregated = {name: float(np.median([row[name] for row in windows])) for name in windows[0]}
    result = diagnose_readings(aggregated, profile)
    result["aggregation"] = "median_across_matched_windows"
    result["n_windows"] = int(n_windows)
    result["window_size"] = int(min(window_size, len(query)))
    result["window_readings"] = windows
    return result


def _score(value: float, record: Dict) -> float:
    center, scale = float(record["center"]), max(float(record["scale"]), 1e-12)
    if record["direction"] == "high":
        return (value - center) / scale
    if record["direction"] == "low":
        return (center - value) / scale
    return abs(value - center) / scale


def _normal_bounds(center: float, scale: float, threshold: float, direction: str):
    if direction == "high":
        return None, center + threshold * scale
    if direction == "low":
        return center - threshold * scale, None
    return center - threshold * scale, center + threshold * scale


def _hash_indices(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _hash_json(value: Dict) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
