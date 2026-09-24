
from __future__ import annotations

from typing import Dict

import geometry


METRIC_SPECS = {
    "h1_skew_over_null": {"path": ("h1", "skew_over_null"), "anomaly": "hubness", "direction": "high"},
    "h1_cv_over_null": {"path": ("h1", "cv_over_null"), "anomaly": "hubness", "direction": "high"},
    "h2_neighbor_sim_ratio": {"path": ("h2", "neighbor_sim_ratio"), "anomaly": "neighborhood_overconcentration", "direction": "high"},
    "h3_second_frac": {"path": ("h3", "second_frac_mean"), "anomaly": "manifold_fragmentation", "direction": "high"},
    "h3_clustering": {"path": ("h3", "clustering_coef_mean"), "anomaly": "weak_local_connectivity", "direction": "low"},
    "h4_top1_top2_gap": {"path": ("h4", "top1_top2_gap"), "anomaly": "score_ambiguity", "direction": "low"},
    "h5_interaction_gap": {"path": ("h5", "interaction_gap"), "anomaly": "weak_interaction", "direction": "low"},
}

AUX_SPECS = {
    "h3_fragmentation": {"path": ("h3", "fragmentation")},
    "h3_n_components": {"path": ("h3", "n_components_mean")},
    "h3_largest_component_frac": {"path": ("h3", "largest_component_frac_mean")},
    "h5_interaction_gap_aux": {"path": ("h5", "interaction_gap")},
    "h5_pair_peak_aux": {"path": ("h5", "pair_peak_leftover_mean")},
}


def configure_h5_primary(name: str) -> None:
    if name == "interaction_gap":
        METRIC_SPECS["h5_interaction_gap"] = {
            "path": ("h5", "interaction_gap"),
            "anomaly": "weak_interaction",
            "direction": "low",
        }
        METRIC_SPECS.pop("h5_pair_peak_leftover", None)
    elif name == "pair_peak":
        METRIC_SPECS["h5_pair_peak_leftover"] = {
            "path": ("h5", "pair_peak_leftover_mean"),
            "anomaly": "weak_interaction",
            "direction": "low",
        }
        METRIC_SPECS.pop("h5_interaction_gap", None)
    else:
        raise ValueError(name)


def raw_diagnostics(query, pool, config: Dict) -> Dict:
    return geometry.run_diagnostics(
        query,
        pool,
        h1_k=int(config.get("h1_k", 10)),
        h2_n=int(config.get("h2_n", 50)),
        h3_knn=int(config.get("h3_knn", 10)),
        h2_sample_queries=int(config.get("h2_sample_queries", 256)),
        h3_sample_queries=int(config.get("h3_sample_queries", 64)),
        h5_sample_queries=int(config.get("h5_sample_queries", 128)),
        pool_pairwise_mean_value=config.get("pool_pairwise_mean"),
        device=config.get("device"),
        batch_q=int(config.get("batch_q", 256)),
    )


def readings_from_raw(raw: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, spec in METRIC_SPECS.items():
        section, key = spec["path"]
        out[name] = float(raw[section][key])
    for name, spec in AUX_SPECS.items():
        section, key = spec["path"]
        out[name] = float(raw[section][key])
    return out


def compute_readings(query, pool, config: Dict) -> Dict[str, float]:
    return readings_from_raw(raw_diagnostics(query, pool, config))


def compute_pool_context(pool, n_sample: int = 1024) -> Dict[str, float]:
    return {"pool_pairwise_mean": geometry.pool_pairwise_mean(pool, n_sample=n_sample)}


def robust_score(value: float, record: Dict) -> float:
    center = float(record["center"])
    scale = max(float(record["scale"]), 1e-12)
    direction = record["direction"]
    if direction == "high":
        return (value - center) / scale
    if direction == "low":
        return (center - value) / scale
    return abs(value - center) / scale


ABS_MIN = {
    "h3_second_frac": 0.15,
}


def diagnose_readings(readings: Dict[str, float], profile: Dict) -> Dict:
    evidence = {}
    anomalies = {}
    for metric, record in profile["metrics"].items():
        value = float(readings[metric])
        score = float(robust_score(value, record))
        threshold = float(record["threshold_score"])
        flagged = bool(score > threshold)
        if metric in ABS_MIN:
            flagged = flagged and (value >= float(ABS_MIN[metric]))
        severity = max(0.0, score / max(threshold, 1e-12))
        item = {
            "metric": metric,
            "anomaly": record["anomaly"],
            "observed": value,
            "normal_lower": record.get("normal_lower"),
            "normal_upper": record.get("normal_upper"),
            "direction": record["direction"],
            "robust_score": score,
            "threshold_score": threshold,
            "severity": severity,
            "flagged": flagged,
        }
        evidence[metric] = item
        if flagged:
            anomaly = record["anomaly"]
            prev = anomalies.get(anomaly)
            if prev is None or severity > prev["severity"]:
                anomalies[anomaly] = {
                    "severity": severity,
                    "primary_metric": metric,
                    "observed": value,
                    "normal_range": [record.get("normal_lower"), record.get("normal_upper")],
                }
    return {
        "profile_id": profile["profile_id"],
        "readings": {k: float(v) for k, v in readings.items()},
        "anomalies": anomalies,
        "evidence": evidence,
        "diagnosis_ground_truth": "not_repair_outcome",
        "familywise_flags_are_channel_policy": False,
    }


def anomaly_robust_score(diagnosis: Dict, anomaly: str):
    scores = [
        float(item["robust_score"])
        for item in (diagnosis.get("evidence") or {}).values()
        if item.get("anomaly") == anomaly
    ]
    return max(scores) if scores else None
