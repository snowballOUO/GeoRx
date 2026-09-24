
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
N1CODE = ROOT.parent / "vendor" / "n1"
LEGACY = ROOT.parent / "vendor" / "legacy"
for p in (str(LEGACY), str(N1CODE), str(CODE)):
    if p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(N1CODE))


if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import h5py  

import batch_common as bc  
import diagnosis as diag_mod  
import profile as profile_mod  
import geometry  
from retrieve import brute_topk  
from h5_score_interaction import anchor_ids, decompose  
from n1 import (  
    SEED_NULL,
    WINDOW_SIZE,
    diag_cfg,
    eval_against_profile,
    fit_n1_profile,
    lamp_pack,
    n1_corpus,
)
from profile import collect_window_readings, fit_profile  
from diagnosis import configure_h5_primary  

try:
    import torch
except ImportError:  
    torch = None




HDF5_ROOT = Path(os.environ["GEORX_DATA_ROOT"]) if os.environ.get("GEORX_DATA_ROOT") else ROOT.parent / "input_hdf5"
MBEIR = Path(os.environ["GEORX_MBEIR_ROOT"]) if os.environ.get("GEORX_MBEIR_ROOT") else ROOT.parent / "input_mbeir"
SPLITS_YAML = LEGACY / "cell_splits.yaml"

PRIMARY_ENCODERS = [
    "clip_sf_large", "blip_ff_large", "openclip_fft", "gme_qwen2vl_2b",
    "vlm2vec_phi3", "e5v_llava_next", "clip_vitb32", "siglip_base", "blip2_vitL",
]
PRIMARY_DATASETS = [
    "mscoco_task0", "visualnews_task0", "fashion200k_task0", "webqa_task1",
    "edis_task2", "nights_task4", "oven_task6", "infoseek_task6",
    "fashioniq_task7", "cirr_task7",
]
SEEDS = [1, 2, 3]







Q_EVAL_N = 500         
N_TRAIN = 512          
N_CAL = 320            
N_FAULT = 320          
Q_DIAG_N = N_TRAIN + N_CAL + N_FAULT          
TOTAL_DRAW = Q_EVAL_N + Q_DIAG_N              
TRAIN_WINDOWS = 32
CAL_WINDOWS = 20

SPHERE_CAL_WINDOWS = int(os.environ.get("SPHERE_CAL_WINDOWS", "32"))
if SPHERE_CAL_WINDOWS < 4:
    raise ValueError("SPHERE_CAL_WINDOWS must be >= 4")
FAULT_WINDOWS = 2
ALPHA = 0.05
H3_SECOND_FRAC_MIN = 0.15  
USEFUL_DR10 = 0.005
RERANK_N = 100
ANALYTIC_METHODS = ["csls_p050", "alpha_qe_a010", "diffusion"]

COMPOSE_ORDER = ["csls_p050", "alpha_qe_a010", "diffusion"]
OPEN_CE_LOADS = {
    "mscoco_task0", "visualnews_task0", "fashion200k_task0",
    "webqa_task1", "edis_task2", "oven_task6", "infoseek_task6",
}
LOCAL_CE_LOADS = {"nights_task4", "fashioniq_task7", "cirr_task7"}
PHI_TO_ANALYTIC = {
    "h1": "csls_p050", "h2": "alpha_qe_a010", "h3": "diffusion",
}


NATIVE_INJECT_STRENGTH = 0.75
H3_TIGHTNESS = 0.90
H3_RESIDUAL = 0.05
H3_BRIDGE_K = 64


NEW_METRIC_SPECS = {
    "h1_skew_over_null": {"path": ("h1", "skew_over_null"), "anomaly": "hubness", "direction": "high"},
    "h1_cv_over_null": {"path": ("h1", "cv_over_null"), "anomaly": "hubness", "direction": "high"},
    "h2_excess_cos": {"path": ("h2", "topn_pairwise_mean"), "anomaly": "neighborhood_overconcentration", "direction": "high"},
    "h3_second_frac": {"path": ("h3", "second_frac_mean"), "anomaly": "manifold_fragmentation", "direction": "high"},
    "h3_clustering": {"path": ("h3", "clustering_coef_mean"), "anomaly": "weak_local_connectivity", "direction": "low"},
    "h4_top1_top2_gap": {"path": ("h4", "top1_top2_gap"), "anomaly": "score_ambiguity", "direction": "low"},
    "h5_score_interaction_share": {"path": ("h5", "interaction_share"), "anomaly": "weak_pair_specific_score_interaction", "direction": "low"},
}


def configure_new_metrics():
    diag_mod.METRIC_SPECS.clear()
    diag_mod.METRIC_SPECS.update(NEW_METRIC_SPECS)
    profile_mod.METRIC_SPECS = diag_mod.METRIC_SPECS
    profile_mod.METRIC_SCALE_FLOOR.pop("h2_neighbor_sim_ratio", None)
    profile_mod.METRIC_SCALE_FLOOR.pop("h5_pair_peak_leftover", None)
    profile_mod.METRIC_SCALE_FLOOR["h2_excess_cos"] = 1e-3
    profile_mod.METRIC_SCALE_FLOOR["h5_score_interaction_share"] = 1e-5
    
    
    
    
    import n1 as n1_mod
    n1_mod.configure_h5_primary = lambda *_args, **_kwargs: None


configure_new_metrics()


TYPE_METRICS = {
    "h1": ("h1_skew_over_null", "hubness"),
    "h2": ("h2_excess_cos", "neighborhood_overconcentration"),
    "h3": ("h3_second_frac", "manifold_fragmentation"),
    "h4": ("h4_top1_top2_gap", "score_ambiguity"),
    "h5": ("h5_score_interaction_share", "weak_pair_specific_score_interaction"),
}





@dataclass
class Context:
    runs: Path
    n1_cache: Path
    logs: Path
    device: object
    batch_q: int
    cells: Dict[str, "bc.Cell"]

    def cell(self, encoder: str, dataset: str) -> "bc.Cell":
        return self.cells[f"{encoder}__{dataset}"]


def build_context(batch_q: int = 128, device_str: str = "cuda") -> Context:
    from retrieve import get_device
    
    
    
    configure_new_metrics()
    splits = bc.load_cell_splits(SPLITS_YAML)
    overrides = dict(splits.get("path_overrides") or {})
    cells = {c.cell_id: c for c in bc.inventory(HDF5_ROOT)}
    for cid, path in overrides.items():
        if cid in cells:
            c = cells[cid]
            path = Path(os.path.expandvars(str(path))).expanduser()
            with h5py.File(path, "r") as f:
                nt = int(f["query/emb_train"].shape[0]) if "query/emb_train" in f else 0
                nq = int(f["query/emb"].shape[0])
                npool = int(f["pool/emb"].shape[0])
                dim = int(f["pool/emb"].shape[1])
            cells[cid] = replace(c, path=Path(path), n_train=nt, n_test=nq, n_pool=npool, dim=dim)
    runs = ROOT / "runs"
    ctx = Context(
        runs=runs,
        n1_cache=runs / "n1_cache",
        logs=runs / "logs",
        device=get_device(device_str),
        batch_q=int(batch_q),
        cells=cells,
    )
    ctx.n1_cache.mkdir(parents=True, exist_ok=True)
    ctx.logs.mkdir(parents=True, exist_ok=True)
    return ctx


def all_pairs(encoders=None, datasets=None) -> List[Tuple[str, str]]:
    enc = set(encoders or PRIMARY_ENCODERS)
    ds = set(datasets or PRIMARY_DATASETS)
    return [(e, d) for e in PRIMARY_ENCODERS for d in PRIMARY_DATASETS if e in enc and d in ds]





def valid_query_indices(cell: "bc.Cell") -> np.ndarray:
    with h5py.File(cell.path, "r") as f:
        qi = np.asarray(f["qrels/query_idx"], dtype=np.int64)
    return np.unique(qi)


def repartition(cell: "bc.Cell", seed: int) -> Dict[str, np.ndarray]:
    valid = valid_query_indices(cell)
    if valid.size < TOTAL_DRAW:
        raise RuntimeError(f"{cell.cell_id}: valid pool {valid.size} < TOTAL_DRAW {TOTAL_DRAW}")
    cseed = bc.cell_seed(seed, cell)
    perm = np.random.default_rng(cseed).permutation(valid)
    a = Q_EVAL_N
    b = a + N_TRAIN
    c = b + N_CAL
    d = c + N_FAULT
    return {
        "eval": perm[:a],
        "train": perm[a:b],
        "calibration": perm[b:c],
        "fault_eval": perm[c:d],
        "n_valid": int(valid.size),
        "_cell_seed": int(cseed),
    }


def _read_rows(dataset, idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        dim = int(dataset.shape[1])
        return np.zeros((0, dim), dtype=np.float32)
    order = np.argsort(idx, kind="mergesort")
    val = np.asarray(dataset[idx[order]], dtype=np.float32)
    out = np.empty_like(val)
    out[order] = val
    return _l2(out)


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)).astype(np.float32, copy=False)


def hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.sort(np.asarray(idx, dtype=np.int64)).tobytes()).hexdigest()


def load_detector_inputs(cell: "bc.Cell", seg: Dict[str, np.ndarray]) -> Dict:
    with h5py.File(cell.path, "r") as f:
        q = f["query/emb"]
        train_q = _read_rows(q, seg["train"])
        cal_q = _read_rows(q, seg["calibration"])
        fault_q = _read_rows(q, seg["fault_eval"])
        pool = _read_rows(f["pool/emb"], np.arange(cell.n_pool))
    
    assert np.intersect1d(seg["eval"], np.concatenate([seg["train"], seg["calibration"], seg["fault_eval"]])).size == 0
    return {
        "train_q": train_q, "cal_q": cal_q, "fault_q": fault_q, "pool": pool,
        "n_valid": int(seg["n_valid"]),
        "manifest": {
            "sampling": "per_seed_permutation_of_valid_test_queries",
            "cell_seed": int(seg["_cell_seed"]),
            "n_valid": int(seg["n_valid"]),
            "q_eval_n": int(len(seg["eval"])),
            "q_diag_n": int(len(seg["train"]) + len(seg["calibration"]) + len(seg["fault_eval"])),
            "n_train": int(len(seg["train"])),
            "n_cal": int(len(seg["calibration"])),
            "n_fault": int(len(seg["fault_eval"])),
            "q_eval_indices_sha256": hash_idx(seg["eval"]),
            "train_indices_sha256": hash_idx(seg["train"]),
            "calibration_indices_sha256": hash_idx(seg["calibration"]),
            "fault_eval_indices_sha256": hash_idx(seg["fault_eval"]),
            "q_diag_eval_disjoint": True,
        },
    }





def _new_window_rows(query, pool, cfg, n_windows, seed, anchors):
    if len(query) == 0:
        raise ValueError("empty query split")
    vals, idx = brute_topk(query, pool, min(50, len(pool)),
                           device=cfg["device"], batch_q=int(cfg["batch_q"]))
    intra = np.empty(len(query), dtype=np.float64)
    second = np.empty(len(query), dtype=np.float64)
    clustering = np.empty(len(query), dtype=np.float64)
    for qi in range(len(query)):
        vec = np.asarray(pool[idx[qi]], dtype=np.float64)
        k = len(vec)
        summed = vec.sum(axis=0)
        intra[qi] = (summed @ summed - np.sum(vec * vec)) / max(k * (k - 1), 1)
        h3 = geometry.h3_nn_graph(pool, idx[qi:qi+1], n=k,
                                  knn=min(10, k - 1), n_sample_queries=1)
        second[qi] = h3["second_frac_mean"]
        clustering[qi] = h3["clustering_coef_mean"]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_windows)):
        chosen = rng.choice(len(query), min(WINDOW_SIZE, len(query)), replace=False)
        h1 = geometry.h1_koccurrence(idx[chosen], len(pool), primary_k=10)
        rows.append({
            "h1_skew_over_null": float(h1["skew_over_null"]),
            "h1_cv_over_null": float(h1["cv_over_null"]),
            "h2_excess_cos": float(intra[chosen].mean() - cfg["pool_pairwise_mean"]),
            "h3_second_frac": float(second[chosen].mean()),
            "h3_clustering": float(clustering[chosen].mean()),
            "h4_top1_top2_gap": float((vals[chosen, 0] - vals[chosen, 1]).mean()),
            "h5_score_interaction_share": float(decompose((query[chosen] @ pool[anchors].T))["interaction_share"]),
        })
    return rows


def _compute_readings_v3(query, pool, cfg):
    local_cfg = dict(cfg)
    if "pool_pairwise_mean" not in local_cfg:
        local_cfg["pool_pairwise_mean"] = geometry.pool_pairwise_mean(pool, n_sample=1024)
    return _new_window_rows(
        np.asarray(query), np.asarray(pool), local_cfg, n_windows=1,
        seed=0, anchors=anchor_ids(len(pool)),
    )[0]





profile_mod.compute_readings = _compute_readings_v3
diag_mod.compute_readings = _compute_readings_v3


def fit_self_profile(train_q, cal_q, pool, device, batch_q: int, seed: int) -> Dict:
    configure_new_metrics()
    cfg = diag_cfg(device, batch_q, pool)
    anchors = anchor_ids(len(pool))
    train_rows = _new_window_rows(train_q, pool, cfg, TRAIN_WINDOWS, seed + 21, anchors)
    cal_rows = _new_window_rows(cal_q, pool, cfg, CAL_WINDOWS, seed + 22, anchors)
    return fit_profile(
        train_rows, cal_rows, alpha=ALPHA,
        split_manifest={"source": "native_test_repartition", "seed": int(seed)},
        config_snapshot={"h2": "h2_excess_cos", "h5": "h5_score_interaction_share",
                         "h5_anchors": int(len(anchors)), "h5_anchor_seed": 42,
                         "window_size": WINDOW_SIZE, "n_train": N_TRAIN,
                         "n_cal": N_CAL, "n_fault": N_FAULT},
    ), cfg


def n1_key(dim: int, n_pool: int) -> str:
    return f"d{int(dim)}_n{int(n_pool)}_c{int(SPHERE_CAL_WINDOWS)}"


def n1_null_seed(dim: int, n_pool: int) -> int:
    return int(SEED_NULL) + int(dim) * 1_000_003 + int(n_pool)


def get_n1_profile(dim: int, n_pool: int, ctx: Context) -> Dict:
    import fcntl
    key = n1_key(dim, n_pool)
    cdir = ctx.n1_cache / key
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / "profile.json"
    with (cdir / "lock").open("a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if path.is_file():
            return json.loads(path.read_text())
        null_seed = n1_null_seed(dim, n_pool)
        pool = n1_corpus(n_pool, dim, null_seed)
        bq = _batch_for(dim, n_pool, ctx.batch_q)
        import n1 as n1_mod
        old_cal = int(n1_mod.N1_CAL_WINDOWS)
        n1_mod.N1_CAL_WINDOWS = int(SPHERE_CAL_WINDOWS)
        try:
            profile, _ = fit_n1_profile(pool, ctx.device, bq, null_seed)
        finally:
            n1_mod.N1_CAL_WINDOWS = old_cal
        
        
        profile.setdefault("config_snapshot", {})["h5_primary"] = "score_interaction_share"
        profile["config_snapshot"]["sphere_cal_windows"] = int(SPHERE_CAL_WINDOWS)
        profile.setdefault("split_manifest", {})["cal_windows"] = int(SPHERE_CAL_WINDOWS)
        profile["profile_id"] = profile_mod._hash_json(profile)
        slim = _slim(profile)
        path.write_text(json.dumps(slim, default=_json))
        _free(pool)
        return slim


def gate_passed(ctx: Context) -> bool:
    path = ctx.runs / "gate" / "GATE.json"
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text()).get("gate_pass"))
    except Exception:
        return False


def diagnose_phi(profile: Dict, fault_q, pool, cfg, seed: int) -> Dict:
    configure_new_metrics()
    anchors = anchor_ids(len(pool))
    rows = _new_window_rows(fault_q, pool, cfg, FAULT_WINDOWS, seed + 4, anchors)
    readings = {name: float(np.median([row[name] for row in rows])) for name in profile["metrics"]}
    d = diag_mod.diagnose_readings(readings, profile)
    phi, per_type = [], {}
    for typ, (metric, anomaly) in TYPE_METRICS.items():
        ev = d["evidence"][metric]
        lamp = anomaly in d["anomalies"]
        if typ == "h3" and lamp and readings[metric] < H3_SECOND_FRAC_MIN:
            lamp = False
        if lamp:
            phi.append(typ)
        per_type[typ] = {"metric": metric, "anomaly": anomaly,
                         "observed": float(readings[metric]),
                         "z": float(ev["robust_score"]),
                         "tau": float(ev["threshold_score"]), "lamp": bool(lamp)}
    return {"phi": phi, "tau": float(next(iter(d["evidence"].values()))["threshold_score"]),
            "per_type": per_type, "readings": readings,
            "scores": {k: float(v["robust_score"]) for k, v in d["evidence"].items()}}


def _batch_for(dim: int, n_pool: int, batch_q: int) -> int:
    if n_pool >= 500_000 and dim >= 2048:
        return min(int(batch_q), 16)
    if n_pool >= 500_000 or dim >= 2048:
        return min(int(batch_q), 32)
    return int(batch_q)


def _slim(profile: Dict) -> Dict:
    out = dict(profile)
    out["metrics"] = {n: {k: r[k] for k in r if k not in ("train_values", "calibration_scores")}
                      for n, r in profile["metrics"].items()}
    fw = dict(profile["familywise_control"]); fw.pop("calibration_max_scores", None)
    out["familywise_control"] = fw
    return out


def _free(*arrs):
    import gc
    for a in arrs:
        del a
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def predicted_P(phi: List[str], dataset: str) -> List[str]:
    out = []
    for t in ("h1", "h2", "h3", "h4", "h5"):
        if t not in phi:
            continue
        if t == "h4":
            out.append("ce_list_local")
        elif t == "h5":
            out.append("ce_pair_blip_itm" if dataset in OPEN_CE_LOADS else "ce_pair_local")
        else:
            out.append(PHI_TO_ANALYTIC[t])
    return out


def literature_P(P: List[str], dataset: str) -> List[str]:
    keep = set(ANALYTIC_METHODS)
    if dataset in OPEN_CE_LOADS:
        keep.add("ce_pair_blip_itm")
    return [m for m in P if m in keep]


def diagnosis_path(ctx: Context, seed: int, cell_id: str) -> Path:
    return ctx.runs / f"S{seed}" / "evaluations" / cell_id / "diagnosis.json"


def load_sealed_diagnosis(ctx: Context, seed: int, cell_id: str) -> Dict:
    path = diagnosis_path(ctx, seed, cell_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing sealed diagnosis {path}")
    payload = json.loads(path.read_text())
    if not verify_seal(payload):
        raise RuntimeError(f"seal mismatch {path}")
    return payload


def load_eval_labeled(cell: "bc.Cell", seg: Dict[str, np.ndarray]) -> Dict:
    eval_idx = np.asarray(seg["eval"], dtype=np.int64)
    train_idx = np.asarray(seg["train"], dtype=np.int64)
    with h5py.File(cell.path, "r") as f:
        eval_q = _read_rows(f["query/emb"], eval_idx)
        ref_q = _read_rows(f["query/emb"], train_idx)
        pool = _read_rows(f["pool/emb"], np.arange(cell.n_pool))
        qi = np.asarray(f["qrels/query_idx"], dtype=np.int64)
        pi = np.asarray(f["qrels/pool_idx"], dtype=np.int64)
    loc = {int(q): i for i, q in enumerate(eval_idx.tolist())}
    positives = [[] for _ in range(len(eval_idx))]
    for q, p in zip(qi.tolist(), pi.tolist()):
        i = loc.get(int(q))
        if i is not None:
            positives[i].append(int(p))
    if any(not g for g in positives):
        raise ValueError(f"{cell.cell_id}: Q_eval query without gold in local pool")
    return {
        "eval_q": eval_q, "pool": pool, "ref_q": ref_q, "positives": positives,
        "eval_idx": eval_idx,
        "q_eval_indices_sha256": hash_idx(eval_idx),
        "qrels_opened": now(),
    }


def recall_hits(top_idx: np.ndarray, positives, k: int) -> np.ndarray:
    return np.asarray(
        [bool(set(gold).intersection(row[:k].tolist())) for row, gold in zip(top_idx, positives)],
        dtype=np.bool_,
    )





def seal(payload: Dict) -> Dict:
    payload = dict(payload)
    payload.pop("seal_sha256", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json).encode()
    payload["seal_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def verify_seal(payload: Dict) -> bool:
    got = payload.get("seal_sha256")
    if not got:
        return False
    body = {k: v for k, v in payload.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=_json).encode()
    return hashlib.sha256(blob).hexdigest() == got


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json))
    tmp.replace(path)  


def locked_csv(path: Path, rows: List[dict], *, fieldnames=None) -> None:
    import csv
    import fcntl
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not rows:
            return
        fields = fieldnames or list(rows[0].keys())
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _json(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_blip_classes():
    import os
    shadowed = sys.modules.get("profile")
    if shadowed is not None and not hasattr(shadowed, "run"):
        sys.modules.pop("profile", None)
    stdlib = os.path.dirname(os.__file__)
    if sys.path[:1] != [stdlib]:
        sys.path.insert(0, stdlib)
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    if shadowed is not None:
        sys.modules["profile"] = shadowed
    return BlipForImageTextRetrieval, BlipProcessor
