
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

import common as C
from n1 import (  
    ALPHA, N1_CAL_WINDOWS, N1_TRAIN_WINDOWS, WINDOW_SIZE,
    diag_cfg, eval_against_profile, lamp_pack, n1_corpus, n1_queries,
)
from plant import plant_world  
from profile import collect_window_readings, diagnose_query_windows, fit_profile  
from h5_score_interaction import contract_queries  

STAGE = "diag_recall"

GEOMS = [
    {"name": "mscoco_768", "d": 768, "n_pool": 5000, "batch_q": 128},
    {"name": "nights_768", "d": 768, "n_pool": 40038, "batch_q": 128},
    {"name": "cirr_4096", "d": 4096, "n_pool": 21551, "batch_q": 32},
]
TARGETS = {"h1": "hubness", "h2": "neighborhood_overconcentration",
           "h3": "manifold_fragmentation", "h4": "score_ambiguity", "h5": "weak_interaction"}



TARGETS["h5"] = "weak_pair_specific_score_interaction"
H5_CONTRACT_STRENGTH = 0.75
H5_SPEC = {
    "level": "interaction_share_v1",
    "mode": "contract_queries",
    "strength": H5_CONTRACT_STRENGTH,
    "pool": "unchanged",
    "clean_direction": "mean(q_train)",
}
METRIC = {k: v[0] for k, v in C.TYPE_METRICS.items()}
FROZEN = json.loads(
    (Path("$GEORX_FROZEN_STRENGTH")
     ).read_text())["frozen_strength"]


FP_WINDOWS = int(os.environ.get("SPHERE_HOLDOUT_WINDOWS", "256"))






FP_MAX = max(1, math.floor(FP_WINDOWS * 3 / 32))


def cell_done(ctx: C.Context, seed: int, cell_id: str) -> bool:
    
    return (ctx.runs / "gate" / "GATE.json").is_file()


def run(ctx: C.Context, seed: int, pairs, log) -> dict:
    gate_dir = ctx.runs / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    if (gate_dir / "GATE.json").is_file():
        prev = json.loads((gate_dir / "GATE.json").read_text())
        log(f"[diag_recall] gate already computed pass={prev.get('gate_pass')} (seed-independent)")
        return {"stage": STAGE, "seed": seed, "gate_pass": prev.get("gate_pass"), "resumed": True}

    plant_rows, fp_rows, w0 = [], [], []
    for geom in GEOMS:
        log(f"[diag_recall] geom {geom['name']} d={geom['d']} n_pool={geom['n_pool']} (fixed sphere)")
        null_seed = C.n1_null_seed(geom["d"], geom["n_pool"])
        pool0 = n1_corpus(geom["n_pool"], geom["d"], null_seed)
        profile = C.get_n1_profile(geom["d"], geom["n_pool"], ctx)  
        cfg = diag_cfg(ctx.device, geom["batch_q"], pool0)
        w0_rec = _w0(geom, profile, pool0, cfg, null_seed)
        w0.append(w0_rec)
        log(f"  W0 flags={w0_rec['flags'] or '—'} dark={w0_rec['dark']}")
        fp = _sphere_self_fp(geom, profile, pool0, cfg, null_seed)
        fp_rows.append(fp)
        log(f"  FP family={fp['fp_family']}/{FP_WINDOWS} per_type={fp['per_type']}")
        for kind in ("h1", "h2", "h3", "h4", "h5"):
            rec = _w1(geom, kind, profile, pool0, cfg, null_seed, ctx.device)
            plant_rows.append(rec)
            log(f"  W1-{kind} D_N1={rec['flags_n1'] or '—'} D_self={rec['flags_self'] or '—'} "
                f"ok_n1={rec['ok_n1']} ok_self={rec['ok_self']}")
        C._free(pool0)

    w0_pass = all(r["dark"] for r in w0)
    fp_ok = all(r["fp_family"] <= FP_MAX for r in fp_rows)
    by_kind = {}
    for kind in TARGETS:
        rs = [r for r in plant_rows if r["kind"] == kind]
        by_kind[kind] = {"w1_pass": all(r["ok_n1"] and r["ok_self"] for r in rs), "n_geom": len(rs)}
    all_hk = all(v["w1_pass"] for v in by_kind.values())
    reportable = [k for k, v in by_kind.items() if v["w1_pass"]]
    
    
    gate_pass = bool(w0_pass and fp_ok and len(reportable) > 0)

    gate_strength = {k: dict(v) for k, v in FROZEN.items()}
    gate_strength["h5"] = dict(H5_SPEC)
    gate = C.seal({
        "stage": STAGE, "gate_pass": gate_pass,
        "w0_pass": w0_pass, "sphere_self_fp_ok": fp_ok, "all_hk_pass": all_hk,
        "reportable_types": reportable, "by_kind": by_kind,
        "geoms": [g["name"] for g in GEOMS], "fp_holdout_windows": FP_WINDOWS,
        "fp_max_at_3_over_32": FP_MAX,
        "frozen_strength": gate_strength, "sphere": "fixed_20260908_reference_seed_independent",
        "note": "Sphere-plant gate for native (plan 5.1) + sphere self FP (6.1). Fixed sphere "
                "reference; tau not searched; frozen strengths from 20260908 w1_ladder. Types not "
                "in reportable_types report readings only (no Phi) in native.",
        "written": C.now(),
    })
    C.write_json(gate_dir / "GATE.json", gate)
    _write_plant_csv(gate_dir, "fixed", plant_rows, w0)
    _write_fp_csv(gate_dir, fp_rows)
    log(f"[diag_recall] GATE pass={gate_pass} (w0={w0_pass} fp={fp_ok} reportable={reportable})")
    return {"stage": STAGE, "seed": seed, "gate_pass": gate_pass, "w0_pass": w0_pass,
            "sphere_self_fp_ok": fp_ok, "reportable_types": reportable, "by_kind": by_kind,
            "written": C.now()}


def _w0(geom, profile, pool0, cfg, null_seed) -> dict:
    eval_q = n1_queries(C.FAULT_WINDOWS * WINDOW_SIZE, geom["d"], null_seed + 30)
    pack = lamp_pack(eval_against_profile(profile, eval_q, pool0, cfg, null_seed + 40, n_windows=C.FAULT_WINDOWS))
    return {"geom": geom["name"], "flags": pack["flags"], "dark": not pack["any_lamp"], "tau": pack["tau"]}


def _sphere_self_fp(geom, profile, pool0, cfg, null_seed) -> dict:
    per_type = {t: 0 for t in C.TYPE_METRICS}
    fam = 0
    q = n1_queries(FP_WINDOWS * WINDOW_SIZE, geom["d"], null_seed + 50)
    for w in range(FP_WINDOWS):
        sub = q[w * WINDOW_SIZE:(w + 1) * WINDOW_SIZE]
        diag = diagnose_query_windows(profile, sub, pool0, n_windows=1, window_size=WINDOW_SIZE,
                                      seed=null_seed + 60 + w, diagnosis_config=cfg)
        flags = set(diag.get("anomalies") or {})
        lit = False
        for t, (metric, anomaly) in C.TYPE_METRICS.items():
            if anomaly in flags:
                per_type[t] += 1
                lit = True
        if lit:
            fam += 1
    return {"geom": geom["name"], "fp_family": fam, "per_type": per_type, "n_windows": FP_WINDOWS}


def _w1(geom, kind, n1_profile, pool0, cfg_n1, null_seed, device) -> dict:
    t0 = time.perf_counter()
    dim, batch_q = geom["d"], geom["batch_q"]
    spec = dict(H5_SPEC) if kind == "h5" else dict(FROZEN[kind])
    level = 0 if kind == "h5" else int(spec.get("level", 0))
    salt = null_seed + 100 + ord(kind[1]) + 17 * level
    q_tr = n1_queries(N1_TRAIN_WINDOWS * WINDOW_SIZE, dim, salt + 1)
    q_ca = n1_queries(N1_CAL_WINDOWS * WINDOW_SIZE, dim, salt + 2)
    q_ev = n1_queries(C.FAULT_WINDOWS * WINDOW_SIZE, dim, salt + 3)
    if kind == "h5":
        
        
        
        
        q_tr_p = contract_queries(q_tr, q_tr, H5_CONTRACT_STRENGTH)
        q_ca_p = contract_queries(q_ca, q_tr, H5_CONTRACT_STRENGTH)
        q_ev_p = contract_queries(q_ev, q_tr, H5_CONTRACT_STRENGTH)
        pool_p = np.array(pool0, dtype=np.float32, copy=True)
    else:
        q_tr_p, pool_p = plant_world(kind, q_tr, pool0, salt, spec, device=device)
        q_ca_p, _ = plant_world(kind, q_ca, pool0, salt, spec, device=device)
        q_ev_p, _ = plant_world(kind, q_ev, pool0, salt, spec, device=device)
    cfg_p = diag_cfg(device, batch_q, pool_p)
    train_rows = collect_window_readings(q_tr_p, pool_p, n_windows=N1_TRAIN_WINDOWS,
                                         window_size=WINDOW_SIZE, seed=salt + 21, diagnosis_config=cfg_p)
    cal_rows = collect_window_readings(q_ca_p, pool_p, n_windows=N1_CAL_WINDOWS,
                                       window_size=WINDOW_SIZE, seed=salt + 22, diagnosis_config=cfg_p)
    self_profile = fit_profile(train_rows, cal_rows, alpha=ALPHA,
                               split_manifest={"world": "w1", "kind": kind, "geom": geom["name"]},
                               config_snapshot={"h5_primary": "score_interaction_share",
                                                "h5_injector": spec})
    cfg_ev = diag_cfg(device, batch_q, pool_p)
    d_n1 = lamp_pack(eval_against_profile(n1_profile, q_ev_p, pool_p, cfg_ev, salt + 40, n_windows=C.FAULT_WINDOWS))
    d_self = lamp_pack(eval_against_profile(self_profile, q_ev_p, pool_p, cfg_p, salt + 41, n_windows=C.FAULT_WINDOWS))
    target = TARGETS[kind]
    return {
        "geom": geom["name"], "kind": kind, "target": target, "spec": spec,
        "flags_n1": d_n1["flags"], "flags_self": d_self["flags"],
        "obs": d_n1["readings"].get(METRIC[kind]), "z_n1": d_n1["scores"].get(METRIC[kind]),
        "tau": d_n1["tau"], "ok_n1": target in d_n1["flags"], "ok_self": target not in d_self["flags"],
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def _write_plant_csv(gate_dir: Path, seed, plant_rows, w0) -> None:
    with (gate_dir / "sphere_plant.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "world", "geom", "kind", "target", "flags_n1", "flags_self",
                    "obs", "z_n1", "tau", "ok_n1", "ok_self"])
        for r in w0:
            w.writerow([seed, "w0", r["geom"], "", "", ";".join(r["flags"]), ";".join(r["flags"]),
                        "", "", _r(r["tau"]), r["dark"], r["dark"]])
        for r in plant_rows:
            w.writerow([seed, "w1", r["geom"], r["kind"], r["target"], ";".join(r["flags_n1"]),
                        ";".join(r["flags_self"]), _r(r["obs"]), _r(r["z_n1"]), _r(r["tau"]),
                        r["ok_n1"], r["ok_self"]])


def _write_fp_csv(gate_dir: Path, fp_rows) -> None:
    
    out = gate_dir / "false_positive_sphere.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "geom", "n_windows"] +
                   [f"fp_{t}" for t in C.TYPE_METRICS] + ["fp_family", "denominator"])
        for r in fp_rows:
            w.writerow(["sphere_self", r["geom"], r["n_windows"]] +
                       [r["per_type"][t] for t in C.TYPE_METRICS] + [r["fp_family"], "window"])


def _r(x):
    return "" if x is None else round(float(x), 4)
