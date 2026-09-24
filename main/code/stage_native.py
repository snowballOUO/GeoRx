
from __future__ import annotations

import csv
import gc
import time
import traceback
from pathlib import Path

import common as C

STAGE = "native"


def cell_done(ctx: C.Context, seed: int, cell_id: str) -> bool:
    path = ctx.runs / f"S{seed}" / "evaluations" / cell_id / "diagnosis.json"
    if not path.is_file():
        return False
    try:
        import json
        return C.verify_seal(json.loads(path.read_text()))
    except Exception:
        return False


def run(ctx: C.Context, seed: int, pairs, log) -> dict:
    seed_dir = ctx.runs / f"S{seed}"
    
    
    if not C.gate_passed(ctx):
        log(f"[native s{seed}] BLOCKED: sphere gate not passed (run diag_recall first).")
        status = {"stage": STAGE, "seed": seed, "state": "blocked_by_gate", "written": C.now()}
        C.write_json(seed_dir / "native_status.json", status)
        return status
    ev_dir = seed_dir / "evaluations"
    ev_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for encoder, dataset in pairs:
        cid = f"{encoder}__{dataset}"
        if cell_done(ctx, seed, cid):
            n_skip += 1
            continue
        cell = ctx.cells[cid]
        t0 = time.perf_counter()
        try:
            _run_cell(ctx, seed, cell, ev_dir / cid, log)
            n_ok += 1
            log(f"[native s{seed}] done {cid} {time.perf_counter()-t0:.1f}s")
        except Exception as exc:
            n_fail += 1
            (ev_dir / cid).mkdir(parents=True, exist_ok=True)
            C.write_json(ev_dir / cid / "error.json",
                         {"cell_id": cid, "seed": seed, "error": repr(exc),
                          "traceback": traceback.format_exc(), "written": C.now()})
            log(f"[native s{seed}] FAIL {cid} {exc!r}")
        finally:
            gc.collect()
            if C.torch is not None and C.torch.cuda.is_available():
                C.torch.cuda.empty_cache()
    _write_csv(ctx, seed, pairs)
    status = {"stage": STAGE, "seed": seed, "n_ok": n_ok, "n_skip": n_skip,
              "n_fail": n_fail, "written": C.now()}
    C.write_json(seed_dir / "native_status.json", status)
    log(f"[native s{seed}] ok={n_ok} skip={n_skip} fail={n_fail}")
    return status


def _run_cell(ctx: C.Context, seed: int, cell, cell_dir: Path, log) -> None:
    seg = C.repartition(cell, seed)
    inp = C.load_detector_inputs(cell, seg)
    self_prof, cfg = C.fit_self_profile(inp["train_q"], inp["cal_q"], inp["pool"],
                                        ctx.device, C._batch_for(cell.dim, cell.n_pool, ctx.batch_q), seed)
    n1_prof = C.get_n1_profile(int(cell.dim), int(cell.n_pool), ctx)
    d_n1 = C.diagnose_phi(n1_prof, inp["fault_q"], inp["pool"], cfg, seed)
    d_self = C.diagnose_phi(self_prof, inp["fault_q"], inp["pool"], cfg, seed)
    types = {}
    for typ, (metric, anomaly) in C.TYPE_METRICS.items():
        n1t, st = d_n1["per_type"][typ], d_self["per_type"][typ]
        types[typ] = {
            "metric": metric, "anomaly": anomaly,
            "observed": n1t["observed"],
            "n1_mu": n1_prof["metrics"][metric]["center"],
            "n1_sigma": n1_prof["metrics"][metric]["scale"],
            "z_n1": n1t["z"], "tau_n1": n1t["tau"], "phi_lamp": n1t["lamp"],
            "z_self": st["z"], "tau_self": st["tau"], "self_lamp": st["lamp"],
        }
    payload = {
        "cell_id": cell.cell_id, "encoder": cell.encoder, "dataset": cell.dataset,
        "seed": int(seed), "dim": int(cell.dim), "n_pool": int(cell.n_pool),
        "n_test": int(cell.n_test), "n_valid": inp["n_valid"],
        "q_diag_n": C.Q_DIAG_N, "q_eval_n": C.Q_EVAL_N,
        "tau_sphere": d_n1["tau"], "tau_self": d_self["tau"],
        "phi": d_n1["phi"], "self_lamps": d_self["phi"],
        "types": types,
        "split_manifest": inp["manifest"],
        "note": "Phi = sphere-referenced lamps on Q_diag fault (H3 needs observed "
                "second_frac>=0.15). self_lamps are the control column, not Phi. "
                "Q_eval is disjoint and untouched here; repair uses it. No qrels read.",
        "written": C.now(),
    }
    C.write_json(cell_dir / "diagnosis.json", C.seal(payload))


def _write_csv(ctx: C.Context, seed: int, pairs) -> None:
    import json
    ev_dir = ctx.runs / f"S{seed}" / "evaluations"
    cols = ["seed", "encoder", "dataset", "dim", "n_pool", "n_test", "n_valid",
            "q_diag_n", "q_eval_n", "tau_sphere"] +\
        [f"phi_{t}" for t in C.TYPE_METRICS] + ["phi_set"] +\
        [f"self_{t}" for t in C.TYPE_METRICS] +\
        [f"z_{t}" for t in C.TYPE_METRICS] + ["obs_h3_second_frac", "seal_sha256"]
    out = ctx.runs / f"S{seed}" / "diagnosis_sets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for encoder, dataset in pairs:
            cid = f"{encoder}__{dataset}"
            p = ev_dir / cid / "diagnosis.json"
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            phi = set(d.get("phi") or [])
            row = [seed, encoder, dataset, d["dim"], d["n_pool"], d["n_test"],
                   d.get("n_valid"), d.get("q_diag_n"), d.get("q_eval_n"),
                   _r(d["tau_sphere"])]
            row += [int(t in phi) for t in C.TYPE_METRICS]
            row += [";".join(d["phi"]) if d["phi"] else "empty"]
            selfl = set(d.get("self_lamps") or [])
            row += [int(t in selfl) for t in C.TYPE_METRICS]
            row += [_r(d["types"][t]["z_n1"]) for t in C.TYPE_METRICS]
            row += [_r(d["types"]["h3"]["observed"]), d.get("seal_sha256", "")]
            w.writerow(row)


def _r(x):
    return "" if x is None else round(float(x), 4)
