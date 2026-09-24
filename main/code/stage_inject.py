
from __future__ import annotations

import csv
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np

import common as C
from unified import INJECTORS, pool_axis

STAGE = "inject"
KINDS = ("h1", "h2", "h3", "h4", "h5")


def cell_done(ctx: C.Context, seed: int, cell_id: str) -> bool:
    path = ctx.runs / f"S{seed}" / "inject" / cell_id / "inject.json"
    if not path.is_file():
        return False
    try:
        d = json.loads(path.read_text())
        return d.get("state") == "complete" and all(k in (d.get("by_kind") or {}) for k in KINDS)
    except Exception:
        return False


def run(ctx: C.Context, seed: int, pairs, log) -> dict:
    seed_dir = ctx.runs / f"S{seed}"
    out_root = seed_dir / "inject"
    out_root.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for encoder, dataset in pairs:
        cid = f"{encoder}__{dataset}"
        if cell_done(ctx, seed, cid):
            n_skip += 1
            continue
        t0 = time.perf_counter()
        try:
            _run_cell(ctx, seed, ctx.cells[cid], out_root / cid, log)
            n_ok += 1
            log(f"[inject s{seed}] done {cid} {time.perf_counter()-t0:.1f}s")
        except Exception as exc:
            n_fail += 1
            (out_root / cid).mkdir(parents=True, exist_ok=True)
            C.write_json(out_root / cid / "error.json",
                         {"cell_id": cid, "seed": seed, "error": repr(exc),
                          "traceback": traceback.format_exc(), "written": C.now()})
            log(f"[inject s{seed}] FAIL {cid} {exc!r}")
        finally:
            gc.collect()
            if C.torch is not None and C.torch.cuda.is_available():
                C.torch.cuda.empty_cache()
    _write_csvs(ctx, seed)
    status = {"stage": STAGE, "seed": seed, "n_ok": n_ok, "n_skip": n_skip,
              "n_fail": n_fail, "written": C.now()}
    C.write_json(seed_dir / "inject_status.json", status)
    log(f"[inject s{seed}] ok={n_ok} skip={n_skip} fail={n_fail}")
    return status


def _run_cell(ctx: C.Context, seed: int, cell, cell_dir: Path, log) -> None:
    diag = C.load_sealed_diagnosis(ctx, seed, cell.cell_id)
    seg = C.repartition(cell, seed)
    
    man = diag["split_manifest"]
    got = {
        "q_eval_indices_sha256": C.hash_idx(seg["eval"]),
        "train_indices_sha256": C.hash_idx(seg["train"]),
        "calibration_indices_sha256": C.hash_idx(seg["calibration"]),
        "fault_eval_indices_sha256": C.hash_idx(seg["fault_eval"]),
    }
    for k, v in got.items():
        if man.get(k) != v:
            raise RuntimeError(f"{cell.cell_id}: split hash {k} drifted vs sealed diagnosis")
    inp = C.load_detector_inputs(cell, seg)
    bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
    self_prof, cfg = C.fit_self_profile(
        inp["train_q"], inp["cal_q"], inp["pool"], ctx.device, bq, seed)
    
    clean = {t: bool(diag["types"][t]["self_lamp"]) for t in KINDS}
    by_kind = {}
    axis = pool_axis(inp["pool"], seed + 7)
    for kind in KINDS:
        t1 = time.perf_counter()
        kw = dict(device=ctx.device, batch_q=bq)
        if kind == "h3":
            kw.update(axis=axis, tightness=C.H3_TIGHTNESS, residual=C.H3_RESIDUAL,
                      bridge_k=C.H3_BRIDGE_K, seed=seed + 30)
        if kind == "h5":
            kw.update(h5_mode="nbhd")
        q2, p2 = INJECTORS[kind][1](
            inp["fault_q"], inp["pool"], C.NATIVE_INJECT_STRENGTH, **kw)
        d = C.diagnose_phi(self_prof, q2, p2, cfg, seed)
        lamps = list(d["phi"])
        zs = {t: d["per_type"][t]["z"] for t in KINDS}
        
        ranked = sorted(KINDS, key=lambda t: float(zs[t] if zs[t] is not None else -1e9), reverse=True)
        by_kind[kind] = {
            "target": INJECTORS[kind][0],
            "lamps": lamps,
            "hit": kind in lamps,
            "top1": ranked[0],
            "top1_ok": ranked[0] == kind,
            "z": zs,
            "observed": {t: d["per_type"][t]["observed"] for t in KINDS},
            "wall_s": round(time.perf_counter() - t1, 2),
        }
        del q2, p2
        gc.collect()
        log(f"[inject s{seed}] {cell.cell_id} {kind} hit={kind in lamps} "
            f"top1={ranked[0]} lamps={lamps or '—'}")
    payload = {
        "state": "complete", "stage": STAGE, "seed": int(seed),
        "cell_id": cell.cell_id, "encoder": cell.encoder, "dataset": cell.dataset,
        "diagnosis_seal_sha256": diag["seal_sha256"],
        "clean_self_lamps": [t for t, on in clean.items() if on],
        "clean_family": any(clean.values()),
        "by_kind": by_kind,
        "strength": C.NATIVE_INJECT_STRENGTH,
        "note": "Self-referenced detector on injected (q',C'). Clean lamps copied "
                "from sealed native self_lamps (same fault windows). No qrels.",
        "written": C.now(),
    }
    C.write_json(cell_dir / "inject.json", payload)


def _write_csvs(ctx: C.Context, seed: int) -> None:
    inj = ctx.runs / f"S{seed}" / "inject"
    rows, clean_rows = [], []
    cofire = {k: {j: 0 for j in KINDS} for k in KINDS}
    n_kind = {k: 0 for k in KINDS}
    for p in sorted(inj.glob("*/inject.json")):
        d = json.loads(p.read_text())
        if d.get("state") != "complete":
            continue
        clean = set(d.get("clean_self_lamps") or [])
        clean_rows.append({
            "seed": seed, "encoder": d["encoder"], "dataset": d["dataset"],
            **{f"fp_{t}": int(t in clean) for t in KINDS},
            "fp_family": int(bool(d.get("clean_family"))),
        })
        for kind in KINDS:
            rec = d["by_kind"][kind]
            lamps = set(rec["lamps"] or [])
            n_kind[kind] += 1
            for j in KINDS:
                if j in lamps:
                    cofire[kind][j] += 1
            rows.append({
                "seed": seed, "encoder": d["encoder"], "dataset": d["dataset"],
                "inject": kind, "hit": int(rec["hit"]), "top1_ok": int(rec["top1_ok"]),
                "top1": rec["top1"], "lamps": ";".join(rec["lamps"]) or "empty",
                **{f"z_{t}": rec["z"][t] for t in KINDS},
            })
    if rows:
        C.locked_csv(ctx.runs / f"S{seed}" / "native_inject.csv", rows)
    if clean_rows:
        C.locked_csv(ctx.runs / f"S{seed}" / "false_positive_native.csv", clean_rows)
    
    import fcntl, os as _os
    path = ctx.runs / f"S{seed}" / "cofire_matrix.csv"
    lock = path.with_suffix(".csv.lock")
    fd = _os.open(str(lock), _os.O_CREAT | _os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["inject"] + list(KINDS) + ["n"])
            for k in KINDS:
                n = max(n_kind[k], 1)
                w.writerow([k] + [round(cofire[k][j] / n, 4) for j in KINDS] + [n_kind[k]])
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        _os.close(fd)


def _csv(path: Path, rows: list[dict]) -> None:
    C.locked_csv(path, rows)
