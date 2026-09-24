

from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

import common as C
from unified import INJECTORS, pool_axis

STAGE = "multi_inject"
KINDS = ("h1", "h2", "h3", "h4", "h5")
ORDER = KINDS  
STRENGTH = C.NATIVE_INJECT_STRENGTH
OUT_ROOT_NAME = "multi_inject"


def all_multi_combos():
    out = []
    for k in range(2, 6):
        out.extend(itertools.combinations(KINDS, k))
    return out  


COMBOS = all_multi_combos()
N_COMBOS = len(COMBOS)  
N_TARGET = 90 * 3 * N_COMBOS  


def combo_key(types) -> str:
    return ";".join(types)


def combo_path(ctx: C.Context, seed: int, cell_id: str, key: str) -> Path:
    safe = key.replace(";", "_")
    return ctx.runs / OUT_ROOT_NAME / f"S{seed}" / cell_id / f"{safe}.json"


def combo_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("state") == "complete"
    except Exception:
        return False


def _parse_combo(s: str):
    parts = tuple(t.strip() for t in s.replace(",", ";").split(";") if t.strip())
    unknown = [t for t in parts if t not in KINDS]
    if unknown:
        raise SystemExit(f"unknown types {unknown}")
    ordered = tuple(t for t in KINDS if t in parts)
    if len(ordered) < 2:
        raise SystemExit(f"need |S|>=2, got {s}")
    return ordered


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.SEEDS))
    ap.add_argument("--encoders", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--cells", nargs="+")
    ap.add_argument("--combos", nargs="+",
                    help="subset of S keys, e.g. h1;h2 h1;h2;h4 (default: all 26)")
    ap.add_argument("--batch-q", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)
    log = _logger(ctx.logs / "multi_inject_workers.log")
    if args.cells:
        pairs = []
        for cid in args.cells:
            if cid not in ctx.cells:
                raise SystemExit(f"unknown cell {cid}")
            e, d = cid.split("__", 1)
            pairs.append((e, d))
    else:
        pairs = C.all_pairs(args.encoders, args.datasets)
    if args.combos:
        combos = [_parse_combo(s) for s in args.combos]
    else:
        combos = list(COMBOS)

    log(f"=== multi_inject start seeds={args.seeds} pairs={len(pairs)} "
        f"combos={len(combos)} device={args.device} batch_q={args.batch_q} "
        f"(3 seeds = independent splits, not a vote; 90×3×26={N_TARGET}) ===")
    overall = {"started": C.now(), "n_ok": 0, "n_skip": 0, "n_fail": 0}
    t_all = time.perf_counter()
    for seed in args.seeds:
        for encoder, dataset in pairs:
            cid = f"{encoder}__{dataset}"
            try:
                st = _run_cell(ctx, seed, ctx.cells[cid], combos, log)
                overall["n_ok"] += st["n_ok"]
                overall["n_skip"] += st["n_skip"]
                overall["n_fail"] += st["n_fail"]
            except Exception as exc:
                overall["n_fail"] += len(combos)
                err_dir = ctx.runs / OUT_ROOT_NAME / f"S{seed}" / cid
                err_dir.mkdir(parents=True, exist_ok=True)
                C.write_json(err_dir / "cell_error.json", {
                    "cell_id": cid, "seed": seed, "error": repr(exc),
                    "traceback": traceback.format_exc(), "written": C.now(),
                })
                log(f"[multi s{seed}] FAIL cell {cid} {exc!r}")
            finally:
                gc.collect()
                if C.torch is not None and C.torch.cuda.is_available():
                    C.torch.cuda.empty_cache()
        _rewrite_csv(ctx, seed)
    _rewrite_all_csv(ctx)
    _write_progress(ctx)
    overall["finished"] = C.now()
    overall["wall_s"] = round(time.perf_counter() - t_all, 1)
    log(f"=== multi_inject done {overall['wall_s']}s ok={overall['n_ok']} "
        f"skip={overall['n_skip']} fail={overall['n_fail']} ===")
    return 0 if overall["n_fail"] == 0 else 1


def _run_cell(ctx: C.Context, seed: int, cell, combos, log) -> dict:
    todo = []
    for types in combos:
        key = combo_key(types)
        path = combo_path(ctx, seed, cell.cell_id, key)
        if combo_done(path):
            continue
        todo.append((types, key, path))
    n_skip = len(combos) - len(todo)
    if not todo:
        return {"n_ok": 0, "n_skip": n_skip, "n_fail": 0}

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
    axis = pool_axis(inp["pool"], seed + 7)
    single = _load_single_inject(ctx, seed, cell.cell_id)

    n_ok = n_fail = 0
    log(f"[multi s{seed}] {cell.cell_id} todo={len(todo)}/{len(combos)} "
        f"n_pool={cell.n_pool} dim={cell.dim} bq={bq}")
    for types, key, path in todo:
        t0 = time.perf_counter()
        try:
            rec = _run_combo(
                cell, seed, types, key, inp, self_prof, cfg, axis, bq,
                ctx.device, single, diag)
            rec["wall_s"] = round(time.perf_counter() - t0, 2)
            rec["written"] = C.now()
            C.write_json(path, rec)
            n_ok += 1
            log(f"[multi s{seed}] {cell.cell_id} {key} phi={rec['phi_set']} "
                f"set_recall={rec['set_recall']} exact={rec['exact_match']} "
                f"{rec['wall_s']}s")
        except Exception as exc:
            n_fail += 1
            C.write_json(path.with_name(path.stem + ".error.json"), {
                "cell_id": cell.cell_id, "seed": seed, "S": key,
                "error": repr(exc), "traceback": traceback.format_exc(),
                "written": C.now(),
            })
            log(f"[multi s{seed}] FAIL {cell.cell_id} {key} {exc!r}")
        finally:
            gc.collect()
            if C.torch is not None and C.torch.cuda.is_available():
                C.torch.cuda.empty_cache()
    return {"n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail}


def _run_combo(cell, seed, types, key, inp, self_prof, cfg, axis, bq, device, single, diag):
    q, p = inp["fault_q"], inp["pool"]
    applied = []
    for kind in ORDER:
        if kind not in types:
            continue
        kw = dict(device=device, batch_q=bq)
        if kind == "h3":
            kw.update(axis=axis, tightness=C.H3_TIGHTNESS, residual=C.H3_RESIDUAL,
                      bridge_k=C.H3_BRIDGE_K, seed=seed + 30)
        if kind == "h5":
            kw.update(h5_mode="nbhd")
        q, p = INJECTORS[kind][1](q, p, STRENGTH, **kw)
        applied.append(kind)
    d = C.diagnose_phi(self_prof, q, p, cfg, seed)
    lamps = list(d["phi"] or [])
    lamp_set = set(lamps)
    s_set = set(types)
    zs = {t: d["per_type"][t]["z"] for t in KINDS}
    ranked = sorted(
        KINDS,
        key=lambda t: float(zs[t] if zs[t] is not None else -1e9),
        reverse=True,
    )
    union = _union_phi(single, types)
    union_set = set(union)
    hits = {t: int(t in lamp_set) for t in KINDS}
    extras = {t: int(t in lamp_set and t not in s_set) for t in KINDS}
    inter = lamp_set & s_set
    rec = {
        "state": "complete", "stage": STAGE, "seed": int(seed),
        "cell_id": cell.cell_id, "encoder": cell.encoder, "dataset": cell.dataset,
        "diagnosis_seal_sha256": diag["seal_sha256"],
        "S": key, "order": ">".join(applied), "strength": STRENGTH,
        "phi": lamps, "phi_set": ";".join(lamps) if lamps else "empty",
        "set_recall": round(len(inter) / max(len(s_set), 1), 4),
        "exact_match": int(lamp_set == s_set),
        "top1": ranked[0], "top1_in_S": int(ranked[0] in s_set),
        "z": zs,
        "observed": {t: d["per_type"][t]["observed"] for t in KINDS},
        "union_phi": ";".join(t for t in KINDS if t in union_set) or "empty",
        "union_exact": int(lamp_set == union_set) if union else None,
        "multi_minus_union": ";".join(t for t in KINDS if t in lamp_set - union_set) or "",
        "union_minus_multi": ";".join(t for t in KINDS if t in union_set - lamp_set) or "",
        "single_inject_missing": int(not single),
        "note": "Real embedding space. Self detector. No qrels.",
    }
    for t in KINDS:
        rec[f"hit_{t}"] = hits[t]
        rec[f"extra_{t}"] = extras[t]
    del q, p
    return rec


def _load_single_inject(ctx: C.Context, seed: int, cell_id: str):
    path = ctx.runs / f"S{seed}" / "inject" / cell_id / "inject.json"
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if d.get("state") != "complete":
        return None
    return d.get("by_kind") or {}


def _union_phi(single, types):
    if not single:
        return []
    seen = set()
    for k in types:
        for t in (single.get(k) or {}).get("lamps") or []:
            seen.add(t)
    return [t for t in KINDS if t in seen]


def _csv_row(d: dict) -> dict:
    return {
        "seed": d["seed"], "encoder": d["encoder"], "dataset": d["dataset"],
        "S": d["S"], "order": d.get("order"), "strength": d.get("strength"),
        "phi_set": d.get("phi_set"), "set_recall": d.get("set_recall"),
        "exact_match": d.get("exact_match"), "top1": d.get("top1"),
        "top1_in_S": d.get("top1_in_S"),
        **{f"hit_{t}": d.get(f"hit_{t}") for t in KINDS},
        **{f"extra_{t}": d.get(f"extra_{t}") for t in KINDS},
        **{f"z_{t}": (d.get("z") or {}).get(t) for t in KINDS},
        "union_phi": d.get("union_phi"), "union_exact": d.get("union_exact"),
        "multi_minus_union": d.get("multi_minus_union"),
        "union_minus_multi": d.get("union_minus_multi"),
        "wall_s": d.get("wall_s"),
    }


def _iter_complete(ctx: C.Context, seed=None):
    root = ctx.runs / OUT_ROOT_NAME
    pattern = f"S{seed}/*/*.json" if seed is not None else "S*/*/*.json"
    for p in sorted(root.glob(pattern)):
        if p.name.endswith("error.json") or p.name == "cell_error.json":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("state") == "complete" and d.get("S"):
            yield d


def _rewrite_csv(ctx: C.Context, seed: int) -> None:
    rows = [_csv_row(d) for d in _iter_complete(ctx, seed)]
    if rows:
        C.locked_csv(ctx.runs / OUT_ROOT_NAME / f"S{seed}_native_multi.csv", rows)


def _rewrite_all_csv(ctx: C.Context) -> None:
    rows = [_csv_row(d) for d in _iter_complete(ctx, None)]
    if rows:
        C.locked_csv(ctx.runs / OUT_ROOT_NAME / "all.csv", rows)


def _write_progress(ctx: C.Context) -> None:
    n = sum(1 for _ in _iter_complete(ctx, None))
    C.write_json(ctx.runs / OUT_ROOT_NAME / "progress.json", {
        "n_complete": n, "n_target": N_TARGET, "frac": round(n / N_TARGET, 4),
        "note": "90 cells × 3 seeds × 26 multi-subsets. 3 = seeds {1,2,3}.",
        "written": C.now(),
    })


def _logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{C.now()} {msg}"
        print(line, flush=True)
        with path.open("a") as fh:
            fh.write(line + "\n")

    return log


if __name__ == "__main__":
    raise SystemExit(main())
