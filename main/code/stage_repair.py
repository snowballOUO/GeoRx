
from __future__ import annotations

import csv
import gc
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

import common as C
import corrections

STAGE = "repair"
KS = (1, 5, 10)


def _skip_ce() -> bool:
    return os.environ.get("SPHERE_SKIP_CE", "1") != "0"


def _fill_ce() -> bool:
    return os.environ.get("SPHERE_FILL_CE", "0") == "1"


def cell_done(ctx: C.Context, seed: int, cell_id: str) -> bool:
    path = ctx.runs / f"S{seed}" / "repair" / f"{cell_id}.json"
    if not path.is_file():
        return False
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    if not d.get("analytic_complete"):
        return False
    if _fill_ce():
        return _ce_rows_complete(d)
    return True


def _ce_rows_complete(d: dict) -> bool:
    dataset = d.get("dataset")
    by = {r.get("method"): r for r in (d.get("rows") or [])}
    for method in _ce_methods(dataset):
        row = by.get(method) or {}
        if row.get("status") == "ok":
            continue
        if _weights_ready(method, dataset):
            return False
    return True


def _weights_ready(method: str, dataset: str) -> bool:
    root = C.ROOT / "runs" / "ce_training" / "weights"
    if method == "ce_pair_blip_itm":
        return (root / "blip_itm_base_coco" / "config.json").is_file()
    return (root / f"{method}__{dataset}.pt").is_file()


def run(ctx: C.Context, seed: int, pairs, log) -> dict:
    seed_dir = ctx.runs / f"S{seed}"
    out = seed_dir / "repair"
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for encoder, dataset in pairs:
        cid = f"{encoder}__{dataset}"
        if cell_done(ctx, seed, cid):
            n_skip += 1
            continue
        t0 = time.perf_counter()
        try:
            _run_cell(ctx, seed, ctx.cells[cid], out / f"{cid}.json", log)
            n_ok += 1
            log(f"[repair s{seed}] done {cid} {time.perf_counter()-t0:.1f}s")
        except Exception as exc:
            n_fail += 1
            payload = {"cell_id": cid, "seed": seed, "error": repr(exc),
                       "traceback": traceback.format_exc(), "written": C.now()}
            existing = out / f"{cid}.json"
            if _fill_ce() and existing.is_file():
                
                C.write_json(out / f"{cid}.ce_error.json", payload)
            else:
                payload["analytic_complete"] = False
                C.write_json(existing, payload)
            log(f"[repair s{seed}] FAIL {cid} {exc!r}")
        finally:
            gc.collect()
            if C.torch is not None and C.torch.cuda.is_available():
                C.torch.cuda.empty_cache()
    _write_csv(ctx, seed)
    status = {"stage": STAGE, "seed": seed, "n_ok": n_ok, "n_skip": n_skip,
              "n_fail": n_fail, "written": C.now()}
    C.write_json(seed_dir / "repair_status.json", status)
    log(f"[repair s{seed}] ok={n_ok} skip={n_skip} fail={n_fail}")
    return status


def _run_cell(ctx: C.Context, seed: int, cell, out_path: Path, log) -> None:
    if _fill_ce() and out_path.is_file():
        try:
            prev = json.loads(out_path.read_text())
        except Exception:
            prev = None
        if prev and prev.get("analytic_complete") and prev.get("cosine_top100"):
            _fill_ce_only(ctx, seed, cell, out_path, prev, log)
            return
    diag = C.load_sealed_diagnosis(ctx, seed, cell.cell_id)
    seg = C.repartition(cell, seed)
    if diag["split_manifest"].get("q_eval_indices_sha256") != C.hash_idx(seg["eval"]):
        raise RuntimeError(f"{cell.cell_id}: Q_eval hash drifted vs sealed diagnosis")
    labeled = C.load_eval_labeled(cell, seg)
    query, pool, positives, ref_q = (
        labeled["eval_q"], labeled["pool"], labeled["positives"], labeled["ref_q"])
    ctx._eval_idx = labeled["eval_idx"]
    bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
    k_out = max(KS + (C.RERANK_N,))
    params = {
        "h1_k": 10, "csls_k": 10, "qe_k": 10, "qe_alpha": 0.10, "qe_power": 3.0,
        "diffusion_knn": 10, "diffusion_steps": 1, "diffusion_beta": 0.10,
        "diffusion_max_pool": 2_000_000,
        "csls_reference_queries": ref_q,
    }
    cos_v, cos_i, st = corrections.search(
        "cosine", query, pool, k_out, params=params, device=ctx.device, batch_q=bq)
    if st != "ok":
        raise RuntimeError(f"cosine failed: {st}")
    base_hits = {k: C.recall_hits(cos_i, positives, k) for k in KS}
    base = {k: float(base_hits[k].mean()) for k in KS}
    h1_counts = np.bincount(cos_i[:, :10].reshape(-1), minlength=pool.shape[0]).astype(np.float64)
    rows = []
    for method in C.ANALYTIC_METHODS:
        t1 = time.perf_counter()
        _, top_i, status = corrections.search(
            method, query, pool, k_out,
            cosine_top_idx=cos_i, cosine_top_val=cos_v, h1_counts=h1_counts,
            params=params, device=ctx.device, batch_q=bq)
        row = _method_row(cell, seed, method, "analytic", "full_corpus",
                          status, base, base_hits, top_i if status == "ok" else None,
                          positives, diag, labeled["qrels_opened"])
        row["method_wall_s"] = round(time.perf_counter() - t1, 2)
        rows.append(row)
        log(f"[repair s{seed}] {cell.cell_id} {method} {status} "
            f"d10={row.get('dR@10')}")
    if _skip_ce() and not _fill_ce():
        rows.extend(_uneval_ce(cell, seed, m, diag, labeled["qrels_opened"], base)
                    for m in _ce_methods(cell.dataset))
        log(f"[repair s{seed}] {cell.cell_id} CE skipped (analytic pass)")
    else:
        rows.extend(_ce_rows(ctx, cell, seed, query, pool, positives, cos_i, base,
                             base_hits, diag, labeled["qrels_opened"], log))
    useful = {r["method"] for r in rows
              if r.get("status") == "ok" and r.get("useful")}
    phi = list(diag.get("phi") or [])
    P = C.predicted_P(phi, cell.dataset)
    payload = {
        "state": "complete", "analytic_complete": True,
        "cell_id": cell.cell_id, "encoder": cell.encoder, "dataset": cell.dataset,
        "seed": int(seed),
        "diagnosis_seal_sha256": diag["seal_sha256"],
        "q_eval_n": int(query.shape[0]), "n_pool": int(pool.shape[0]),
        "q_eval_indices_sha256": labeled["q_eval_indices_sha256"],
        "qrels_opened": labeled["qrels_opened"],
        "phi": phi, "P": P, "T_all": sorted(useful),
        "baseline": {f"R@{k}": base[k] for k in KS},
        "cosine_top100": cos_i[:, :C.RERANK_N].tolist(),
        "eval_idx": labeled["eval_idx"].tolist(),
        "rows": rows, "written": C.now(),
    }
    C.write_json(out_path, payload)


def _fill_ce_only(ctx, seed, cell, out_path, prev, log) -> None:
    diag = C.load_sealed_diagnosis(ctx, seed, cell.cell_id)
    eval_idx = np.asarray(prev.get("eval_idx"), dtype=np.int64)
    ctx._eval_idx = eval_idx
    cos_i = np.asarray(prev["cosine_top100"], dtype=np.int64)
    base = {k: float(prev["baseline"][f"R@{k}"]) for k in KS}
    positives = _positives_only(cell, eval_idx)
    base_hits = {k: C.recall_hits(cos_i, positives, k) for k in KS}
    kept_analytic = [r for r in (prev.get("rows") or [])
                     if not str(r.get("method", "")).startswith("ce_")]
    already = {r["method"]: r for r in (prev.get("rows") or [])
               if str(r.get("method", "")).startswith("ce_") and r.get("status") == "ok"}
    ce_rows = _ce_rows(ctx, cell, seed, None, None, positives, cos_i, base,
                       base_hits, diag, prev.get("qrels_opened"), log,
                       skip_ok=already)
    rows = kept_analytic + ce_rows
    useful = {r["method"] for r in rows
              if r.get("status") == "ok" and r.get("useful")}
    prev["rows"] = rows
    prev["T_all"] = sorted(useful)
    prev["ce_filled"] = C.now()
    prev["written"] = C.now()
    C.write_json(out_path, prev)


def _positives_only(cell, eval_idx) -> list:
    import h5py
    eval_idx = np.asarray(eval_idx, dtype=np.int64)
    with h5py.File(cell.path, "r") as f:
        qi = np.asarray(f["qrels/query_idx"], dtype=np.int64)
        pi = np.asarray(f["qrels/pool_idx"], dtype=np.int64)
    loc = {int(q): i for i, q in enumerate(eval_idx.tolist())}
    positives = [[] for _ in range(len(eval_idx))]
    for q, p in zip(qi.tolist(), pi.tolist()):
        i = loc.get(int(q))
        if i is not None:
            positives[i].append(int(p))
    return positives


def _method_row(cell, seed, method, provenance, scope, status, base, base_hits,
                top_i, positives, diag, qrels_opened) -> dict:
    row = {
        "seed": int(seed), "encoder": cell.encoder, "dataset": cell.dataset,
        "method": method, "method_provenance": provenance, "scoring_scope": scope,
        "status": status,
        "R@1_base": base[1], "R@5_base": base[5], "R@10_base": base[10],
        "diagnosis_seal_sha256": diag["seal_sha256"],
        "qrels_opened": qrels_opened,
    }
    if status != "ok" or top_i is None:
        row["useful"] = False
        return row
    for k in KS:
        hits = C.recall_hits(top_i, positives, k)
        row[f"R@{k}"] = float(hits.mean())
        row[f"dR@{k}"] = float(hits.mean() - base[k])
    row["useful"] = bool(row["dR@10"] >= C.USEFUL_DR10)
    return row


def _ce_rows(ctx, cell, seed, query, pool, positives, cos_i, base, base_hits,
             diag, qrels_opened, log, skip_ok=None) -> list:
    skip_ok = skip_ok or {}
    rows = []
    try:
        import ce_rerank
    except Exception as exc:
        log(f"[repair s{seed}] {cell.cell_id} CE import skip {exc!r}")
        return [_uneval_ce(cell, seed, m, diag, qrels_opened, base)
                for m in _ce_methods(cell.dataset)]
    for method in _ce_methods(cell.dataset):
        if method in skip_ok:
            rows.append(skip_ok[method])
            log(f"[repair s{seed}] {cell.cell_id} {method} keep-ok")
            continue
        t1 = time.perf_counter()
        try:
            top_i, status, prov = ce_rerank.rerank(
                method, cell.dataset, query, pool, positives, cos_i[:, :C.RERANK_N],
                ctx, seed)
        except Exception as exc:
            status, top_i, prov = f"error:{exc!r}", None, "locally_trained"
        if status != "ok":
            row = _uneval_ce(cell, seed, method, diag, qrels_opened, base)
            row["status"] = status
            row["method_provenance"] = prov
        else:
            row = _method_row(cell, seed, method, prov, "rerank_top100",
                              "ok", base, base_hits, top_i, positives, diag, qrels_opened)
        row["method_wall_s"] = round(time.perf_counter() - t1, 2)
        rows.append(row)
        log(f"[repair s{seed}] {cell.cell_id} {method} {row['status']}")
    return rows


def _ce_methods(dataset: str) -> list:
    if dataset in C.OPEN_CE_LOADS:
        return ["ce_pair_blip_itm", "ce_list_local"]
    return ["ce_pair_local", "ce_list_local"]


def _uneval_ce(cell, seed, method, diag, qrels_opened, base) -> dict:
    return {
        "seed": int(seed), "encoder": cell.encoder, "dataset": cell.dataset,
        "method": method,
        "method_provenance": ("pretrained_literature" if method == "ce_pair_blip_itm"
                              else "locally_trained"),
        "scoring_scope": "rerank_top100",
        "status": "unevaluable_no_weights",
        "R@1_base": base[1], "R@5_base": base[5], "R@10_base": base[10],
        "useful": False,
        "diagnosis_seal_sha256": diag["seal_sha256"],
        "qrels_opened": qrels_opened,
    }


def _write_csv(ctx: C.Context, seed: int) -> None:
    rows = []
    for p in sorted((ctx.runs / f"S{seed}" / "repair").glob("*.json")):
        d = json.loads(p.read_text())
        if not d.get("analytic_complete"):
            continue
        for r in d.get("rows") or []:
            rows.append(r)
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    C.locked_csv(ctx.runs / f"S{seed}" / "repair_delta.csv", rows, fieldnames=fields)
