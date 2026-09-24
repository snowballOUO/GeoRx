
from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np

import common as C
import compose as CMP
import corrections

STAGE = "compose"
KS = (1, 5, 10)


def cell_done(ctx: C.Context, seed: int, cell_id: str) -> bool:
    path = ctx.runs / f"S{seed}" / "compose" / f"{cell_id}.json"
    if not path.is_file():
        return False
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return bool(d.get("compose_complete"))


def run(ctx: C.Context, seed: int, pairs, log) -> dict:
    out = ctx.runs / f"S{seed}" / "compose"
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for encoder, dataset in pairs:
        cid = f"{encoder}__{dataset}"
        if cell_done(ctx, seed, cid):
            n_skip += 1
            continue
        t0 = time.perf_counter()
        try:
            skipped = _run_cell(ctx, seed, ctx.cells[cid], out / f"{cid}.json", log)
            if skipped:
                n_skip += 1
                log(f"[compose s{seed}] skip {cid} {skipped}")
            else:
                n_ok += 1
                log(f"[compose s{seed}] done {cid} {time.perf_counter()-t0:.1f}s")
        except Exception as exc:
            n_fail += 1
            C.write_json(out / f"{cid}.error.json", {
                "cell_id": cid, "seed": seed, "error": repr(exc),
                "traceback": traceback.format_exc(), "written": C.now(),
            })
            log(f"[compose s{seed}] FAIL {cid} {exc!r}")
        finally:
            gc.collect()
            if C.torch is not None and C.torch.cuda.is_available():
                C.torch.cuda.empty_cache()
    _write_csv(ctx, seed)
    status = {"stage": STAGE, "seed": seed, "n_ok": n_ok, "n_skip": n_skip,
              "n_fail": n_fail, "written": C.now()}
    C.write_json(ctx.runs / f"S{seed}" / "compose_status.json", status)
    log(f"[compose s{seed}] ok={n_ok} skip={n_skip} fail={n_fail}")
    return status


def _run_cell(ctx: C.Context, seed: int, cell, out_path: Path, log) -> str:
    diag = C.load_sealed_diagnosis(ctx, seed, cell.cell_id)
    phi = list(diag.get("phi") or [])
    P = C.predicted_P(phi, cell.dataset)
    chain = CMP.analytic_chain(P)
    if len(phi) < 2:
        return "phi_lt_2"
    if len(chain) < 2:
        return f"analytic_chain_lt_2:{';'.join(chain) if chain else 'empty'}"

    repair_p = ctx.runs / f"S{seed}" / "repair" / f"{cell.cell_id}.json"
    if not repair_p.is_file():
        raise FileNotFoundError(f"missing analytic repair {repair_p}")
    repair = json.loads(repair_p.read_text())
    if not repair.get("analytic_complete"):
        raise RuntimeError(f"{cell.cell_id}: analytic repair incomplete")

    seg = C.repartition(cell, seed)
    if diag["split_manifest"].get("q_eval_indices_sha256") != C.hash_idx(seg["eval"]):
        raise RuntimeError(f"{cell.cell_id}: Q_eval hash drifted vs sealed diagnosis")
    if repair.get("q_eval_indices_sha256") != C.hash_idx(seg["eval"]):
        raise RuntimeError(f"{cell.cell_id}: Q_eval hash drifted vs repair JSON")

    labeled = C.load_eval_labeled(cell, seg)
    query, pool, positives, ref_q = (
        labeled["eval_q"], labeled["pool"], labeled["positives"], labeled["ref_q"])
    bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
    k_out = max(KS + (C.RERANK_N,))
    params = {
        "h1_k": 10, "csls_k": 10, "qe_k": 10, "qe_alpha": 0.10, "qe_power": 3.0,
        "diffusion_knn": 10, "diffusion_steps": 1, "diffusion_beta": 0.10,
        "diffusion_max_pool": 2_000_000,
        "csls_reference_queries": ref_q,
    }
    t_cos = time.perf_counter()
    cos_v, cos_i, st = corrections.search(
        "cosine", query, pool, k_out, params=params, device=ctx.device, batch_q=bq)
    if st != "ok":
        raise RuntimeError(f"cosine failed: {st}")
    base = {k: float(C.recall_hits(cos_i, positives, k).mean()) for k in KS}
    cosine_wall = round(time.perf_counter() - t_cos, 2)

    t_ch = time.perf_counter()
    packed = CMP.run_chain(
        query, pool, chain, k=k_out, params=params, device=ctx.device,
        batch_q=bq, cosine_top_idx=cos_i, cosine_top_val=cos_v)
    chain_wall = round(time.perf_counter() - t_ch, 2)
    if packed["status"] != "ok":
        raise RuntimeError(f"compose chain failed: {packed['status']}")

    top_i = packed["top_idx"]
    compose_r = {k: float(C.recall_hits(top_i, positives, k).mean()) for k in KS}
    compose_d = {k: compose_r[k] - base[k] for k in KS}

    indep = {r["method"]: r for r in (repair.get("rows") or [])}
    csls_step_d10 = None
    csls_match = None
    if chain[0] == "csls_p050":
        csls_i = packed["steps"][0].get("top_idx")
        if csls_i is None:
            raise RuntimeError("CSLS step missing top_idx")
        csls_r10 = float(C.recall_hits(csls_i, positives, 10).mean())
        csls_step_d10 = csls_r10 - base[10]
        indep_d = (indep.get("csls_p050") or {}).get("dR@10")
        csls_match = (indep_d is not None and abs(csls_step_d10 - float(indep_d)) < 1e-12)

    def _indep_d(m):
        row = indep.get(m) or {}
        if row.get("status") != "ok":
            return None
        return row.get("dR@10")

    d_csls = _indep_d("csls_p050")
    d_qe = _indep_d("alpha_qe_a010")
    d_dif = _indep_d("diffusion")
    members_in_chain = []
    for m in chain:
        v = _indep_d(m)
        if v is not None:
            members_in_chain.append(v)
    best = max(members_in_chain) if members_in_chain else None
    d_comp = compose_d[10]
    delta_vs_best = None if best is None else (d_comp - best)
    skipped_in_P = [m for m in P if m not in chain]

    qe_src = None
    for step in packed["steps"]:
        if step["method"] == "alpha_qe_a010":
            qe_src = step.get("neighbor_source")

    payload = {
        "compose_complete": True,
        "cell_id": cell.cell_id, "encoder": cell.encoder, "dataset": cell.dataset,
        "seed": int(seed),
        "diagnosis_seal_sha256": diag["seal_sha256"],
        "q_eval_indices_sha256": labeled["q_eval_indices_sha256"],
        "repair_json_sha256_ignored": None,
        "phi": phi, "P": P, "chain": chain,
        "neighbor_source_qe": qe_src,
        "skipped_in_P": skipped_in_P,
        "baseline": {f"R@{k}": base[k] for k in KS},
        "compose": {**{f"R@{k}": compose_r[k] for k in KS},
                    **{f"dR@{k}": compose_d[k] for k in KS}},
        "independent": {
            "csls_p050": d_csls, "alpha_qe_a010": d_qe, "diffusion": d_dif,
        },
        "dR@10_best_single": best,
        "delta_vs_best": delta_vs_best,
        "beat_best": bool(delta_vs_best is not None and delta_vs_best >= C.USEFUL_DR10),
        "csls_step_dR@10": csls_step_d10,
        "csls_match_indep": csls_match,
        "baseline_match_repair": abs(base[10] - float(repair["baseline"]["R@10"])) < 1e-12,
        "steps": [
            {k: v for k, v in step.items() if k not in ("top_idx", "top_val")}
            for step in packed["steps"]
        ],
        "cosine_wall_s": cosine_wall, "chain_wall_s": chain_wall,
        "qrels_opened": labeled["qrels_opened"],
        "written": C.now(),
        "notes": (
            "Appendix sequential compose. Does not enter T or channel hit. "
            "CE in P is skipped in this stage."
        ),
    }
    C.write_json(out_path, payload)
    log(f"[compose s{seed}] {cell.cell_id} chain={chain} "
        f"d10={d_comp:.4f} best_single={best} delta_vs_best={delta_vs_best} "
        f"csls_match={csls_match} qe_src={qe_src}")
    return ""


def _write_csv(ctx: C.Context, seed: int) -> None:
    rows = []
    for p in sorted((ctx.runs / f"S{seed}" / "compose").glob("*.json")):
        if p.name.endswith(".error.json"):
            continue
        d = json.loads(p.read_text())
        if not d.get("compose_complete"):
            continue
        comp = d.get("compose") or {}
        indep = d.get("independent") or {}
        rows.append({
            "seed": d["seed"], "encoder": d["encoder"], "dataset": d["dataset"],
            "phi_set": ";".join(d.get("phi") or []) or "empty",
            "P": ";".join(d.get("P") or []) or "empty",
            "chain": ">".join(d.get("chain") or []),
            "neighbor_source_qe": d.get("neighbor_source_qe"),
            "skipped_in_P": ";".join(d.get("skipped_in_P") or []) or "empty",
            "R@10_base": (d.get("baseline") or {}).get("R@10"),
            "R@10_compose": comp.get("R@10"),
            "dR@10_compose": comp.get("dR@10"),
            "dR@10_csls_indep": indep.get("csls_p050"),
            "dR@10_qe_indep": indep.get("alpha_qe_a010"),
            "dR@10_diff_indep": indep.get("diffusion"),
            "dR@10_best_single": d.get("dR@10_best_single"),
            "delta_vs_best": d.get("delta_vs_best"),
            "beat_best": int(bool(d.get("beat_best"))),
            "csls_match_indep": d.get("csls_match_indep"),
            "baseline_match_repair": d.get("baseline_match_repair"),
            "csls_step_dR@10": d.get("csls_step_dR@10"),
            "cosine_wall_s": d.get("cosine_wall_s"),
            "chain_wall_s": d.get("chain_wall_s"),
        })
    fields = [
        "seed", "encoder", "dataset", "phi_set", "P", "chain",
        "neighbor_source_qe", "skipped_in_P",
        "R@10_base", "R@10_compose", "dR@10_compose",
        "dR@10_csls_indep", "dR@10_qe_indep", "dR@10_diff_indep",
        "dR@10_best_single", "delta_vs_best", "beat_best",
        "csls_match_indep", "baseline_match_repair", "csls_step_dR@10",
        "cosine_wall_s", "chain_wall_s",
    ]
    if rows:
        C.locked_csv(ctx.runs / f"S{seed}" / "compose_delta.csv", rows, fieldnames=fields)
