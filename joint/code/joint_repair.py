

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CONFIG_PATH = Path(__file__).resolve().parent / "joint_config.json"
with CONFIG_PATH.open(encoding="utf-8") as fh:
    JCFG = json.load(fh)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = Path(os.environ.get("JOINT_BASE_ROOT", str(PACKAGE_ROOT / "main")))
BASE_CODE = BASE_ROOT / "code"
if str(BASE_CODE) not in sys.path:
    sys.path.insert(0, str(BASE_CODE))

import common as C  
import corrections  
import ce_rerank  


KS = (1, 5, 10, 100)
RERANK_N = int(JCFG["default"]["ce_top_n"])
FOUR_METHODS = ("csls_p050", "alpha_qe_a010", "ce_list_local", "ce_pair")
FOUR_TARGET = frozenset(("h1", "h2", "h4", "h5"))
H2H5_TARGET = frozenset(("h2", "h5"))


def now() -> str:
    return C.now()


def sha256_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def log_line(log_path: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def method_names(dataset: str) -> Tuple[str, str]:
    point = "ce_pair_blip_itm" if dataset in C.OPEN_CE_LOADS else "ce_pair_local"
    return point, "ce_list_local"


def params_for(ref_q: np.ndarray) -> Dict:
    d = JCFG["default"]
    return {
        "h1_k": int(d["csls_k"]),
        "csls_k": int(d["csls_k"]),
        "hub_lambda": float(d["csls_lambda"]),
        "qe_k": int(d["qe_k"]),
        "qe_alpha": float(d["qe_alpha"]),
        "qe_power": float(d["qe_power"]),
        "diffusion_knn": 10,
        "diffusion_steps": 1,
        "diffusion_beta": 0.10,
        "diffusion_max_pool": 2_000_000,
        "csls_reference_queries": ref_q,
    }


def recall_row(top_idx: np.ndarray, positives: Sequence[Sequence[int]], base: Mapping[int, float]) -> Dict:
    row = {}
    for k in KS:
        value = float(C.recall_hits(top_idx[:, :k], positives, k).mean())
        row[f"R@{k}"] = value
        row[f"dR@{k}"] = value - float(base[k])
    row["useful"] = bool(row["dR@10"] >= C.USEFUL_DR10)
    return row


def restrict_to_cosine(method_top: np.ndarray, cosine_top: np.ndarray) -> np.ndarray:
    out = np.empty_like(cosine_top)
    width = cosine_top.shape[1]
    for q in range(cosine_top.shape[0]):
        allowed = set(int(x) for x in cosine_top[q].tolist())
        selected = []
        for x in method_top[q].tolist():
            x = int(x)
            if x in allowed and x not in selected:
                selected.append(x)
        selected.extend(int(x) for x in cosine_top[q].tolist() if int(x) not in selected)
        out[q] = np.asarray(selected[:width], dtype=np.int64)
    return out


def fuse_rankings(rankings: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights have different lengths")
    if not rankings:
        raise ValueError("empty ranking list")
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("weights must be non-negative and non-zero")
    weights /= weights.sum()
    base = rankings[0]
    nq, width = base.shape
    fused = np.zeros((nq, width), dtype=np.float32)
    for ranking, weight in zip(rankings, weights.tolist()):
        if ranking.shape != base.shape:
            raise ValueError("all rankings must have the same shape")
        for q in range(nq):
            pos = {int(x): i for i, x in enumerate(ranking[q].tolist())}
            fused[q] += float(weight) * np.asarray(
                [1.0 - pos.get(int(x), width) / max(width - 1, 1) for x in base[q]],
                dtype=np.float32,
            )
    out = np.empty_like(base)
    for q in range(nq):
        
        
        order = np.lexsort((base[q], -fused[q]))
        out[q] = base[q, order]
    return out


def ce_rerank_with_root(method: str, dataset: str, top_idx: np.ndarray,
                        query: np.ndarray, pool: np.ndarray,
                        positives: Sequence[Sequence[int]], ctx, seed: int):
    pair_root = Path(os.environ.get("JOINT_PAIR_WEIGHTS_ROOT", JCFG["pair_weights_root"]))
    list_root = Path(os.environ.get("JOINT_LISTWISE_WEIGHTS_ROOT", JCFG["listwise_weights_root"]))
    old = ce_rerank.WEIGHTS
    list_train_module = None
    old_list_train_weights = None
    if method == "ce_list_local":
        
        
        
        
        
        import importlib
        list_train_module = importlib.import_module("ce_list_train")
        old_list_train_weights = list_train_module.WEIGHTS
        list_train_module.WEIGHTS = list_root
    ce_rerank.WEIGHTS = pair_root if method.startswith("ce_pair") else list_root
    try:
        return ce_rerank.rerank(method, dataset, query, pool, positives, top_idx, ctx, seed)
    finally:
        ce_rerank.WEIGHTS = old
        if list_train_module is not None:
            list_train_module.WEIGHTS = old_list_train_weights


def single_methods(methods: Sequence[str], query, pool, positives, cosine_top,
                   params, ctx, cell, seed, base, log_path) -> Tuple[Dict, Dict]:
    rows: Dict[str, Dict] = {}
    tops: Dict[str, np.ndarray] = {}
    for method in methods:
        t0 = time.perf_counter()
        if method in ("csls_p050", "alpha_qe_a010"):
            _, top, status = corrections.search(
                method, query, pool, RERANK_N, cosine_top_idx=cosine_top,
                params=params, device=ctx.device, batch_q=ctx.batch_q,
            )
            provenance, scope = "analytic", "full_corpus"
        else:
            top, status, provenance = ce_rerank_with_root(
                method, cell.dataset, cosine_top, query, pool, positives, ctx, seed
            )
            scope = "rerank_top100"
        row = {"method": method, "status": status, "method_provenance": provenance,
               "scoring_scope": scope, "wall_s": round(time.perf_counter() - t0, 2)}
        if status == "ok" and top is not None:
            tops[method] = np.asarray(top, dtype=np.int64)
            row.update(recall_row(tops[method], positives, base))
        else:
            row.update({f"R@{k}": None for k in KS})
            row.update({f"dR@{k}": None for k in KS})
            row["useful"] = False
        rows[method] = row
        log_line(log_path, f"single {cell.cell_id} S{seed} {method} {status} "
                 f"dR10={row.get('dR@10')}")
    return rows, tops


def sequence_result(order: Sequence[str], rankings: Mapping[str, np.ndarray],
                    gamma: float, positives, base) -> Tuple[np.ndarray, Dict]:
    if not order:
        raise ValueError("empty sequence")
    current = rankings[order[0]]
    steps = [{"method": order[0], "status": "cached_single"}]
    for method in order[1:]:
        current = fuse_rankings([current, rankings[method]], [1.0 - gamma, gamma])
        steps.append({"method": method, "status": "cached_single", "gamma": float(gamma)})
    row = {"method": "sequence:" + ">".join(order),
           "status": "ok", "scoring_scope": "shared_cosine_top100_rank_fusion",
           "gamma": float(gamma), "steps": steps}
    row.update(recall_row(current, positives, base))
    return current, row


def weights_four() -> Dict[str, List[float]]:
    return {
        "equal": [0.25, 0.25, 0.25, 0.25],
        "csls_heavy": [0.5, 1 / 6, 1 / 6, 1 / 6],
        "qe_heavy": [1 / 6, 0.5, 1 / 6, 1 / 6],
        "listwise_heavy": [1 / 6, 1 / 6, 0.5, 1 / 6],
        "pointwise_heavy": [1 / 6, 1 / 6, 1 / 6, 0.5],
    }


def weights_two() -> Dict[str, List[float]]:
    return {"qe025": [0.75, 0.25], "equal": [0.5, 0.5], "ce075": [0.25, 0.75]}


def evaluate_cell(ctx, output_root: Path, group: str, cell_id: str, seed: int,
                  log_path: Path) -> Path:
    out_dir = output_root / "runs" / f"S{seed}" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cell_id}.json"
    if out_path.is_file():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
            if previous.get("state") == "complete":
                return out_path
        except Exception:
            pass
    encoder, dataset = cell_id.split("__", 1)
    cell = ctx.cells[cell_id]
    diag = C.load_sealed_diagnosis(ctx, seed, cell_id)
    phi = frozenset(diag.get("phi") or [])
    target = FOUR_TARGET if group == "four" else H2H5_TARGET
    if phi != target:
        raise RuntimeError(f"{cell_id} S{seed}: phi={sorted(phi)} != {sorted(target)}")
    seg = C.repartition(cell, seed)
    if diag["split_manifest"].get("q_eval_indices_sha256") != C.hash_idx(seg["eval"]):
        raise RuntimeError(f"{cell_id} S{seed}: Q_eval hash drift")
    labeled = C.load_eval_labeled(cell, seg)
    query, pool, positives, ref_q = labeled["eval_q"], labeled["pool"], labeled["positives"], labeled["ref_q"]
    ctx._eval_idx = labeled["eval_idx"]
    params = params_for(ref_q)
    t0 = time.perf_counter()
    cos_val, cos_idx, status = corrections.search(
        "cosine", query, pool, RERANK_N, params=params, device=ctx.device, batch_q=ctx.batch_q)
    if status != "ok":
        raise RuntimeError(f"cosine failed: {status}")
    base = {k: float(C.recall_hits(cos_idx[:, :k], positives, k).mean()) for k in KS}
    base_row = {"method": "cosine", "status": "ok", "scoring_scope": "full_corpus",
                "useful": False}
    for k in KS:
        base_row[f"R@{k}"] = base[k]
        base_row[f"dR@{k}"] = 0.0
    rows = [base_row]
    if group == "four":
        single_list = ["csls_p050", "alpha_qe_a010", "ce_list_local", method_names(dataset)[0]]
    else:
        single_list = ["alpha_qe_a010", method_names(dataset)[0]]
    singles, tops = single_methods(single_list, query, pool, positives, cos_idx,
                                   params, ctx, cell, seed, base, log_path)
    for method, row in singles.items():
        row = dict(row); row["method"] = method; rows.append(row)

    if group == "four":
        point = method_names(dataset)[0]
        parallel_methods = ["csls_p050", "alpha_qe_a010", "ce_list_local", point]
        rankings = {}
        for m in parallel_methods:
            if m not in tops:
                raise RuntimeError(f"missing single-method result for {m} on {cell_id} S{seed}")
            rankings[m] = restrict_to_cosine(tops[m], cos_idx)
        for tag, w in weights_four().items():
            top = fuse_rankings([rankings[m] for m in parallel_methods], w)
            r = recall_row(top, positives, base)
            rows.append({"method": f"joint_parallel:{tag}", "status": "ok",
                         "scoring_scope": "shared_cosine_top100", "weights": dict(zip(parallel_methods, w)), **r})
        for analytic_order in (("csls_p050", "alpha_qe_a010"),
                               ("alpha_qe_a010", "csls_p050")):
            for ce_order in (("ce_list_local", point), (point, "ce_list_local")):
                
                
                for gamma in (0.25, float(JCFG["default"]["gamma"]), 0.75, 1.0):
                    _, r = sequence_result(tuple(analytic_order) + tuple(ce_order),
                                           rankings, gamma, positives, base)
                    rows.append(r)
    else:
        point = method_names(dataset)[0]
        rankings = {
            "alpha_qe_a010": restrict_to_cosine(tops["alpha_qe_a010"], cos_idx),
            point: restrict_to_cosine(tops[point], cos_idx),
        }
        qtop = rankings["alpha_qe_a010"]
        ctop = rankings[point]
        for tag, w in weights_two().items():
            top = fuse_rankings([qtop, ctop], w)
            r = recall_row(top, positives, base)
            rows.append({"method": f"joint_parallel:{tag}", "status": "ok",
                         "scoring_scope": "shared_cosine_top100",
                         "weights": {"alpha_qe_a010": w[0], point: w[1]}, **r})
        for order in (("alpha_qe_a010", point), (point, "alpha_qe_a010")):
            for gamma in (0.25, float(JCFG["default"]["gamma"]), 0.75, 1.0):
                _, r = sequence_result(order, rankings, gamma, positives, base)
                rows.append(r)
    payload = {
        "state": "complete", "group": group, "cell_id": cell_id,
        "encoder": encoder, "dataset": dataset, "seed": int(seed),
        "phi": sorted(phi), "target_phi": sorted(target),
        "diagnosis_seal_sha256": diag["seal_sha256"],
        "q_eval_indices_sha256": labeled["q_eval_indices_sha256"],
        "q_eval_n": int(query.shape[0]), "n_pool": int(pool.shape[0]),
        "baseline": {f"R@{k}": base[k] for k in KS},
        "rows": rows, "wall_s": round(time.perf_counter() - t0, 2),
        "config_sha256": sha256_json(JCFG), "written": now(),
    }
    C.write_json(out_path, payload)
    log_line(log_path, f"complete {group} {cell_id} S{seed} wall={payload['wall_s']}s")
    return out_path


def write_csv(output_root: Path, group: str) -> Path:
    records = []
    for path in sorted((output_root / "runs").glob(f"S*/{group}/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("state") != "complete":
            continue
        for row in payload.get("rows", []):
            records.append({"group": group, "cell_id": payload["cell_id"],
                            "seed": payload["seed"], "phi": ";".join(payload["phi"]),
                            "n_pool": payload["n_pool"], **row})
    out = output_root / "runs" / f"{group}_joint_delta.csv"
    if records:
        fields = sorted({k for r in records for k in r})
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    return out


def selected(group: str) -> Dict[str, List[int]]:
    if group == "four":
        return {cell: [1, 2, 3] for cell in JCFG["four_cells"]}
    if group == "h2h5":
        return {k: [int(x) for x in v] for k, v in JCFG["h2h5_seeds"].items()}
    raise ValueError(group)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=("four", "h2h5"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-q", type=int, default=32)
    ap.add_argument("--output-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "joint" / "runs")
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args(argv)
    if args.device.startswith("cpu") and C.torch is not None:
        C.torch.set_num_threads(int(os.environ.get("JOINT_CPU_THREADS", "40")))
    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)
    chosen = selected(args.group)
    if args.cells:
        chosen = {k: v for k, v in chosen.items() if k in set(args.cells)}
    if args.seeds:
        allowed = set(args.seeds)
        chosen = {k: [s for s in v if s in allowed] for k, v in chosen.items()}
    log_path = args.output_root / "runs" / "logs" / f"joint_{args.group}_{os.getpid()}.log"
    log_line(log_path, f"start group={args.group} cells={chosen} device={args.device}")
    for cell_id, seeds in chosen.items():
        for seed in seeds:
            evaluate_cell(ctx, args.output_root, args.group, cell_id, seed, log_path)
            gc.collect()
            if C.torch is not None and C.torch.cuda.is_available():
                C.torch.cuda.empty_cache()
    out_csv = write_csv(args.output_root, args.group)
    log_line(log_path, f"done group={args.group} csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
