

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

import common as C
from retrieve import brute_topk
from unified import pool_axis


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("GEORX_SOURCE_ROOT", str(PACKAGE_ROOT / "main")))
METHOD = "protected_support_v4_exact_h4"
KINDS = ("h1", "h2", "h3", "h4", "h5")
PAIRS = tuple(itertools.combinations(KINDS, 2))
DEFAULT_CELLS = (
    "clip_sf_large__mscoco_task0",
    "clip_sf_large__nights_task4",
    "e5v_llava_next__cirr_task7",
)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def at_query_cosine(centroid, query, target):
    c = np.asarray(centroid, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    c0 = float(np.clip(c @ q, -1.0 + 1e-6, 1.0 - 1e-6))
    tgt = float(min(max(target, c0 + 1e-5), 0.999))
    denom = max(1.0 - c0 * c0, 1e-12)
    a = float(np.sqrt(max((1.0 - tgt * tgt) / denom, 0.0)))
    b = tgt - a * c0
    return normalize((a * c + b * q)[None])[0]


def set_query_cosine(vector, query, target):
    v = normalize(np.asarray(vector, dtype=np.float32)[None])[0]
    q = normalize(np.asarray(query, dtype=np.float32)[None])[0]
    c = float(np.clip(v @ q, -1.0, 1.0))
    ortho = v - c * q
    norm = float(np.linalg.norm(ortho))
    if norm < 1e-8:
        return v
    u = ortho / norm
    target = float(np.clip(target, -0.999999, 0.999999))
    return normalize((target * q + np.sqrt(max(1.0 - target * target, 0.0)) * u)[None])[0]


def parse_combo(text: str):
    xs = tuple(x.strip().lower() for x in text.replace("+", ";").replace(",", ";").split(";") if x.strip())
    if len(xs) != 2 or len(set(xs)) != 2 or any(x not in KINDS for x in xs):
        raise argparse.ArgumentTypeError(text)
    return tuple(sorted(xs, key=KINDS.index))


def source_diag(seed, cell_id):
    path = SOURCE / "runs" / f"S{seed}" / "evaluations" / cell_id / "diagnosis.json"
    d = json.loads(path.read_text())
    if not C.verify_seal(d):
        raise RuntimeError(f"seal mismatch {path}")
    return d


def check_split(diag, seg, cell_id):
    got = {
        "q_eval_indices_sha256": C.hash_idx(seg["eval"]),
        "train_indices_sha256": C.hash_idx(seg["train"]),
        "calibration_indices_sha256": C.hash_idx(seg["calibration"]),
        "fault_eval_indices_sha256": C.hash_idx(seg["fault_eval"]),
    }
    for k, v in got.items():
        if diag["split_manifest"].get(k) != v:
            raise RuntimeError(f"{cell_id}: split drift {k}")


def query_groups(nq: int, pair):
    ids = np.arange(nq, dtype=np.int64)
    return {pair[0]: ids[0::2], pair[1]: ids[1::2]}


def _unique_anchors(clean_idx, qids, k, reserved, n_pool, rng):
    out = np.empty((len(qids), k), dtype=np.int64)
    available = None
    for i, qi in enumerate(qids):
        chosen = []
        for doc in clean_idx[int(qi)].tolist():
            doc = int(doc)
            if doc not in reserved:
                chosen.append(doc)
                reserved.add(doc)
                if len(chosen) == k:
                    break
        if len(chosen) < k:
            if available is None or len(available) < k - len(chosen):
                available = np.asarray([j for j in range(n_pool) if j not in reserved], dtype=np.int64)
                rng.shuffle(available)
            take = available[: k - len(chosen)].tolist()
            available = available[k - len(chosen):]
            chosen.extend(take)
            reserved.update(take)
        out[i] = chosen
    return out


def build_registry(q, p, pair, clean_idx, axis, seed):
    rng = np.random.default_rng(seed + 1600)
    groups = query_groups(len(q), pair)
    reserved = set()
    reg = {"groups": groups, "supports": {}, "axis": axis}
    for kind in pair:
        qids = groups[kind]
        if kind == "h1":
            center = normalize(p.mean(axis=0, keepdims=True))[0]
            ranking = np.argsort(-(p @ center))
            hub = next(int(x) for x in ranking if int(x) not in reserved)
            reserved.add(hub)
            reg["supports"][kind] = {"hub": hub}
        elif kind == "h2":
            reg["supports"][kind] = {"anchors": _unique_anchors(clean_idx, qids, 5, reserved, len(p), rng)}
        elif kind == "h3":
            proj = p @ axis
            candidates = []
            seen = set()
            for qi in qids:
                for doc in clean_idx[int(qi)]:
                    doc = int(doc)
                    if doc not in reserved and doc not in seen:
                        candidates.append(doc); seen.add(doc)
            pos = [j for j in candidates if proj[j] >= 0]
            neg = [j for j in candidates if proj[j] < 0]
            for source, sign_pos in ((pos, True), (neg, False)):
                need = 256 - len(source)
                if need > 0:
                    pool_ids = np.flatnonzero(proj >= 0 if sign_pos else proj < 0)
                    pool_ids = np.asarray([int(j) for j in pool_ids if int(j) not in reserved and int(j) not in seen])
                    rng.shuffle(pool_ids)
                    source.extend(pool_ids[:need].tolist())
                    seen.update(source)
            n_each = min(256, len(pos), len(neg))
            support = np.asarray(pos[:n_each] + neg[:n_each], dtype=np.int64)
            signs = np.asarray([1] * n_each + [-1] * n_each, dtype=np.int8)
            reserved.update(support.tolist())
            reg["supports"][kind] = {"ids": support, "signs": signs}
        elif kind == "h4":
            reg["supports"][kind] = {"anchors": _unique_anchors(clean_idx, qids, 2, reserved, len(p), rng)}
        elif kind == "h5":
            anchors = _unique_anchors(clean_idx, qids, 5, reserved, len(p), rng)
            masks = []
            for qi, docs in zip(qids, anchors):
                had = np.abs(q[int(qi)] * p[int(docs[0])])
                masks.append(had >= float(np.quantile(had, 0.90)))
            reg["supports"][kind] = {"anchors": anchors, "masks": np.asarray(masks, dtype=bool)}
    return reg


def inject(kind, q, p, reg, strength):
    t = float(strength)
    q2 = np.array(q, dtype=np.float32, copy=True)
    p2 = np.array(p, dtype=np.float32, copy=True)
    qids = reg["groups"][kind]
    spec = reg["supports"][kind]
    if kind == "h1":
        hub = normalize(p2[[spec["hub"]]])[0]
        q2[qids] = normalize((1.0 - t) * q2[qids] + t * hub)
    elif kind == "h2":
        for qi, docs in zip(qids, spec["anchors"]):
            mean = normalize(p2[docs].mean(axis=0, keepdims=True))[0]
            p2[docs] = normalize((1.0 - t) * p2[docs] + t * mean)
            q2[int(qi)] = normalize(((1.0 - t) * q2[int(qi)] + t * normalize(p2[docs].mean(axis=0, keepdims=True))[0])[None])[0]
    elif kind == "h3":
        ids, signs = spec["ids"], spec["signs"]
        u = reg["axis"]
        work = p2[ids] + t * signs[:, None] * u
        work = normalize(work)
        pos, neg = signs > 0, signs < 0
        cpos = normalize(work[pos].mean(axis=0, keepdims=True))[0]
        cneg = normalize(work[neg].mean(axis=0, keepdims=True))[0]
        tight = C.H3_TIGHTNESS
        work[pos] = normalize((1.0 - tight) * work[pos] + tight * cpos)
        work[neg] = normalize((1.0 - tight) * work[neg] + tight * cneg)
        p2[ids] = work
        cpos = normalize(work[pos].mean(axis=0, keepdims=True))[0]
        cneg = normalize(work[neg].mean(axis=0, keepdims=True))[0]
        mid = normalize((cpos + cneg)[None])[0]
        target = 0.99
        bridge_n = min(32, C.H3_BRIDGE_K)
        pos_local = np.flatnonzero(pos)[:bridge_n]
        neg_local = np.flatnonzero(neg)[:bridge_n]
        work[pos_local] = at_query_cosine(cpos, mid, target)
        work[neg_local] = at_query_cosine(cneg, mid, target)
        p2[ids] = work
        proj = q2[qids] - (q2[qids] @ u)[:, None] * u
        q2[qids] = normalize(cpos + cneg + C.H3_RESIDUAL * proj)
    elif kind == "h4":
        for qi, docs in zip(qids, spec["anchors"]):
            mid = normalize(p2[docs].mean(axis=0, keepdims=True))[0]
            p2[docs] = normalize((1.0 - t) * p2[docs] + t * mid)
            q2[int(qi)] = normalize(((1.0 - t) * q2[int(qi)] + t * normalize(p2[docs].mean(axis=0, keepdims=True))[0])[None])[0]
            scores = p2[docs] @ q2[int(qi)]
            target = float(np.mean(scores))
            p2[int(docs[0])] = set_query_cosine(p2[int(docs[0])], q2[int(qi)], target)
            p2[int(docs[1])] = set_query_cosine(p2[int(docs[1])], q2[int(qi)], target)
    elif kind == "h5":
        for qi, docs, mask in zip(qids, spec["anchors"], spec["masks"]):
            q2[int(qi), mask] *= 1.0 - t
            for doc in docs:
                p2[int(doc), mask] *= 1.0 - t
        q2[qids] = normalize(q2[qids])
        p2[spec["anchors"].reshape(-1)] = normalize(p2[spec["anchors"].reshape(-1)])
    return q2, p2


def pairwise_sim(x):
    gram = x @ x.T
    n = len(x)
    return float((gram.sum() - np.trace(gram)) / max(n * (n - 1), 1))


def peak_share(q, p):
    had = np.abs(q * p)
    k = max(1, int(np.ceil(had.shape[1] * 0.10)))
    top = np.partition(had, had.shape[1] - k, axis=1)[:, -k:]
    return top.sum(axis=1) / np.maximum(had.sum(axis=1), 1e-12)


def verify_geometry(clean_q, clean_p, q, p, reg, pair, device, batch_q):
    out = {}
    max_k = 50 if "h3" in pair else 10
    _, final_idx = brute_topk(q, p, min(max_k, len(p)), device=device, batch_q=batch_q)
    for kind in pair:
        qids = reg["groups"][kind]
        spec = reg["supports"][kind]
        if kind == "h1":
            hub = int(spec["hub"])
            clean_share = float(np.mean(brute_topk(clean_q[qids], clean_p, 1, device=device, batch_q=batch_q)[1][:, 0] == hub))
            final_share = float(np.mean(final_idx[qids, 0] == hub))
            out[kind] = {"realized": bool(final_share > clean_share), "clean_hub_top1_share": clean_share, "final_hub_top1_share": final_share}
        elif kind == "h2":
            clean_sim = np.asarray([pairwise_sim(clean_p[docs]) for docs in spec["anchors"]])
            final_sim = np.asarray([pairwise_sim(p[docs]) for docs in spec["anchors"]])
            retained = np.asarray([len(set(docs.tolist()) & set(final_idx[int(qi), :10].tolist())) for qi, docs in zip(qids, spec["anchors"])])
            out[kind] = {"realized": bool(np.all(final_sim > clean_sim) and np.any(retained > 0)), "all_groups_contracted": bool(np.all(final_sim > clean_sim)), "mean_clean_similarity": float(clean_sim.mean()), "mean_final_similarity": float(final_sim.mean()), "mean_anchors_in_top10": float(retained.mean())}
        elif kind == "h3":
            ids, signs = spec["ids"], spec["signs"]
            pos, neg = signs > 0, signs < 0
            cpos = normalize(p[ids[pos]].mean(axis=0, keepdims=True))[0]
            cneg = normalize(p[ids[neg]].mean(axis=0, keepdims=True))[0]
            own = min(float(np.mean(p[ids[pos]] @ cpos)), float(np.mean(p[ids[neg]] @ cneg)))
            cross = float(cpos @ cneg)
            both = []
            pos_ids, neg_ids = set(ids[pos].tolist()), set(ids[neg].tolist())
            for qi in qids:
                got = set(final_idx[int(qi), :50].tolist())
                both.append(bool(got & pos_ids) and bool(got & neg_ids))
            out[kind] = {"realized": bool(own > cross and any(both)), "min_within_centroid_cosine": own, "cross_centroid_cosine": cross, "queries_retrieving_both_islands": float(np.mean(both))}
        elif kind == "h4":
            clean_gap, final_gap, retained = [], [], []
            for qi, docs in zip(qids, spec["anchors"]):
                clean_scores = clean_p[docs] @ clean_q[int(qi)]
                final_scores = p[docs] @ q[int(qi)]
                clean_gap.append(abs(float(clean_scores[0] - clean_scores[1])))
                final_gap.append(abs(float(final_scores[0] - final_scores[1])))
                retained.append(len(set(docs.tolist()) & set(final_idx[int(qi), :10].tolist())))
            clean_gap, final_gap = np.asarray(clean_gap), np.asarray(final_gap)
            retained = np.asarray(retained)
            out[kind] = {"realized": bool(np.max(final_gap) <= 1e-5 and np.all(retained == 2)), "all_pairs_exact_tie": bool(np.max(final_gap) <= 1e-5), "max_final_pair_gap": float(final_gap.max()), "mean_clean_pair_gap": float(clean_gap.mean()), "mean_final_pair_gap": float(final_gap.mean()), "pairs_both_in_top10": float(np.mean(retained == 2))}
        elif kind == "h5":
            top = spec["anchors"][:, 0]
            clean_peak = peak_share(clean_q[qids], clean_p[top])
            final_peak = peak_share(q[qids], p[top])
            out[kind] = {"realized": bool(np.all(final_peak < clean_peak)), "all_peak_shares_reduced": bool(np.all(final_peak < clean_peak)), "mean_clean_peak_share": float(clean_peak.mean()), "mean_final_peak_share": float(final_peak.mean())}
    return out


def run_record(ctx, cell, seed, pair, order, inp, profile, cfg, clean_idx, axis):
    reg = build_registry(inp["fault_q"], inp["pool"], pair, clean_idx, axis, seed)
    q, p = inp["fault_q"], inp["pool"]
    t0 = time.perf_counter()
    for kind in order:
        q, p = inject(kind, q, p, reg, C.NATIVE_INJECT_STRENGTH)
    geom = verify_geometry(inp["fault_q"], inp["pool"], q, p, reg, pair, ctx.device, C._batch_for(cell.dim, cell.n_pool, ctx.batch_q))
    realized = [k for k in pair if geom[k]["realized"]]
    diagnosis = C.diagnose_phi(profile, q, p, cfg, seed)
    phi = list(diagnosis["phi"] or [])
    called = set(pair); realized_set = set(realized); phi_set = set(phi)
    rec = {
        "state": "complete", "method": METHOD, "seed": seed,
        "cell_id": cell.cell_id, "encoder": cell.encoder, "dataset": cell.dataset,
        "pair": ";".join(pair), "order": ">".join(order),
        "strength": C.NATIVE_INJECT_STRENGTH,
        "geometry": geom, "realized_set": realized,
        "called_realization": len(realized_set) / len(called),
        "phi": phi,
        "called_recall": len(phi_set & called) / len(called),
        "realized_recall": (len(phi_set & realized_set) / len(realized_set)) if realized_set else None,
        "exact_called": int(phi_set == called),
        "z": {k: diagnosis["per_type"][k]["z"] for k in KINDS},
        "observed": {k: diagnosis["per_type"][k]["observed"] for k in KINDS},
        "wall_s": round(time.perf_counter() - t0, 3), "written": C.now(),
    }
    del q, p
    return rec


def out_path(seed, cell_id, pair, order):
    return ROOT / "runs" / METHOD / f"S{seed}" / cell_id / f"{'_'.join(pair)}__{'_then_'.join(order)}.json"


def rewrite_csv():
    records = []
    for path in sorted((ROOT / "runs" / METHOD).glob("S*/*/*.json")):
        d = json.loads(path.read_text())
        if d.get("state") == "complete": records.append(d)
    if not records: return
    rows = []
    for d in records:
        row = {k: d.get(k) for k in ("seed", "encoder", "dataset", "pair", "order", "strength", "called_realization", "called_recall", "realized_recall", "exact_called", "wall_s")}
        row["realized_set"] = ";".join(d["realized_set"]) or "empty"
        row["phi"] = ";".join(d["phi"]) or "empty"
        for kind in d["pair"].split(";"):
            row[f"realized_{kind}"] = int(d["geometry"][kind]["realized"])
        rows.append(row)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    out = ROOT / "runs" / f"{METHOD}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(out)


def main(argv=None):
    global METHOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--cells", nargs="+")
    ap.add_argument("--encoders", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--combos", nargs="+", type=parse_combo)
    ap.add_argument("--orders", choices=("canonical", "both"), default="both")
    ap.add_argument("--run-name", default=METHOD)
    ap.add_argument("--status-tag", default="")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--rewrite-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-q", type=int, default=32)
    args = ap.parse_args(argv)
    METHOD = args.run_name
    if args.rewrite_only:
        rewrite_csv()
        return 0
    pairs = list(dict.fromkeys(args.combos or PAIRS))
    cells = args.cells
    if cells is None:
        cells = [f"{encoder}__{dataset}" for encoder, dataset in C.all_pairs(args.encoders, args.datasets)]
    order_count = 1 if args.orders == "canonical" else 2
    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)
    status = {"started": C.now(), "method": METHOD, "orders": args.orders,
              "n_ok": 0, "n_skip": 0, "n_fail": 0, "errors": []}
    for seed in args.seeds:
        for cell_id in cells:
            cell = ctx.cells[cell_id]
            diag = source_diag(seed, cell_id); seg = C.repartition(cell, seed); check_split(diag, seg, cell_id)
            inp = C.load_detector_inputs(cell, seg)
            bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
            profile, cfg = C.fit_self_profile(inp["train_q"], inp["cal_q"], inp["pool"], ctx.device, bq, seed)
            axis = pool_axis(inp["pool"], seed + 7)
            _, clean_idx = brute_topk(inp["fault_q"], inp["pool"], min(64, cell.n_pool), device=ctx.device, batch_q=bq)
            for pair in pairs:
                orders = (pair,) if args.orders == "canonical" else (pair, tuple(reversed(pair)))
                for order in orders:
                    path = out_path(seed, cell_id, pair, order)
                    if path.is_file():
                        try:
                            if json.loads(path.read_text()).get("state") == "complete": status["n_skip"] += 1; continue
                        except Exception: pass
                    try:
                        rec = run_record(ctx, cell, seed, pair, order, inp, profile, cfg, clean_idx, axis)
                        C.write_json(path, rec); status["n_ok"] += 1
                        print(f"s{seed} {cell_id} {rec['order']} realized={rec['realized_set']} phi={rec['phi']}", flush=True)
                    except Exception as exc:
                        status["n_fail"] += 1
                        status["errors"].append({"seed": seed, "cell_id": cell_id, "pair": pair, "order": order, "error": repr(exc), "traceback": traceback.format_exc()})
                    finally:
                        gc.collect()
                        if C.torch is not None and C.torch.cuda.is_available(): C.torch.cuda.empty_cache()
            del inp, profile, cfg, axis, clean_idx
    if not args.no_rewrite:
        rewrite_csv()
    status.update({"finished": C.now(), "n_target": len(args.seeds) * len(cells) * len(pairs) * order_count})
    suffix = f"_{args.status_tag}" if args.status_tag else ""
    C.write_json(ROOT / "runs" / f"status{suffix}.json", status)
    print(json.dumps(status, indent=2, default=str))
    return 0 if status["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
