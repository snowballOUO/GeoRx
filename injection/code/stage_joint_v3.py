
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import stage_protected_pairs as P

C = P.C
ROOT = Path(__file__).resolve().parents[1]
METHOD = "same_query_joint_v3"
COVERAGE = 0.90  
TOL = 1e-5


def norm(x):
    return P.normalize(x)


def rotate_plane(x, a, b, chunk=8192):
    cosine = float(np.clip(a @ b, -1, 1))
    v = b - cosine * a
    sine = float(np.linalg.norm(v))
    if sine < 1e-8:
        return x.copy()
    v /= sine
    out = np.empty_like(x)
    for lo in range(0, len(x), chunk):
        z = x[lo:lo+chunk]
        za, zv = z @ a, z @ v
        out[lo:lo+chunk] = z + ((cosine-1)*za-sine*zv)[:, None]*a + (sine*za+(cosine-1)*zv)[:, None]*v
    return out


def construct(q0, p0, pair, clean_idx, axis, seed):
    pair = tuple(sorted(pair))  
    t = C.NATIVE_INJECT_STRENGTH
    q, p = q0.copy(), p0.copy()
    meta = {"all_query_ids": list(range(len(q))), "strength": t,
            "construction": "shared_support_joint", "pair": list(pair)}
    if "h3" in pair:
        old_groups = P.query_groups
        try:
            P.query_groups = lambda n, kinds: {k: np.arange(n) for k in kinds}
            reg = P.build_registry(q0, p0, ("h3",), clean_idx, axis, seed)
        finally:
            P.query_groups = old_groups
        q, p = P.inject("h3", q, p, reg, t)
        spec = reg["supports"]["h3"]
        ids, signs = spec["ids"], spec["signs"]
        islands = [ids[signs > 0], ids[signs < 0]]
        bridges = [x[:min(32, C.H3_BRIDGE_K)] for x in islands]
        centers = [norm(p[b].mean(0, keepdims=True))[0] for b in bridges]
        center = norm((centers[0]+centers[1])[None])[0]
        if "h2" in pair:
            for island, c in zip(islands, centers):
                p[island] = norm((1-t)*p[island]+t*c)
        if "h1" in pair:
            q = norm((1-t)*q+t*center)
        if "h4" in pair:
            delta = centers[0]-centers[1]
            delta /= max(float(np.linalg.norm(delta)), 1e-8)
            q = norm(q-(q @ delta)[:, None]*delta)
        meta["support"] = np.concatenate(bridges).tolist()
        meta["islands"] = [x.tolist() for x in islands]
    else:
        center0 = norm(p0.mean(0, keepdims=True))[0]
        selected = np.argsort(-(p0 @ center0))[:64]
        center = p0[selected[0]].copy() if "h1" in pair else norm(p0[selected].mean(0, keepdims=True))[0]
        if "h2" in pair and "h4" in pair:
            p[selected] = norm((1-t)*p0[selected]+t*center)
            selected = selected[np.argsort(-(p[selected] @ center))]
        if "h4" in pair:
            tie = selected[:2]
            center = norm(p[tie].mean(0, keepdims=True))[0]
            p[tie] = norm((1-t)*p[tie]+t*center)
            meta["tie_ids"] = tie.tolist()
        if "h2" in pair and "h4" not in pair:
            start = 2 if "h4" in pair else (1 if "h1" in pair else 0)
            ids = selected[start:]
            p[ids] = norm((1-t)*p0[ids]+t*center)
        q = norm((1-t)*q0+t*center)
        if "h4" in pair:
            delta = p[selected[0]]-p[selected[1]]
            delta /= max(float(np.linalg.norm(delta)), 1e-8)
            q = norm(q-(q @ delta)[:, None]*delta)
            if "h2" in pair:
                
                
                
                
                
                residual = p[selected]-(p[selected] @ center)[:, None]*center
                _, singular, vt = np.linalg.svd(residual, full_matrices=False)
                basis = vt[singular > 1e-6]
                q = norm(q-(q @ basis.T) @ basis)
        meta["support"] = selected.tolist() if "h2" in pair else selected[:(2 if "h4" in pair else 1)].tolist()
    before_h5_q = q.copy()
    
    
    
    if "h5" in pair:
        a = norm(q.mean(0, keepdims=True))[0]
        mask = a*a >= np.quantile(a*a, 0.90)
        b = a.copy()
        b[mask] *= 1-t
        b = norm(b[None])[0]
        q, p = rotate_plane(q, a, b), rotate_plane(p, a, b)
        meta["h5_rotation_cosine"] = float(a @ b)
        meta["h5_transport"] = "orthogonal_transport_of_native_attenuated_mean"
    meta["mean_query_cosine_to_clean"] = float(np.mean(np.sum(q*q0, axis=1)))
    meta["max_query_norm_error"] = float(np.max(np.abs(np.linalg.norm(q, axis=1)-1)))
    return q, p, meta


def neighbor_similarity(p, idx):
    v = p[idx].astype(np.float64)
    n = v.shape[1]
    return ((v.sum(1)**2).sum(1)-(v*v).sum((1, 2)))/(n*(n-1))


def second_component(p, idx):
    out = []
    for ids in idx:
        v = p[ids]
        s = v @ v.T
        np.fill_diagonal(s, -np.inf)
        nn = np.argpartition(-s, 10, axis=1)[:, :10]
        adj = np.zeros((len(ids), len(ids)), dtype=bool)
        adj[np.arange(len(ids))[:, None], nn] = True
        adj |= adj.T.copy()
        seen, sizes = set(), []
        for start in range(len(ids)):
            if start in seen:
                continue
            todo, count = [start], 0
            seen.add(start)
            while todo:
                cur = todo.pop(); count += 1
                for nb in np.flatnonzero(adj[cur]):
                    if int(nb) not in seen:
                        seen.add(int(nb)); todo.append(int(nb))
            sizes.append(count)
        sizes.sort(reverse=True)
        out.append(sizes[1]/len(ids) if len(sizes)>1 else 0.0)
    return np.asarray(out)


def peak_left(q, p):
    prod = q*p
    return prod.max(1)-prod.mean(1)


def audit(q0, p0, q, p, pair, meta, clean_vals, clean_idx, device, batch_q):
    vals, idx = P.brute_topk(q, p, 50, device=device, batch_q=batch_q)
    flags, geometry = {}, {}
    for k in pair:
        extra = {}
        if k == "h1":
            support = np.asarray(meta["support"])
            before = np.mean(np.isin(clean_idx[:, :10], support).any(1))
            mask = np.isin(idx[:, :10], support).any(1)
            counts0 = np.bincount(clean_idx[:, :10].ravel(), minlength=len(p))
            counts = np.bincount(idx[:, :10].ravel(), minlength=len(p))
            concentration0 = float(np.sum((counts0/counts0.sum())**2))
            concentration = float(np.sum((counts/counts.sum())**2))
            valid = concentration > concentration0 and mask.mean() > before
            extra = {"clean_occurrence_concentration": concentration0, "final_occurrence_concentration": concentration,
                     "clean_support_query_coverage": float(before)}
        elif k == "h2":
            before, after = neighbor_similarity(p0, clean_idx[:, :50]), neighbor_similarity(p, idx)
            mask = after > before+1e-6
            valid = after.mean() > before.mean()
            extra = {"clean_top50_pairwise_mean": float(before.mean()), "final_top50_pairwise_mean": float(after.mean())}
        elif k == "h3":
            second = second_component(p, idx)
            mask = second > 0
            valid = True
            extra = {"second_component_fraction_mean": float(second.mean()), "second_component_fraction_min": float(second.min())}
        elif k == "h4":
            gap = vals[:, 0]-vals[:, 1]
            mask = gap <= TOL
            valid = True
            extra = {"final_top1_top2_gap_mean": float(gap.mean()), "final_top1_top2_gap_max": float(gap.max()),
                     "clean_top1_top2_gap_mean": float(np.mean(clean_vals[:, 0]-clean_vals[:, 1]))}
        elif k == "h5":
            before = peak_left(q0, p0[clean_idx[:, 0]])
            after = peak_left(q, p[idx[:, 0]])
            mask = after < before
            valid = after.mean() < before.mean()
            extra = {"clean_actual_top1_peak_left_mean": float(before.mean()), "final_actual_top1_peak_left_mean": float(after.mean())}
        flags[k] = mask & valid
        geometry[k] = {"realized": bool(valid and mask.mean() >= COVERAGE),
                       "query_coverage": float(np.mean(flags[k])), **extra}
    joint = np.logical_and.reduce(list(flags.values()))
    return geometry, {"n_queries": len(q), "same_query_target_coverage": 1.0,
                      "joint_geometry_coverage": float(joint.mean()),
                      "query_geometry_masks": {k: v.astype(int).tolist() for k, v in flags.items()},
                      "gate_coverage": COVERAGE}, idx


def run_record(ctx, cell, seed, pair, order, inp, profile, cfg, clean_idx, axis):
    started = time.perf_counter()
    q0, p0 = inp["fault_q"], inp["pool"]
    bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
    q, p, meta = construct(q0, p0, pair, clean_idx, axis, seed)
    clean_vals = np.sum(q0[:, None]*p0[clean_idx[:, :50]], axis=-1)
    geometry, coverage, _ = audit(q0, p0, q, p, pair, meta, clean_vals, clean_idx, ctx.device, bq)
    diagnosis = C.diagnose_phi(profile, q, p, cfg, seed)
    phi = set(diagnosis["phi"] or [])
    realized = {k for k in pair if geometry[k]["realized"]}
    return {"state": "complete", "method": METHOD, "seed": seed, "cell_id": cell.cell_id,
            "encoder": cell.encoder, "dataset": cell.dataset, "pair": ";".join(pair),
            "order": "joint", "strength": C.NATIVE_INJECT_STRENGTH,
            "geometry": geometry, "coverage": coverage, "construction": meta,
            "realized_set": sorted(realized), "phi": sorted(phi),
            "called_realization": len(realized)/2, "called_recall": len(phi & set(pair))/2,
            "realized_recall": len(phi & realized)/len(realized) if realized else None,
            "exact_called": int(phi == set(pair)), "tau": diagnosis["tau"],
            "z": {k: diagnosis["per_type"][k]["z"] for k in P.KINDS},
            "observed": {k: diagnosis["per_type"][k]["observed"] for k in P.KINDS},
            "wall_s": time.perf_counter()-started, "written": C.now()}


if __name__ == "__main__":
    P.ROOT = ROOT
    P.METHOD = METHOD
    P.run_record = run_record
    raise SystemExit(P.main())
