

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

import ce_list_train
import common as C
import compose as CMP
import corrections

PHI_CASES = ("h1;h2", "h1;h4", "h2;h4", "h1;h2;h4")
KS = (1, 5, 10)
CSLS_LAMS = (0.25, 0.50, 1.00)
QE_ALPHAS = (0.05, 0.10, 0.20)
CE_NS = (50, 100)
FUSION_BETAS = (0.25, 0.50, 0.75)
RRF_KS = (10, 60)
OUT_ROOT = C.ROOT / "runs" / "joint_phi_sweep"



VARIANT_COMBO = {
    "single_csls": "h1",
    "csls_on_cosine_shortlist": "h1",
    "single_qe": "h2",
    "single_h4_on_cosine": "h4",
    "csls_then_qe": "h1;h2",
    "qe_then_csls": "h1;h2",
    "csls_qe_csls": "h1;h2",
    "zfusion_csls_qe": "h1;h2",
    "rrf_csls_qe": "h1;h2",
    "csls_then_h4": "h1;h4",
    "h4_then_csls": "h1;h4",
    "zfusion_csls_h4": "h1;h4",
    "rrf_csls_h4": "h1;h4",
    "h4_then_qe": "h2;h4",
    "qe_then_h4": "h2;h4",
    "zfusion_qe_h4": "h2;h4",
    "rrf_qe_h4": "h2;h4",
    "csls_qe_h4": "h1;h2;h4",
    "csls_h4_qe": "h1;h2;h4",
    "qe_csls_h4": "h1;h2;h4",
    "qe_h4_csls": "h1;h2;h4",
    "h4_csls_qe": "h1;h2;h4",
    "h4_qe_csls": "h1;h2;h4",
    "zfusion_csls_qe_h4": "h1;h2;h4",
    "rrf_csls_qe_h4": "h1;h2;h4",
}


def _recall(top_i, positives):
    return {k: float(C.recall_hits(top_i, positives, k).mean()) for k in KS}


def _canon_phi(phi) -> str:
    s = set(phi or [])
    return ";".join(t for t in ("h1", "h2", "h3", "h4", "h5") if t in s) or "empty"


def list_work_items(seeds):
    items = []
    for seed in seeds:
        path = C.ROOT / "runs" / f"S{seed}" / "diagnosis_sets.csv"
        with path.open() as fh:
            for r in csv.DictReader(fh):
                phi = r["phi_set"]
                if phi in PHI_CASES:
                    items.append((int(seed), f"{r['encoder']}__{r['dataset']}", phi))
    items.sort()
    return items


def _csls_short_scores(query, pool, r_c, lam, short_idx, device, batch_q):
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    r_t = torch.from_numpy(np.ascontiguousarray(r_c)).to(device)
    nq, n_short = short_idx.shape
    out = np.empty((nq, n_short), dtype=np.float64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        sl = torch.from_numpy(np.ascontiguousarray(short_idx[s:e])).to(device)
        cos = (q_t[s:e].unsqueeze(1) * p_t[sl]).sum(dim=-1)
        out[s:e] = (cos - float(lam) * r_t[sl]).detach().cpu().numpy()
    return out


def _cos_short_scores(query, pool, short_idx, device, batch_q):
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    nq, n_short = short_idx.shape
    out = np.empty((nq, n_short), dtype=np.float64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        sl = torch.from_numpy(np.ascontiguousarray(short_idx[s:e])).to(device)
        out[s:e] = (q_t[s:e].unsqueeze(1) * p_t[sl]).sum(dim=-1).detach().cpu().numpy()
    return out


def _zrow(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = np.maximum(x.std(axis=1, keepdims=True), 1e-8)
    return (x - mu) / sd


def _rerank(short_idx, scores, k_out):
    nq = short_idx.shape[0]
    new_i = np.empty((nq, k_out), dtype=np.int64)
    for qi in range(nq):
        order = np.argsort(-scores[qi], kind="stable")
        ranked = short_idx[qi, order]
        m = min(k_out, ranked.size)
        new_i[qi, :m] = ranked[:m]
        if m < k_out:
            new_i[qi, m:] = ranked[m - 1]
    return new_i


def _ranks_of(short_idx, ordered_idx):
    nq, n_short = short_idx.shape
    out = np.empty((nq, n_short), dtype=np.float64)
    default = ordered_idx.shape[1] + 1
    for qi in range(nq):
        pos = {int(v): r for r, v in enumerate(ordered_idx[qi].tolist(), start=1)}
        out[qi] = [pos.get(int(v), default) for v in short_idx[qi].tolist()]
    return out


class Runner:
    def __init__(self, ctx, seed, cell, log):
        self.ctx = ctx
        self.seed = seed
        self.cell = cell
        self.log = log
        self.rows = []
        self._h4 = {}
        diag = C.load_sealed_diagnosis(ctx, seed, cell.cell_id)
        self.phi = list(diag.get("phi") or [])
        self.phi_set = _canon_phi(self.phi)
        seg = C.repartition(cell, seed)
        labeled = C.load_eval_labeled(cell, seg)
        ctx._eval_idx = labeled["eval_idx"]
        self.query = labeled["eval_q"]
        self.pool = labeled["pool"]
        self.positives = labeled["positives"]
        self.ref_q = labeled["ref_q"]
        self.bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
        self.k_out = max(KS + (C.RERANK_N,))
        self.params = {
            "h1_k": 10, "csls_k": 10, "qe_k": 10, "qe_alpha": 0.10, "qe_power": 3.0,
            "csls_reference_queries": self.ref_q,
        }
        _, self.cos_i, st = corrections.search(
            "cosine", self.query, self.pool, self.k_out, params=self.params,
            device=ctx.device, batch_q=self.bq)
        if st != "ok":
            raise RuntimeError(st)
        self.base = _recall(self.cos_i, self.positives)
        self.r_c = None
        self._csls_idx = {}
        self._qe_idx = {}

    def emit(self, variant, params, top_i, wall, kind, note=""):
        if variant not in VARIANT_COMBO:
            raise KeyError(f"unregistered variant {variant!r}")
        rec = _recall(top_i, self.positives)
        d = {k: rec[k] - self.base[k] for k in KS}
        row = {
            "seed": self.seed, "encoder": self.cell.encoder, "dataset": self.cell.dataset,
            "cell_id": self.cell.cell_id, "phi_set": self.phi_set,
            "combo": VARIANT_COMBO[variant],
            "h4_method": "ce_list_local_list_mixer",
            "h2_method": "alpha_qe",
            "variant": variant, "kind": kind, "params": params,
            "R@1_base": self.base[1], "R@5_base": self.base[5], "R@10_base": self.base[10],
            "R@1": rec[1], "R@5": rec[5], "R@10": rec[10],
            "dR@1": d[1], "dR@5": d[5], "dR@10": d[10],
            "useful": int(d[10] >= C.USEFUL_DR10),
            "wall_s": round(wall, 3), "note": note,
            "scope": "exploratory_joint_phi_sweep",
        }
        self.rows.append(row)
        self.log(f"  {variant} {params} dR10={d[10]:+.4f}")
        return row

    def csls(self, lam):
        if lam in self._csls_idx:
            return self._csls_idx[lam], 0.0
        t0 = time.perf_counter()
        p = dict(self.params)
        if lam == 0.50:
            _, top_i, st = corrections.search(
                "csls_p050", self.query, self.pool, self.k_out, params=p,
                cosine_top_idx=self.cos_i, device=self.ctx.device, batch_q=self.bq)
        else:
            _, top_i, st = corrections._csls(
                self.query, self.pool, self.k_out, p, self.ctx.device, self.bq,
                penalty_lambda=lam)
        if st != "ok":
            raise RuntimeError(st)
        if self.r_c is None:
            self.r_c = p["_csls_r_c_cache"][next(iter(p["_csls_r_c_cache"]))]
            self.params["_csls_r_c_cache"] = p["_csls_r_c_cache"]
        self._csls_idx[lam] = top_i
        return top_i, time.perf_counter() - t0

    def qe_from_idx(self, neigh_idx, alpha):
        t0 = time.perf_counter()
        p = {**self.params, "qe_alpha": float(alpha)}
        _, top_i, q2 = CMP.alpha_qe_from_neighbors(
            self.query, self.pool, self.k_out, neigh_idx, p,
            self.ctx.device, self.bq)
        return top_i, q2, time.perf_counter() - t0

    def qe_indep(self, alpha):
        key = float(alpha)
        if key in self._qe_idx:
            return self._qe_idx[key]
        t0 = time.perf_counter()
        p = {**self.params, "qe_alpha": float(alpha)}
        _, top_i, st = corrections._alpha_qe(
            self.query, self.pool, self.k_out, p, self.ctx.device, self.bq)
        if st != "ok":
            raise RuntimeError(st)
        wall = time.perf_counter() - t0
        self._qe_idx[key] = (top_i, wall)
        return top_i, wall

    def csls_on_query(self, query, lam):
        t0 = time.perf_counter()
        p = dict(self.params)
        _, top_i, st = corrections._csls(
            query, self.pool, self.k_out, p, self.ctx.device, self.bq,
            penalty_lambda=lam)
        if st != "ok":
            raise RuntimeError(st)
        if self.r_c is None:
            self.r_c = p["_csls_r_c_cache"][next(iter(p["_csls_r_c_cache"]))]
            self.params["_csls_r_c_cache"] = p["_csls_r_c_cache"]
        return top_i, time.perf_counter() - t0

    def h4(self, short_idx):
        key = hashlib.sha256(np.ascontiguousarray(short_idx).tobytes()).hexdigest()
        if key in self._h4:
            return self._h4[key]
        t0 = time.perf_counter()
        sc, st = ce_list_train.scores_on_shortlist(
            self.cell.dataset, short_idx, self.ctx)
        if st != "ok":
            raise RuntimeError(f"H4 list_mixer failed: {st}")
        self._h4[key] = (sc, time.perf_counter() - t0)
        return self._h4[key]

    def close(self):
        del self.query, self.pool
        self._h4.clear()
        torch.cuda.empty_cache()


def run_h1h2(R: Runner):
    for lam in CSLS_LAMS:
        top, w = R.csls(lam)
        R.emit("single_csls", f"lambda={lam:g}", top, w, "single")
    for a in QE_ALPHAS:
        top, w = R.qe_indep(a)
        R.emit("single_qe", f"alpha={a:g}", top, w, "single")
    for lam, a in product(CSLS_LAMS, QE_ALPHAS):
        csls_i, w_c = R.csls(lam)
        qe_from_csls, q2_from_csls, w_q = R.qe_from_idx(csls_i, a)
        R.emit("csls_then_qe", f"lambda={lam:g},alpha={a:g}", qe_from_csls, w_c + w_q, "joint")
        _, q2_from_cos, w_q2 = R.qe_from_idx(R.cos_i, a)
        c2, w_c2 = R.csls_on_query(q2_from_cos, lam)
        R.emit("qe_then_csls", f"lambda={lam:g},alpha={a:g}", c2, w_q2 + w_c2, "joint")
        c4, w_c4 = R.csls_on_query(q2_from_csls, lam)
        R.emit("csls_qe_csls", f"lambda={lam:g},alpha={a:g}", c4, w_c + w_q + w_c4, "joint")
    short = R.cos_i[:, :100]
    if R.r_c is None:
        R.csls(0.50)
    for lam, a in product(CSLS_LAMS, QE_ALPHAS):
        sc_c = _csls_short_scores(R.query, R.pool, R.r_c, lam, short, R.ctx.device, R.bq)
        qe_i, q2, _ = R.qe_from_idx(R.cos_i, a)
        sc_q = _cos_short_scores(q2, R.pool, short, R.ctx.device, R.bq)
        zc, zq = _zrow(sc_c), _zrow(sc_q)
        for b in FUSION_BETAS:
            t0 = time.perf_counter()
            R.emit("zfusion_csls_qe", f"lambda={lam:g},alpha={a:g},beta={b:g},N=100",
                   _rerank(short, (1 - b) * zc + b * zq, R.k_out),
                   time.perf_counter() - t0, "joint")
        csls_i, _ = R.csls(lam)
        rc = _ranks_of(short, csls_i)
        rq = _ranks_of(short, qe_i)
        for krrf in RRF_KS:
            t0 = time.perf_counter()
            sc = 1.0 / (krrf + rc) + 1.0 / (krrf + rq)
            R.emit("rrf_csls_qe", f"lambda={lam:g},alpha={a:g},k_rrf={krrf},N=100",
                   _rerank(short, sc, R.k_out), time.perf_counter() - t0, "joint")


def run_h1h4(R: Runner):
    for lam in CSLS_LAMS:
        top, w = R.csls(lam)
        R.emit("single_csls", f"lambda={lam:g}", top, w, "single")
    short_cos = R.cos_i[:, :100]
    h4_rank = {}
    for n in CE_NS:
        sl = short_cos[:, :n]
        sc, wce = R.h4(sl)
        t0 = time.perf_counter()
        h4_rank[n] = _rerank(sl, sc, R.k_out)
        R.emit("single_h4_on_cosine", f"N={n}", h4_rank[n],
               wce + time.perf_counter() - t0, "single")
    sc_cos, _ = R.h4(short_cos)
    if R.r_c is None:
        R.csls(0.50)
    t0 = time.perf_counter()
    rec_i = _rerank(
        short_cos,
        _csls_short_scores(R.query, R.pool, R.r_c, 0.50, short_cos, R.ctx.device, R.bq),
        R.k_out)
    R.emit("csls_on_cosine_shortlist", "lambda=0.5,N=100", rec_i,
           time.perf_counter() - t0, "single",
           note="CSLS restricted to cosine top-100; not H4")
    for n in CE_NS:
        sl = h4_rank[n][:, :n]
        for lam in CSLS_LAMS:
            scs = _csls_short_scores(R.query, R.pool, R.r_c, lam, sl, R.ctx.device, R.bq)
            t0 = time.perf_counter()
            R.emit("h4_then_csls", f"lambda={lam:g},N={n}",
                   _rerank(sl, scs, R.k_out),
                   time.perf_counter() - t0, "joint")
    for lam in CSLS_LAMS:
        csls_i, w1 = R.csls(lam)
        for n in CE_NS:
            sl = csls_i[:, :n]
            sc, w2 = R.h4(sl)
            t0 = time.perf_counter()
            R.emit("csls_then_h4", f"lambda={lam:g},N={n}",
                   _rerank(sl, sc, R.k_out),
                   w1 + w2 + time.perf_counter() - t0, "joint")
        sc_c = _csls_short_scores(R.query, R.pool, R.r_c, lam, short_cos, R.ctx.device, R.bq)
        zc, ze = _zrow(sc_c), _zrow(sc_cos)
        for b in FUSION_BETAS:
            t0 = time.perf_counter()
            R.emit("zfusion_csls_h4", f"lambda={lam:g},beta={b:g},N=100",
                   _rerank(short_cos, (1 - b) * zc + b * ze, R.k_out),
                   time.perf_counter() - t0, "joint")
        rc = _ranks_of(short_cos, csls_i)
        re = _ranks_of(short_cos, h4_rank[100])
        for krrf in RRF_KS:
            t0 = time.perf_counter()
            R.emit("rrf_csls_h4", f"lambda={lam:g},k_rrf={krrf},N=100",
                   _rerank(short_cos, 1 / (krrf + rc) + 1 / (krrf + re), R.k_out),
                   time.perf_counter() - t0, "joint")


def run_h2h4(R: Runner):
    for a in QE_ALPHAS:
        top, w = R.qe_indep(a)
        R.emit("single_qe", f"alpha={a:g}", top, w, "single")
    short_cos = R.cos_i[:, :100]
    h4_rank = {}
    for n in CE_NS:
        sl = short_cos[:, :n]
        sc, wce = R.h4(sl)
        h4_rank[n] = _rerank(sl, sc, R.k_out)
        t0 = time.perf_counter()
        R.emit("single_h4_on_cosine", f"N={n}", h4_rank[n],
               wce + time.perf_counter() - t0, "single")
        for a in QE_ALPHAS:
            qe_i, _, wq = R.qe_from_idx(h4_rank[n], a)
            R.emit("h4_then_qe", f"alpha={a:g},N={n}", qe_i, wce + wq, "joint")
    sc_cos, _ = R.h4(short_cos)
    for a in QE_ALPHAS:
        qe_i, w1 = R.qe_indep(a)
        for n in CE_NS:
            sl = qe_i[:, :n]
            sc, w2 = R.h4(sl)
            t0 = time.perf_counter()
            R.emit("qe_then_h4", f"alpha={a:g},N={n}",
                   _rerank(sl, sc, R.k_out),
                   w1 + w2 + time.perf_counter() - t0, "joint")
    for a in QE_ALPHAS:
        qe_i, q2, _ = R.qe_from_idx(R.cos_i, a)
        sc_q = _cos_short_scores(q2, R.pool, short_cos, R.ctx.device, R.bq)
        zq, ze = _zrow(sc_q), _zrow(sc_cos)
        for b in FUSION_BETAS:
            t0 = time.perf_counter()
            R.emit("zfusion_qe_h4", f"alpha={a:g},beta={b:g},N=100",
                   _rerank(short_cos, (1 - b) * zq + b * ze, R.k_out),
                   time.perf_counter() - t0, "joint")
        rq = _ranks_of(short_cos, qe_i)
        re = _ranks_of(short_cos, h4_rank[100])
        for krrf in RRF_KS:
            t0 = time.perf_counter()
            R.emit("rrf_qe_h4", f"alpha={a:g},k_rrf={krrf},N=100",
                   _rerank(short_cos, 1 / (krrf + rq) + 1 / (krrf + re), R.k_out),
                   time.perf_counter() - t0, "joint")


def run_h1h2h4(R: Runner):
    run_h1h2(R)
    short_cos = R.cos_i[:, :100]
    h4_rank = {}
    for n in CE_NS:
        sl = short_cos[:, :n]
        sc, wce = R.h4(sl)
        h4_rank[n] = _rerank(sl, sc, R.k_out)
        t0 = time.perf_counter()
        R.emit("single_h4_on_cosine", f"N={n}", h4_rank[n],
               wce + time.perf_counter() - t0, "single")
    sc_cos, _ = R.h4(short_cos)
    if R.r_c is None:
        R.csls(0.50)
    for lam, a, n in product(CSLS_LAMS, QE_ALPHAS, CE_NS):
        csls_i, w1 = R.csls(lam)
        qe_i, _, w2 = R.qe_from_idx(csls_i, a)
        sl = qe_i[:, :n]
        sc, w3 = R.h4(sl)
        t0 = time.perf_counter()
        R.emit("csls_qe_h4", f"lambda={lam:g},alpha={a:g},N={n}",
               _rerank(sl, sc, R.k_out),
               w1 + w2 + w3 + time.perf_counter() - t0, "joint")
    lam, a = 0.50, 0.10
    csls_i, _ = R.csls(lam)
    qe_i, q2, _ = R.qe_from_idx(R.cos_i, a)
    for n in CE_NS:
        sl = csls_i[:, :n]
        sc, w = R.h4(sl)
        h4i = _rerank(sl, sc, R.k_out)
        qei, _, wq = R.qe_from_idx(h4i, a)
        R.emit("csls_h4_qe", f"lambda={lam:g},alpha={a:g},N={n}", qei, w + wq, "joint")
        c2, wc = R.csls_on_query(q2, lam)
        sl = c2[:, :n]
        sc, w = R.h4(sl)
        t0 = time.perf_counter()
        R.emit("qe_csls_h4", f"lambda={lam:g},alpha={a:g},N={n}",
               _rerank(sl, sc, R.k_out), wc + w + time.perf_counter() - t0, "joint")
        sl = qe_i[:, :n]
        sc, w = R.h4(sl)
        h4i = _rerank(sl, sc, R.k_out)[:, :n]
        scs = _csls_short_scores(q2, R.pool, R.r_c, lam, h4i, R.ctx.device, R.bq)
        t0 = time.perf_counter()
        R.emit("qe_h4_csls", f"lambda={lam:g},alpha={a:g},N={n}",
               _rerank(h4i, scs, R.k_out), w + time.perf_counter() - t0, "joint")
        sl = short_cos[:, :n]
        sc, w = R.h4(sl)
        h4i = _rerank(sl, sc, R.k_out)[:, :n]
        scs = _csls_short_scores(R.query, R.pool, R.r_c, lam, h4i, R.ctx.device, R.bq)
        csls_h4 = _rerank(h4i, scs, R.k_out)
        qei, _, wq = R.qe_from_idx(csls_h4, a)
        R.emit("h4_csls_qe", f"lambda={lam:g},alpha={a:g},N={n}", qei, w + wq, "joint")
        qei, q2b, wq = R.qe_from_idx(h4_rank[n], a)
        c3, wc = R.csls_on_query(q2b, lam)
        R.emit("h4_qe_csls", f"lambda={lam:g},alpha={a:g},N={n}", c3, wq + wc, "joint")
    extra_mixes = [("equal", (1 / 3, 1 / 3, 1 / 3))]
    for trip in ((0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5)):
        extra_mixes.append((f"w{trip}", trip))
    for lam, a in product(CSLS_LAMS, QE_ALPHAS):
        sc_c = _csls_short_scores(R.query, R.pool, R.r_c, lam, short_cos, R.ctx.device, R.bq)
        qe_ia, q2a, _ = R.qe_from_idx(R.cos_i, a)
        sc_q = _cos_short_scores(q2a, R.pool, short_cos, R.ctx.device, R.bq)
        zc, zq, ze = _zrow(sc_c), _zrow(sc_q), _zrow(sc_cos)
        mixes = extra_mixes if (lam == 0.50 and a == 0.10) else extra_mixes[:1]
        for name, (w1, w2, w3) in mixes:
            t0 = time.perf_counter()
            R.emit("zfusion_csls_qe_h4", f"lambda={lam:g},alpha={a:g},{name},N=100",
                   _rerank(short_cos, w1 * zc + w2 * zq + w3 * ze, R.k_out),
                   time.perf_counter() - t0, "joint")
        rc = _ranks_of(short_cos, R.csls(lam)[0])
        rq = _ranks_of(short_cos, qe_ia)
        re = _ranks_of(short_cos, h4_rank[100])
        for krrf in RRF_KS:
            t0 = time.perf_counter()
            R.emit("rrf_csls_qe_h4", f"lambda={lam:g},alpha={a:g},k_rrf={krrf},N=100",
                   _rerank(short_cos, 1 / (krrf + rc) + 1 / (krrf + rq) + 1 / (krrf + re), R.k_out),
                   time.perf_counter() - t0, "joint")


def _mark_beats(rows):
    best_atom, best_name = {}, {}
    for r in rows:
        if r["kind"] != "single":
            continue
        atom = r["combo"]
        d = float(r["dR@10"])
        if d > best_atom.get(atom, float("-inf")):
            best_atom[atom] = d
            best_name[atom] = f"{r['variant']}({r['params']})"
    for r in rows:
        d = float(r["dR@10"])
        if r["kind"] == "single":
            r["dR@10_best_single"] = d
            r["delta_vs_best_single"] = 0.0
            r["beats_best_single"] = 0
            r["beats_best_by_useful"] = 0
            r["best_single_desc"] = ""
            continue
        atoms = [a for a in r["combo"].split(";") if a]
        best = max((best_atom.get(a, float("-inf")) for a in atoms), default=float("-inf"))
        desc = "; ".join(
            f"{a}={best_name[a]}:{best_atom[a]:+.4f}" if a in best_atom else f"{a}=NA"
            for a in atoms)
        r["dR@10_best_single"] = best
        r["delta_vs_best_single"] = d - best
        r["beats_best_single"] = int(np.isfinite(best) and d > best + 1e-15)
        r["beats_best_by_useful"] = int(np.isfinite(best) and (d - best) >= C.USEFUL_DR10)
        r["best_single_desc"] = desc
    return best_atom


def run_item(ctx, seed, cid, log):
    cell = ctx.cells[cid]
    out_p = OUT_ROOT / "cells" / f"S{seed}_{cid}.csv"
    if out_p.is_file():
        log(f"[skip] {cid} s{seed} already written")
        return
    R = Runner(ctx, seed, cell, log)
    log(f"[cell] S{seed} {cid} phi={R.phi_set} baseR10={R.base[10]:.4f}")
    try:
        if R.phi_set == "h1;h2":
            run_h1h2(R)
        elif R.phi_set == "h1;h4":
            run_h1h4(R)
        elif R.phi_set == "h2;h4":
            run_h2h4(R)
        elif R.phi_set == "h1;h2;h4":
            run_h1h2h4(R)
        else:
            raise RuntimeError(R.phi_set)
        _mark_beats(R.rows)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fields = list(R.rows[0].keys())
        tmp = out_p.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(R.rows)
        tmp.replace(out_p)
        err = OUT_ROOT / "cells" / f"S{seed}_{cid}.err.txt"
        if err.is_file():
            err.unlink()
        nbeat = sum(int(r["beats_best_single"]) for r in R.rows)
        log(f"[done] {cid} s{seed} rows={len(R.rows)} joints_beat_single={nbeat}")
    finally:
        R.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-q", type=int, default=64)
    args = ap.parse_args(argv)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis.strip() == "3":
        raise SystemExit("refuse GPU3 (official CE fill)")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)

    def log(msg: str) -> None:
        print(f"{C.now()} {msg}", flush=True)

    items = list_work_items(args.seeds)
    mine = [it for i, it in enumerate(items) if i % args.n_shards == args.shard]
    log(f"[sweep] shard {args.shard}/{args.n_shards} n={len(mine)}/{len(items)}")
    for seed, cid, phi in mine:
        try:
            run_item(ctx, seed, cid, log)
        except Exception as exc:
            err = OUT_ROOT / "cells" / f"S{seed}_{cid}.err.txt"
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(f"{exc!r}\n")
            log(f"[FAIL] S{seed} {cid} {exc!r}")
    log("[sweep] shard finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
