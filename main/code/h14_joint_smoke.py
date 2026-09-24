

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch

import common as C
import corrections
import ce_rerank
import ce_train


SMOKE_CELLS = [
    "clip_sf_large__nights_task4",      
    "blip2_vitL__oven_task6",           
    "clip_vitb32__infoseek_task6",      
]
KS = (1, 5, 10)
CSLS_LAMBDAS = (0.25, 0.50, 1.00)
FUSION_BETAS = (0.25, 0.50, 0.75)
RRF_KS = (10, 60)
SHORT_NS = (50, 100)


def _recall(top_i, positives):
    return {k: float(C.recall_hits(top_i, positives, k).mean()) for k in KS}


def _ce_method(dataset: str) -> str:
    if dataset in C.LOCAL_CE_LOADS:
        return "ce_list_local"
    return "ce_pair_blip_itm"


def _csls_short_scores(query, pool, r_c, lam, short_idx, device, batch_q):
    p_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device)
    q_t = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    r_t = torch.from_numpy(np.ascontiguousarray(r_c)).to(device)
    nq, n_short = short_idx.shape
    out = np.empty((nq, n_short), dtype=np.float64)
    for s in range(0, nq, batch_q):
        e = min(s + batch_q, nq)
        sl = torch.from_numpy(np.ascontiguousarray(short_idx[s:e])).to(device)
        gathered = p_t[sl]
        cos = (q_t[s:e].unsqueeze(1) * gathered).sum(dim=-1)
        rc = r_t[sl]
        out[s:e] = (cos - float(lam) * rc).detach().cpu().numpy()
    return out


def _zrow(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.maximum(sd, 1e-8)
    return (x - mu) / sd


def _rerank_from_scores(short_idx, scores, k_out):
    nq, n_short = short_idx.shape
    new_i = np.empty((nq, k_out), dtype=np.int64)
    for qi in range(nq):
        order = np.argsort(-scores[qi], kind="stable")
        ranked = short_idx[qi, order]
        m = min(k_out, ranked.size)
        new_i[qi, :m] = ranked[:m]
        if m < k_out:
            new_i[qi, m:] = ranked[m - 1]
    return new_i


def _rrf(rank_a, rank_b, k_rrf):
    return 1.0 / (k_rrf + rank_a) + 1.0 / (k_rrf + rank_b)


def _ranks_of(short_idx, ordered_idx):
    nq, n_short = short_idx.shape
    out = np.empty((nq, n_short), dtype=np.float64)
    for qi in range(nq):
        pos = {int(v): r for r, v in enumerate(ordered_idx[qi].tolist(), start=1)}
        default = ordered_idx.shape[1] + 1
        out[qi] = [pos.get(int(v), default) for v in short_idx[qi].tolist()]
    return out


def ce_scores_on_shortlist(method, dataset, short_idx, ctx):
    if method == "ce_pair_blip_itm":
        return _blip_scores(dataset, short_idx, ctx)
    return _local_scores(method, dataset, short_idx, ctx)


def _blip_scores(dataset, short_idx, ctx):
    import torch
    from PIL import Image

    pack = ce_rerank._cached_blip(ctx)
    model, processor, device = pack["model"], pack["processor"], pack["device"]
    eval_idx = ctx._eval_idx
    queries = ce_rerank._cached_jsonl(ctx, "q", dataset, C.MBEIR / ce_rerank.QUERY_JSONL[dataset])
    cands = ce_rerank._cached_jsonl(ctx, "c", dataset, C.MBEIR / ce_rerank.CAND_JSONL[dataset])
    nq, n_short = short_idx.shape
    scores = np.full((nq, n_short), -1e9, dtype=np.float64)
    with torch.no_grad():
        for qi in range(nq):
            qrec = queries[int(eval_idx[qi])]
            texts, images, keep_j = [], [], []
            for j, cj in enumerate(short_idx[qi].tolist()):
                crec = cands[int(cj)]
                text, img_path = ce_rerank._pair_text_image(qrec, crec, dataset)
                if img_path is None or not img_path.is_file():
                    continue
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                    texts.append(text.strip() or ".")
                    keep_j.append(j)
                except Exception:
                    continue
            if not keep_j:
                continue
            scs = []
            bs = 32
            for s in range(0, len(keep_j), bs):
                inputs = processor(
                    images=images[s:s + bs], text=texts[s:s + bs],
                    return_tensors="pt", padding=True, truncation=True,
                    max_length=40,
                ).to(device)
                out = model(**inputs)
                logit = out.itm_score[:, 1] if out.itm_score.dim() == 2 else out.itm_score.reshape(-1)
                scs.append(logit.float().cpu().numpy())
            sc = np.concatenate(scs)
            for j, val in zip(keep_j, sc):
                scores[qi, j] = float(val)
    return scores, "ok"


def _local_scores(method, dataset, short_idx, ctx):
    ckpt = C.ROOT / "runs" / "ce_training" / "weights" / f"{method}__{dataset}.pt"
    if not ckpt.is_file():
        return None, "unevaluable_no_weights"
    from PIL import Image

    blob = torch.load(ckpt, map_location="cpu")
    dest = C.ROOT / "runs" / "ce_training" / "weights" / "blip_itm_base_coco"
    device = ctx.device
    pack = getattr(ctx, "_local_ce_pack", None)
    if pack is None:
        BlipForImageTextRetrieval, BlipProcessor = C.load_blip_classes()
        processor = BlipProcessor.from_pretrained(str(dest))
        vision = BlipForImageTextRetrieval.from_pretrained(str(dest)).vision_model.to(device).eval()
        pack = {"processor": processor, "vision": vision}
        ctx._local_ce_pack = pack
    processor, vision = pack["processor"], pack["vision"]
    head = ce_train.PairHead(int(blob["dim"])).to(device)
    head.load_state_dict(blob["state"])
    head.eval()
    queries = ce_rerank._cached_jsonl(ctx, "q", dataset, C.MBEIR / ce_rerank.QUERY_JSONL[dataset])
    cands = ce_rerank._cached_jsonl(ctx, "c", dataset, C.MBEIR / ce_rerank.CAND_JSONL[dataset])
    eval_idx = ctx._eval_idx
    cache = {}

    def feat(rel):
        if rel in cache:
            return cache[rel]
        img = Image.open(C.MBEIR / rel).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            cache[rel] = vision(**inputs).pooler_output[0]
        return cache[rel]

    nq, n_short = short_idx.shape
    scores = np.full((nq, n_short), -1e9, dtype=np.float64)
    with torch.no_grad():
        for qi in range(nq):
            qrec = queries[int(eval_idx[qi])]
            qimg = qrec.get("query_img_path")
            if not qimg:
                continue
            try:
                qf = feat(qimg)
            except Exception:
                continue
            ids_j, vecs = [], []
            for j, cj in enumerate(short_idx[qi].tolist()):
                cimg = cands[int(cj)].get("img_path")
                if not cimg:
                    continue
                try:
                    vecs.append(feat(cimg))
                    ids_j.append(j)
                except Exception:
                    continue
            if not ids_j:
                continue
            a = qf.unsqueeze(0).expand(len(ids_j), -1)
            b = torch.stack(vecs)
            sc = head(a, b).cpu().numpy()
            for j, val in zip(ids_j, sc):
                scores[qi, j] = float(val)
    return scores, "ok"


def emit(rows, *, seed, cell, phi, ce_m, variant, params, base, rec, wall, note=""):
    d = {k: rec[k] - base[k] for k in KS}
    row = {
        "seed": seed, "encoder": cell.encoder, "dataset": cell.dataset,
        "phi_set": ";".join(phi) if phi else "empty",
        "ce_method": ce_m, "variant": variant, "params": params,
        "R@1_base": base[1], "R@5_base": base[5], "R@10_base": base[10],
        "R@1": rec[1], "R@5": rec[5], "R@10": rec[10],
        "dR@1": d[1], "dR@5": d[5], "dR@10": d[10],
        "useful": int(d[10] >= C.USEFUL_DR10),
        "wall_s": round(wall, 3), "note": note,
        "scope": "exploratory_h14_joint_smoke",
    }
    rows.append(row)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cells", nargs="+", default=SMOKE_CELLS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-q", type=int, default=64)
    args = ap.parse_args(argv)
    ctx = C.build_context(batch_q=args.batch_q, device_str=args.device)

    def log(msg: str) -> None:
        print(f"{C.now()} {msg}", flush=True)

    rows = []
    k_out = max(KS + (C.RERANK_N,))
    for cid in args.cells:
        cell = ctx.cells[cid]
        diag = C.load_sealed_diagnosis(ctx, args.seed, cid)
        phi = list(diag.get("phi") or [])
        seg = C.repartition(cell, args.seed)
        labeled = C.load_eval_labeled(cell, seg)
        ctx._eval_idx = labeled["eval_idx"]
        query, pool, positives, ref_q = (
            labeled["eval_q"], labeled["pool"], labeled["positives"], labeled["ref_q"])
        bq = C._batch_for(cell.dim, cell.n_pool, ctx.batch_q)
        ce_m = _ce_method(cell.dataset)
        params = {
            "h1_k": 10, "csls_k": 10, "qe_k": 10, "qe_power": 3.0,
            "csls_reference_queries": ref_q,
        }
        t0 = time.perf_counter()
        cos_v, cos_i, st = corrections.search(
            "cosine", query, pool, k_out, params=params, device=ctx.device, batch_q=bq)
        if st != "ok":
            raise RuntimeError(st)
        base = _recall(cos_i, positives)
        log(f"[h14] {cid} phi={';'.join(phi)} ce={ce_m} "
            f"base R1/5/10={base[1]:.3f}/{base[5]:.3f}/{base[10]:.3f} "
            f"cosine {time.perf_counter()-t0:.1f}s")

        csls_idx = {}
        r_c = None
        for lam in CSLS_LAMBDAS:
            t1 = time.perf_counter()
            p = dict(params)
            name = {0.25: "csls_p025", 0.50: "csls_p050", 1.00: "csls_p100"}[lam]
            if lam == 0.50:
                top_v, top_i, st = corrections.search(
                    "csls_p050", query, pool, k_out, params=p,
                    cosine_top_idx=cos_i, cosine_top_val=cos_v,
                    device=ctx.device, batch_q=bq)
            else:
                top_v, top_i, st = corrections._csls(
                    query, pool, k_out, p, ctx.device, bq,
                    cosine_top_val=cos_v, penalty_lambda=lam)
            if st != "ok":
                raise RuntimeError(st)
            rec = _recall(top_i, positives)
            csls_idx[lam] = top_i
            if r_c is None:
                r_c = p["_csls_r_c_cache"][next(iter(p["_csls_r_c_cache"]))]
            emit(rows, seed=args.seed, cell=cell, phi=phi, ce_m=ce_m,
                 variant="csls_full", params=f"lambda={lam}",
                 base=base, rec=rec, wall=time.perf_counter() - t1)
            log(f"  csls λ={lam:g} dR1/5/10={rec[1]-base[1]:+.4f}/{rec[5]-base[5]:+.4f}/{rec[10]-base[10]:+.4f}")

        t1 = time.perf_counter()
        rec = _recall(csls_idx[0.50][:, :100], positives)  
        short_cos = cos_i[:, :100]
        rec_short = _recall(
            _rerank_from_scores(
                short_cos,
                _csls_short_scores(query, pool, r_c, 0.50, short_cos, ctx.device, bq),
                k_out),
            positives)
        emit(rows, seed=args.seed, cell=cell, phi=phi, ce_m=ce_m,
             variant="csls_on_cosine_shortlist", params="lambda=0.5,N=100",
             base=base, rec=rec_short, wall=time.perf_counter() - t1,
             note="CE→CSLS set-equivalent; CE unused")
        log(f"  csls_on_cosine_100 dR10={rec_short[10]-base[10]:+.4f}")

        t_ce = time.perf_counter()
        ce_cos, st = ce_scores_on_shortlist(ce_m, cell.dataset, short_cos, ctx)
        if st != "ok":
            log(f"  CE cosine-shortlist {st}; skip CE variants")
            del query, pool
            continue
        log(f"  CE scores on cosine top-100 {time.perf_counter()-t_ce:.1f}s")

        for n_short in SHORT_NS:
            t1 = time.perf_counter()
            sl = short_cos[:, :n_short]
            sc = ce_cos[:, :n_short]
            rec = _recall(_rerank_from_scores(sl, sc, k_out), positives)
            emit(rows, seed=args.seed, cell=cell, phi=phi, ce_m=ce_m,
                 variant="ce_on_cosine", params=f"N={n_short}",
                 base=base, rec=rec, wall=time.perf_counter() - t1)
            log(f"  ce_on_cosine N={n_short} dR10={rec[10]-base[10]:+.4f}")

        t_ce2 = time.perf_counter()
        csls100 = csls_idx[0.50][:, :100]
        ce_csls, st = ce_scores_on_shortlist(ce_m, cell.dataset, csls100, ctx)
        if st != "ok":
            log(f"  CE on CSLS shortlist {st}")
            ce_csls = None
        else:
            log(f"  CE scores on CSLS top-100 {time.perf_counter()-t_ce2:.1f}s")
            for n_short in SHORT_NS:
                t1 = time.perf_counter()
                sl = csls100[:, :n_short]
                sc = ce_csls[:, :n_short]
                rec = _recall(_rerank_from_scores(sl, sc, k_out), positives)
                emit(rows, seed=args.seed, cell=cell, phi=phi, ce_m=ce_m,
                     variant="ce_on_csls", params=f"lambda=0.5,N={n_short}",
                     base=base, rec=rec, wall=time.perf_counter() - t1,
                     note="sequential CSLS→CE")
                log(f"  ce_on_csls N={n_short} dR10={rec[10]-base[10]:+.4f}")

        csls_sc_cos = _csls_short_scores(query, pool, r_c, 0.50, short_cos, ctx.device, bq)
        z_c = _zrow(csls_sc_cos)
        z_e = _zrow(ce_cos)
        for beta in FUSION_BETAS:
            t1 = time.perf_counter()
            sc = (1.0 - beta) * z_c + beta * z_e
            rec = _recall(_rerank_from_scores(short_cos, sc, k_out), positives)
            emit(rows, seed=args.seed, cell=cell, phi=phi, ce_m=ce_m,
                 variant="zfusion_csls_ce", params=f"lambda=0.5,beta={beta},N=100",
                 base=base, rec=rec, wall=time.perf_counter() - t1)
            log(f"  zfusion β={beta:g} dR10={rec[10]-base[10]:+.4f}")

        rank_csls = _ranks_of(short_cos, csls_idx[0.50])
        ce_order = _rerank_from_scores(short_cos, ce_cos, 100)
        rank_ce = _ranks_of(short_cos, ce_order)
        for k_rrf in RRF_KS:
            t1 = time.perf_counter()
            sc = _rrf(rank_csls, rank_ce, k_rrf)
            rec = _recall(_rerank_from_scores(short_cos, sc, k_out), positives)
            emit(rows, seed=args.seed, cell=cell, phi=phi, ce_m=ce_m,
                 variant="rrf_csls_ce", params=f"k_rrf={k_rrf},N=100",
                 base=base, rec=rec, wall=time.perf_counter() - t1)
            log(f"  rrf k={k_rrf} dR10={rec[10]-base[10]:+.4f}")

        del query, pool
        torch.cuda.empty_cache()

    out = ctx.runs / f"S{args.seed}" / "h14_joint_smoke.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f"[h14] wrote {out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
