
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
import ce_rerank
from ce_train import official_train_jsonl, official_val_jsonl

STAGING = C.ROOT / "runs" / "ce_training" / "h4_listwise"
WEIGHTS = C.ROOT / "runs" / "ce_training" / "weights"
UNION_TRAIN = C.MBEIR / "cand_pool/global/mbeir_union_train_cand_pool.jsonl"
UNION_VAL = C.MBEIR / "cand_pool/global/mbeir_union_val_cand_pool.jsonl"


TASK_ID = {
    "mscoco_task0": "0", "visualnews_task0": "0", "fashion200k_task0": "0",
    "webqa_task1": "1", "edis_task2": "2", "nights_task4": "4",
    "oven_task6": "6", "infoseek_task6": "6",
    "fashioniq_task7": "7", "cirr_task7": "7",
}
ALL_LOADS = list(TASK_ID)


class ListMixer(nn.Module):

    def __init__(self, dim: int, nhead: int = 8, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(dim, 1)

    def forward(self, q, c):
        
        n = c.shape[1]
        tok = self.in_proj(torch.cat([q.unsqueeze(1).expand(-1, n, -1), c], dim=-1))
        return self.out(self.encoder(tok)).squeeze(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", nargs="+", default=ALL_LOADS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-neg", type=int, default=7)
    ap.add_argument("--n-train-lists", type=int, default=8192)
    ap.add_argument("--n-val-lists", type=int, default=512)
    ap.add_argument("--img-batch", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--bank-cap", type=int, default=200000)
    args = ap.parse_args(argv)
    STAGING.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    for load in args.loads:
        _train_load(load, device, args)
    return 0


def _missing(x) -> bool:
    return x is None or str(x).strip() in ("", "None", "none")


def _slim(rec: dict) -> dict:
    return {
        "did": str(rec.get("did")),
        "txt": rec.get("txt") or rec.get("cand_txt") or "",
        "img_path": rec.get("img_path") or rec.get("cand_img_path"),
        "modality": rec.get("modality"),
    }


def _load_queries(path: Path, task_id: str | None) -> list:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task_id is not None and str(rec.get("task_id")) != str(task_id):
                continue
            rows.append(rec)
    return rows


def _collect_pool_simple(union_path: Path, pos_dids: set, prefixes: set, bank_cap: int):
    recs = {}
    banks = {p: [] for p in prefixes}
    with union_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            did = str(rec.get("did"))
            pref = did.split(":")[0]
            keep_pos = did in pos_dids
            keep_bank = pref in banks and len(banks[pref]) < bank_cap
            if not (keep_pos or keep_bank):
                continue
            recs[did] = _slim(rec)
            
            
            if keep_bank:
                banks[pref].append(did)
    return recs, banks


def _sample_lists(queries, recs, banks, n_lists, n_neg, rng):
    lists = []
    n_try = n_lists * 8
    for _ in range(n_try):
        if len(lists) >= n_lists:
            break
        q = queries[rng.randrange(len(queries))]
        pos = [str(d) for d in (q.get("pos_cand_list") or []) if str(d) in recs]
        if not pos:
            continue
        gold = pos[rng.randrange(len(pos))]
        pref = gold.split(":")[0]
        bank = banks.get(pref) or []
        pos_set = set(pos)
        negs = []
        if len(bank) < n_neg + 1:
            continue
        for _n in range(n_neg * 12):
            if len(negs) >= n_neg:
                break
            d = bank[rng.randrange(len(bank))]
            if d in pos_set or d in negs:
                continue
            negs.append(d)
        if len(negs) < n_neg:
            continue
        q_side = {
            "txt": q.get("query_txt") or "",
            "img_path": q.get("query_img_path"),
        }
        cands = [recs[gold], *[recs[d] for d in negs]]
        lists.append((q_side, cands))
    return lists


def _record_ok(rec: dict) -> bool:
    return (not _missing(rec.get("img_path"))) or (not _missing(rec.get("txt")))


class FrozenBlip:
    def __init__(self, device):
        dest = WEIGHTS / "blip_itm_base_coco"
        if not (dest / "config.json").is_file():
            raise FileNotFoundError(f"missing {dest}; download BLIP-ITM first")
        BlipForImageTextRetrieval, BlipProcessor = C.load_blip_classes()
        self.processor = BlipProcessor.from_pretrained(str(dest))
        self.model = BlipForImageTextRetrieval.from_pretrained(str(dest)).to(device).eval()
        self.device = device
        self.dim = int(self.model.vision_model.config.hidden_size)
        self.img_cache = {}
        self.txt_cache = {}

    def encode_records(self, recs: list, img_batch: int) -> torch.Tensor:
        self.prefill([recs], img_batch, log=None, tag="")
        return torch.stack([self.encode_one(rec) for rec in recs])

    def encode_one(self, rec: dict, img_batch: int = 1) -> torch.Tensor:
        del img_batch
        parts = []
        img = rec.get("img_path")
        txt = rec.get("txt") or rec.get("query_txt") or ""
        if not _missing(img):
            feat = self.img_cache.get(str(img))
            if feat is None:
                feat = self._image_batch([str(img)])[0]
                self.img_cache[str(img)] = feat
            parts.append(feat)
        if not _missing(txt):
            feat = self.txt_cache.get(str(txt))
            if feat is None:
                feat = self._text_batch([str(txt)])[0]
                self.txt_cache[str(txt)] = feat
            parts.append(feat)
        if not parts:
            return torch.zeros(self.dim, device=self.device)
        return torch.stack(parts).mean(0)

    @torch.no_grad()
    def _image_batch(self, rels: list) -> list:
        from PIL import Image
        imgs = []
        ok = []
        for rel in rels:
            try:
                imgs.append(Image.open(C.MBEIR / rel).convert("RGB"))
                ok.append(True)
            except Exception:
                imgs.append(None)
                ok.append(False)
        out = [torch.zeros(self.dim, device=self.device) for _ in rels]
        good_i = [i for i, f in enumerate(ok) if f]
        if not good_i:
            return out
        batch_img = [imgs[i] for i in good_i]
        inputs = self.processor(images=batch_img, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device)
        feat = self.model.vision_model(pixel_values=pixel).pooler_output.detach()
        for j, i in enumerate(good_i):
            out[i] = feat[j]
        return out

    @torch.no_grad()
    def _text_batch(self, texts: list) -> list:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True,
                                truncation=True, max_length=40)
        input_ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        hidden = self.model.text_encoder(input_ids=input_ids, attention_mask=mask)[0]
        feat = hidden[:, 0].detach()
        return [feat[i] for i in range(feat.shape[0])]

    def prefill(self, list_groups, img_batch: int, log, tag: str):
        imgs, texts = [], []
        for lists in list_groups:
            for qrec, crecs in lists:
                for rec in (qrec, *crecs):
                    img = rec.get("img_path")
                    txt = rec.get("txt") or rec.get("query_txt") or ""
                    if not _missing(img) and str(img) not in self.img_cache:
                        imgs.append(str(img))
                    if not _missing(txt) and str(txt) not in self.txt_cache:
                        texts.append(str(txt))
        imgs = list(dict.fromkeys(imgs))
        texts = list(dict.fromkeys(texts))
        if log:
            log(f"  {tag} unique images={len(imgs)} texts={len(texts)}")
        for s in range(0, len(imgs), img_batch):
            chunk = imgs[s:s + img_batch]
            feats = self._image_batch(chunk)
            for rel, feat in zip(chunk, feats):
                self.img_cache[rel] = feat
            if log and (s // img_batch) % 20 == 0:
                log(f"  {tag} images {min(s+img_batch, len(imgs))}/{len(imgs)}")
        tb = max(img_batch, 32)
        for s in range(0, len(texts), tb):
            chunk = texts[s:s + tb]
            feats = self._text_batch(chunk)
            for t, feat in zip(chunk, feats):
                self.txt_cache[t] = feat

    def encode_lists(self, lists, img_batch: int, log, tag: str):
        self.prefill([lists], img_batch, log, tag)
        q_feats, c_feats = [], []
        n_drop = 0
        for qrec, crecs in lists:
            try:
                q = self.encode_one(qrec)
                cs = torch.stack([self.encode_one(c) for c in crecs])
            except Exception:
                n_drop += 1
                continue
            q_feats.append(q)
            c_feats.append(cs)
        if not q_feats:
            raise RuntimeError(f"{tag}: no lists encoded")
        if log:
            log(f"  {tag} stacked lists={len(q_feats)} drop={n_drop}")
        return torch.stack(q_feats), torch.stack(c_feats), n_drop


def _train_load(load: str, device, args) -> None:
    out = STAGING / f"ce_list_local__{load}.pt"
    if out.is_file() and not args.overwrite:
        print(f"skip existing {out}", flush=True)
        return
    print(f"=== H4 listwise {load} device={device} ===", flush=True)
    print("  neg=same-prefix pool except this query's pos_cand_list", flush=True)
    train_path = official_train_jsonl(load)
    val_path = official_val_jsonl(load)
    print(f"  train_jsonl={train_path}", flush=True)
    print(f"  val_jsonl={val_path}", flush=True)
    tid = TASK_ID[load]
    train_q = _load_queries(train_path, tid)
    val_q = _load_queries(val_path, None)
    print(f"  n_train_q={len(train_q)} n_val_q={len(val_q)} task_id={tid}", flush=True)
    if not train_q:
        raise RuntimeError(f"{load}: zero train queries after task filter")

    def pos_dids(rows):
        s = set()
        for q in rows:
            for d in q.get("pos_cand_list") or []:
                s.add(str(d))
        return s

    train_pos = pos_dids(train_q)
    val_pos = pos_dids(val_q)
    prefixes = {d.split(":")[0] for d in train_pos}
    print(f"  train_pos={len(train_pos)} prefixes={sorted(prefixes)}", flush=True)
    print("  scanning union train/val pools...", flush=True)
    recs_tr, banks_tr = _collect_pool_simple(UNION_TRAIN, train_pos, prefixes, args.bank_cap)
    recs_va, banks_va = _collect_pool_simple(UNION_VAL, val_pos, prefixes, min(args.bank_cap, 80000))
    hit_tr = sum(d in recs_tr for d in train_pos)
    hit_va = sum(d in recs_va for d in val_pos)
    print(f"  train pool recs={len(recs_tr)} pos_hit={hit_tr}/{len(train_pos)} "
          f"bank={[ (p, len(banks_tr[p])) for p in sorted(banks_tr) ]}", flush=True)
    print(f"  val pool recs={len(recs_va)} pos_hit={hit_va}/{len(val_pos)} "
          f"bank={[ (p, len(banks_va[p])) for p in sorted(banks_va) ]}", flush=True)
    if hit_tr < 0.5 * max(len(train_pos), 1):
        raise RuntimeError(f"{load}: too few positives in union train pool")

    rng_tr = random.Random(0)
    rng_va = random.Random(1)
    train_lists = _sample_lists(train_q, recs_tr, banks_tr, args.n_train_lists, args.n_neg, rng_tr)
    val_lists = _sample_lists(val_q, recs_va, banks_va, args.n_val_lists, args.n_neg, rng_va)
    print(f"  lists train={len(train_lists)} val={len(val_lists)}", flush=True)
    if len(train_lists) < 256:
        raise RuntimeError(f"{load}: only {len(train_lists)} train lists")
    if len(val_lists) < 64:
        print(f"  {load}: few val lists ({len(val_lists)}); hold out 256 from train", flush=True)
        val_lists = train_lists[-256:]
        train_lists = train_lists[:-256]

    blip = FrozenBlip(device)
    print("  encoding train lists...", flush=True)
    q_tr, c_tr, drop_tr = blip.encode_lists(train_lists, args.img_batch, print, load)
    print(f"  train tensor {tuple(q_tr.shape)} {tuple(c_tr.shape)} drop={drop_tr}", flush=True)
    print("  encoding val lists...", flush=True)
    q_va, c_va, drop_va = blip.encode_lists(val_lists, args.img_batch, print, load + ".val")
    print(f"  val tensor {tuple(q_va.shape)} {tuple(c_va.shape)} drop={drop_va}", flush=True)
    del blip
    torch.cuda.empty_cache()

    mixer = ListMixer(q_tr.shape[-1]).to(device)
    opt = torch.optim.AdamW(mixer.parameters(), lr=1e-3)
    best, best_state, wait = 1e9, None, 0
    n_tr = q_tr.shape[0]
    steps = args.steps
    batch = min(args.batch, n_tr)
    rng = random.Random(0)
    for step in range(1, steps + 1):
        idx = [rng.randrange(n_tr) for _ in range(batch)]
        q = q_tr[idx]
        c = c_tr[idx]
        tgt = torch.zeros(batch, dtype=torch.long, device=device)
        loss = F.cross_entropy(mixer(q, c), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps:
            mixer.eval()
            with torch.no_grad():
                vloss, n = 0.0, 0
                for s in range(0, q_va.shape[0], batch):
                    e = min(s + batch, q_va.shape[0])
                    logits = mixer(q_va[s:e], c_va[s:e])
                    vt = torch.zeros(e - s, dtype=torch.long, device=device)
                    vloss += float(F.cross_entropy(logits, vt))
                    n += 1
            mixer.train()
            vmean = vloss / max(n, 1)
            print(f"  {load} step {step} train={float(loss):.4f} val_ce={vmean:.4f}", flush=True)
            if vmean < best - 1e-4:
                best, best_state, wait = vmean, {k: v.detach().cpu().clone()
                                                for k, v in mixer.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 8:
                    print(f"  {load} early stop at {step}", flush=True)
                    break
    if best_state is not None:
        mixer.load_state_dict(best_state)
    blob = {
        "kind": "list_mixer",
        "dim": int(q_tr.shape[-1]),
        "n_neg": int(args.n_neg),
        "state": {k: v.detach().cpu() for k, v in mixer.state_dict().items()},
        "load": load,
        "val_ce": float(best),
        "n_train_lists": int(q_tr.shape[0]),
        "n_val_lists": int(q_va.shape[0]),
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "note": ("H4 true listwise: Transformer over 1 gold + 7 negs; "
                 "negatives = same-prefix pool except this query's golds; "
                 "frozen BLIP-ITM vision+text; official train/val only"),
    }
    tmp = out.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.replace(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (STAGING / f"ce_list_local__{load}.sha256").write_text(sha + "\n")
    print(f"wrote {out} sha={sha[:12]} val_ce={best:.4f}", flush=True)
    _append_md(load, train_path, val_path, sha, best)
    del mixer, q_tr, c_tr, q_va, c_va
    torch.cuda.empty_cache()


def _append_md(load, train_path, val_path, sha, val_ce) -> None:
    path = STAGING / "TRAINING.md"
    header = (
        "# H4 listwise CE (plan 8.5.4)\n\n"
        "No open-source multimodal listwise weights. Frozen official "
        "`Salesforce/blip-itm-base-coco` vision+text; train ListMixer "
        "(Transformer over the candidate list, list cross-entropy). "
        "Official train + val only. Negatives: same-prefix candidates except "
        "this query's pos_cand_list (other queries' golds allowed). "
        "Staging directory: not installed over the GPU3 fill checkpoints "
        "until that job finishes.\n\n"
    )
    if not path.is_file():
        path.write_text(header)
    with path.open("a") as fh:
        fh.write(
            f"- `{load}` list_mixer; train `{train_path.name}` val `{val_path.name}`; "
            f"sha={sha[:16]} val_ce={val_ce:.4f} written={C.now()}\n"
        )


def load_list_mixer_blob(dataset: str, path: Path | None = None) -> dict:
    ckpt = path or (STAGING / f"ce_list_local__{dataset}.pt")
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if blob.get("kind") != "list_mixer":
        raise RuntimeError(
            f"{ckpt} kind={blob.get('kind')!r}; H4 must be list_mixer, "
            "not PairHead / ITM / ce_pair_local")
    return blob


def scores_on_shortlist(dataset, short_idx, ctx, blob=None):
    blob = blob or load_list_mixer_blob(dataset)
    pack = getattr(ctx, "_h4_list_pack", None)
    if pack is None:
        pack = {"blip": FrozenBlip(ctx.device), "mixers": {}}
        ctx._h4_list_pack = pack
    blip = pack["blip"]
    mixers = pack.setdefault("mixers", {})
    if dataset not in mixers:
        mixer = ListMixer(int(blob["dim"])).to(ctx.device)
        mixer.load_state_dict(blob["state"])
        mixer.eval()
        mixers[dataset] = mixer
    mixer = mixers[dataset]
    queries = ce_rerank._cached_jsonl(ctx, "q", dataset, C.MBEIR / ce_rerank.QUERY_JSONL[dataset])
    cands = ce_rerank._cached_jsonl(ctx, "c", dataset, C.MBEIR / ce_rerank.CAND_JSONL[dataset])
    eval_idx = getattr(ctx, "_eval_idx", None)
    if eval_idx is None:
        return None, "error:missing_eval_idx"
    nq, n_short = short_idx.shape
    q_sides, crecs_all = [], []
    for qi in range(nq):
        qrec = queries[int(eval_idx[qi])]
        q_sides.append({"txt": qrec.get("query_txt") or "",
                        "img_path": qrec.get("query_img_path")})
        crecs_all.append([_slim(cands[int(cj)]) for cj in short_idx[qi].tolist()])
    blip.prefill([list(zip(q_sides, crecs_all))], 64, None, "h4_infer")
    scores = np.full((nq, n_short), -1e9, dtype=np.float64)
    q_feats, c_feats, ok = [], [], []
    with torch.no_grad():
        for qi in range(nq):
            try:
                q_feats.append(blip.encode_one(q_sides[qi]))
                c_feats.append(torch.stack([blip.encode_one(c) for c in crecs_all[qi]]))
                ok.append(qi)
            except Exception:
                continue
        if not ok:
            return None, "error:h4_encode_fail_all"
        q_t = torch.stack(q_feats)
        c_t = torch.stack(c_feats)
        bs = 16
        outs = []
        for s in range(0, q_t.shape[0], bs):
            outs.append(mixer(q_t[s:s + bs], c_t[s:s + bs]).float().cpu())
        scored = torch.cat(outs, dim=0).numpy()
        for row, qi in enumerate(ok):
            scores[qi] = scored[row]
    n_bad = int((scores <= -1e8).all(axis=1).sum())
    if n_bad > max(5, int(0.05 * nq)):
        return None, f"error:h4_encode_fail_{n_bad}/{nq}"
    return scores, "ok"


def score_top100(method, dataset, top100, blob, ctx):
    del method
    dest = WEIGHTS / "blip_itm_base_coco"
    device = ctx.device
    pack = getattr(ctx, "_h4_list_pack", None)
    if pack is None:
        blip = FrozenBlip(device)
        pack = {"blip": blip}
        ctx._h4_list_pack = pack
    blip = pack["blip"]
    mixer = ListMixer(int(blob["dim"])).to(device)
    mixer.load_state_dict(blob["state"])
    mixer.eval()
    queries = ce_rerank._cached_jsonl(ctx, "q", dataset, C.MBEIR / ce_rerank.QUERY_JSONL[dataset])
    cands = ce_rerank._cached_jsonl(ctx, "c", dataset, C.MBEIR / ce_rerank.CAND_JSONL[dataset])
    eval_idx = getattr(ctx, "_eval_idx", None)
    if eval_idx is None:
        return None, "error:missing_eval_idx"
    new_idx = np.array(top100, copy=True)
    with torch.no_grad():
        for qi in range(top100.shape[0]):
            qrec = queries[int(eval_idx[qi])]
            q_side = {"txt": qrec.get("query_txt") or "", "img_path": qrec.get("query_img_path")}
            ids, crecs = [], []
            for cj in top100[qi].tolist():
                crecs.append(_slim(cands[int(cj)]))
                ids.append(int(cj))
            try:
                qf = blip.encode_one(q_side).unsqueeze(0)
                cf = blip.encode_records(crecs, 32).unsqueeze(0)
                sc = mixer(qf, cf)[0].cpu().numpy()
            except Exception:
                continue
            order = np.argsort(-sc)
            ranked = [ids[i] for i in order]
            rest = [c for c in top100[qi].tolist() if c not in ranked]
            new_idx[qi] = (ranked + rest)[:top100.shape[1]]
    return new_idx, "ok"


if __name__ == "__main__":
    raise SystemExit(main())
