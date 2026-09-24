
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

WEIGHTS = C.ROOT / "runs" / "ce_training" / "weights"
LOCAL_LOADS = ["nights_task4", "fashioniq_task7", "cirr_task7"]
HF_MODEL = "Salesforce/blip-itm-base-coco"


def official_train_jsonl(load: str) -> Path:
    stem = load.split("_task")[0]
    p = C.MBEIR / f"query/train/mbeir_{stem}_train.jsonl"
    if p.is_file():
        return p
    alt = C.MBEIR / f"query/train/mbeir_{load}_train.jsonl"
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"no official train jsonl for {load}: {p} or {alt}")


def official_val_jsonl(load: str) -> Path:
    p = C.MBEIR / f"query/val/mbeir_{load}_val.jsonl"
    if p.is_file():
        return p
    stem = load.split("_task")[0]
    alt = C.MBEIR / f"query/val/mbeir_{stem}_val.jsonl"
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"no official val jsonl for {load}: {p} or {alt}")


class PairHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=-1)).squeeze(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train-local", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if args.download:
        _download()
    if args.train_local:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        for load in LOCAL_LOADS:
            _train_load(load, device, args.steps, args.batch)
    return 0


def _download() -> None:
    BlipForImageTextRetrieval, BlipProcessor = C.load_blip_classes()
    dest = WEIGHTS / "blip_itm_base_coco"
    if (dest / "config.json").is_file():
        print(f"already have {dest}", flush=True)
        return
    print(f"downloading {HF_MODEL} -> {dest}", flush=True)
    proc = BlipProcessor.from_pretrained(HF_MODEL)
    model = BlipForImageTextRetrieval.from_pretrained(HF_MODEL)
    dest.mkdir(parents=True, exist_ok=True)
    proc.save_pretrained(dest)
    model.save_pretrained(dest)
    blob = b"".join(p.read_bytes() for p in sorted(dest.glob("*")) if p.is_file() and p.stat().st_size < 50_000_000)
    (WEIGHTS / "blip_itm_base_coco.sha256").write_text(
        hashlib.sha256(blob).hexdigest() + "\n")
    print("download done", flush=True)


def _train_load(load: str, device, steps: int, batch: int) -> None:
    from PIL import Image
    BlipForImageTextRetrieval, BlipProcessor = C.load_blip_classes()

    dest = WEIGHTS / "blip_itm_base_coco"
    if not (dest / "config.json").is_file():
        raise FileNotFoundError("run --download first")
    out_pair = WEIGHTS / f"ce_pair_local__{load}.pt"
    out_list = WEIGHTS / f"ce_list_local__{load}.pt"
    if out_pair.is_file() and out_list.is_file():
        print(f"skip trained {load}", flush=True)
        return
    print(f"train {load}", flush=True)
    processor = BlipProcessor.from_pretrained(str(dest))
    vision = BlipForImageTextRetrieval.from_pretrained(str(dest)).vision_model.to(device).eval()
    train_path = official_train_jsonl(load)
    val_path = official_val_jsonl(load)
    print(f"  train_jsonl={train_path}", flush=True)
    print(f"  val_jsonl={val_path}", flush=True)
    train_q = _jsonl(train_path)
    val_q = _jsonl(val_path)
    cand_path = C.MBEIR / ce_rerank.CAND_JSONL[load]
    cands = _jsonl(cand_path)
    did_to_i = {}
    for i, c in enumerate(cands):
        did_to_i[str(c.get("did"))] = i
    
    feat_cache = {}

    def embed_path(rel):
        rel = str(rel)
        if rel in feat_cache:
            return feat_cache[rel]
        path = C.MBEIR / rel
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = vision(**inputs).pooler_output[0].detach().cpu()
        feat_cache[rel] = feat
        if len(feat_cache) > 20000:
            feat_cache.pop(next(iter(feat_cache)))
        return feat

    def pair_of(qrec, cand_i):
        qimg = qrec.get("query_img_path")
        cimg = cands[cand_i].get("img_path")
        if not qimg or not cimg:
            return None
        try:
            return embed_path(qimg), embed_path(cimg)
        except Exception:
            return None

    def sample_batch(queries, n, rng):
        xs, ys = [], []
        for _ in range(n * 4):
            if len(xs) >= n:
                break
            q = queries[rng.randrange(len(queries))]
            pos_ids = [did_to_i[str(d)] for d in (q.get("pos_cand_list") or []) if str(d) in did_to_i]
            if not pos_ids:
                continue
            if rng.random() < 0.5:
                cand_i, y = pos_ids[rng.randrange(len(pos_ids))], 1.0
            else:
                cand_i, y = rng.randrange(len(cands)), 0.0
            got = pair_of(q, cand_i)
            if got is None:
                continue
            xs.append(got)
            ys.append(y)
        if not xs:
            return None
        a = torch.stack([p[0] for p in xs]).to(device)
        b = torch.stack([p[1] for p in xs]).to(device)
        y = torch.tensor(ys, dtype=torch.float32, device=device)
        return a, b, y

    def sample_list_batch(queries, n_neg, rng):
        for _ in range(40):
            q = queries[rng.randrange(len(queries))]
            pos_ids = [did_to_i[str(d)] for d in (q.get("pos_cand_list") or []) if str(d) in did_to_i]
            if not pos_ids:
                continue
            pos_i = pos_ids[rng.randrange(len(pos_ids))]
            negs = []
            for _n in range(n_neg * 8):
                if len(negs) >= n_neg:
                    break
                j = rng.randrange(len(cands))
                if j in pos_ids or j in negs:
                    continue
                negs.append(j)
            if len(negs) < n_neg:
                continue
            got = [pair_of(q, i) for i in [pos_i, *negs]]
            if any(g is None for g in got):
                continue
            a = torch.stack([p[0] for p in got]).to(device)
            b = torch.stack([p[1] for p in got]).to(device)
            tgt = torch.tensor([0], dtype=torch.long, device=device)
            return a, b, tgt
        return None

    dim = int(vision.config.hidden_size)
    head = PairHead(dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    rng = random.Random(0)
    best, best_state, wait = 1e9, None, 0
    n_skip = 0
    for step in range(1, steps + 1):
        batch_t = sample_batch(train_q, batch, rng)
        if batch_t is None:
            n_skip += 1
            if n_skip >= 200 and step == n_skip:
                raise RuntimeError(f"{load}: 200 consecutive empty train batches; "
                                   "check pos_cand_list vs cand did")
            continue
        n_skip = 0
        a, b, y = batch_t
        loss = F.binary_cross_entropy_with_logits(head(a, b), y)
        list_t = sample_list_batch(train_q, 7, rng)
        if list_t is not None:
            la, lb, tgt = list_t
            loss = loss + 0.5 * F.cross_entropy(head(la, lb).unsqueeze(0), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps:
            
            rng_v = random.Random(1)
            vloss, n = 0.0, 0
            with torch.no_grad():
                for _ in range(20):
                    vb = sample_batch(val_q, batch, rng_v)
                    if vb is None:
                        continue
                    va, vb_, vy = vb
                    vloss += float(F.binary_cross_entropy_with_logits(head(va, vb_), vy))
                    n += 1
            vmean = vloss / max(n, 1)
            print(f"  {load} step {step} train={float(loss):.4f} val={vmean:.4f}", flush=True)
            if vmean < best:
                best, best_state, wait = vmean, {k: v.detach().cpu() for k, v in head.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 8:
                    break
    head.load_state_dict(best_state or head.state_dict())
    blob = {
        "kind": "pair_head", "dim": dim, "state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "load": load, "val_loss": best,
        "note": "official train+val only; frozen BLIP vision; pair BCE + list CE (1+7)",
    }
    torch.save(blob, out_pair)
    torch.save({**blob, "kind": "list_head",
                "note": "same head; trained with list CE over 1 gold + 7 negs (independent pair scores, list loss)"},
               out_list)
    sha = hashlib.sha256(out_pair.read_bytes()).hexdigest()
    (WEIGHTS / f"ce_pair_local__{load}.sha256").write_text(sha + "\n")
    lsha = hashlib.sha256(out_list.read_bytes()).hexdigest()
    (WEIGHTS / f"ce_list_local__{load}.sha256").write_text(lsha + "\n")
    print(f"wrote {out_pair} sha={sha[:12]} val={best:.4f}", flush=True)
    _append_training_md(load, train_path, val_path, sha, lsha, best)
    del vision, head
    torch.cuda.empty_cache()


def score_top100(method, dataset, top100, blob, ctx):
    from PIL import Image

    dest = WEIGHTS / "blip_itm_base_coco"
    device = ctx.device
    pack = getattr(ctx, "_local_ce_pack", None)
    if pack is None:
        BlipForImageTextRetrieval, BlipProcessor = C.load_blip_classes()
        processor = BlipProcessor.from_pretrained(str(dest))
        vision = BlipForImageTextRetrieval.from_pretrained(str(dest)).vision_model.to(device).eval()
        pack = {"processor": processor, "vision": vision}
        ctx._local_ce_pack = pack
    processor, vision = pack["processor"], pack["vision"]
    dim = int(blob["dim"])
    head = PairHead(dim).to(device)
    head.load_state_dict(blob["state"])
    head.eval()
    queries = ce_rerank._cached_jsonl(ctx, "q", dataset, C.MBEIR / ce_rerank.QUERY_JSONL[dataset])
    cands = ce_rerank._cached_jsonl(ctx, "c", dataset, C.MBEIR / ce_rerank.CAND_JSONL[dataset])
    eval_idx = getattr(ctx, "_eval_idx", None)
    if eval_idx is None:
        return None, "error:missing_eval_idx"
    cache = {}

    def feat(rel):
        if rel in cache:
            return cache[rel]
        img = Image.open(C.MBEIR / rel).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            cache[rel] = vision(**inputs).pooler_output[0]
        return cache[rel]

    new_idx = np.array(top100, copy=True)
    with torch.no_grad():
        for qi in range(top100.shape[0]):
            qrec = queries[int(eval_idx[qi])]
            qimg = qrec.get("query_img_path")
            if not qimg:
                continue
            try:
                qf = feat(qimg)
            except Exception:
                continue
            ids, vecs = [], []
            for cj in top100[qi].tolist():
                cimg = cands[int(cj)].get("img_path")
                if not cimg:
                    continue
                try:
                    vecs.append(feat(cimg))
                    ids.append(int(cj))
                except Exception:
                    continue
            if not ids:
                continue
            a = qf.unsqueeze(0).expand(len(ids), -1)
            b = torch.stack(vecs)
            sc = head(a, b).cpu().numpy()
            if method == "ce_list_local":
                sc = sc - sc.max()  
            order = np.argsort(-sc)
            ranked = [ids[i] for i in order]
            rest = [c for c in top100[qi].tolist() if c not in ranked]
            new_idx[qi] = (ranked + rest)[:top100.shape[1]]
    return new_idx, "ok"


def _append_training_md(load, train_path, val_path, pair_sha, list_sha, val_loss) -> None:
    path = C.ROOT / "runs" / "ce_training" / "TRAINING.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"- `{load}` frozen BLIP vision + PairHead; official train `{train_path.name}` "
        f"val `{val_path.name}`; pair BCE + list CE (1 gold + 7 neg); "
        f"pair_sha={pair_sha[:16]} list_sha={list_sha[:16]} val_loss={val_loss:.4f} "
        f"written={C.now()}\n"
    )
    header = (
        "# Local CE training (plan 8.5.5)\n\n"
        "Official train queries + qrels/train only. Val for early stop. "
        "Never opened repair_test / Q_eval qrels.\n\n"
    )
    if not path.is_file():
        path.write_text(header)
    with path.open("a") as fh:
        fh.write(line)


def _jsonl(path: Path):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
