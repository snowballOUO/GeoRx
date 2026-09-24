
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import common as C

HF_MODEL = "Salesforce/blip-itm-base-coco"
WEIGHTS = C.ROOT / "runs" / "ce_training" / "weights"
MBEIR = C.MBEIR

QUERY_JSONL = {
    "mscoco_task0": "query/test/mbeir_mscoco_task0_test.jsonl",
    "visualnews_task0": "query/test/mbeir_visualnews_task0_test.jsonl",
    "fashion200k_task0": "query/test/mbeir_fashion200k_task0_test.jsonl",
    "webqa_task1": "query/test/mbeir_webqa_task1_test.jsonl",
    "edis_task2": "query/test/mbeir_edis_task2_test.jsonl",
    "nights_task4": "query/test/mbeir_nights_task4_test.jsonl",
    "oven_task6": "query/test/mbeir_oven_task6_test.jsonl",
    "infoseek_task6": "query/test/mbeir_infoseek_task6_test.jsonl",
    "fashioniq_task7": "query/test/mbeir_fashioniq_task7_test.jsonl",
    "cirr_task7": "query/test/mbeir_cirr_task7_test.jsonl",
}
CAND_JSONL = {
    "mscoco_task0": "cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl",
    "visualnews_task0": "cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl",
    "fashion200k_task0": "cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl",
    "webqa_task1": "cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl",
    "edis_task2": "cand_pool/local/mbeir_edis_task2_cand_pool.jsonl",
    "nights_task4": "cand_pool/local/mbeir_nights_task4_cand_pool.jsonl",
    "oven_task6": "cand_pool/local/mbeir_oven_task6_cand_pool.jsonl",
    "infoseek_task6": "cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl",
    "fashioniq_task7": "cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl",
    "cirr_task7": "cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl",
}


def rerank(method, dataset, query, pool, positives, top100, ctx, seed):
    del query, pool, positives  
    if method == "ce_pair_blip_itm":
        if dataset not in C.OPEN_CE_LOADS:
            return None, "unevaluable_wrong_load", "pretrained_literature"
        return _blip_itm(dataset, top100, ctx, seed)
    if method in ("ce_pair_local", "ce_list_local"):
        ckpt = WEIGHTS / f"{method}__{dataset}.pt"
        if not ckpt.is_file():
            return None, "unevaluable_no_weights", "locally_trained"
        return _local_rerank(method, dataset, top100, ckpt, ctx)
    return None, f"unknown_method:{method}", "locally_trained"


def _blip_itm(dataset, top100, ctx, seed):
    import torch
    from PIL import Image

    model_dir = WEIGHTS / "blip_itm_base_coco"
    if not (model_dir / "config.json").is_file():
        return None, "unevaluable_no_weights", "pretrained_literature"
    eval_idx = getattr(ctx, "_eval_idx", None)
    if eval_idx is None:
        return None, "error:missing_eval_idx", "pretrained_literature"
    pack = _cached_blip(ctx)
    model, processor, device = pack["model"], pack["processor"], pack["device"]
    queries = _cached_jsonl(ctx, "q", dataset, MBEIR / QUERY_JSONL[dataset])
    cands = _cached_jsonl(ctx, "c", dataset, MBEIR / CAND_JSONL[dataset])
    new_idx = np.array(top100, copy=True)
    with torch.no_grad():
        for qi in range(top100.shape[0]):
            qrec = queries[int(eval_idx[qi])]
            texts, images, keep = [], [], []
            for cj in top100[qi].tolist():
                crec = cands[int(cj)]
                text, img_path = _pair_text_image(qrec, crec, dataset)
                if img_path is None or not img_path.is_file():
                    continue
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                    texts.append(text.strip() or ".")
                    keep.append(int(cj))
                except Exception:
                    continue
            if not keep:
                continue
            scores = []
            bs = 32
            for s in range(0, len(keep), bs):
                batch_img = images[s:s + bs]
                batch_txt = texts[s:s + bs]
                inputs = processor(images=batch_img, text=batch_txt,
                                   return_tensors="pt", padding=True,
                                   truncation=True, max_length=40).to(device)
                out = model(**inputs)
                logit = out.itm_score[:, 1] if out.itm_score.dim() == 2 else out.itm_score.reshape(-1)
                scores.append(logit.float().cpu().numpy())
            sc = np.concatenate(scores)
            order = np.argsort(-sc)
            ranked = [keep[i] for i in order]
            rest = [c for c in top100[qi].tolist() if c not in ranked]
            new_idx[qi, :len(ranked) + len(rest)] = (ranked + rest)[:top100.shape[1]]
    return new_idx, "ok", "pretrained_literature"


def _cached_blip(ctx):
    pack = getattr(ctx, "_blip_itm_pack", None)
    if pack is not None:
        return pack
    BlipForImageTextRetrieval, BlipProcessor = C.load_blip_classes()
    model_dir = WEIGHTS / "blip_itm_base_coco"
    processor = BlipProcessor.from_pretrained(str(model_dir))
    model = BlipForImageTextRetrieval.from_pretrained(str(model_dir))
    device = ctx.device
    model.to(device).eval()
    pack = {"model": model, "processor": processor, "device": device}
    ctx._blip_itm_pack = pack
    return pack


def _cached_jsonl(ctx, kind, dataset, path: Path):
    store = getattr(ctx, "_jsonl_cache", None)
    if store is None:
        store = {}
        ctx._jsonl_cache = store
    key = f"{kind}:{dataset}"
    if key not in store:
        store[key] = _load_jsonl(path)
    return store[key]


def _local_rerank(method, dataset, top100, ckpt, ctx):
    import torch
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if blob.get("kind") == "list_mixer":
        import ce_list_train
        idx, status = ce_list_train.score_top100(method, dataset, top100, blob, ctx)
        return idx, status, "locally_trained"
    import ce_train
    idx, status = ce_train.score_top100(method, dataset, top100, blob, ctx)
    return idx, status, "locally_trained"


def _pair_text_image(qrec, crec, dataset):
    img_root = MBEIR
    qtxt = (qrec.get("query_txt") or "") or ""
    ctxt = (crec.get("txt") or crec.get("cand_txt") or "") or ""
    qimg = qrec.get("query_img_path")
    cimg = crec.get("img_path") or crec.get("cand_img_path")
    if dataset in ("mscoco_task0", "visualnews_task0", "fashion200k_task0"):
        text, img = qtxt, cimg
    elif dataset in ("webqa_task1", "edis_task2"):
        text, img = (qtxt + " " + ctxt).strip(), cimg
    elif dataset in ("oven_task6", "infoseek_task6"):
        text, img = (qtxt + " " + ctxt).strip(), qimg
    else:
        return "", None
    if not img:
        return text, None
    return text, img_root / img


def _load_jsonl(path: Path):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
