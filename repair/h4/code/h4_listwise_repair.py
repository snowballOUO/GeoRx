


from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
OLD_ROOT = Path(os.environ.get("GEORX_SOURCE_ROOT", str(PACKAGE_ROOT / "main")))
NEW_ROOT = Path(os.environ.get("GEORX_H4_OUTPUT_ROOT", str(PACKAGE_ROOT / "repair" / "h4" / "runs")))
OLD_CODE = OLD_ROOT / "code"
H4_ROOT = Path(os.environ.get("GEORX_LISTWISE_WEIGHTS_ROOT", str(PACKAGE_ROOT / "weights" / "listwise")))



sys.path.insert(0, str(OLD_CODE))
import common as C  
import ce_list_train  
import stage_repair  


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_blob(dataset: str, cache: dict) -> tuple[dict, Path, str]:
    if dataset not in cache:
        path = H4_ROOT / f"ce_list_local__{dataset}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        blob = ce_list_train.torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("kind") != "list_mixer":
            raise RuntimeError(f"{path}: expected kind=list_mixer, got {blob.get('kind')!r}")
        cache[dataset] = (blob, path, _sha256(path))
    return cache[dataset]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _score_top100_fixed(dataset: str, top100: np.ndarray, blob: dict, ctx):
    
    torch = ce_list_train.torch
    pack = getattr(ctx, "_h4_fixed_pack", None)
    if pack is None:
        pack = {"blip": ce_list_train.FrozenBlip(ctx.device), "mixers": {}}
        ctx._h4_fixed_pack = pack
    blip = pack["blip"]
    mixer = pack["mixers"].get(dataset)
    if mixer is None:
        mixer = ce_list_train.ListMixer(int(blob["dim"])).to(ctx.device)
        mixer.load_state_dict(blob["state"])
        mixer.eval()
        pack["mixers"][dataset] = mixer
    queries = ce_list_train.ce_rerank._cached_jsonl(
        ctx, "q", dataset, C.MBEIR / ce_list_train.ce_rerank.QUERY_JSONL[dataset])
    cands = ce_list_train.ce_rerank._cached_jsonl(
        ctx, "c", dataset, C.MBEIR / ce_list_train.ce_rerank.CAND_JSONL[dataset])
    eval_idx = getattr(ctx, "_eval_idx", None)
    if eval_idx is None:
        return None, "error:missing_eval_idx", {"rank_changed_rows": 0, "rank_changed_entries": 0}
    new_idx = np.array(top100, copy=True)
    n_bad = 0
    changed_rows = 0
    changed_entries = 0
    with torch.no_grad():
        for qi in range(top100.shape[0]):
            qrec = queries[int(eval_idx[qi])]
            q_side = {"txt": qrec.get("query_txt") or "",
                      "img_path": qrec.get("query_img_path")}
            ids = [int(cj) for cj in top100[qi].tolist()]
            crecs = [ce_list_train._slim(cands[cj]) for cj in ids]
            try:
                
                blip.prefill([[(q_side, crecs)]], 32, None, "h4_infer")
                qf = blip.encode_one(q_side).unsqueeze(0)
                cf = torch.stack([blip.encode_one(c) for c in crecs]).unsqueeze(0)
                sc = mixer(qf, cf)[0].float().cpu().numpy()
                order = np.argsort(-sc, kind="stable")
                ranked = np.asarray([ids[i] for i in order], dtype=np.int64)
                new_idx[qi] = ranked
                diff = int(np.sum(ranked != top100[qi]))
                if diff:
                    changed_rows += 1
                    changed_entries += diff
            except Exception:
                n_bad += 1
    if n_bad > max(5, int(0.05 * top100.shape[0])):
        return None, f"error:h4_encode_fail_{n_bad}/{top100.shape[0]}", {
            "rank_changed_rows": changed_rows, "rank_changed_entries": changed_entries,
        }
    return new_idx, "ok", {"rank_changed_rows": changed_rows,
                            "rank_changed_entries": changed_entries,
                            "rank_failed_rows": n_bad}


def _row(cell, seed: int, status: str, base: dict, top_i, positives, old: dict,
         weight_path: Path, weight_sha: str, wall_s: float) -> dict:
    row = {
        "seed": int(seed), "encoder": cell.encoder, "dataset": cell.dataset,
        "method": "ce_list_local", "method_provenance": "h4_true_list_mixer",
        "scoring_scope": "rerank_top100", "status": status,
        "weight_path": str(weight_path), "weight_sha256": weight_sha,
        "weight_kind": "list_mixer", "source_repair_json": old.get("_path"),
        "R@1_base": float(base["R@1"]), "R@5_base": float(base["R@5"]),
        "R@10_base": float(base["R@10"]), "useful": False,
        "method_wall_s": round(float(wall_s), 2),
    }
    if status != "ok" or top_i is None:
        return row
    for k in (1, 5, 10):
        hits = C.recall_hits(top_i, positives, k)
        row[f"R@{k}"] = float(hits.mean())
        row[f"dR@{k}"] = float(hits.mean() - base[f"R@{k}"])
    row["useful"] = bool(row["dR@10"] >= C.USEFUL_DR10)
    return row


def run_worker(seed_list: list[int], datasets: list[str], device: str,
               batch_q: int) -> int:
    ctx = C.build_context(batch_q=batch_q, device_str=device)
    blob_cache: dict = {}
    summary = {"started": C.now(), "pid": os.getpid(), "device": device,
               "seeds": seed_list, "datasets": datasets,
               "n_done": 0, "n_skip": 0, "n_fail": 0, "errors": []}
    for seed in seed_list:
        old_dir = OLD_ROOT / "runs" / f"S{seed}" / "repair"
        out_dir = NEW_ROOT / "runs" / f"S{seed}" / "repair"
        for old_path in sorted(old_dir.glob("*.json")):
            try:
                old = json.loads(old_path.read_text())
            except Exception:
                continue
            if not old.get("analytic_complete"):
                continue
            dataset = str(old.get("dataset"))
            encoder = str(old.get("encoder"))
            if dataset not in datasets:
                continue
            cid = f"{encoder}__{dataset}"
            out_path = out_dir / f"{cid}.json"
            if out_path.is_file():
                try:
                    prev = json.loads(out_path.read_text())
                    if prev.get("status") == "ok":
                        summary["n_skip"] += 1
                        continue
                except Exception:
                    pass
            t0 = time.perf_counter()
            try:
                cell = ctx.cells[cid]
                top100 = np.asarray(old["cosine_top100"], dtype=np.int64)
                eval_idx = np.asarray(old["eval_idx"], dtype=np.int64)
                if top100.shape[0] != len(eval_idx):
                    raise RuntimeError(f"{cid}: top100/eval_idx length mismatch")
                ctx._eval_idx = eval_idx
                positives = stage_repair._positives_only(cell, eval_idx)
                blob, weight_path, weight_sha = _load_blob(dataset, blob_cache)
                top_i, status, rank_stats = _score_top100_fixed(dataset, top100, blob, ctx)
                base = old["baseline"]
                row = _row(cell, seed, status, base, top_i, positives, {
                    "_path": str(old_path)}, weight_path, weight_sha,
                    time.perf_counter() - t0)
                row.update(rank_stats)
                payload = {
                    "state": "complete", "stage": "h4_true_listwise_repair",
                    "seed": int(seed), "cell_id": cid, "encoder": encoder,
                    "dataset": dataset, "input_repair_json": str(old_path),
                    "input_diagnosis_seal_sha256": old.get("diagnosis_seal_sha256"),
                    "q_eval_n": len(eval_idx),
                    "q_eval_indices_sha256": old.get("q_eval_indices_sha256"),
                    "baseline": base, "weight_path": str(weight_path),
                    "weight_sha256": weight_sha, "weight_kind": blob.get("kind"),
                    "rank_stats": rank_stats,
                    "rows": [row], "status": status,
                    "written": C.now(),
                }
                _write_json(out_path, payload)
                summary["n_done"] += 1
                print(f"[{C.now()}] done S{seed} {cid} status={status} "
                      f"dR10={row.get('dR@10')} wall={row['method_wall_s']}s", flush=True)
                
                
                
                
                
                pack = getattr(ctx, "_h4_fixed_pack", None)
                if pack is not None:
                    pack["blip"].img_cache.clear()
                    pack["blip"].txt_cache.clear()
                    if ce_list_train.torch.cuda.is_available():
                        ce_list_train.torch.cuda.empty_cache()
            except Exception as exc:  
                summary["n_fail"] += 1
                summary["errors"].append({"seed": seed, "cell_id": cid,
                                          "error": repr(exc)})
                _write_json(out_dir / f"{cid}.error.json", {
                    "state": "error", "seed": seed, "cell_id": cid,
                    "error": repr(exc), "written": C.now(),
                })
                print(f"[{C.now()}] FAIL S{seed} {cid}: {exc!r}", flush=True)
    summary["finished"] = C.now()
    name = os.environ.get("H4_WORKER_NAME", f"worker_{os.getpid()}")
    _write_json(NEW_ROOT / "runs" / f"{name}.summary.json", summary)
    return 0 if summary["n_fail"] == 0 else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-q", type=int, default=32)
    args = ap.parse_args(argv)
    return run_worker(args.seeds, args.datasets, args.device, args.batch_q)


if __name__ == "__main__":
    raise SystemExit(main())
