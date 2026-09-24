
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

LEGACY = Path(__file__).resolve().parents[2] / "vendor" / "legacy"
sys.path.insert(0, str(LEGACY))
import batch_common as bc  

HDF5_ROOT = Path(os.environ["GEORX_DATA_ROOT"]) if os.environ.get("GEORX_DATA_ROOT") else Path(__file__).resolve().parents[2] / "input_hdf5"
SEALED = Path(os.environ.get("GEORX_SEALED_DIAG_ROOT", str(Path(__file__).resolve().parents[2] / "external" / "sealed_diagnosis")))
SPLITS_YAML = LEGACY / "cell_splits.yaml"
OUT = Path(__file__).resolve().parents[1] / "runs" / "split_hash_check.md"
SEED_SPLITS = 20260825


def hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def reproduce(cell) -> dict:
    seed = bc.cell_seed(SEED_SPLITS, cell)
    rng = np.random.default_rng(seed)
    pool_idx = np.arange(cell.n_pool, dtype=np.int64)  
    if cell.n_train:
        order = rng.permutation(cell.n_train)
        a = int(cell.n_train * 0.60)
        b = int(cell.n_train * 0.80)
        train_idx, cal_idx, fault_idx = order[:a], order[a:b], order[b:]
        test_order = rng.permutation(cell.n_test)
        d = int(cell.n_test * 0.50)
        e = int(cell.n_test * 0.75)
        diagnosis_test_idx = test_order[:d]
        repair_dev_idx = test_order[d:e]
        repair_test_idx = test_order[e:]
    else:
        order = rng.permutation(cell.n_test)
        a = int(cell.n_test * 0.45)
        b = int(cell.n_test * 0.65)
        c = int(cell.n_test * 0.80)
        d = int(cell.n_test * 0.85)
        e = int(cell.n_test * 0.925)
        train_idx = order[:a]
        cal_idx = order[a:b]
        fault_idx = order[b:c]
        diagnosis_test_idx = order[c:d]
        repair_dev_idx = order[d:e]
        repair_test_idx = order[e:]
    return {
        "train_indices_sha256": hash_idx(train_idx),
        "calibration_indices_sha256": hash_idx(cal_idx),
        "fault_eval_indices_sha256": hash_idx(fault_idx),
        "diagnosis_test_indices_sha256": hash_idx(diagnosis_test_idx),
        "repair_dev_indices_sha256": hash_idx(repair_dev_idx),
        "repair_test_indices_sha256": hash_idx(repair_test_idx),
        "pool_sample_indices_sha256": hash_idx(pool_idx),
    }


def main() -> None:
    splits = bc.load_cell_splits(SPLITS_YAML)
    overrides = dict(splits.get("path_overrides") or {})
    primary_enc = set(splits["primary_encoders"])
    primary_ds = set(splits["primary_datasets"])
    cells = {c.cell_id: c for c in bc.inventory(HDF5_ROOT)}
    for cid, path in overrides.items():
        if cid in cells:
            c = cells[cid]
            path = Path(os.path.expandvars(str(path))).expanduser()
            with h5py.File(path, "r") as f:
                nt = int(f["query/emb_train"].shape[0]) if "query/emb_train" in f else 0
                nq = int(f["query/emb"].shape[0])
                npool = int(f["pool/emb"].shape[0])
                dim = int(f["pool/emb"].shape[1])
            cells[cid] = bc.Cell(c.encoder, c.dataset, Path(path), nt, nq, npool, dim, c.modality)

    rows, n_ok, n_total = [], 0, 0
    for enc in sorted(primary_enc):
        for ds in sorted(primary_ds):
            cid = f"{enc}__{ds}"
            n_total += 1
            cell = cells.get(cid)
            prof = SEALED / cid / "profile.json"
            if cell is None:
                rows.append((cid, "MISSING_HDF5", "")); continue
            if not prof.exists():
                rows.append((cid, "MISSING_SEALED", "")); continue
            repro = reproduce(cell)
            sealed = json.loads(prof.read_text())["split_manifest"]
            mism = [k for k in repro if repro[k] != sealed.get(k)]
            if not mism:
                n_ok += 1
                rows.append((cid, "OK", ""))
            else:
                rows.append((cid, "MISMATCH", ",".join(k.replace("_indices_sha256", "") for k in mism)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# message (message, seed 20260825)",
        "",
        "message `complete_9enc_v1` message + pool message。",
        "message test message、message 1/2/3，message、message。",
        "",
        f"- message：**{n_ok}/{n_total}**",
        "",
        "| (encoder, message) | message | message |",
        "|---|---|---|",
        *[f"| {c} | {s} | {m} |" for c, s, m in rows],
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"split-hash check: {n_ok}/{n_total} match -> {OUT}")
    for c, s, m in rows:
        if s != "OK":
            print(f"  {s}: {c} {m}")


if __name__ == "__main__":
    main()
