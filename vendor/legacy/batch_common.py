
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import yaml


@dataclass(frozen=True)
class Cell:
    encoder: str
    dataset: str
    path: Path
    n_train: int
    n_test: int
    n_pool: int
    dim: int
    modality: str

    @property
    def cell_id(self) -> str:
        return f"{self.encoder}__{self.dataset}"


def load_cell_splits(path: Path) -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def hitrate_role(cell: Cell, splits: Dict) -> str:
    
    
    if cell.dataset in set(splits.get("never_hitrate_datasets") or []):
        return "never_hitrate"
    primary_enc = set(splits.get("primary_encoders") or [])
    primary_ds = set(splits.get("primary_datasets") or [])
    if (not primary_enc or cell.encoder in primary_enc) and (not primary_ds or cell.dataset in primary_ds):
        return "heldout_hitrate"
    return "audit_only"


def inventory(root: Path) -> list[Cell]:
    cells = []
    for path in sorted(Path(root).glob("*/*.hdf5")):
        if path.is_symlink():
            continue
        with h5py.File(path, "r") as f:
            cells.append(
                Cell(
                    encoder=path.parent.name,
                    dataset=path.stem,
                    path=path,
                    n_train=int(f["query/emb_train"].shape[0]) if "query/emb_train" in f else 0,
                    n_test=int(f["query/emb"].shape[0]),
                    n_pool=int(f["pool/emb"].shape[0]),
                    dim=int(f["pool/emb"].shape[1]),
                    modality=str(f.attrs.get("modality", "")),
                )
            )
    return cells


def cell_seed(base_seed: int, cell: Cell) -> int:
    suffix = int(hashlib.sha256(cell.cell_id.encode()).hexdigest()[:8], 16)
    return int((int(base_seed) + suffix) % (2**32 - 1))


def load_diagnosis_samples(
    cell: Cell,
    *,
    base_seed: int,
    pool_cap: int,
    window_size: int,
    train_windows: int,
    calibration_windows: int,
    test_windows: int,
    fault_windows: int,
    action_query_cap: int,
    csls_reference_cap: int = 2048,
    pool_mode: str = "full",
) -> Dict:
    seed = cell_seed(base_seed, cell)
    rng = np.random.default_rng(seed)
    with h5py.File(cell.path, "r") as f:
        if pool_mode == "full":
            pool_idx = np.arange(cell.n_pool, dtype=np.int64)
        elif pool_mode == "capped":
            pool_idx = _stratified_block_indices(cell.n_pool, min(cell.n_pool, pool_cap), rng)
        else:
            raise ValueError(f"unknown pool_mode {pool_mode}")
        pool = _read_rows(f["pool/emb"], pool_idx)
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
            mode = "native_train_plus_disjoint_diagnosis_repair_tests"
            train_ds = cal_ds = fault_ds = f["query/emb_train"]
            test_ds = f["query/emb"]
            csls_ref_source = "query/emb_train"
            csls_ref_idx = np.sort(train_idx[: min(len(train_idx), csls_reference_cap)])
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
            mode = "official_queries_repartitioned_45_20_15_5_7.5_7.5"
            train_ds = cal_ds = fault_ds = test_ds = f["query/emb"]
            csls_ref_source = "query/emb"
            csls_ref_idx = np.sort(train_idx[: min(len(train_idx), csls_reference_cap)])

        train_take = min(len(train_idx), max(window_size * train_windows, window_size))
        cal_take = min(len(cal_idx), max(window_size * calibration_windows, window_size))
        fault_take = min(len(fault_idx), max(window_size * fault_windows, window_size))
        test_take = min(len(diagnosis_test_idx), max(window_size * test_windows, window_size))
        train_q = _read_rows(train_ds, train_idx[:train_take])
        cal_q = _read_rows(cal_ds, cal_idx[:cal_take])
        fault_q = _read_rows(fault_ds, fault_idx[:fault_take])
        test_q = _read_rows(test_ds, diagnosis_test_idx[:test_take])

    diagnosis_test_action_idx = np.sort(
        diagnosis_test_idx[: min(len(diagnosis_test_idx), action_query_cap)]
    )
    repair_dev_action_idx = np.sort(repair_dev_idx[: min(len(repair_dev_idx), action_query_cap)])
    repair_test_action_idx = np.sort(repair_test_idx[: min(len(repair_test_idx), action_query_cap)])
    split_manifest = {
        "seed": seed,
        "mode": mode,
        "native_train_available": bool(cell.n_train),
        "train_indices_sha256": _hash_indices(train_idx),
        "calibration_indices_sha256": _hash_indices(cal_idx),
        "fault_eval_indices_sha256": _hash_indices(fault_idx),
        "diagnosis_test_indices_sha256": _hash_indices(diagnosis_test_idx),
        "repair_dev_indices_sha256": _hash_indices(repair_dev_idx),
        "repair_test_indices_sha256": _hash_indices(repair_test_idx),
        "pool_sample_indices_sha256": _hash_indices(pool_idx),
        "pool_sample_indices": None if pool_mode == "full" else pool_idx.tolist(),
        "n_train_partition": int(len(train_idx)),
        "n_calibration_partition": int(len(cal_idx)),
        "n_fault_eval_partition": int(len(fault_idx)),
        "n_diagnosis_test_partition": int(len(diagnosis_test_idx)),
        "n_repair_dev_partition": int(len(repair_dev_idx)),
        "n_repair_test_partition": int(len(repair_test_idx)),
        "n_pool_diagnosis": int(len(pool_idx)),
        "n_pool_full": int(cell.n_pool),
        "pool_mode": pool_mode,
        "pool_equals_repair_pool": True,
        "pool_sampling": "full_pool" if pool_mode == "full" else "eight_stratified_contiguous_blocks_or_full_pool",
        "repair_dev_query_indices": repair_dev_action_idx.tolist(),
        "repair_test_query_indices": repair_test_action_idx.tolist(),
        "diagnosis_test_query_indices": diagnosis_test_action_idx.tolist(),
        "csls_reference_source": csls_ref_source,
        "csls_reference_query_indices": csls_ref_idx.tolist(),
        "csls_reference_disjoint_from_action": True,
    }
    return {
        "pool": pool,
        "pool_idx": pool_idx,
        "train": train_q,
        "calibration": cal_q,
        "fault": fault_q,
        "test": test_q,
        "manifest": split_manifest,
    }


def _read_rows(dataset, indices: np.ndarray) -> np.ndarray:
    idx = np.sort(np.asarray(indices, dtype=np.int64))
    value = np.asarray(dataset[idx], dtype=np.float32)
    return _normalize(value)


def _stratified_block_indices(n: int, size: int, rng, n_blocks: int = 8) -> np.ndarray:
    if size >= n:
        return np.arange(n, dtype=np.int64)
    blocks = min(int(n_blocks), int(size))
    base, remainder = divmod(int(size), blocks)
    pieces = []
    for block in range(blocks):
        take = base + (1 if block < remainder else 0)
        region_start = (block * n) // blocks
        region_end = ((block + 1) * n) // blocks
        room = max(0, region_end - region_start - take)
        start = region_start + (int(rng.integers(0, room + 1)) if room else 0)
        pieces.append(np.arange(start, start + take, dtype=np.int64))
    return np.concatenate(pieces)


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)).astype(np.float32, copy=False)


def _hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
