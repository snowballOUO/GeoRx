

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(os.environ.get("GEORX_SOURCE_ROOT", str(PACKAGE_ROOT / "main")))
OLD_IMPL = Path(os.environ.get("GEORX_H4_IMPL", str(Path(__file__).resolve().parent / "h4_listwise_repair.py")))
NEW_ROOT = Path(os.environ.get("GEORX_H4_OUTPUT_ROOT", str(PACKAGE_ROOT / "repair" / "h4" / "runs")))
WEIGHTS_ROOT = Path(os.environ.get("GEORX_LISTWISE_WEIGHTS_ROOT", str(PACKAGE_ROOT / "weights" / "listwise")))


def load_impl():
    spec = importlib.util.spec_from_file_location("h4_repair_fixed_impl", OLD_IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {OLD_IMPL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    module.OLD_ROOT = OFFICIAL_ROOT
    module.NEW_ROOT = NEW_ROOT
    module.H4_ROOT = WEIGHTS_ROOT
    return module


def parse_job(raw: str) -> tuple[list[int], list[str]]:
    try:
        dataset_part, seed_part = raw.split(":", 1)
        datasets = [x for x in dataset_part.split(",") if x]
        seeds = [int(x) for x in seed_part.split(",") if x]
    except Exception as exc:  
        raise ValueError(f"invalid --job {raw!r}; expected dataset[,dataset]:seed[,seed]") from exc
    if not datasets or not seeds:
        raise ValueError(f"invalid --job {raw!r}; datasets and seeds are required")
    return seeds, datasets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True,
                    help="dataset[,dataset]:seed[,seed], repeatable")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch-q", type=int, default=32)
    ap.add_argument("--worker-name", required=True)
    args = ap.parse_args(argv)

    impl = load_impl()
    failures = 0
    for index, raw in enumerate(args.job):
        seeds, datasets = parse_job(raw)
        
        
        os.environ["H4_WORKER_NAME"] = f"{args.worker_name}_job{index}"
        rc = int(impl.run_worker(seeds, datasets, args.device, args.batch_q))
        failures += int(rc != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
