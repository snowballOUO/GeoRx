

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "code" / "joint_repair.py"
PYTHON = os.environ.get("GEORX_PYTHON", "python3")


COMMANDS = [
    
    ("A-gpu0", "gpu0", "four", ["clip_vitb32__fashion200k_task0"], [1, 2, 3]),
    ("A-gpu1", "gpu1", "four", ["blip2_vitL__fashion200k_task0"], [1, 2, 3]),
    ("A-gpu2", "gpu2", "four", ["clip_vitb32__oven_task6"], [1, 2, 3]),
    ("A-gpu3", "gpu3", "h2h5", ["gme_qwen2vl_2b__visualnews_task0"], [1, 2, 3]),
    
    
    ("B-cpu0", "cpu0", "h2h5",
     ["clip_sf_large__webqa_task1", "blip_ff_large__webqa_task1"], [1, 2, 3]),
    ("B-gpu3", "gpu3", "h2h5", ["clip_vitb32__visualnews_task0"], [1, 3]),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true", help="actually start workers")
    ap.add_argument("--phase", choices=("A", "B"),
                    help="phase to print/start; required together with --start")
    args = ap.parse_args(argv)
    if args.start and args.phase is None:
        ap.error("--start requires --phase A or --phase B")
    phase = args.phase
    commands = [x for x in COMMANDS if phase is None or x[0].startswith(phase + "-")]
    for label, tag, group, cells, seeds in commands:
        device = "cuda" if tag.startswith("gpu") else "cpu"
        cmd = [PYTHON, str(WORKER), "--group", group, "--device", device,
               "--batch-q", "32", "--cells", *cells, "--seeds", *map(str, seeds)]
        log_dir = ROOT / "runs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"launcher_{label.lower().replace('-', '_')}.out"
        print(label, "nohup", " ".join(cmd), ">", str(log_path), "2>&1 &", flush=True)
        if args.start:
            env = {"CUDA_VISIBLE_DEVICES": tag[-1]} if tag.startswith("gpu") else {}
            if tag.startswith("cpu"):
                env.update({
                    "JOINT_CPU_THREADS": "40",
                    "OMP_NUM_THREADS": "40",
                    "MKL_NUM_THREADS": "40",
                    "OPENBLAS_NUM_THREADS": "40",
                    "NUMEXPR_NUM_THREADS": "40",
                })
            with log_path.open("ab") as log_fh:
                subprocess.Popen(["nohup", *cmd], cwd=str(ROOT),
                                 env={**os.environ, **env},
                                 stdout=log_fh, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    if args.start:
        print(f"phase {phase} workers started")
    else:
        print("dry plan only; no worker started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
