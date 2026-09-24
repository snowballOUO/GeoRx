
from __future__ import annotations

import json
from pathlib import Path

import common as C


COST = {
    "edis_task2": 170, "webqa_task1": 105, "visualnews_task0": 95,
    "oven_task6": 95, "infoseek_task6": 95, "fashion200k_task0": 60,
    "mscoco_task0": 35, "fashioniq_task7": 35, "cirr_task7": 32,
    "nights_task4": 32,
}
CPU_DATASETS = {"nights_task4", "fashioniq_task7", "cirr_task7"}


def main() -> None:
    
    
    root = Path(__file__).resolve().parent
    jobs = [{"seed": s, "cell_id": f"{e}__{d}"}
            for s in C.SEEDS for e, d in C.all_pairs()]
    cpu = [j for j in jobs if j["seed"] == 3 and
           j["cell_id"].split("__", 1)[1] in CPU_DATASETS]
    cpu_keys = {(j["seed"], j["cell_id"]) for j in cpu}
    gpu = [j for j in jobs if (j["seed"], j["cell_id"]) not in cpu_keys]
    bins = [{"jobs": [], "cost": 0, "device": "cuda", "gpu": i}
            for i in range(4)]
    for j in sorted(gpu, key=lambda x: COST[x["cell_id"].split("__", 1)[1]], reverse=True):
        b = min(bins, key=lambda x: x["cost"])
        b["jobs"].append(j)
        b["cost"] += COST[j["cell_id"].split("__", 1)[1]]
    workers = {f"gpu{i}": b for i, b in enumerate(bins)}
    workers["cpu_seed3_light"] = {"jobs": cpu, "cost": sum(COST[j["cell_id"].split("__", 1)[1]] for j in cpu),
                                   "device": "cpu", "gpu": None}
    plan = {"version": 1, "description": "270 native cells; GPU-heavy weighted split; CPU only seed3 light datasets",
            "total_jobs": len(jobs), "cpu_jobs": len(cpu), "gpu_jobs": len(gpu),
            "workers": workers}
    out = root / "runs" / "native_shard_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2))
    for name, w in workers.items():
        print(name, "jobs=", len(w["jobs"]), "estimated_cost=", w["cost"])
    print("wrote", out)


if __name__ == "__main__":
    main()
