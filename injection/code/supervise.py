
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("GEORX_PYTHON", sys.executable)
SMOKE = "smoke_v3"
FULL = "same_query_joint_v3_full90"
SMOKE_CELLS = ["clip_sf_large__mscoco_task0", "clip_sf_large__nights_task4",
               "e5v_llava_next__cirr_task7", "clip_sf_large__fashion200k_task0"]
SHARDS = [("edis_task2",), ("oven_task6", "infoseek_task6"),
          ("visualnews_task0", "webqa_task1"),
          ("mscoco_task0", "nights_task4", "fashion200k_task0", "fashioniq_task7", "cirr_task7")]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"code").glob("*.py"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("smoke", "full"), required=True)
    a = ap.parse_args()
    lock = (ROOT/"runs"/"supervisor.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = SMOKE if a.phase == "smoke" else FULL
    expected = 120 if a.phase == "smoke" else 2700
    manifest = hashes()
    if a.phase == "full":
        gate = json.loads((ROOT/"runs"/f"{SMOKE}_GATE.json").read_text())
        inv = json.loads((ROOT/"runs"/"invariants.json").read_text())
        smoke_status = json.loads((ROOT/"runs"/f"{SMOKE}_supervisor.json").read_text())
        
        
        
        
        
        assert inv["passed"] and smoke_status["state"] in ("complete", "gate_failed")
        for fn in ("stage_joint.py", "stage_protected_pairs.py", "common.py"):
            assert manifest[fn] == smoke_status["source_sha256"][fn], f"source changed since smoke: {fn}"
    state = {"state": "running", "phase": a.phase, "run_name": name, "expected": expected,
             "started": time.time(), "pid": os.getpid(), "source_sha256": manifest, "workers": []}
    if a.phase == "full":
        state["smoke_geometry_gate_passed"] = bool(gate["passed"])
        state["smoke_geometry_gate_warning"] = (not bool(gate["passed"]))
    out = ROOT/"runs"/f"{name}_supervisor.json"
    processes = []
    for gpu in range(4):
        filters = ["--cells", SMOKE_CELLS[gpu]] if a.phase == "smoke" else ["--datasets", *SHARDS[gpu]]
        cmd = ["taskset", "-c", "40-79", PY, "-u", str(ROOT/"code"/"stage_joint.py"),
               "--seeds", "1", "2", "3", "--orders", "canonical", "--run-name", name,
               "--batch-q", str([8,16,16,32][gpu]), "--status-tag", f"{name}_gpu{gpu}", "--no-rewrite", *filters]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        log = ROOT/"runs"/"logs"/f"{name}_gpu{gpu}_{int(time.time())}.out"
        handle = log.open("w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((proc, handle))
        state["workers"].append({"gpu": gpu, "pid": proc.pid, "log": str(log), "cmd": cmd})
    write(out, state)
    while any(p.poll() is None for p, _ in processes):
        state["completed_records"] = len(list((ROOT/"runs"/name).glob("S*/*/*.json")))
        state["updated"] = time.time()
        write(out, state)
        time.sleep(15)
    for worker, (proc, handle) in zip(state["workers"], processes):
        handle.close()
        worker["returncode"] = proc.returncode
    state["finished"] = time.time()
    codes = [p.returncode for p, _ in processes]
    if any(codes):
        state["state"] = "failed"
        write(out, state)
        return 1
    consolidation = subprocess.run([PY, str(ROOT/"code"/"stage_joint.py"), "--run-name", name, "--rewrite-only"], check=False)
    summary = subprocess.run([PY, str(ROOT/"code"/"analyze_full90.py"), "--run-name", name,
                              "--expected", str(expected), "--write"], check=False)
    code = consolidation.returncode or summary.returncode
    for gpu in range(4):
        shard = json.loads((ROOT/"runs"/f"status_{name}_gpu{gpu}.json").read_text())
        if shard['n_fail'] or shard['n_ok']+shard['n_skip'] != shard['n_target']:
            code = 1
    if a.phase == "smoke":
        gate = subprocess.run([PY, str(ROOT/"code"/"check_smoke.py"), "--run-name", name,
                               "--expected", str(expected), "--write"], check=False)
        code = code or gate.returncode
    comparison = subprocess.run([PY, str(ROOT/"code"/"report_effects.py"), "--run-name", name], check=False)
    code = code or comparison.returncode
    state["state"] = "complete" if code == 0 else "gate_failed"
    state["completed_records"] = len(list((ROOT/"runs"/name).glob("S*/*/*.json")))
    write(out, state)
    print(json.dumps(state, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
