
import argparse
import json
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    files = sorted((ROOT/"runs"/a.run_name).glob("S*/*/*.json"))
    rows = [json.loads(p.read_text()) for p in files]
    cells = ("clip_sf_large__mscoco_task0", "clip_sf_large__nights_task4",
             "e5v_llava_next__cirr_task7", "clip_sf_large__fashion200k_task0")
    expected_grid = {(s,c,';'.join(p)) for s in (1,2,3) for c in cells
                     for p in itertools.combinations(('h1','h2','h3','h4','h5'),2)}
    actual_grid = {(r['seed'],r['cell_id'],r['pair']) for r in rows}
    grid_ok = actual_grid == expected_grid and len(actual_grid) == len(rows) if a.expected == 120 else len(actual_grid) == len(rows)
    failures = []
    for r in rows:
        good = (r["state"] == "complete" and r["strength"] == 0.75
                and r["coverage"]["same_query_target_coverage"] == 1.0
                and r["coverage"]["joint_geometry_coverage"] >= 0.90
                and len(r["realized_set"]) == 2
                and r["construction"]["max_query_norm_error"] < 1e-5)
        print(r["seed"], r["cell_id"], r["pair"],
              "joint=", round(r["coverage"]["joint_geometry_coverage"], 4),
              "per_type=", {k: round(v["query_coverage"], 4) for k, v in r["geometry"].items()},
              "recall=", r["called_recall"], "PASS" if good else "FAIL")
        if not good:
            failures.append({"seed": r["seed"], "cell_id": r["cell_id"],
                             "pair": r["pair"], "geometry": r["geometry"], "coverage": r["coverage"]["joint_geometry_coverage"]})
    report = {"passed": len(rows) == a.expected and grid_ok and not failures, "records": len(rows),
              "exact_grid": grid_ok,
              "expected": a.expected, "failures": failures,
              "target_diagnosis_recall": sum(r["called_recall"] for r in rows)/len(rows) if rows else None,
              "note": "Recall is reported, never used as a smoke acceptance criterion."}
    print(json.dumps(report, indent=2))
    if a.write:
        (ROOT/"runs"/f"{a.run_name}_GATE.json").write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
