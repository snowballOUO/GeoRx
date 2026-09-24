

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("h1", "h2", "h3", "h4", "h5")


def pct(a, b):
    return 100.0 * a / b if b else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="protected_support_v4_exact_h4_full90")
    ap.add_argument("--expected", type=int, default=2700)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    run_root = ROOT / "runs" / args.run_name
    records = []
    for path in sorted(run_root.glob("S*/*/*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if d.get("state") == "complete":
            records.append(d)

    by_type = {k: {"target": 0, "realized": 0, "diagnosed": 0,
                   "realized_and_diagnosed": 0, "realized_but_missed": 0,
                   "not_realized_but_diagnosed": 0} for k in KINDS}
    by_pair = defaultdict(lambda: {"records": 0, "targets": 0, "realized": 0,
                                   "diagnosed": 0, "realized_and_diagnosed": 0,
                                   "realized_but_missed": 0, "exact": 0})
    extra = 0
    for d in records:
        pair = tuple(d["pair"].split(";"))
        targets, phi = set(pair), set(d.get("phi") or [])
        realized = set(d.get("realized_set") or [])
        row = by_pair["+".join(pair)]
        row["records"] += 1
        row["targets"] += 2
        row["realized"] += len(targets & realized)
        row["diagnosed"] += len(targets & phi)
        row["realized_and_diagnosed"] += len(targets & realized & phi)
        row["realized_but_missed"] += len((targets & realized) - phi)
        row["exact"] += int(phi == targets)
        extra += len(phi - targets)
        for k in pair:
            by_type[k]["target"] += 1
            by_type[k]["realized"] += int(k in realized)
            by_type[k]["diagnosed"] += int(k in phi)
            by_type[k]["realized_and_diagnosed"] += int(k in realized and k in phi)
            by_type[k]["realized_but_missed"] += int(k in realized and k not in phi)
            by_type[k]["not_realized_but_diagnosed"] += int(k not in realized and k in phi)

    n_targets = 2 * len(records)
    result = {
        "run_name": args.run_name,
        "records": len(records),
        "expected": args.expected,
        "complete": len(records) == args.expected,
        "geometry_realization_pct": pct(sum(x["realized"] for x in by_type.values()), n_targets),
        "target_diagnosis_recall_pct": pct(sum(x["diagnosed"] for x in by_type.values()), n_targets),
        "recall_given_geometry_realized_pct": pct(
            sum(x["realized_and_diagnosed"] for x in by_type.values()),
            sum(x["realized"] for x in by_type.values())),
        "realized_but_missed": sum(x["realized_but_missed"] for x in by_type.values()),
        "not_realized_targets": n_targets - sum(x["realized"] for x in by_type.values()),
        "not_realized_but_diagnosed": sum(x["not_realized_but_diagnosed"] for x in by_type.values()),
        "exact_set_pct": pct(sum(x["exact"] for x in by_pair.values()), len(records)),
        "extra_lamps_per_record": extra / len(records) if records else None,
        "by_type": {},
        "by_pair": {},
    }
    for k, x in by_type.items():
        result["by_type"][k] = {**x, "realization_pct": pct(x["realized"], x["target"]),
                                      "diagnosis_recall_pct": pct(x["diagnosed"], x["target"]),
                                      "recall_given_geometry_realized_pct": pct(x["realized_and_diagnosed"], x["realized"])}
    for k, x in sorted(by_pair.items()):
        result["by_pair"][k] = {**x, "realization_pct": pct(x["realized"], x["targets"]),
                                      "diagnosis_recall_pct": pct(x["diagnosed"], x["targets"]),
                                      "recall_given_geometry_realized_pct": pct(x["realized_and_diagnosed"], x["realized"]),
                                      "exact_set_pct": pct(x["exact"], x["records"])}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.write:
        out = ROOT / "runs" / f"{args.run_name}_summary.json"
        out.write_text(text + "\n")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
