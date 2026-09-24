
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("GEORX_BASE_INJECTION_ROOT", str(Path(__file__).resolve().parents[2] / "injection" / "runs" / "baseline")))


def key(r):
    return (r['seed'], r['cell_id'], r['pair'])


def load(root):
    return [json.loads(p.read_text()) for p in sorted(root.glob('S*/*/*.json'))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    a = ap.parse_args()
    rows = load(ROOT/'runs'/a.run_name)
    old = {key(r): r for r in load(BASE)}
    groups = defaultdict(list)
    for r in rows:
        groups[r['pair']].append(r)
    result = {'run_name': a.run_name, 'records': len(rows), 'by_pair': {}, 'by_cell': {}}
    for pair, rr in sorted(groups.items()):
        matched = [r for r in rr if key(r) in old]
        result['by_pair'][pair] = {
            'records': len(rr),
            'new_target_recall': sum(r['called_recall'] for r in rr)/len(rr),
            'old_target_recall_matched': sum(old[key(r)]['called_recall'] for r in matched)/len(matched) if matched else None,
            'same_query_geometry_coverage_mean': sum(r['coverage']['joint_geometry_coverage'] for r in rr)/len(rr),
            'same_query_geometry_coverage_min': min(r['coverage']['joint_geometry_coverage'] for r in rr),
            'exact_target_set_rate': sum(r['exact_called'] for r in rr)/len(rr),
            'extra_lamps_mean': sum(len(set(r['phi'])-set(r['pair'].split(';'))) for r in rr)/len(rr)}
    for r in rows:
        result['by_cell'].setdefault(r['cell_id'], []).append(r['called_recall'])
    result['by_cell'] = {k: {'records': len(v), 'target_recall': sum(v)/len(v)} for k,v in result['by_cell'].items()}
    result['caveat'] = 'Geometry criteria differ from v4; compare diagnosis recall with the frozen detector, and inspect the new per-query coverage separately. Shared supports may create extra anomaly types. Numerical injection parameter 0.75 is not a guarantee of equal physical displacement under different constructions.'
    out = ROOT/'runs'/f'{a.run_name}_comparison.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
