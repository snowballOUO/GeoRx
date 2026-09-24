
from __future__ import annotations

import csv
import json
from pathlib import Path

import common as C

STAGE = "eval"


def cell_done(ctx: C.Context, seed: int, cell_id: str) -> bool:
    
    return (ctx.runs / f"S{seed}" / "channel_hit.csv").is_file()


def run(ctx: C.Context, seed: int, pairs, log) -> dict:
    rows_lit, rows_loc = [], []
    n_missing = 0
    for encoder, dataset in pairs:
        cid = f"{encoder}__{dataset}"
        repair_p = ctx.runs / f"S{seed}" / "repair" / f"{cid}.json"
        if not repair_p.is_file():
            n_missing += 1
            continue
        d = json.loads(repair_p.read_text())
        if not d.get("analytic_complete"):
            n_missing += 1
            continue
        phi = list(d.get("phi") or [])
        P_all = C.predicted_P(phi, dataset)
        P_lit = C.literature_P(P_all, dataset)
        rows_ok = [r for r in d.get("rows") or [] if r.get("status") == "ok"]
        T_all = {r["method"] for r in rows_ok if r.get("useful")}
        T_lit = {m for m in T_all if m in set(C.ANALYTIC_METHODS) | (
            {"ce_pair_blip_itm"} if dataset in C.OPEN_CE_LOADS else set())}
        uneval = [r["method"] for r in d.get("rows") or []
                  if str(r.get("status", "")).startswith("unevaluable")]
        rows_lit.append(_hit_row(seed, encoder, dataset, "literature_only",
                                 phi, P_lit, sorted(T_lit), uneval))
        rows_loc.append(_hit_row(seed, encoder, dataset, "with_local_ce",
                                 phi, P_all, sorted(T_all), uneval))
    all_rows = rows_lit + rows_loc
    path = ctx.runs / f"S{seed}" / "channel_hit.csv"
    if all_rows:
        fields = list(all_rows[0].keys())
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
    exact = [r["exact_match"] for r in rows_lit]
    status = {
        "stage": STAGE, "seed": seed, "n_rows": len(rows_lit),
        "n_missing_repair": n_missing,
        "literature_exact_match": (sum(exact) / len(exact)) if exact else None,
        "written": C.now(),
    }
    C.write_json(ctx.runs / f"S{seed}" / "eval_status.json", status)
    log(f"[eval s{seed}] literature exact-match="
        f"{status['literature_exact_match']} missing={n_missing}")
    _maybe_write_table90(ctx)
    return status


def _hit_row(seed, encoder, dataset, scope, phi, P, T, uneval) -> dict:
    Ps, Ts = set(P), set(T)
    union = Ps | Ts
    jacc = 1.0 if not union else len(Ps & Ts) / len(union)
    return {
        "seed": seed, "encoder": encoder, "dataset": dataset, "scope": scope,
        "phi_set": ";".join(phi) if phi else "empty",
        "P": ";".join(P) if P else "empty",
        "T": ";".join(T) if T else "empty",
        "exact_match": int(Ps == Ts),
        "jaccard": round(jacc, 4),
        "unevaluable_types": ";".join(uneval) if uneval else "",
    }


def _maybe_write_table90(ctx: C.Context) -> None:
    tables = {}
    for seed in C.SEEDS:
        p = ctx.runs / f"S{seed}" / "channel_hit.csv"
        if not p.is_file():
            return
        with p.open() as fh:
            tables[seed] = [r for r in csv.DictReader(fh) if r["scope"] == "literature_only"]
    lines = ["# TABLE90 literature-only exact match", "",
             "| encoder | dataset | S1 Φ | S1 match | S2 Φ | S2 match | S3 Φ | S3 match |",
             "|---|---|---|---|---|---|---|---|"]
    key = lambda r: (r["encoder"], r["dataset"])
    index = {s: {key(r): r for r in tables[s]} for s in C.SEEDS}
    keys = [key(r) for r in tables[1]]
    matches = {s: 0 for s in C.SEEDS}
    for enc, ds in keys:
        cells = []
        for s in C.SEEDS:
            r = index[s][(enc, ds)]
            cells += [r["phi_set"], r["exact_match"]]
            matches[s] += int(r["exact_match"])
        lines.append(f"| {enc} | {ds} | " + " | ".join(cells) + " |")
    n = len(keys) or 1
    lines += ["",
              f"S1 exact {matches[1]}/{n} = {matches[1]/n:.3f}",
              f"S2 exact {matches[2]}/{n} = {matches[2]/n:.3f}",
              f"S3 exact {matches[3]}/{n} = {matches[3]/n:.3f}",
              f"mean {(sum(matches[s]/n for s in C.SEEDS)/3):.3f}"]
    (ctx.runs / "TABLE90.md").write_text("\n".join(lines) + "\n")
    summary = [
        "# SUMMARY",
        "",
        "Opportunity-geometry sphere is not a retrieval ideal. Empty Φ is not R@k optimal.",
        "Injection recall does not prove native-Φ accuracy. Native sphere lamps are not false positives.",
        "Channel hit rate is method-set exact match P=T, not ΔR macro-average.",
        "Cross-encoder is top-100 rerank; locally trained weights are appendix-only.",
        "Headline is 90 cells including MSCOCO and NIGHTS. never_hitrate_datasets is empty.",
        "",
        f"Literature exact-match mean "
        f"{(sum(matches[s]/n for s in C.SEEDS)/3):.3f} "
        f"(S1={matches[1]/n:.3f}, S2={matches[2]/n:.3f}, S3={matches[3]/n:.3f}).",
        "",
    ]
    (ctx.runs / "SUMMARY.md").write_text("\n".join(summary))
