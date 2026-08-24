"""Split-half stability (review item 7.1): partition the n=100 confirmatory
queries into disjoint halves across 200 random splits; report per-operator
mean MRR +- std across splits, and the sign stability of CombSUM - RRF.

Data: per_query filtered_results.json from the three n=100 benchmarks.
Output: appendix_stats/split_half_stability.{md,json}
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

RUNS = {
    "hotpotqa": ("benchmark_20260824_034107", "run_20260824_032535"),
    "musique": ("benchmark_20260824_034226", "run_20260824_033236"),
    "nq_rear": ("benchmark_20260824_034248", "run_20260824_033353"),
}
OPS = ["combsum", "rrf", "zscore", "linear"]


def load_op(bench, op, gold):
    out = {}
    pq = bench / f"op_{op}" / "per_query"
    for qdir in sorted(pq.iterdir()):
        if not qdir.is_dir():
            continue
        fr = json.loads((qdir / "filtered_results.json").read_text(encoding="utf-8"))
        qi = int(fr["query_idx"])
        g = set(gold.get(str(qi), []))
        ranked = [d for d, _ in (fr.get("filtered_ranked") or [])]
        mrr = 0.0
        for i, d in enumerate(ranked, 1):
            if d in g:
                mrr = 1.0 / i
                break
        out[qi] = mrr
    return out


def main():
    lines = [
        "# Split-Half Stability of the n=100 Confirmatory Core\n",
        "200 random disjoint 50/50 query partitions (seed=42). Reported: "
        "per-operator mean MRR across halves, its split-to-split std, and the "
        "fraction of splits where CombSUM beats RRF on BOTH halves (sign "
        "stability of the paired difference).\n",
    ]
    results = {}
    for ds, (bench_name, run_name) in RUNS.items():
        bench = PROJ / f"outputs/{ds}_benchmark/benchmarks/{bench_name}"
        gold = json.loads((PROJ / f"outputs/{ds}_benchmark/runs/{run_name}/query_gold.json").read_text(encoding="utf-8"))
        arrays = {op: load_op(bench, op, gold) for op in OPS}
        qids = sorted(set.intersection(*[set(a) for a in arrays.values()]))
        rng = np.random.default_rng(42)
        op_half_means = {op: [] for op in OPS}
        both_sign = 0
        n_valid = 0
        for _ in range(200):
            perm = rng.permutation(len(qids))
            half_a = [qids[i] for i in perm[:len(qids) // 2]]
            half_b = [qids[i] for i in perm[len(qids) // 2:]]
            means_a = {op: float(np.mean([arrays[op][q] for q in half_a])) for op in OPS}
            means_b = {op: float(np.mean([arrays[op][q] for q in half_b])) for op in OPS}
            for op in OPS:
                op_half_means[op].extend([means_a[op], means_b[op]])
            if means_a["combsum"] > means_a["rrf"] and means_b["combsum"] > means_b["rrf"]:
                both_sign += 1
            n_valid += 1
        res = {}
        for op in OPS:
            arr = np.array(op_half_means[op])
            res[op] = {"mean": round(float(arr.mean()), 4),
                       "std_across_splits": round(float(arr.std()), 4)}
        res["combsum_gt_rrf_both_halves_pct"] = round(100 * both_sign / n_valid, 1)
        results[ds] = res

    lines.append("| Dataset | Operator | mean MRR (split-half) | std across 400 halves |")
    lines.append("|---------|----------|----------------------:|----------------------:|")
    for ds, res in results.items():
        for op in OPS:
            r = res[op]
            lines.append(f"| {ds} | {op} | {r['mean']:.3f} | {r['std_across_splits']:.3f} |")
        lines.append(f"| **{ds}** | **CombSUM>RRF on both halves** | **{res['combsum_gt_rrf_both_halves_pct']}%** | |")

    out_md = OUT_DIR / "split_half_stability.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "split_half_stability.json").write_text(json.dumps(results, indent=2),
                                                       encoding="utf-8")
    print(f"wrote {out_md}")
    for ds, res in results.items():
        print(ds, "->", res["combsum_gt_rrf_both_halves_pct"], "% both-halves")


if __name__ == "__main__":
    main()
