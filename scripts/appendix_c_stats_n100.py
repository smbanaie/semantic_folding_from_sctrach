"""Statistical analysis for the n=100 confirmatory core (Appendix C, V2 refresh).

Extends scripts/appendix_c_stats.py to the n=100 runs produced by
Final-Improvements-V2 S1/S2 (hotpotqa/musique/nq_rear, SF+SPLADE, 7 operators).

Protocol identical to the n=50 tables:
  1. paired bootstrap 95% CI (10k resamples, seed=42)
  2. two-sided Wilcoxon signed-rank per operator pair (21 pairs/dataset),
     sign-flip permutation fallback
  3. Holm-Bonferroni across the 21 p-values within each dataset

INPUT: outputs/<ds>_benchmark/benchmarks/<n100 dir>/op_*/summary.json
OUTPUT: appendix_stats/appendix_c_<ds>_n100.md (+ .json)
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
OUT = PROJ / "docs/papers/Journal A/appendix_stats"
N_RESAMPLES = 10_000
ALPHA = 0.05

DATASETS = {
    "hotpotqa": PROJ / "outputs/hotpotqa_benchmark/benchmarks/benchmark_20260824_034107",
    "musique": PROJ / "outputs/musique_benchmark/benchmarks/benchmark_20260824_034226",
    "nq_rear": PROJ / "outputs/nq_rear_benchmark/benchmarks/benchmark_20260824_034248",
}
EXPECT_N = 100


def load_arrays(bench):
    data = {}
    for op_dir in sorted(bench.glob("op_*")):
        s = op_dir / "summary.json"
        if not s.exists():
            continue
        d = json.loads(s.read_text(encoding="utf-8"))
        if d.get("num_queries") != EXPECT_N:
            return None  # mixed-n guard
        arr = d.get("per_query_mrr")
        if not arr:
            return None
        data[d["operator"]] = np.asarray(arr, dtype=float)
    return data or None


def bootstrap_ci(x, n=N_RESAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_p(a, b):
    diff = a - b
    nz = diff[diff != 0]
    if len(nz) < 3:
        # sign-flip permutation fallback on the mean difference
        obs = abs(diff.mean())
        rng = np.random.default_rng(42)
        signs = rng.choice([-1.0, 1.0], size=(10_000, len(diff)))
        null = np.abs((signs * diff).mean(axis=1))
        return float((np.sum(null >= obs - 1e-12) + 1) / 10_001)
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    for ds, bench in DATASETS.items():
        data = load_arrays(bench)
        if not data:
            print(f"[skip] {ds}: no clean n={EXPECT_N} arrays at {bench.name}")
            continue
        ops = sorted(data)
        lines = [
            f"# Appendix C — {ds} (SF+SPLADE, n={EXPECT_N}, 7 operators)\n",
            "Paired bootstrap 95% CIs; Wilcoxon signed-rank pairwise tests with "
            "Holm correction across all 21 comparisons. Generated from "
            f"`{bench.name}`.\n",
            "| Operator | MRR | 95% CI |", "|----------|----:|--------|",
        ]
        for op in ops:
            x = data[op]
            lo, hi = bootstrap_ci(x)
            lines.append(f"| {op} | {x.mean():.3f} | [{lo:.3f}, {hi:.3f}] |")

        lines += ["\n## Pairwise tests\n",
                  "| Pair | ΔMRR | raw p | Holm p | sig |",
                  "|------|-----:|------:|-------:|-----|"]
        pairs = list(combinations(ops, 2))
        raw = []
        deltas = []
        for a, b in pairs:
            deltas.append(data[a].mean() - data[b].mean())
            raw.append(paired_p(data[a], data[b]))
        adj = holm(raw)
        for (a, b), dm, rp, ap in zip(pairs, deltas, raw, adj):
            sig = "yes" if ap < ALPHA else "no"
            lines.append(f"| {a} vs {b} | {dm:+.3f} | {rp:.4f} | {ap:.4f} | {sig} |")

        nsig = sum(1 for ap in adj if ap < ALPHA)
        lines.append(f"\n{nsig}/{len(pairs)} comparisons survive Holm at α={ALPHA}.\n")

        out_md = OUT / f"appendix_c_{ds}_n100.md"
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (OUT / f"appendix_c_{ds}_n100.json").write_text(
            json.dumps({op: {"mrr": float(v.mean()), "values": v.tolist()}
                        for op, v in data.items()}, indent=2), encoding="utf-8")
        print(f"wrote {out_md} ({nsig}/{len(pairs)} survive Holm)")


if __name__ == "__main__":
    main()
