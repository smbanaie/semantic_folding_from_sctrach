"""Statistical analysis for the n=50 7-operator matrix (Appendix C).

WHY THIS EXISTS
===============
SIGIR reviewer point #21 demanded a real statistical protocol - paired tests
with multiple-comparison correction - instead of the paper's earlier
"planned" placeholder (Appendix C). This script delivers it from REAL
per-query MRR arrays produced by the n=50 benchmark runs; nothing is
simulated or fabricated.

WHAT IT DOES
============
For each dataset with n=50 seven-operator benchmark output:
  1. Paired bootstrap 95% confidence intervals (10,000 resamples, seed=42).
  2. Two-sided Wilcoxon signed-rank test between every operator pair
     (21 pairs per dataset), falling back to a sign-flip permutation test
     when scipy cannot handle the difference distribution.
  3. Holm-Bonferroni family-wise correction across the 21 pairwise
     p-values within each dataset.

INPUTS
======
outputs/<dataset>_benchmark/benchmarks/benchmark_*/op_*/summary.json
  reads "operator", "num_queries" and "per_query_mrr" fields; summaries
  whose num_queries differs from EXPECT_N (50) are skipped so stale n=10
  probe runs can never be mixed into an n=50 analysis.

USAGE
=====
    .venv/Scripts/python scripts/appendix_c_stats.py

Datasets are auto-discovered from outputs/<ds>_benchmark/benchmarks by taking
the most recent benchmark dirs that contain op_*/summary.json files.

OUTPUTS
=======
docs/papers/Journal A/appendix_stats/appendix_c_<dataset>.md
  MRR + 95% CI table per operator, and pairwise Wilcoxon table:
  delta MRR, raw p, Holm-adjusted p, significance verdict.
These tables are transcribed into Appendix C of the journal manuscript.

CAVEATS
=======
With one gold passage per query and n=50 only large effects survive Holm;
this is expected and reported honestly rather than tuned away.
Bootstrap is seeded (42) so reruns are byte-identical.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
OUT = PROJ / "docs/papers/Journal A/appendix_stats"
OUT.mkdir(parents=True, exist_ok=True)

N_RESAMPLES = 10000
ALPHA = 0.05

# dataset -> list of benchmark dirs containing op_*/summary.json at n=50
DATASETS = {
    "hotpotqa": [
        PROJ / "outputs/hotpotqa_benchmark/benchmarks/benchmark_20260823_144631",
        PROJ / "outputs/hotpotqa_benchmark/benchmarks/benchmark_20260823_152138",
    ],
    "musique": None,
    "nq_rear": None,
}


def load_per_query_mrr(bench_dirs, expect_n=None):
    """Load per_query_mrr arrays for each operator from one or more bench dirs.
    If expect_n is set, skip summaries whose num_queries != expect_n (avoids
    mixing stale n=10 runs with fresh n=50 runs)."""
    data = {}
    for d in bench_dirs:
        if d is None or not d.exists():
            continue
        for opdir in sorted(d.glob("op_*")):
            sf = opdir / "summary.json"
            if not sf.exists():
                continue
            s = json.loads(sf.read_text(encoding="utf-8"))
            if expect_n is not None and s.get("num_queries") != expect_n:
                continue
            op = s.get("operator", opdir.name.replace("op_", ""))
            if op in data:
                continue  # keep first occurrence silently when filtered by n
            pq = s.get("per_query_mrr")
            if pq:
                data[op] = np.array(pq, dtype=float)
    return data


def bootstrap_ci(vals, n_resamples=N_RESAMPLES, alpha=ALPHA):
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    n = len(vals)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = vals[idx].mean()
    return float(vals.mean()), float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def holm_bonferroni(pvals):
    """Return Holm-adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = min(1.0, (m - i) * ranked[i])
        running_max = max(running_max, val)
        adj[order[i]] = running_max
    return adj


def wilcoxon_paired(a, b):
    """Paired Wilcoxon signed-rank; returns (statistic, p). Falls back to
    paired permutation test if all differences are zero."""
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        # sign-flip permutation fallback
        rng = np.random.default_rng(42)
        obs = abs(diff.mean())
        n = len(diff)
        cnt = 0
        trials = 10000
        for _ in range(trials):
            signs = rng.choice([-1.0, 1.0], size=n)
            if abs((signs * diff).mean()) >= obs:
                cnt += 1
        return obs, cnt / trials


def analyse(dataset_name, bench_dirs, expect_n=50):
    data = load_per_query_mrr(bench_dirs, expect_n=expect_n)
    ops = sorted(data.keys())
    if len(ops) < 2:
        print(f"  {dataset_name}: insufficient operators at n={expect_n} ({ops}) — skipping")
        return

    lines = []
    lines.append(f"## Appendix C — {dataset_name} (n={len(next(iter(data.values())))}, SF+SPLADE, 7 operators)\n")
    lines.append("| Operator | MRR | 95% CI |")
    lines.append("|----------|----:|--------|")

    ci_table = {}
    for op in ops:
        mean, lo, hi = bootstrap_ci(data[op])
        ci_table[op] = (mean, lo, hi)
        lines.append(f"| {op} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] |")

    lines.append("\n### Pairwise Wilcoxon signed-rank tests (Holm-adjusted)\n")
    lines.append("| Pair | ΔMRR | raw p | Holm-adjusted p | significant? |")
    lines.append("|------|-----:|------:|----------------:|--------------|")

    pairs = list(combinations(ops, 2))
    raw_p = []
    deltas = []
    for a_op, b_op in pairs:
        a, b = data[a_op], data[b_op]
        delta = float(a.mean() - b.mean())
        _, p = wilcoxon_paired(a, b)
        raw_p.append(p)
        deltas.append(delta)

    adj_p = holm_bonferroni(raw_p)

    sig_count = 0
    for (a_op, b_op), delta, rp, ap in zip(pairs, deltas, raw_p, adj_p):
        sig = "yes" if ap < ALPHA else "no"
        if ap < ALPHA:
            sig_count += 1
        lines.append(f"| {a_op} vs {b_op} | {delta:+.3f} | {rp:.4f} | {ap:.4f} | {sig} |")

    lines.append(f"\n**Significant comparisons (Holm α={ALPHA}): {sig_count}/{len(pairs)}**\n")
    lines.append("*Bootstrap CI: 10,000 resamples, seed=42. Wilcoxon signed-rank two-sided; "
                 "Holm-Bonferroni family-wise correction applied per dataset.*\n")

    out_file = OUT / f"appendix_c_{dataset_name}.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out_file}")
    print(f"  operators: {dict((k, round(v[0], 4)) for k, v in ci_table.items())}")


def main():
    for ds, dirs in DATASETS.items():
        print(f"=== {ds} ===")
        if dirs is None:
            # auto-discover: find latest benchmarks with op_*/summary.json
            base = PROJ / f"outputs/{ds}_benchmark/benchmarks"
            if not base.exists():
                print(f"  no benchmark dir for {ds}")
                continue
            candidates = sorted(base.glob("benchmark_*"), reverse=True)[:6]
            found_dirs = [c for c in candidates if any(c.glob("op_*/summary.json"))]
            if not found_dirs:
                print(f"  no valid op summaries for {ds}")
                continue
            dirs = found_dirs
        analyse(ds, dirs)


if __name__ == "__main__":
    main()
