"""Operator x RetrieverPair factorial interaction screen (V2-S6; reviewer #29,
resolver Experiment 6).

Question: does the OPERATOR x PAIR interaction carry signal beyond main
effects? This directly tests the paper's central claim that operator
effectiveness depends on the joint score geometry of the pair.

Design constraints (honest about power):
  - Per-query MRR arrays come from the four-pair runs. Available at n>=50:
      hotpotqa : SF+SPLADE (n=100 new), SF+DPR (n=50 Aug-22),
                 BM25+SPLADE (n=50), BM25+DPR (n=50)
      nq_rear  : SF+SPLADE (n=100 new), SF+DPR (n=50), BM25 pairs (n=50)
    Queries are the SAME query ids within a dataset across pairs only where
    runs overlap; we use the common prefix of query indices and require the
    same gold sets.
  - Operators compared: rrf vs combsum (the two families the claim is about).
  - Test: within each query i and dataset d, compute paired difference
        delta_i = MRR_combsum(i) - MRR_rrf(i)
    for each pair p. Interaction contrast for pair p vs baseline q:
        D_i = delta_i(p) - delta_i(q)
    Under H0 (no interaction), sign of D_i is exchangeable -> two-sided
    sign-flip permutation test on mean(D), stratified by dataset.
    10,000 resamples, seed=42.

This is an interaction SCREEN: with n=50-100 the test cannot be confirmatory;
we report effect sizes (mean D, Cohen's dz) and exact p_perm transparently.

Output: appendix_stats/factorial_interaction.md (+ .json)
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

# run dirs providing per_query fused results (op_linear/op_rrf/op_combsum)
# pair identities verified against each benchmark's config.yml (signal_a/retriever_b)
RUNS = {
    ("hotpotqa", "SF+SPLADE", 100): (
        PROJ / "outputs/hotpotqa_benchmark/benchmarks/benchmark_20260824_034107",
        PROJ / "outputs/hotpotqa_benchmark/runs/run_20260824_032535/query_gold.json"),
    ("hotpotqa", "SF+DPR", 50): (
        PROJ / "outputs/hotpotqa_benchmark/benchmarks/benchmark_20260822_150019",
        None),
    ("hotpotqa", "BM25+SPLADE", 50): (
        PROJ / "outputs/hotpotqa_benchmark/benchmarks/benchmark_20260822_152459",
        None),
    ("hotpotqa", "BM25+DPR", 50): (
        PROJ / "outputs/hotpotqa_benchmark/benchmarks/benchmark_20260822_152011",
        None),
    ("nq_rear", "SF+SPLADE", 100): (
        PROJ / "outputs/nq_rear_benchmark/benchmarks/benchmark_20260824_034248",
        PROJ / "outputs/nq_rear_benchmark/runs/run_20260824_033353/query_gold.json"),
    ("nq_rear", "SF+DPR", 50): (
        PROJ / "outputs/nq_rear_benchmark/benchmarks/benchmark_20260822_151029",
        None),
}
OPS = ["rrf", "combsum"]


def load_per_query_mrr(bench_dir, op, gold):
    """Per-query MRR array aligned by query index from per_query/<idx>/filtered_results.json."""
    pq_dir = bench_dir / f"op_{op}" / "per_query"
    if not pq_dir.exists():
        return None
    out = {}
    for qdir in sorted(pq_dir.iterdir()):
        if not qdir.is_dir():
            continue
        fr = qdir / "filtered_results.json"
        if not fr.exists():
            continue
        d = json.loads(fr.read_text(encoding="utf-8"))
        qi = int(d["query_idx"])
        gset = set(gold.get(str(qi), []))
        if not gset:
            continue
        # fused ranking from filtered_ranked (already restricted to pool)
        ranked = [doc for doc, _ in (d.get("filtered_ranked") or [])]
        mrr = 0.0
        for i, doc in enumerate(ranked, 1):
            if doc in gset:
                mrr = 1.0 / i
                break
        out[qi] = {"mrr": mrr, "gold": sorted(gset)}
    return out


def find_gold_for(bench_dir):
    """Gold file discovery: benchmarks store config.yml pointing at run_dir."""
    import yaml
    cfg = bench_dir / "config.yml"
    if cfg.exists():
        y = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        rd = y.get("phase2", {}).get("run_dir")
        if rd:
            p = Path(rd) / "query_gold.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
    return None


def cohens_dz(diffs):
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1))


def perm_signflip(diffs, n_resamples=10_000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    obs = abs(d.mean())
    if len(d) == 0 or obs == 0:
        return obs, 1.0
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(d)))
    means = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(means >= obs - 1e-12) + 1) / (n_resamples + 1))
    return float(obs), min(1.0, p)


def main():
    lines = [
        "# Operator x Pair Factorial Interaction Screen (S6)\n",
        "Contrast: CombSUM minus RRF (per query), differenced between retriever "
        "pairs. H0: the pair difference-of-differences has zero mean "
        "(sign-flip permutation, 10k resamples, seed=42, two-sided).\n"
        "This is a screening analysis, not a powered confirmatory test.\n",
        "| Dataset | Pair A | Pair B | n | mean Δ(A) | mean Δ(B) | mean D=A−B | dz | p_perm |",
        "|---------|--------|--------|--:|----------:|----------:|-----------:|---:|-------:|",
    ]
    results = {}

    for ds in ("hotpotqa", "nq_rear"):
        data = {}
        for (d2, pair, expected_n), (bench, gold_path) in RUNS.items():
            if d2 != ds:
                continue
            gold = (json.loads(gold_path.read_text(encoding="utf-8"))
                    if gold_path and Path(gold_path).exists()
                    else find_gold_for(bench))
            if gold is None:
                print(f"[skip] {ds}/{pair}: no gold")
                continue
            arrays = {}
            for op in OPS:
                arr = load_per_query_mrr(bench, op, gold)
                if arr is None:
                    arrays = None
                    break
                arrays[op] = arr
            if not arrays:
                print(f"[skip] {ds}/{pair}: no per_query data")
                continue
            data[pair] = arrays

        pairs = list(data.keys())
        if len(pairs) < 2:
            continue
        base = pairs[0]  # conventionally SF+SPLADE listed first
        for other in pairs[1:]:
            qs = (set(data[base][OPS[0]]) & set(data[base][OPS[1]])
                  & set(data[other][OPS[0]]) & set(data[other][OPS[1]]))
            qs = sorted(qs)
            if len(qs) < 20:
                continue
            delta_a = [data[base]["combsum"][i]["mrr"] - data[base]["rrf"][i]["mrr"]
                       for i in qs]
            delta_b = [data[other]["combsum"][i]["mrr"] - data[other]["rrf"][i]["mrr"]
                       for i in qs]
            D = [a - b for a, b in zip(delta_a, delta_b)]
            obs, p = perm_signflip(D)
            dz = cohens_dz(D)
            key = f"{ds}:{base}-vs-{other}"
            results[key] = {
                "n": len(qs), "mean_delta_A": round(float(np.mean(delta_a)), 4),
                "mean_delta_B": round(float(np.mean(delta_b)), 4),
                "mean_D": round(float(np.mean(D)), 4),
                "dz": round(dz, 3), "p_perm": p}
            lines.append(
                f"| {ds} | {base} | {other} | {len(qs)} | "
                f"{np.mean(delta_a):+.3f} | {np.mean(delta_b):+.3f} | "
                f"{np.mean(D):+.3f} | {dz:+.2f} | {p:.4f} |")

    out_md = OUT_DIR / "factorial_interaction.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "factorial_interaction.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
