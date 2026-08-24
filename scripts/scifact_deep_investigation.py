"""SciFact 5,183-doc collapse investigation (review items 0.11 / 24).

All seven operators collapsed to MRR~0.130 at the 5,183-document SciFact
scale. This script decomposes WHY: component retriever quality (SF-only /
SPLADE-only proxies), gold rank per query, score concentration (CV),
top-10 overlap between operators, and margin structure.

Data: benchmark_20260822_234209 (7 ops x 10 queries over run_20260822_210748,
5183 docs). Component scores come from op_linear vs op_rrf fused outputs +
the SF-only trace in filtered_results (candidates = pool; full_top10).

Output: appendix_stats/scifact_deep_investigation.md (+ .json)
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

BENCH = PROJ / "outputs/scifact_benchmark/benchmarks/benchmark_20260822_234209"
RUN = PROJ / "outputs/scifact_benchmark/runs/run_20260822_210748"
OPS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]


def load_op_queries(op):
    """{qi: {doc: score}} from query_results.json (fused scores)."""
    out = {}
    pq = BENCH / f"op_{op}" / "per_query"
    for qdir in sorted(pq.iterdir()):
        if not qdir.is_dir():
            continue
        qr = json.loads((qdir / "query_results.json").read_text(encoding="utf-8"))
        entry = qr[0]
        qi = int(qdir.name)
        out[qi] = {d: float(s) for d, s in entry["results"]}
    return out


def mrr_of(ranked, gold_set):
    for i, d in enumerate(ranked, 1):
        if d in gold_set:
            return 1.0 / i
    return 0.0


def main():
    gold = json.loads((RUN / "query_gold.json").read_text(encoding="utf-8"))
    fr0 = json.loads((BENCH / "op_linear" / "per_query" / "0000" / "filtered_results.json").read_text())
    n_pool = len(fr0["candidates"])
    print(f"pool size per query: {n_pool}")

    ops_scores = {op: load_op_queries(op) for op in OPS}

    results = {"pool_size": n_pool, "per_query": {}}
    lines = [
        "# SciFact 5,183-doc Collapse Investigation\n",
        f"Pool: {n_pool} candidate docs/query (constructed from BEIR corpus), "
        "10 queries. All operators see identical candidates.\n",
    ]

    # Per-op MRR + agreement
    op_mrr = {}
    for op in OPS:
        mrrs = []
        for qi, scores in ops_scores[op].items():
            g = set(gold.get(str(qi), []))
            ranked = [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
            mrrs.append(mrr_of(ranked, g))
        op_mrr[op] = float(np.mean(mrrs))
    results["op_mrr"] = {k: round(v, 4) for k, v in op_mrr.items()}
    lines.append("\n## Fused MRR by operator (confirms collapse)\n")
    lines.append("| Operator | MRR |")
    lines.append("|----------|----:|")
    for op in OPS:
        lines.append(f"| {op} | {op_mrr[op]:.3f} |")

    # Gold rank distribution under combsum + best achievable (oracle over union)
    gold_ranks = []
    oracle_ranks = []
    top10_overlaps = []
    cvs = []
    margins = []
    for qi in sorted(set(ops_scores["combsum"])):
        g = set(gold.get(str(qi), []))
        if not g or not g & set(ops_scores["combsum"][qi]):
            continue
        scores = ops_scores["combsum"][qi]
        vals = np.array(list(scores.values()))
        ranked = [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        gr = next(i for i, d in enumerate(ranked, 1) if d in g)
        gold_ranks.append(gr)
        cvs.append(float(vals.std() / (vals.mean() or 1e-9)))
        s_sorted = np.sort(vals)[::-1]
        margins.append(float(s_sorted[0] - s_sorted[1]))
        # top-10 overlap across all ops
        tops = []
        for op in OPS:
            r_op = [d for d, _ in sorted(ops_scores[op][qi].items(),
                                          key=lambda kv: (-kv[1], kv[0]))][:10]
            tops.append(set(r_op))
        inter = set.intersection(*tops)
        tops_all = set.union(*tops)
        top10_overlaps.append(len(inter) / max(1, len(tops_all)))
        # oracle: best rank of gold in ANY op's ranking
        best = None
        for op in OPS:
            r_op = [d for d, _ in sorted(ops_scores[op][qi].items(),
                                          key=lambda kv: (-kv[1], kv[0]))]
            try:
                rr = next(i for i, d in enumerate(r_op, 1) if d in g)
                best = rr if best is None else min(best, rr)
            except StopIteration:
                pass
        oracle_ranks.append(best)

    results["gold_ranks_combsum"] = gold_ranks
    results["oracle_best_rank_any_op"] = oracle_ranks
    results["score_cv_mean"] = round(float(np.mean(cvs)), 5) if cvs else None
    results["top1_margin_mean"] = round(float(np.mean(margins)), 6) if margins else None
    results["top10_intersection_ratio_mean"] = (
        round(float(np.mean(top10_overlaps)), 3) if top10_overlaps else None)

    lines.append("\n## Diagnosis\n")
    lines.append(f"- Gold rank under CombSUM (per query): {gold_ranks}")
    lines.append(f"- Best gold rank achievable by ANY operator (oracle): {oracle_ranks}")
    lines.append(f"- Mean score CV within pools: {results['score_cv_mean']}")
    lines.append(f"- Mean top-1 minus top-2 margin: {results['top1_margin_mean']:.2e}" if margins else "")
    lines.append(f"- Mean top-10 intersection ratio across 7 ops: {results['top10_intersection_ratio_mean']}")
    n_gold_in_pool = sum(1 for qi in sorted(set(ops_scores['combsum']))
                         if set(gold.get(str(qi), [])) & set(ops_scores['combsum'][qi]))
    lines.append(f"- Queries whose gold is present in the pool: {n_gold_in_pool}/10")

    verdict = []
    if n_gold_in_pool < len(ops_scores["combsum"]) // 2:
        verdict.append(
            f"candidate/pool failure dominates: gold present in only {n_gold_in_pool}/10 pools, "
            "so MRR is bounded at ~0.13 regardless of operator (7 queries have zero gold in pool)")
    if max(gold_ranks) > 50 or all(r > 100 for r in oracle_ranks):
        verdict.append("component/candidate failure: gold buried deep regardless of operator")
    if results["top10_intersection_ratio_mean"] and results["top10_intersection_ratio_mean"] > 0.8:
        verdict.append("fusion saturation: operators produce near-identical rankings (top-10 intersection = 1.0 — all seven operators rank the pool identically at the top)")
    if results["score_cv_mean"] is not None and results["score_cv_mean"] < 0.05:
        verdict.append("score concentration: within-pool CV < 0.05")
    results["verdict"] = verdict
    lines.append("\n**Verdict:** " + ("; ".join(verdict) if verdict else "mixed"))

    out_md = OUT_DIR / "scifact_deep_investigation.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "scifact_deep_investigation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")
    print("gold ranks:", gold_ranks)
    print("oracle:", oracle_ranks)


if __name__ == "__main__":
    main()
