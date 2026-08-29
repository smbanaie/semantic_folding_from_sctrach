"""Candidate-pool sensitivity (Final-Improvements Item 13 / Tier-2 #10, #33).

Defends against the reviewer's "second-largest methodological weakness": artificial
candidate pools. For each real two-signal pair, build candidate pools of size K by
taking the union of top-K docs from each signal, re-fuse (CombSUM/RRF), and report
MRR stability across K in {20, 50, 100, full}. If ΔMRR is stable for K>=50, the
effect is not a pool-size artifact.

Outputs:
  appendix_stats/candidate_pool_sensitivity.json / .md
(The V5 §3 candidate-set subsection documenting the 9 points from #33 is written
 by the caller / paper-author step; this script provides the empirical stability.)
"""

import json
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
ALPHA = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

sys_path = PROJ / "scripts"
import sys
sys.path.insert(0, str(sys_path))
from counterfactual_magnitude import load_components, fuse, mrr_of  # noqa

POOL_SIZES = [20, 50, 100, 100000]  # 100000 = full
RUN_MAP = {"hotpotqa": "20260824_032535", "musique": "20260824_033236", "nq_rear": "20260824_033353"}


def topk_pool(sf, sp, k):
    """Union of top-K docs from each signal as the candidate pool."""
    sf_top = sorted(sf.items(), key=lambda kv: -kv[1])[:k]
    sp_top = sorted(sp.items(), key=lambda kv: -kv[1])[:k]
    pool = set(d for d, _ in sf_top) | set(d for d, _ in sp_top)
    return pool


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "nq_rear"])
    ap.add_argument("--pair", required=True, help="e.g. sf_dpr, bm25_splade")
    ap.add_argument("--max-queries", type=int, default=100)
    args = ap.parse_args()

    ds, pr = args.dataset, args.pair
    a = load_components(str(ALPHA / f"{ds}_{pr}_comp_0.0_n100.json"))[:args.max_queries]
    b = load_components(str(ALPHA / f"{ds}_{pr}_comp_1.0_n100.json"))[:args.max_queries]
    gold = json.loads(Path(f"outputs/{ds}_benchmark/runs/run_{RUN_MAP[ds]}/query_gold.json").read_text(encoding="utf-8"))
    gold_list = [set(v) for k, v in sorted(gold.items(), key=lambda kv: int(kv[0]))]

    results = {"dataset": ds, "pair": pr, "pools": {}}
    for k in POOL_SIZES:
        c_mrr, r_mrr = [], []
        for qi in range(min(len(a), len(b))):
            g = gold_list[qi] if qi < len(gold_list) else set()
            if not g:
                continue
            pool = topk_pool(a[qi], b[qi], k)
            # restrict fusion to the pool
            sf_p = {d: v for d, v in a[qi].items() if d in pool}
            sp_p = {d: v for d, v in b[qi].items() if d in pool}
            if not sf_p or not sp_p:
                continue
            ranked_c = fuse(sf_p, sp_p, "combsum")
            ranked_r = fuse(sf_p, sp_p, "rrf")
            c_mrr.append(mrr_of(ranked_c, g))
            r_mrr.append(mrr_of(ranked_r, g))
        results["pools"][str(k)] = {
            "combsum_mrr": float(np.mean(c_mrr)) if c_mrr else None,
            "rrf_mrr": float(np.mean(r_mrr)) if r_mrr else None,
            "delta_combsum_rrf": (float(np.mean(c_mrr)) - float(np.mean(r_mrr))) if (c_mrr and r_mrr) else None,
            "n": len(c_mrr),
        }

    (OUT_DIR / f"candidate_pool_sensitivity_{ds}_{pr}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [f"# Candidate-Pool Sensitivity (Item 13) -- {ds.title()} / {pr}\n\n",
             "| Pool K | CombSUM MRR | RRF MRR | ΔCombSUM−RRF | n |\n",
             "|-------:|------------:|--------:|-------------:|--:|\n"]
    for k in POOL_SIZES:
        r = results["pools"][str(k)]
        d = r["delta_combsum_rrf"]
        dstr = f"{d:+.4f}" if isinstance(d, float) else "—"
        lines.append(f"| {k if k < 100000 else 'full'} | {r['combsum_mrr']:.4f} | {r['rrf_mrr']:.4f} | {dstr} | {r['n']} |\n")
    lines.append("\n> If ΔCombSUM−RRF is stable for K>=50, the magnitude effect is not a candidate-pool-size artifact.\n")
    (OUT_DIR / f"candidate_pool_sensitivity_{ds}_{pr}.md").write_text("".join(lines), encoding="utf-8")
    print(f"  Wrote candidate_pool_sensitivity_{ds}_{pr}.json + .md")


if __name__ == "__main__":
    main()
