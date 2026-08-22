"""
Synthetic Magnitude-Control Experiment (RQ3, Reviewer #7/#8).

Goal: isolate score magnitude as a causal factor by holding RANK fixed while
manipulating MAGNITUDE, then applying all fusion operators. If a rank-only
operator (RRF/Borda) cannot distinguish the conditions but a score-space operator
(CombSUM/CombMNZ/Linear) can, magnitude is causally operative — independent of
ranking correlation. This is the clean causal test the reviewer demanded.

Run:
  .venv\Scripts\python semantic_folding\synthetic_magnitude_experiment.py \
      --output results/synthetic_magnitude_<ts>.json

Reads operators from fusion_operators.fuse().
"""
import argparse
import json
from pathlib import Path

from semantic_folding.fusion_operators import fuse

OPERATORS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]


def make_scores(a_score, b_score, a_rank=1, b_rank=2, n_distract=5):
    """Build two retriever score dicts with A ranked above B but a_score/b_score
    set by the condition. Distractors fill ranks 3..n to exercise rank-only ops."""
    sa, sb = {}, {}
    # Retriever A: A top, B second
    sa["docA"] = a_score
    sa["docB"] = b_score * 0.5  # A's view of B weaker
    sb["docA"] = a_score * 0.5
    sb["docB"] = b_score
    # distractors (low scores, ranks 3+)
    for i in range(n_distract):
        sa[f"d{i}"] = 1.0 / (i + 3)
        sb[f"d{i}"] = 1.0 / (i + 3)
    return sa, sb


def condition_set():
    """Rank(A)=1, Rank(B)=2 fixed; vary magnitude margin."""
    return [
        ("large_45_12", 45.0, 12.0),
        ("large_40_15", 40.0, 15.0),
        ("med_35_20", 35.0, 20.0),
        ("small_30_25", 30.0, 25.0),
        ("tiny_21_19", 21.0, 19.0),
        ("rev_12_45", 12.0, 45.0),
        ("rev_18_30", 18.0, 30.0),
    ]


def run():
    results = {"conditions": [], "operators": OPERATORS}
    for name, a, b in condition_set():
        sa, sb = make_scores(a, b)
        row = {"condition": name, "scoreA": a, "scoreB": b, "margin": a - b,
               "A_above_B": {}}
        for op in OPERATORS:
            fused = fuse(op, sa, sb, alpha=0.3, rrf_k=60)
            # rank dict by fused score desc
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            order = [d for d, _ in ranked]
            a_above_b = order.index("docA") < order.index("docB")
            row["A_above_B"][op] = a_above_b
        results["conditions"].append(row)
    return results


def rank_invariant_check():
    """Verify RRF invariance under monotonic transforms (Proposition 1 evidence)."""
    import math
    sa = {"docA": 45.0, "docB": 12.0, "d0": 1.0, "d1": 0.5}
    sb = {"docA": 22.5, "docB": 24.0, "d0": 1.0, "d1": 0.5}
    base = fuse("rrf", sa, sb, rrf_k=60)
    transforms = {
        "log": lambda x: math.log(x + 1),
        "sqrt": lambda x: math.sqrt(x),
        "exp": lambda x: math.exp(x / 10.0),
        "sigmoid": lambda x: 1 / (1 + math.exp(-x / 20.0)),
    }
    inv = {}
    for tname, f in transforms.items():
        sa_t = {k: f(v) for k, v in sa.items()}
        sb_t = {k: f(v) for k, v in sb.items()}
        ft = fuse("rrf", sa_t, sb_t, rrf_k=60)
        # compare rank order
        inv[tname] = (sorted(base, key=lambda d: base[d], reverse=True) ==
                      sorted(ft, key=lambda d: ft[d], reverse=True))
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    res = run()
    res["rrf_rank_invariant"] = rank_invariant_check()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    # console summary
    print("Synthetic magnitude-control results:")
    for c in res["conditions"]:
        flags = " ".join(f"{op}={'✓' if c['A_above_B'][op] else '✗'}"
                         for op in OPERATORS)
        print(f"  {c['condition']:12s} margin={c['margin']:+6.1f}  {flags}")
    print("RRF rank-invariance under transforms:",
          res["rrf_rank_invariant"])


if __name__ == "__main__":
    main()
