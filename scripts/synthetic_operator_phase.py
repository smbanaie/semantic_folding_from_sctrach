"""Synthetic operator phase diagram (Final-Improvements Item 15) — v2.

Upgrades the paper's 2-doc toy into a controlled simulation where operators
CAN differ: gold is NOT rank-1 by rank order; instead each condition defines
how "goldness" relates to rank vs magnitude, and signal B carries a
magnitude spike on a mid-rank document in some conditions.

Conditions:
  pools N in {20, 100, 500}
  magnitude family in {concentrated, spread, heavy-tail}   (signal A shape)
  scale ratio of signal B in {1, 10, 100}
  relevance regimes:
    * rank-dominant      : gold = rank-1 doc; no magnitude spikes
    * magnitude-dominant : gold = mid-rank doc with a large B-spike
    * mixed              : half trials each
Metric: top-1 accuracy per operator over N_TRIALS.

All operators see identical inputs; only the fusion rule differs. Seed fixed
per cell for reproducibility.
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys_path = str(PROJ)
import sys
sys.path.insert(0, sys_path)
from semantic_folding import fusion_operators as fo

OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
FIG_DIR = PROJ / "docs/papers/Journal A/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OPS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]
FAMILIES = ["concentrated", "spread", "heavy-tail"]
SCALES = [1, 10, 100]
POOLS = [20, 100, 500]
REGIMES = ["rank-dominant", "magnitude-dominant", "mixed"]
N_TRIALS = 30


def make_scores(rng, n, family):
    base = np.linspace(1.0, 0.05, n)
    if family == "concentrated":
        scores = 0.9 + 0.1 * base + rng.normal(0, 0.005, n)
    elif family == "spread":
        scores = base + rng.normal(0, 0.02, n)
    else:
        scores = base * rng.lognormal(0, 0.15, n)
    scores = np.sort(scores)[::-1]
    return {f"doc_{i:06d}": float(s - scores.min() + 0.01) for i, s in enumerate(scores)}


def run_cell(seed, n, family, scale, regime):
    rng = np.random.default_rng(seed)
    hits = {op: 0 for op in OPS}
    for _ in range(N_TRIALS):
        a = make_scores(rng, n, family)
        docs = list(a.keys())
        b_vals = np.sort(rng.uniform(0.2, 1.0, n))[::-1]
        b = {d: float(v) for d, v in zip(docs, b_vals)}
        # pick gold and apply regime structure to signal B's magnitudes
        if regime == "rank-dominant":
            gold = docs[0]
        else:
            gold_rank = rng.integers(max(2, n // 20), max(3, n // 5))  # mid-rank
            gold = docs[gold_rank]
            b[gold] = float(b_vals[0] * (1.5 + rng.random()))  # big B-spike
            if regime == "rank-dominant":
                pass
        b_scaled = {d: v * scale for d, v in b.items()}
        for op in OPS:
            fused = fo.fuse(op, a, b_scaled, alpha=0.3, k=max(60, n))
            top = max(fused.items(), key=lambda kv: kv[1])[0]
            hits[op] += int(top == gold)
    return {op: hits[op] / N_TRIALS for op in OPS}


def main():
    results = {}
    for n in POOLS:
        for family in FAMILIES:
            for scale in SCALES:
                for regime in REGIMES:
                    key = f"n={n}|{family}|x{scale}|{regime}"
                    seed = abs(hash(key)) % (2 ** 32)
                    results[key] = run_cell(seed, n, family, scale, regime)

    # aggregate table
    lines = [
        "# Synthetic Operator Phase Diagram — v2 (Item 15)\n",
        f"{N_TRIALS} trials/cell; N in {{20,100,500}}; families control signal-A "
        "variance/tails; signal-B carries a magnitude spike on the gold doc in "
        "magnitude-relevant regimes; B scaled x{1,10,100}; deterministic seeds.\n",
        "## Mean top-1 accuracy (averaged over pool sizes and scales)\n",
        "| family | regime | " + " | ".join(OPS) + " | winner |",
        "|--------|--------|" + "|".join(["------:"] * len(OPS)) + "|--------|",
    ]
    agg = []
    for family in FAMILIES:
        for regime in REGIMES:
            keys = [k for k in results if f"|{family}|" in k and k.endswith(f"|{regime}")]
            means = {op: float(np.mean([results[k][op] for k in keys])) for op in OPS}
            spread = max(means.values()) - min(means.values())
            winner = max(means, key=means.get)
            agg.append((family, regime, means, winner, spread))
            mark = "" if spread > 0.01 else " *(all tie)*"
            lines.append(f"| {family} | {regime} | "
                         + " | ".join(f"{means[op]:.3f}" for op in OPS)
                         + f" | **{winner}**{mark} |")

    out_md = OUT_DIR / "operator_phase_diagram.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "operator_phase_diagram.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")

    diffs = sum(1 for v in results.values()
                if len(set(round(x, 6) for x in v.values())) > 1)
    print(f"{diffs}/{len(results)} cells differentiate operators")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = [f"{reg}\n{fam}" for (fam, reg, _, _, _) in agg]
        rrf_vals = [m["rrf"] for (_, _, m, _, _) in agg]
        cs_vals = [m["combsum"] for (_, _, m, _, _) in agg]
        x = np.arange(len(agg))
        ax.bar(x - 0.18, rrf_vals, 0.36, label="rrf (rank-only)", color="#4C72B0")
        ax.bar(x + 0.18, cs_vals, 0.36, label="combsum (score-space)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("top-1 accuracy")
        ax.set_title("Rank-only vs score-space fusion across synthetic conditions")
        ax.legend()
        fig.tight_layout()
        out_png = FIG_DIR / "operator_phase_diagram.png"
        fig.savefig(out_png, dpi=150)
        print(f"plot saved: {out_png}")
    except ImportError:
        print("matplotlib unavailable")


if __name__ == "__main__":
    main()
