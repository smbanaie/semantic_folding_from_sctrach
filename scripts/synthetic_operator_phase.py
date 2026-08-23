"""Synthetic operator phase diagram (Final-Improvements Item 15).

Controlled simulation where fusion operators CAN differ, upgrading the
paper's 2-document toy (§7.2): gold is NOT rank-1 by rank order. Each
condition defines how "goldness" relates to rank vs magnitude; in
magnitude-relevant regimes signal B carries a large spike on a mid-rank doc.

Conditions:
  pools N            : {20, 100, 500}
  magnitude family   : {concentrated, spread, heavy-tail}  (signal-A shape)
  signal-B scale     : {1, 10, 100}
  relevance regime   : rank-dominant      gold = rank-1, no spikes
                       magnitude-dominant gold = mid-rank doc + big B-spike
                       mixed              half trials each

Metric: top-1 accuracy per operator over N_TRIALS per cell.
Deterministic: seed derived from the condition key.

Usage:  .venv/Scripts/python scripts/synthetic_operator_phase.py
Output: appendix_stats/operator_phase_diagram.{md,json}
        figures/operator_phase_diagram.png (if matplotlib present)
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from semantic_folding import fusion_operators as fo  # noqa: E402

OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
FIG_DIR = PROJ / "docs/papers/Journal A/figures"

OPS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]
FAMILIES = ["concentrated", "spread", "heavy-tail"]
SCALES = [1, 10, 100]
POOLS = [20, 100, 500]
REGIMES = ["rank-dominant", "magnitude-dominant", "mixed"]
N_TRIALS = 30


def make_scores(rng, n, family):
    """Strictly decreasing scores shaped by the distribution family."""
    base = np.linspace(1.0, 0.05, n)
    if family == "concentrated":
        scores = 0.9 + 0.1 * base + rng.normal(0, 0.005, n)
    elif family == "spread":
        scores = base + rng.normal(0, 0.02, n)
    else:  # heavy-tail
        scores = base * rng.lognormal(0, 0.15, n)
    scores = np.sort(scores)[::-1]
    return {f"doc_{i:06d}": float(s - scores.min() + 0.01) for i, s in enumerate(scores)}


def run_cell(seed, n, family, scale, regime):
    rng = np.random.default_rng(seed)
    hits = {op: 0 for op in OPS}
    lo_rank = max(2, n // 20)          # gold lives mid-rank, never top-2
    hi_rank = max(3, n // 5)
    for _ in range(N_TRIALS):
        a = make_scores(rng, n, family)
        docs = list(a.keys())
        b_vals = np.sort(rng.uniform(0.2, 1.0, n))[::-1]
        b = {d: float(v) for d, v in zip(docs, b_vals)}
        if regime == "rank-dominant":
            gold = docs[0]
        else:
            gold = docs[int(rng.integers(lo_rank, hi_rank))]
            b[gold] = float(b_vals[0] * (1.5 + rng.random()))  # magnitude spike
        b_scaled = {d: v * scale for d, v in b.items()}
        k = max(60, n)
        for op in OPS:
            fused = fo.fuse(op, a, b_scaled, alpha=0.3, k=k)
            top = max(fused.items(), key=lambda kv: kv[1])[0]
            hits[op] += int(top == gold)
    return {op: hits[op] / N_TRIALS for op in OPS}


def run_all():
    results = {}
    for n in POOLS:
        for family in FAMILIES:
            for scale in SCALES:
                for regime in REGIMES:
                    key = f"n={n}|{family}|x{scale}|{regime}"
                    seed = zlib.crc32(key.encode("utf-8"))  # stable across processes
                    results[key] = run_cell(seed, n, family, scale, regime)
    return results


def aggregate(results):
    rows = []
    for family in FAMILIES:
        for regime in REGIMES:
            keys = [k for k in results
                    if f"|{family}|" in k and k.endswith(f"|{regime}")]
            means = {op: float(np.mean([results[k][op] for k in keys])) for op in OPS}
            spread = max(means.values()) - min(means.values())
            winner = max(means, key=means.get)
            rows.append({"family": family, "regime": regime,
                         "means": means, "winner": winner, "spread": spread})
    return rows


def write_report(results, rows):
    lines = [
        "# Synthetic Operator Phase Diagram (Item 15)\n",
        f"{N_TRIALS} trials/cell; N in {{{','.join(map(str, POOLS))}}}; "
        "families shape signal A; signal B carries a magnitude spike on the "
        f"gold doc in magnitude-relevant regimes; B scaled x{{{','.join(map(str, SCALES))}}}; "
        "deterministic seeds.\n",
        "## Mean top-1 accuracy (averaged over pool sizes and scales)\n",
        "| family | regime | " + " | ".join(OPS) + " | winner |",
        "|--------|--------|" + "|".join(["------:"] * len(OPS)) + "|--------|",
    ]
    for r in rows:
        mark = "" if r["spread"] > 0.01 else " *(all tie)*"
        cells = " | ".join(f"{r['means'][op]:.3f}" for op in OPS)
        lines.append(f"| {r['family']} | {r['regime']} | {cells} | **{r['winner']}**{mark} |")

    out_md = OUT_DIR / "operator_phase_diagram.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "operator_phase_diagram.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")
    diffs = sum(1 for v in results.values()
                if len(set(round(x, 6) for x in v.values())) > 1)
    print(f"{diffs}/{len(results)} cells differentiate operators")


def write_plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plot")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r['regime']}\n{r['family']}" for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["means"]["rrf"] for r in rows], 0.36,
           label="rrf (rank-only)", color="#4C72B0")
    ax.bar(x + 0.18, [r["means"]["combsum"] for r in rows], 0.36,
           label="combsum (score-space)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Rank-only vs score-space fusion across synthetic conditions")
    ax.legend()
    fig.tight_layout()
    out_png = FIG_DIR / "operator_phase_diagram.png"
    fig.savefig(out_png, dpi=150)
    print(f"plot saved: {out_png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = run_all()
    rows = aggregate(results)
    write_report(results, rows)
    write_plot(rows)


if __name__ == "__main__":
    main()
