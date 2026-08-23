"""Score-Margin vs Fusion-Error analysis (Final-Improvements Item 14).

Reviewer Priority 2: for each query compute the gold-vs-best-negative margin
in each signal, then measure how often rank-only fusion (RRF) mis-ranks gold
while score-space fusion (CombSUM) succeeds — binned by margin.

Data: real component scores from alpha-sweep endpoint runs (SF=maxnorm SF,
SPLADE=maxnorm SPLADE), with query_gold.json for gold doc ids.

Outputs:
  docs/papers/Journal A/appendix_stats/margin_vs_error.md   (table)
  docs/papers/Journal A/figures/margin_vs_error.png         (plot, if matplotlib)
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys_paths = str(PROJ)
import sys
sys.path.insert(0, sys_paths)
from semantic_folding import fusion_operators as fo

ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
FIG_DIR = PROJ / "docs/papers/Journal A/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "hotpotqa": (
        ALPHA_DIR / "hotpotqa_comp_1.0.json",
        ALPHA_DIR / "hotpotqa_comp_0.0.json",
        PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/query_gold.json",
    ),
    "musique": (
        ALPHA_DIR / "musique_comp_1.0.json",
        ALPHA_DIR / "musique_comp_0.0.json",
        PROJ / "outputs/musique_benchmark/runs/run_20260822_191925/query_gold.json",
    ),
    "scifact": (
        ALPHA_DIR / "scifact_comp_1.0.json",
        ALPHA_DIR / "scifact_comp_0.0.json",
        PROJ / "outputs/scifact_benchmark/runs/run_20260822_191507/query_gold.json",
    ),
}

MARGIN_BINS = [(-0.50, -0.10), (-0.10, 0.00), (0.00, 0.10), (0.10, 0.30), (0.30, 1.01)]
BIN_LABELS = ["neg[<-.10]", "small[-.10,0)", "pos[0,.10)", "med[.10,.30)", "large[.30+]"]


def load(path):
    return [dict(q["results"]) for q in json.loads(path.read_text(encoding="utf-8"))]


def fused_top(fused):
    return max(fused.items(), key=lambda kv: kv[1])[0]


def main():
    all_rows = []
    plot_data = {}
    for ds, (sf_path, sp_path, gold_path) in DATASETS.items():
        sf = load(sf_path)
        sp = load(sp_path)
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        rows = []
        for qi in range(len(sf)):
            g = set(gold.get(str(qi), []))
            if not g or not sf[qi] or not sp[qi]:
                continue

            def margin(scores):
                gold_best = max((s for d, s in scores.items() if d in g), default=None)
                neg_best = max((s for d, s in scores.items() if d not in g), default=0.0)
                if gold_best is None:
                    return None
                mx = max(abs(s) for s in scores.values()) or 1.0
                return (gold_best - neg_best) / mx  # normalized margin (can be negative)

            m_sf = margin(sf[qi])
            m_sp = margin(sp[qi])
            # joint margin: mean of the two signals' margins (fusion sees both)
            if m_sf is None or m_sp is None:
                m_joint = None
            else:
                m_joint = (m_sf + m_sp) / 2
            rrf_top = fused_top(fo.fuse("rrf", sf[qi], sp[qi], alpha=0.3, k=60))
            comb_top = fused_top(fo.fuse("combsum", sf[qi], sp[qi], alpha=0.3, k=60))
            rrf_wrong = rrf_top not in g
            comb_right = comb_top in g
            rows.append({
                "query": qi,
                "margin_sf": round(m_sf, 3) if m_sf is not None else None,
                "margin_splade": round(m_sp, 3) if m_sp is not None else None,
                "margin_joint": round(m_joint, 3) if m_joint is not None else None,
                "rrf_wrong": rrf_wrong,
                "comb_right": comb_right,
                "rescue": rrf_wrong and comb_right,
                "both_fail": rrf_wrong and not comb_right,
            })
        all_rows.append((ds, rows))
        plot_data[ds] = rows

    # Build table: per dataset, rescue rate by mean-signal margin bin
    # Build table: per dataset, rescue rate by joint margin bin
    lines = [
        "# Score Margin vs Fusion Error (Item 14)\n",
        "Per query, per signal: margin = (best gold score − best non-gold score)/max|score| "
        "(negative = gold scores *below* a distractor in that signal). Joint margin = mean of the two signals' margins. "
        "'Rescue' = RRF top-1 misses gold while CombSUM top-1 hits it; 'both fail' = neither recovers gold.\n",
    ]
    summary_for_plot = {}
    for ds, rows in all_rows:
        lines.append(f"\n### {ds} (n={len(rows)} queries)\n")
        lines.append("| Joint-margin bin | #queries | #rescues | rescue rate | #both-fail |")
        lines.append("|------------------|---------:|---------:|------------:|-----------:|")
        rates = []
        for (lo, hi), lab in zip(MARGIN_BINS, BIN_LABELS):
            bucket = [r for r in rows if r["margin_joint"] is not None and lo <= r["margin_joint"] < hi]
            n_resc = sum(1 for r in bucket if r["rescue"])
            n_fail = sum(1 for r in bucket if r["both_fail"])
            rate = n_resc / len(bucket) if bucket else None
            rates.append((rate, len(bucket)))
            rate_s = f"{rate:.2f}" if rate is not None else "—"
            lines.append(f"| {lab} | {len(bucket)} | {n_resc} | {rate_s} | {n_fail} |")
        summary_for_plot[ds] = rates
        total_resc = sum(1 for r in rows if r["rescue"])
        disc = sum(1 for r in rows if r["rrf_wrong"] != r["comb_right"])
        neg_side = [r for r in rows if r["margin_joint"] is not None and r["margin_joint"] < 0]
        resc_neg = sum(1 for r in neg_side if r["rescue"])
        pos_side = [r for r in rows if r["margin_joint"] is not None and 0 <= r["margin_joint"]]
        resc_pos = sum(1 for r in pos_side if r["rescue"])
        lines.append(f"\nOverall: {total_resc}/{len(rows)} rescues. "
                     f"Negative-joint-margin queries: {resc_neg}/{len(neg_side)} rescued. "
                     f"Non-negative-margin queries: {resc_pos}/{len(pos_side)} rescued. "
                     f"RRF/CombSUM top-1 disagreement: {disc}/{len(rows)} queries.\n")

    out_md = OUT_DIR / "margin_vs_error.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(BIN_LABELS))
        width = 0.26
        for i, (ds, rates) in enumerate(summary_for_plot.items()):
            vals = [0 if v is None else v for (v, cnt) in rates]
            ax.bar(x + (i - 1) * width, vals, width, label=ds)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_xlabel("Joint normalized margin: mean(gold−distractor margins of both signals)")
        ax.set_ylabel("P(RRF top-1 wrong ∧ CombSUM top-1 correct)")
        ax.set_title("Rank-only fusion failure concentrates at small or negative margins")
        ax.legend()
        fig.tight_layout()
        out_png = FIG_DIR / "margin_vs_error.png"
        fig.savefig(out_png, dpi=150)
        print(f"plot saved: {out_png}")
    except ImportError:
        print("matplotlib unavailable — table only")


if __name__ == "__main__":
    main()
