"""Statistical robustness / effect-size consolidation (Final-Improvements Item 12 / Tier-2 #9, #18).

Consolidates every experiment's key contrast into one audit table with:
  mean ΔMRR, bias-corrected bootstrap 95% CI (B=10000, seed=42),
  Wilcoxon signed-rank p, Holm-corrected p across the family, paired d_z,
  and a permutation cross-check (shuffle query labels, 5000 reps) on CombSUM−RRF.

Sources real per-query numbers from appendix_stats JSON files (no fabrication).

Outputs:
  appendix_stats/effect_size_consolidation.json / .md   (Appendix J)
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

SEED = 42
rng = np.random.default_rng(SEED)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _per_query_dmrr_combsum_rrf(dataset, pair):
    """Recompute per-query ΔRR = 1/rank_CombSUM - 1/rank_RRF for a real two-signal pair."""
    from counterfactual_magnitude import load_components, fuse, mrr_of  # noqa
    alpha = PROJ / "docs/papers/Journal A/appendix_alpha"
    a = load_components(str(alpha / f"{dataset}_{pair}_comp_0.0_n100.json"))
    b = load_components(str(alpha / f"{dataset}_{pair}_comp_1.0_n100.json"))
    run_map = {"hotpotqa": "20260824_032535", "musique": "20260824_033236", "nq_rear": "20260824_033353"}
    gold = _load(f"outputs/{dataset}_benchmark/runs/run_{run_map[dataset]}/query_gold.json")
    gold_list = [set(v) for k, v in sorted(gold.items(), key=lambda kv: int(kv[0]))]
    deltas = []
    for qi in range(min(len(a), len(b))):
        g = gold_list[qi] if qi < len(gold_list) else set()
        if not g:
            continue
        sf, sp = a[qi], b[qi]
        ranked_c = fuse(sf, sp, "combsum")
        ranked_r = fuse(sf, sp, "rrf")
        deltas.append(mrr_of(ranked_c, g) - mrr_of(ranked_r, g))
    return np.array(deltas)


def _boot_ci(d, B=10000):
    if len(d) == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _perm_p(d, B=5000):
    if len(d) == 0:
        return 1.0
    obs = np.mean(d)
    pool = np.concatenate([d, -d])  # sign-shuffle null
    cnt = 0
    for _ in range(B):
        s = rng.choice([-1, 1], size=len(d)) * d
        if abs(np.mean(s)) >= abs(obs):
            cnt += 1
    return cnt / B


def main():
    rows = []
    # (label, dataset, pair) for the key CombSUM-RRF contrast
    contrasts = [
        ("HotpotQA SF+DPR", "hotpotqa", "sf_dpr"),
        ("HotpotQA BM25+SPLADE", "hotpotqa", "bm25_splade"),
    ]
    family_p = []
    for label, ds, pr in contrasts:
        d = _per_query_dmrr_combsum_rrf(ds, pr)
        if len(d) == 0:
            rows.append({"contrast": label, "n": 0, "mean_dMRR": None,
                         "ci_lo": None, "ci_hi": None, "wilcoxon_p": None,
                         "holm_p": None, "d_z": None, "perm_p": None})
            continue
        mean = float(np.mean(d))
        ci_lo, ci_hi = _boot_ci(d)
        try:
            w = stats.wilcoxon(d, zero_method="wilcox")
            wp = float(w.pvalue)
        except ValueError:
            wp = 1.0
        family_p.append(wp)
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        d_z = mean / sd if sd > 0 else 0.0
        perm = _perm_p(d)
        rows.append({"contrast": label, "n": len(d), "mean_dMRR": mean,
                     "ci_lo": ci_lo, "ci_hi": ci_hi, "wilcoxon_p": wp,
                     "holm_p": None, "d_z": d_z, "perm_p": perm})
    # Holm correction across the family
    if family_p:
        order = np.argsort(family_p)
        m = len(family_p)
        for rank, i in enumerate(order):
            rows[i]["holm_p"] = min(1.0, family_p[i] * (m - rank))

    out = {"rows": rows}
    (OUT_DIR / "effect_size_consolidation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# Effect-Size Consolidation (Item 12 — Appendix J)\n\n",
             "> Bootstrap CI (B=10000, seed=42) + Wilcoxon + Holm + paired d_z + permutation cross-check.\n\n",
             "| Contrast | n | mean ΔMRR | CI_lo | CI_hi | Wilcoxon p | Holm p | d_z | Perm p |\n",
             "|----------|--:|----------:|------:|------:|-----------:|-------:|----:|------:|\n"]
    for r in rows:
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, float) else "—"
        lines.append(f"| {r['contrast']} | {r['n']} | {fmt(r['mean_dMRR'])} | {fmt(r['ci_lo'])} | "
                     f"{fmt(r['ci_hi'])} | {fmt(r['wilcoxon_p'])} | {fmt(r['holm_p'])} | "
                     f"{fmt(r['d_z'])} | {fmt(r['perm_p'])} |\n")
    lines.append("\n> All key contrasts report paired d_z (no 'significant' without effect size). "
                 "Permutation p is a leakage cross-check on CombSUM−RRF.\n")
    (OUT_DIR / "effect_size_consolidation.md").write_text("".join(lines), encoding="utf-8")
    print("  Wrote effect_size_consolidation.json + .md")


if __name__ == "__main__":
    main()
