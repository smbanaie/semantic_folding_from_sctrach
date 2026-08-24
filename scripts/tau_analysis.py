"""Tau analysis (Final-Improvements-V2 S4; reviewer #15/#16, resolver Exp 11/12).

Separates two quantities the manuscript previously conflated:
  tau_signal   = Kendall tau between the two component rankings (SF vs SPLADE)
                 for one query — the "complementarity diagnostic".
  tau_operator = Kendall tau between two operators' fused rankings (e.g. RRF
                 vs CombSUM) for one query — an operator-agreement measure.

Also computes:
  fusion gain  = MRR(fused) - max(MRR(A), MRR(B))          per query/operator
  top-k disagreement metrics (top-1 / top-3 overlap, gold-rank difference)
  complementarity 4-cell table: A-correct/B-correct x operator MRR

Correlates Fusion Gain with tau_signal AND with local disagreement separately,
per dataset (never pooled across datasets).

Data: real SF+SPLADE endpoint traces + gold (same artifacts as §7.5).
Output: appendix_stats/tau_analysis.md (+ .json)
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from semantic_folding import fusion_operators as fo  # noqa: E402

OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
ALPHA = PROJ / "docs/papers/Journal A/appendix_alpha"

TRACES = {
    "hotpotqa": (
        ALPHA / "hotpotqa_comp_1.0.json",
        ALPHA / "hotpotqa_comp_0.0.json",
        PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/query_gold.json",
    ),
    "musique": (
        ALPHA / "musique_comp_1.0.json",
        ALPHA / "musique_comp_0.0.json",
        PROJ / "outputs/musique_benchmark/runs/run_20260822_191925/query_gold.json",
    ),
    "scifact": (
        ALPHA / "scifact_comp_1.0.json",
        ALPHA / "scifact_comp_0.0.json",
        PROJ / "outputs/scifact_benchmark/runs/run_20260822_191507/query_gold.json",
    ),
}
OPS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]


def load_scores(path):
    return [dict(q["results"]) for q in json.loads(path.read_text(encoding="utf-8"))]


def ranked(fused):
    return [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def mrr(ranked_list, gold):
    for i, d in enumerate(ranked_list, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def mrr_of(ranked_list, gold):
    for i, d in enumerate(ranked_list, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def main():
    lines = [
        "# Tau Analysis: signal-level vs operator-level agreement (S4)\n",
        "tau_signal  = Kendall(SF ranking, SPLADE ranking) per query — component "
        "complementarity diagnostic.\n"
        "tau_operator= Kendall between two operators' fused rankings — operator "
        "agreement, NOT a complementarity measure.\n"
        "Fusion Gain = MRR(fused) - max(MRR(A), MRR(B)) per query.\n",
    ]
    results = {}

    for ds, (sf_path, sp_path, gold_path) in TRACES.items():
        sf = load_scores(sf_path)
        sp = load_scores(sp_path)
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        n_q = len(sf)

        per_query = []
        fused_ranked = {op: [] for op in OPS}
        for qi in range(n_q):
            g = set(gold.get(str(qi), []))
            if not g:
                continue
            ra = ranked(sf[qi])
            rb = ranked(sp[qi])
            common = set(ra[:60]) | set(rb[:60])
            tau_s = kendalltau(
                [ra.index(d) if d in ra else 999 for d in sorted(common)],
                [rb.index(d) if d in rb else 999 for d in sorted(common)],
            ).statistic
            _ma = mrr(ra, g)
            _mb = mrr(rb, g)
            m_a, m_b = _ma, _mb
            row = {
                "query": qi,
                "tau_signal": None if np.isnan(tau_s) else round(float(tau_s), 3),
                "mrr_a": m_a, "mrr_b": m_b,
                "a_correct_top1": ra and ra[0] in g,
                "b_correct_top1": rb and rb[0] in g,
                "top1_disagree": bool(ra) and bool(rb) and ra[0] != rb[0],
                "top3_overlap": len(set(ra[:3]) & set(rb[:3])) / 3,
                "gold_rank_diff": abs(
                    (ra.index(next(iter(g))) + 1 if next(iter(g)) in ra else 999)
                    - (rb.index(next(iter(g))) + 1 if next(iter(g)) in rb else 999)),
                "ops": {},
            }
            for op in OPS:
                rl = ranked(fo.fuse(op, sf[qi], sp[qi], alpha=0.3, k=60))
                fused_ranked[op].append(rl)
                fg = mrr(rl, g) - max(m_a, m_b)
                row["ops"][op] = {"mrr": mrr(rl, g), "fusion_gain": round(fg, 4)}
            per_query.append(row)

        # tau_operator matrices (query-averaged pairwise)
        tau_op_table = {}
        for a, b in combinations(OPS, 2):
            taus = []
            for qi, rl_a in enumerate(fused_ranked[a]):
                rl_b = fused_ranked[b][qi]
                common = sorted(set(rl_a[:30]) | set(rl_b[:30]))
                va = [rl_a.index(d) if d in rl_a else 999 for d in common]
                vb = [rl_b.index(d) if d in rl_b else 999 for d in common]
                t = kendalltau(va, vb).statistic
                if not np.isnan(t):
                    taus.append(t)
            tau_op_table[f"{a}|{b}"] = round(float(np.mean(taus)), 3) if taus else None

        # correlations: fusion gain (combsum & rrf) vs tau_signal and top1_disagree
        def corr(sub, key_fn):
            xs = [key_fn(r) for r in sub]
            ys = [r["ops"]["combsum"]["fusion_gain"] for r in sub]
            from scipy.stats import spearmanr
            rho, p = spearmanr(xs, ys)
            return (round(float(rho), 3) if not np.isnan(rho) else None,
                    round(float(p), 4) if not np.isnan(p) else None)

        valid = [r for r in per_query if r["tau_signal"] is not None]
        rho_tau, p_tau = corr(valid, lambda r: r["tau_signal"])
        dis = [r for r in per_query]
        rho_dis, p_dis = corr(dis, lambda r: 1.0 if r["top1_disagree"] else 0.0)

        # 4-cell complementarity table (top-1 correctness of each signal)
        cells = {"TT": [], "TF": [], "FT": [], "FF": []}
        for r in per_query:
            key = ("T" if r["a_correct_top1"] else "F") + ("T" if r["b_correct_top1"] else "F")
            cells[key].append(r)
        cell_rows = {}
        for key, rs in cells.items():
            if not rs:
                continue
            cell_rows[key] = {
                op: round(float(np.mean([r["ops"][op]["mrr"] for r in rs])), 3)
                for op in OPS
            }
            cell_rows[key]["n"] = len(rs)

        results[ds] = {
            "n_queries": len(per_query),
            "mean_tau_signal": round(float(np.mean(
                [r["tau_signal"] for r in valid])), 3) if valid else None,
            "rho_fusgain_vs_tau_signal": {"rho": rho_tau, "p": p_tau},
            "rho_fusgain_vs_top1_disagree": {"rho": rho_dis, "p": p_dis},
            "tau_operator_mean": tau_op_table,
            "cells": cell_rows,
        }

    # ---- write report ----
    for ds, res in results.items():
        lines.append(f"\n## {ds} (n={res['n_queries']} queries)\n")
        lines.append(f"- mean tau_signal: **{res['mean_tau_signal']}**")
        rt = res["rho_fusgain_vs_tau_signal"]
        rd = res["rho_fusgain_vs_top1_disagree"]
        lines.append(f"- Fusion Gain(combsum) vs tau_signal: rho={rt['rho']}, p={rt['p']}")
        lines.append(f"- Fusion Gain(combsum) vs top-1 disagreement: rho={rd['rho']}, p={rd['p']}")
        lines.append("\n### Complementarity 4-cell table (A=top-1 correct? B=top-1 correct?)\n")
        lines.append("| Cell (A,B) | n | " + " | ".join(OPS) + " | mean MRR over ops |")
        lines.append("|-----|--:|" + "---:|" * (len(OPS) + 1))
        for key in ("TT", "TF", "FT", "FF"):
            cr = res["cells"].get(key)
            if not cr:
                continue
            vals = [cr.get(op) for op in OPS]
            mean_all = float(np.mean([v for v in vals if v is not None]))
            lines.append(f"| {key} | {cr['n']} | "
                         + " | ".join("—" if v is None else f"{v:.2f}" for v in vals)
                         + f" | {mean_all:.3f} |")

    out_md = OUT_DIR / "tau_analysis.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "tau_analysis.json").write_text(json.dumps(results, indent=2),
                                               encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
