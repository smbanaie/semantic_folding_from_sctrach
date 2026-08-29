"""Top-rank ΔRR decomposition (SIGIR-Final-Tasks Item 4).

Answers #38-4: "why a small number of queries produce the MRR difference." Decomposes
the total CombSUM−RRF MRR gap into per-query contributions ΔRR_q = RR_CombSUM,q − RR_RRF,q
and measures how concentrated the gain is at the decision boundary (Type A/B), not spread
evenly. Reuses the SF+SPLADE component traces (same as Items 1–2); no re-index.

Outputs: docs/papers/Journal A/appendix_stats/toprank_decomposition_n{N}.{json,md}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys.path.insert(0, str(PROJ))
import counterfactual_magnitude as cm
import geometry_predictor as gp

ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["hotpotqa", "musique", "nq_rear", "scifact", "2wikimultihopqa"]
NQ_REAR_GOLD = PROJ / "outputs/nq_rear_benchmark/runs/run_20260824_033353/query_gold.json"
HOTPOT_GOLD = PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/query_gold.json"


def comp_path(ds, n):
    if n == 100:
        return ALPHA_DIR / f"{ds}_comp_1.0_n100.json", ALPHA_DIR / f"{ds}_comp_0.0_n100.json"
    return ALPHA_DIR / f"{ds}_comp_1.0.json", ALPHA_DIR / f"{ds}_comp_0.0.json"


def gold_for(ds, n):
    ds_map = cm.DATASETS[f"n{n}"]
    if ds not in ds_map:
        return None
    return load_gold(ds_map[ds])


def load_gold(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = [raw[str(i)] if str(i) in raw else raw[i] for i in range(len(raw))]
        return [set(v) for v in items]
    return [set((q.get("gold") or q.get("gold_ids") or q.get("answer_docs")) or []) for q in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, choices=[10, 100])
    ap.add_argument("--ds", default=None, help="single dataset; omit for all")
    args = ap.parse_args()
    N = args.n
    datasets = [args.ds] if args.ds else DATASETS

    results = {}
    for ds in datasets:
        c1, c0 = comp_path(ds, N)
        if not (c1.exists() and c0.exists()):
            print(f"SKIP {ds}: missing {c1.name}/{c0.name}")
            continue
        sf_list = cm.load_components(c1)
        sp_list = cm.load_components(c0)
        gold = gold_for(ds, N)
        if gold is None:
            print(f"SKIP {ds}: no gold for n={N}")
        n = min(len(sf_list), len(sp_list), len(gold))
        deltas = []
        types = []
        margins = []
        for qi in range(n):
            f_sf, f_sp = sf_list[qi], sp_list[qi]
            r_cs = gp.fuse(f_sf, f_sp, "combsum")
            r_rr = gp.fuse(f_sf, f_sp, "rrf")
            d_cs = gp.mrr_of(r_cs, gold[qi])
            d_rr = gp.mrr_of(r_rr, gold[qi])
            dr = d_cs - d_rr
            deltas.append(dr)
            rc = gp.gold_rank_in(r_cs, gold[qi])
            rr = gp.gold_rank_in(r_rr, gold[qi])
            types.append(gp.classify_type(rc, rr))
            feats = gp.geometry_features(f_sf, f_sp, gold[qi])
            margins.append(feats["joint_margin"])
        deltas = np.array(deltas)
        margins = np.array(margins)
        abs_d = np.abs(deltas)
        total = abs_d.sum()
        # concentration: top-k% share of total |ΔRR|
        order = np.argsort(abs_d)[::-1]
        sorted_abs = abs_d[order]
        cum = np.cumsum(sorted_abs) / (total if total > 0 else 1.0)
        top10_share = float(cum[int(0.1 * n) - 1]) if n >= 10 else float(cum[0])
        top20_share = float(cum[int(0.2 * n) - 1]) if n >= 5 else 1.0
        n_zero = int((deltas == 0).sum())
        n_pos = int((deltas > 0).sum())
        n_neg = int((deltas < 0).sum())
        # cross-tab with type
        type_counts = {t: int((np.array(types) == t).sum()) for t in ["A", "B", "C", "D"]}
        # mean margin by type
        mean_margin_by_type = {}
        for t in ["A", "B", "C", "D"]:
            idx = np.where(np.array(types) == t)[0]
            mean_margin_by_type[t] = float(margins[idx].mean()) if len(idx) else None
        # H6: >=80% of |ΔRR| from <20% queries
        h6_pass = top20_share >= 0.80
        results[ds] = {
            "n": n,
            "total_abs_delta": float(total),
            "mean_delta": float(deltas.mean()),
            "n_zero": n_zero, "n_pos": n_pos, "n_neg": n_neg,
            "top10_share_hi_delta": top10_share,
            "top20_share_hi_delta": top20_share,
            "h6_ge_80pct_in_20pct": h6_pass,
            "type_counts": type_counts,
            "mean_joint_margin_by_type": mean_margin_by_type,
        }
        print(f"  {ds}: n={n} meanΔRR={deltas.mean():+.4f} |ΔRR| top20share={top20_share:.3f} "
              f"zero={n_zero} pos={n_pos} neg={n_neg} types={type_counts} H6={'PASS' if h6_pass else 'fail'}")
    out = {"sample_size": N, "results": results}
    name = f"toprank_decomposition_n{N}"
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # markdown
    lines = [f"# Top-rank ΔRR decomposition — n={N}", ""]
    lines.append("| dataset | n | mean ΔRR | top20% share of |ΔRR| | H6 (≥80% in 20%) | #zero | #pos | #neg | Type A/B/C/D |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds, r in results.items():
        tm = r["mean_joint_margin_by_type"]
        tc = r["type_counts"]
        lines.append(f"| {ds} | {r['n']} | {r['mean_delta']:+.4f} | {r['top20_share_hi_delta']:.3f} | "
                     f"{'PASS' if r['h6_ge_80pct_in_20pct'] else 'fail'} | {r['n_zero']} | {r['n_pos']} | {r['n_neg']} | "
                     f"A={tc['A']}/B={tc['B']}/C={tc['C']}/D={tc['D']} |")
    lines.append("")
    lines.append("Mean joint_margin by type (negative-margin regime = where magnitude fusion wins):")
    for ds, r in results.items():
        tm = r["mean_joint_margin_by_type"]
        lines.append(f"  {ds}: A={tm['A']}, B={tm['B']}, C={tm['C']}, D={tm['D']}")
    (OUT_DIR / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {name}.json/.md")


if __name__ == "__main__":
    main()
