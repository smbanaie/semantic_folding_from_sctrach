"""Item 6 — Normalization ablation (SIGIR-Final-Reviews #11).

Does the relevance-aligned-magnitude effect come from *absolute scale* or *within-retriever
separation*? Re-normalize each signal four ways before the same CombSUM/RRF fusion and test
whether the effect (CombSUM > RRF; positive World- degradation) survives.

Schemes (applied per-signal, independently to A and B):
  raw          : s as-is
  minmax       : (s-min)/(max-min)   preserves separation, kills absolute scale
  zscore       : (s-mu)/sigma        preserves separation, zero mean/unit var
  ranknorm     : percentile rank     discards within-signal geometry G_within

For each (A_scheme x B_scheme) cell and each dataset (SF+SPLADE n=100):
  - MRR_combsum, MRR_rrf, deltaMRR
  - top-1 change count (combsum vs rrf)
  - Kendall tau(combsum, rrf)
  - World- degradation = MRR_combsum(orig) - MRR_combsum(worldneg)  (effect operative?)

Output: docs/papers/Journal A/appendix_stats/normalization_ablation.{json,md}
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

ALPHA = PROJ / "docs/papers/Journal A/appendix_alpha"
STAT = PROJ / "docs/papers/Journal A/appendix_stats"
STAT.mkdir(parents=True, exist_ok=True)

N = 100
SEED = 42
RHO = 1.5
SCHEMES = ["raw", "minmax", "zscore", "ranknorm"]


def normalize(scores, scheme):
    if not scores:
        return {}
    if scheme == "raw":
        return dict(scores)
    vals = np.array(list(scores.values()), dtype=float)
    docs = list(scores.keys())
    if scheme == "minmax":
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-12:
            return {d: 0.0 for d in docs}
        out = (vals - lo) / (hi - lo)
    elif scheme == "zscore":
        sd = vals.std()
        if sd < 1e-12:
            return {d: 0.0 for d in docs}
        out = (vals - vals.mean()) / sd
    elif scheme == "ranknorm":
        order = np.argsort(np.argsort(-vals))  # rank 0..n-1 (0 = highest)
        out = (len(vals) - 1 - order) / max(1, len(vals) - 1)  # percentile 1..0
    else:
        raise ValueError(scheme)
    return {d: float(v) for d, v in zip(docs, out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="hotpotqa")
    args = ap.parse_args()
    ds = args.ds
    c1, c0 = cm.comp_path(ds, N)
    sf_raw = gp.load_components(c1)
    sp_raw = gp.load_components(c0)
    gold_path = cm.DATASETS[f"n{N}"][ds]
    gold = gp.load_gold(gold_path)
    rng = np.random.default_rng(SEED)

    cells = {}
    md = [f"# Normalization ablation — {ds} (n={N}, SF+SPLADE)",
          "",
          "| A \\ B | scheme | MRR_cs | MRR_rrf | ΔMRR | top1_chg | τ(cs,rrf) | World− deg |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for sa in SCHEMES:
        for sb in SCHEMES:
            # per-query loop
            mrr_cs_list, mrr_rrf_list, top1_chg, tau_list, worldneg_list = [], [], 0, [], []
            for qi in range(N):
                g = gold[qi]
                if not g:
                    continue
                sfn = normalize(sf_raw[qi], sa)
                spn = normalize(sp_raw[qi], sb)
                if not sfn or not spn:
                    continue
                r_cs = gp.fuse(sfn, spn, "combsum")
                r_rr = gp.fuse(sfn, spn, "rrf")
                mrr_cs_list.append(gp.mrr_of(r_cs, g))
                mrr_rrf_list.append(gp.mrr_of(r_rr, g))
                if gp.gold_rank_in(r_cs, g) != gp.gold_rank_in(r_rr, g):
                    top1_chg += 1
                tau_list.append(gp.kendall_tau_scores(
                    {d: r_cs.index(d) + 1 for d in set(r_cs)},
                    {d: r_rr.index(d) + 1 for d in set(r_rr)}))
                # World- intervention on normalized signals
                worlds = cm.build_worlds(sfn, spn, g, RHO, rng)
                wsf, wsp = worlds["worldneg"]
                mrr_wneg = gp.mrr_of(gp.fuse(wsf, wsp, "combsum"), g)
                worldneg_list.append(gp.mrr_of(r_cs, g) - mrr_wneg)
            cell = {
                "A": sa, "B": sb,
                "mrr_combsum": float(np.mean(mrr_cs_list)),
                "mrr_rrf": float(np.mean(mrr_rrf_list)),
                "deltaMRR": float(np.mean(mrr_cs_list) - np.mean(mrr_rrf_list)),
                "top1_change": top1_chg,
                "tau_cs_rrf": float(np.mean(tau_list)),
                "worldneg_degradation": float(np.mean(worldneg_list)),
            }
            cells[f"{sa}|{sb}"] = cell
            md.append(f"| {sa} | {sb} | {cell['mrr_combsum']:.4f} | {cell['mrr_rrf']:.4f} | "
                      f"{cell['deltaMRR']:+.4f} | {top1_chg} | {cell['tau_cs_rrf']:.3f} | "
                      f"{cell['worldneg_degradation']:+.4f} |")
    md.append("")
    md.append("World- deg > 0 => magnitude effect operative under that normalization. "
              "raw/raw is the paper's baseline.")
    (STAT / "normalization_ablation.json").write_text(
        json.dumps({"dataset": ds, "n": N, "cells": cells}, indent=2), encoding="utf-8")
    (STAT / "normalization_ablation.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
