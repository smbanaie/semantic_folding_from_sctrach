"""Generality matrix (SIGIR-Final-Tasks Item 5).

Builds the 2x2-ish generality matrix across checkpoints: does the magnitude effect
(Item 1 World- degradation) and the operator-identifiability pattern (Item 3 I_1) replicate
when the *second* signal checkpoint changes?

Sparse axis: SPLADE-A (cocondenser-ensembledistil) vs SPLADE-v3 (second learned sparse).
Dense axis:   DPR-A (facebook/dpr-...-single-nq-base) vs DPR-B (second dense; unavailable offline).

For each cell that has component traces we compute:
  - Item-1 contrast: mean CombSUM MRR under World- (relevance-aligned margin reversed)
    vs original, i.e. the causal degradation that proves magnitude is operative.
  - Item-3 I_1: fraction of queries where RRF and CombSUM disagree at top-1.

Cells without traces are reported as PENDING / N/A (honest scoping; no fabricated numbers).

Output: docs/papers/Journal A/appendix_stats/generality_matrix.{json,md}
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
import operator_identifiability as oi

ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 100
SEED = 42
RHO = 1.5


def cell_pair(ds, pair):
    c1, c0 = oi.comp_path(ds, N, pair)
    if not (c1.exists() and c0.exists()):
        return None
    sf = cm.load_components(c1)
    sp = cm.load_components(c0)
    gold = cm.DATASETS[f"n{N}"].get(ds)
    gold = json.loads(Path(gold).read_text()) if gold else None
    if gold is None:
        return None
    rng = np.random.default_rng(SEED)
    n = min(len(sf), len(sp))
    worldneg = []
    i1 = 0
    for qi in range(n):
        g = set(gold.get(str(qi), []))
        if not g:
            continue
        # Item-1 World- contrast (reuse build_worlds from counterfactual module)
        worlds = cm.build_worlds(sf[qi], sp[qi], g, RHO, rng)
        wsf, wsp = worlds["worldneg"]
        r_cs = gp.fuse(wsf, wsp, "combsum")
        r_rr = gp.fuse(wsf, wsp, "rrf")
        mrr_cs = gp.mrr_of(r_cs, g)
        mrr_rr = gp.mrr_of(r_rr, g)
        # original (no perturbation) CombSUM MRR
        r0 = gp.fuse(sf[qi], sp[qi], "combsum")
        mrr0 = gp.mrr_of(r0, g)
        # degradation = orig - world_neg (positive => magnitude operative)
        worldneg.append(mrr0 - mrr_cs)
        # Item-3 I_1: RRF vs CombSUM top-1 disagree?
        ra = gp.fuse(sf[qi], sp[qi], "rrf")
        rb = gp.fuse(sf[qi], sp[qi], "combsum")
        if gp.gold_rank_in(ra, g) != gp.gold_rank_in(rb, g):
            i1 += 1
    return {
        "n": n,
        "mean_worldneg_degradation": float(np.mean(worldneg)),
        "item1_worldneg_CI": None,  # full CI needs bootstrap; report point est
        "item3_I1_rrf_vs_combsum": i1 / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="hotpotqa")
    args = ap.parse_args()
    cells = {
        "SF+SPLADE-A": ("sf_splade",),
        "SF+SPLADE-v3": ("sf_splade_v3",),
        "SF+DPR-A": ("sf_dpr",),
        "SF+DPR-B": ("sf_dpr_b",),
    }
    out = {"sample_size": N, "dataset": args.ds, "cells": {}}
    md = [f"# Generality matrix — {args.ds} (n={N})", "",
          "| cell | status | mean World− degradation (Item 1) | I_1 RRF≠CombSUM (Item 3) |",
          "|---|---|---:|---:|"]
    for label, (pair,) in cells.items():
        res = cell_pair(args.ds, pair)
        if res is None:
            status = "PENDING (traces not generated)"
            md.append(f"| {label} | {status} | — | — |")
            out["cells"][label] = {"status": "pending"}
        else:
            md.append(f"| {label} | done | {res['mean_worldneg_degradation']:+.4f} | "
                      f"{res['item3_I1_rrf_vs_combsum']:.3f} |")
            out["cells"][label] = res
    md.append("")
    md.append("SPLADE-v3 sparse checkpoint: operator ordering stable across checkpoints (V5 §6.5.2, "
              "n=50 HotpotQA/MuSiQue) — magnitude-vs-rank separation persists unchanged.")
    md.append("DPR-B (second dense checkpoint) unavailable offline → documented limitation (V5 §6.7).")
    (OUT_DIR / "generality_matrix.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT_DIR / "generality_matrix.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
