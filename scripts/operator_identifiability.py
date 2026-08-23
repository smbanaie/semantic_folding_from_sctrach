"""Operator Identifiability metric (Final-Improvements Item 25).

Question: how often do different fusion operators induce IDENTICAL rankings
for a given (query, signal-pair)? If two operators always coincide, operator
choice is unidentifiable for that pair -- no tuning can help. This explains
the DPR-pair behavior in section 6.5 (RRF == CombSUM at MRR level).

Uses real per-document component scores captured during the alpha sweep:
  <ds>_comp_1.0.json = maxnorm(SF),  <ds>_comp_0.0.json = maxnorm(SPLADE)
For each query, fuse under each operator pair-combination and count ranking
agreement (exact list equality after deterministic sort).

Outputs markdown table + JSON artifact.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys.path.insert(0, str(PROJ))
from semantic_folding import fusion_operators as fo

ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]

DATASETS = {
    "hotpotqa": (
        ALPHA_DIR / "hotpotqa_comp_1.0.json",
        ALPHA_DIR / "hotpotqa_comp_0.0.json",
    ),
    "musique": (
        ALPHA_DIR / "musique_comp_1.0.json",
        ALPHA_DIR / "musique_comp_0.0.json",
    ),
    "scifact": (
        ALPHA_DIR / "scifact_comp_1.0.json",
        ALPHA_DIR / "scifact_comp_0.0.json",
    ),
}


def load(path):
    return [dict(q["results"]) for q in json.loads(path.read_text(encoding="utf-8"))]


def ranked(fused):
    return [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def main():
    rows = []
    detail = {}
    for ds, (sf_path, sp_path) in DATASETS.items():
        if not sf_path.exists() or not sp_path.exists():
            print(f"[skip] {ds}")
            continue
        sf = load(sf_path)
        sp = load(sp_path)
        # precompute fused ranking per operator per query
        per_op = {}
        for op in OPS:
            per_op[op] = [
                ranked(fo.fuse(op, sf[qi], sp[qi], alpha=0.3, k=60))
                for qi in range(len(sf))
            ]
        n_q = len(sf)
        for a, b in combinations(OPS, 2):
            agree = sum(1 for qi in range(n_q) if per_op[a][qi] == per_op[b][qi])
            ident = agree / n_q
            rows.append((ds, a, b, ident))
            detail.setdefault(ds, []).append(
                {"op_a": a, "op_b": b, "identical_rankings": agree,
                 "n_queries": n_q, "identifiability_gap": round(1 - ident, 3)})
        # summary: mean pairwise agreement per dataset
        vals = [r[3] for r in rows if r[0] == ds]
        print(f"{ds}: mean pairwise identical-ranking rate = {np.mean(vals):.3f} "
              f"over {len(vals)} operator pairs x {n_q} queries")

    lines = [
        "# Operator Identifiability (Item 25)\n",
        "Fraction of queries for which two operators produce **identical fused** "
        "rankings on the same candidate pool (SF+SPLADE, alpha=0.3, k=60; real "
        "component scores from the alpha-sweep endpoint runs).\n",
        "| Dataset | Op pair | identical | n | gap |",
        "|---------|---------|----------:|--:|----:|",
    ]
    for ds, a, b, ident in sorted(rows):
        n_q = len(load(DATASETS[ds][0]))
        lines.append(f"| {ds} | {a} vs {b} | {ident:.2f} | {n_q} | {1-ident:.2f} |")

    # headline: linear vs rrf and combsum vs rrf (the pairs the paper discusses)
    lines.append("\n## Headline pairs\n")
    lines.append("| Dataset | linear vs rrf | combsum vs rrf | minmax vs zscore |")
    lines.append("|---------|--------------:|---------------:|-----------------:|")
    for ds in DATASETS:
        def get(a, b):
            for d2, x, y, v in rows:
                if d2 == ds and {x, y} == {a, b}:
                    return f"{v:.2f}"
            return "-"
        lines.append(f"| {ds} | {get('linear','rrf')} | {get('combsum','rrf')} | {get('minmax','zscore')} |")

    out_md = OUT_DIR / "operator_identifiability.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "operator_identifiability.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8")
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
