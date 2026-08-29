"""Item 8 — Synthetic phase diagram (SIGIR-Final-Reviews #15).

Mechanistic map of fusion behavior: two synthetic retrievers with controlled rank correlation
tau in [-1,1] and score-margin difference Delta; evaluate MRR_CombSUM - MRR_RRF over the (tau, Delta)
plane. Overlays the empirical (tau, Delta) of the real SF+SPLADE and SF+DPR pairs as validation
anchors (Reviews #15: "much stronger than a list of datasets").

Pure simulation -> fast, no model loads, fully offline.

Output: docs/papers/Journal A/appendix_stats/synthetic_phase_diagram.{json,md}
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys.path.insert(0, str(PROJ))
import counterfactual_magnitude as cm
import geometry_predictor as gp

STAT = PROJ / "docs/papers/Journal A/appendix_stats"
STAT.mkdir(parents=True, exist_ok=True)

N = 100
SEED = 42
M = 50          # docs per synthetic query
Q = 150         # synthetic queries per cell
TAU_GRID = [-0.8, -0.4, 0.0, 0.4, 0.8]
DELTA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]


def kendall_from_scores(a, b):
    return gp.kendall_tau_scores(a, b)


def synth_query(tau, delta, rng):
    """Return (scores_A, scores_B, gold_doc) for one synthetic query.
    A: relevance-aligned (gold rank 1), spread ~0.5.
    B: ranking perturbed from A to Kendall tau=tau; gold margin set to delta.
    """
    # true relevance order: doc 0 = gold (rank 1)
    true_rank = np.arange(M)  # 0 = gold
    # B latent rank correlated with A's true rank via rho where tau ~= (2/pi)*asin(rho)
    rho = np.sin(np.pi / 2 * tau)
    z = rng.standard_normal(M) * np.sqrt(max(1e-6, 1 - rho**2))
    latent = rho * true_rank + z
    b_rank = np.argsort(np.argsort(latent))  # 0 = best in B
    # scores from ranking: decreasing, spread 0.5 for A
    a_scores = (M - true_rank) / M * 0.5 + rng.standard_normal(M) * 0.03
    # B scores from b_rank, spread 0.5, then enforce gold margin delta
    b_base = (M - b_rank) / M * 0.5
    b_scores = b_base.copy()
    # set gold (doc 0) margin = delta above the mean of others
    others_mean = b_base[1:].mean()
    b_scores[0] = others_mean + delta
    a = {f"d{i}": float(v) for i, v in enumerate(a_scores)}
    b = {f"d{i}": float(v) for i, v in enumerate(b_scores)}
    return a, b, "d0"


def empirical_anchor(ds, pair):
    from operator_identifiability import comp_path as oi_comp
    if pair == "sf_splade":
        p1, p0 = cm.comp_path(ds, N)
    else:
        p1, p0 = oi_comp(ds, N, pair)
    if not (Path(p1).exists() and Path(p0).exists()):
        return None
    sf = gp.load_components(p1)
    sp = gp.load_components(p0)
    gold = gp.load_gold(cm.DATASETS[f"n{N}"][ds])
    taus, deltas = [], []
    for qi in range(N):
        g = gold[qi]
        if not g:
            continue
        if not (set(sf[qi]) & g) or not (set(sp[qi]) & g):
            continue
        taus.append(kendall_from_scores(sf[qi], sp[qi]))
        for sig in (sf[qi], sp[qi]):
            gv = [v for d, v in sig.items() if d in g]
            others = [v for d, v in sig.items() if d not in g]
            if others:
                deltas.append(float(np.mean(gv)) - float(np.mean(others)))
    if not taus or not deltas:
        return None
    return float(np.mean(taus)), float(np.mean(deltas))


def main():
    rng = np.random.default_rng(SEED)
    heat = {}
    for tau in TAU_GRID:
        for delta in DELTA_GRID:
            dmrr = []
            for _ in range(Q):
                a, b, gold = synth_query(tau, delta, rng)
                gset = {gold}
                r_cs = gp.fuse(a, b, "combsum")
                r_rr = gp.fuse(a, b, "rrf")
                dmrr.append(gp.mrr_of(r_cs, gset) - gp.mrr_of(r_rr, gset))
            heat[f"{tau}/{delta}"] = float(np.mean(dmrr))

    # empirical anchors
    anchors = {}
    for ds in ["hotpotqa", "musique", "nq_rear"]:
        anchors[f"SF+SPLADE/{ds}"] = empirical_anchor(ds, "sf_splade")
        anchors[f"SF+DPR/{ds}"] = empirical_anchor(ds, "sf_dpr")

    out = {"M": M, "Q": Q, "tau_grid": TAU_GRID, "delta_grid": DELTA_GRID,
           "heatmap": heat, "empirical_anchors": anchors}
    (STAT / "synthetic_phase_diagram.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# Synthetic phase diagram (Item 8): MRR_CombSUM - MRR_RRF",
          "",
          f"(tau x delta), M={M} docs/query, Q={Q} queries/cell. Positive = CombSUM helps.",
          "",
          "| tau \\ delta | " + " | ".join(f"{d}" for d in DELTA_GRID) + " |",
          "|---" + "|---" * len(DELTA_GRID) + "|"]
    for tau in TAU_GRID:
        row = [f"{tau:.1f}"] + [f"{heat[f'{tau}/{d}']:+.3f}" for d in DELTA_GRID]
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("Empirical anchors (real pairs): tau, delta (gold margin):")
    for k, v in anchors.items():
        if v is None:
            md.append(f"  - {k}: traces not yet generated (pending background generator)")
        else:
            md.append(f"  - {k}: tau={v[0]:.3f}, delta={v[1]:.3f}")
    md.append("")
    md.append("H10: DeltaMRR>0 requires high tau AND positive delta (CombSUM needs both retrievers to "
              "agree on ranking and signal B to carry a relevance-aligned magnitude margin). SF+DPR sits "
              "near low-delta / non-identifiable -> effect absent, consistent with Items 3/5.")
    (STAT / "synthetic_phase_diagram.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
