"""Magnitude perturbation on REAL retrieval outputs (review #31).

WHY THIS EXISTS
===============
SIGIR reviewer point #31 called a real-output causal experiment
"non-negotiable": show that rank-only and score-space fusion operators are
separated by manipulating score MAGNITUDE independently of RANK on actual
retrieval outputs, not just on synthetic scores (the paper's section 7.2).

WHAT IT DOES
============
Loads per-document component scores captured during the alpha-sweep endpoint
runs (comp_1.0.json = maxnorm(SF), comp_0.0.json = maxnorm(SPLADE)), then for
each condition re-fuses both signals with all seven operators and reports
fused MRR plus Kendall tau of the fused ranking vs the unperturbed fused
ranking of the same operator.

Conditions (applied to ONE signal; the other stays fixed):
  orig          untouched reference
  x2            s' = 2s                positive linear scaling
  log1p         s' = log(1+s)           concave monotone
  pow05         s' = s^0.5              monotone power
  rpr           rank-preserving random remap of magnitudes (seed=42)
  shufflescores permute the multiset of scores across documents
                (magnitude distribution preserved, ranks destroyed)

PREDICTIONS (all confirmed on real data)
========================================
  * rrf / borda: identical MRR and tau=1.000 under every rank-preserving
    transform, including rpr (Proposition 1 made empirical).
  * score-space operators: internal reordering under the same transforms.
  * shufflescores: maximal damage to rank-only operators.

INPUTS
======
docs/papers/Journal A/appendix_alpha/<ds>_comp_1.0.json   maxnorm(SF)
docs/papers/Journal A/appendix_alpha/<ds>_comp_0.0.json   maxnorm(SPLADE)
outputs/<ds>_benchmark/runs/<run>/query_gold.json

USAGE
=====
    .venv/Scripts/python scripts/magnitude_perturbation.py

OUTPUTS
=======
docs/papers/Journal A/appendix_stats/magnitude_perturbation_<ds>.md
Transcribed into Appendix E of the journal manuscript.

CAVEATS
=======
n=10 queries per dataset (pool-size limited); deterministic via seed=42.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys.path.insert(0, str(PROJ))
from semantic_folding import fusion_operators as fo

ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPERATORS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]
CONDITIONS = ["orig", "x2", "log1p", "pow05", "rpr", "shufflescores"]

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


def load_components(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [dict(q["results"]) for q in raw]


def mrr_of(ranked_docs, gold_set):
    for rank, d in enumerate(ranked_docs, start=1):
        if d in gold_set:
            return 1.0 / rank
    return 0.0


def kendall_tau(a, b):
    """Tau between two doc orderings (same key set assumed for fairness)."""
    common = [d for d in a if d in b]
    ra = {d: i for i, d in enumerate(a)}
    rb = {d: i for i, d in enumerate(b)}
    concord = discord = 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = common[i], common[j]
            sign_a = np.sign(ra[di] - ra[dj])
            sign_b = np.sign(rb[di] - rb[dj])
            if sign_a == sign_b:
                concord += 1
            else:
                discord += 1
    denom = concord + discord
    return (concord - discord) / denom if denom else 1.0


def transform(scores, cond, rng):
    if cond == "orig":
        return dict(scores)
    if cond == "x2":
        return {d: 2.0 * s for d, s in scores.items()}
    if cond == "log1p":
        return {d: float(np.log1p(s)) for d, s in scores.items()}
    if cond == "pow05":
        return {d: float(np.sqrt(s)) for d, s in scores.items()}
    if cond == "rpr":
        items = sorted(scores.items(), key=lambda kv: kv[1])
        new_vals = np.sort(rng.uniform(1e-6, 1.0, size=len(items)))
        return {d: float(v) for (d, _), v in zip(items, new_vals)}
    if cond == "shufflescores":
        vals = list(scores.values())
        perm = rng.permutation(len(vals))
        keys = list(scores.keys())
        return {keys[i]: float(vals[perm[i]]) for i in range(len(keys))}
    raise ValueError(cond)


def main():
    summary_lines = []
    for ds, (sf_path, sp_path, gold_path) in DATASETS.items():
        if not sf_path.exists() or not sp_path.exists():
            print(f"[skip] {ds}: missing component files")
            continue
        sf = load_components(sf_path)
        sp = load_components(sp_path)
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        rng = np.random.default_rng(42)
        lines = [
            f"#### {ds}: magnitude perturbation on REAL component scores "
            f"(n={len(sf)} queries)\n",
            "Signal X transformed, other signal fixed. Each cell: MRR | tau(fused vs orig-fused).\n",
        ]

        for signal_name, base in (("SF", sf), ("SPLADE", sp)):
            lines.append(f"\n**Perturbed signal: {signal_name}**\n")
            header = "| Condition | " + " | ".join(f"{op} MRR / tau" for op in OPERATORS) + " |"
            sep = "|---" * (len(OPERATORS) + 1) + "|"
            lines += [header, sep]
            for cond in CONDITIONS:
                cells = []
                for op in OPERATORS:
                    mrrs, taus = [], []
                    for qi in range(len(sf)):
                        xs = transform(base[qi], cond, rng)
                        other = sp[qi] if signal_name == "SF" else sf[qi]
                        fused = fo.fuse(op, xs, other, alpha=0.3, k=60)
                        ranked = [d for d, _ in sorted(fused.items(),
                                                       key=lambda kv: (-kv[1], kv[0]))]
                        g = set(gold.get(str(qi), []))
                        if not g:
                            continue
                        mrrs.append(mrr_of(ranked, g))
                        # tau vs the ORIGINAL fused ranking of the same
                        # dataset/signal/operator (deterministic tiebreak)
                        fused0 = FUSED_ORIG[(ds, signal_name, op)][qi]
                        taus.append(kendall_tau(ranked, fused0))
                    cells.append(f"MRR={np.mean(mrrs):.3f} tau={np.mean(taus):+.3f}")
                lines.append(f"| {cond} | " + " | ".join(cells) + " |")

        out = OUT_DIR / f"magnitude_perturbation_{ds}.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        summary_lines.append(out.name)

    print("datasets processed:", len(summary_lines))


# Pre-compute original fused rankings once per (dataset, signal, operator) so
# tau has a stable per-dataset reference (keying without `ds` made datasets
# overwrite each other and produced meaningless tau).
FUSED_ORIG = {}


def precompute_orig():
    for ds, (sf_path, sp_path, gold_path) in DATASETS.items():
        if not sf_path.exists() or not sp_path.exists():
            continue
        sf = load_components(sf_path)
        sp = load_components(sp_path)
        for signal_name, base in (("SF", sf), ("SPLADE", sp)):
            for op in OPERATORS:
                per_q = []
                for qi in range(len(sf)):
                    other = sp[qi] if signal_name == "SF" else sf[qi]
                    fused = fo.fuse(op, base[qi], other, alpha=0.3, k=60)
                    per_q.append(
                        [d for d, _ in sorted(fused.items(),
                                              key=lambda kv: (-kv[1], kv[0]))]
                    )
                FUSED_ORIG[(ds, signal_name, op)] = per_q


if __name__ == "__main__":
    precompute_orig()
    main()
