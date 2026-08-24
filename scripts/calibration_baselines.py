"""Calibration baselines (review item 0.8).

Question: is CombSUM's advantage caused by magnitude itself, or simply by
better calibration? We apply a battery of per-signal normalizations BEFORE
fusion, then fuse with combsum/combmnz/linear and compare against raw.

Baselines added (beyond the paper's existing min-max/z-score):
  l2        : s / ||s||_2
  rank_gauss: inverse-normal transform of ranks (van der Waerden)
  sigmoid   : 1 / (1 + exp(-(s - median(s)) / IQR(s)))
  quantile  : map to uniform [0,1] via empirical CDF
  softmax   : softmax with temperature = std(s) (shift-invariant)

Data: real SF+SPLADE component traces (comp_1.0 = maxnorm SF,
comp_0.0 = SPLADE), gold from the index run. n=10 queries per dataset —
exploratory, same regime as §7.4.

Output: appendix_stats/calibration_baselines.md (+ .json)
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from semantic_folding import fusion_operators as fo  # noqa: E402

ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

TRACES = {
    "hotpotqa": (
        ALPHA_DIR / "hotpotqa_comp_1.0.json",
        ALPHA_DIR / "hotpotqa_comp_0.0.json",
        PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/query_gold.json"),
    "musique": (
        ALPHA_DIR / "musique_comp_1.0.json",
        ALPHA_DIR / "musique_comp_0.0.json",
        PROJ / "outputs/musique_benchmark/runs/run_20260822_191925/query_gold.json"),
    "scifact": (
        ALPHA_DIR / "scifact_comp_1.0.json",
        ALPHA_DIR / "scifact_comp_0.0.json",
        PROJ / "outputs/scifact_benchmark/runs/run_20260822_191507/query_gold.json"),
}

NORMALIZERS = ["raw", "minmax", "zscore", "l2", "rank_gauss", "sigmoid", "quantile", "softmax"]
FUSERS = ["combsum", "combmnz", "linear"]


def load_scores(path):
    return [dict(q["results"]) for q in json.loads(path.read_text(encoding="utf-8"))]


def normalize(scores, kind, rng):
    if kind == "raw":
        return dict(scores)
    vals = np.array(list(scores.values()), dtype=float)
    keys = list(scores.keys())
    if kind == "minmax":
        rng_ = vals.max() - vals.min()
        newv = (vals - vals.min()) / rng_ if rng_ > 0 else np.zeros_like(vals)
    elif kind == "zscore":
        sd = vals.std()
        newv = (vals - vals.mean()) / sd if sd > 0 else np.zeros_like(vals)
    elif kind == "l2":
        n2 = float(np.linalg.norm(vals))
        newv = vals / n2 if n2 > 0 else np.zeros_like(vals)
    elif kind == "rank_gauss":
        order = vals.argsort().argsort()
        u = (order + 0.5) / len(vals)
        from scipy.stats import norm
        newv = norm.ppf(u)
    elif kind == "sigmoid":
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        scale = iqr if iqr > 0 else (vals.std() or 1.0)
        newv = 1.0 / (1.0 + np.exp(-(vals - np.median(vals)) / scale))
    elif kind == "quantile":
        order = vals.argsort().argsort()
        newv = (order + 0.5) / len(vals)
    elif kind == "softmax":
        t = vals.std() or 1.0
        z = (vals - vals.max()) / t
        e = np.exp(z)
        newv = e / e.sum()
    else:
        raise ValueError(kind)
    return {k: float(v) for k, v in zip(keys, newv)}


def mrr_of(ranked, gold_set):
    for i, d in enumerate(ranked, 1):
        if d in gold_set:
            return 1.0 / i
    return 0.0


def main():
    lines = [
        "# Calibration Baselines: magnitude vs calibration (review item 0.8)\n",
        "Per-signal normalization applied before fusion; fused with three "
        "score-space operators. n=10 exploratory queries per dataset "
        "(same traces as \u00a77.4). If calibration alone explains CombSUM's "
        "advantage, some normalizer should match or beat raw.\n",
    ]
    results = {}
    for ds, (sf_path, sp_path, gold_path) in TRACES.items():
        sf = load_scores(sf_path)
        sp = load_scores(sp_path)
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        rng = np.random.default_rng(42)
        rows = {}
        for nkind in NORMALIZERS:
            row = {}
            for fop in FUSERS:
                mrrs = []
                for qi in range(len(sf)):
                    g = set(gold.get(str(qi), []))
                    if not g:
                        continue
                    a = normalize(sf[qi], nkind, rng)
                    b = normalize(sp[qi], nkind, rng)
                    params = {"alpha": 0.3} if fop == "linear" else {}
                    fused = fo.fuse(fop, a, b, k=60, **params)
                    ranked = [d for d, _ in sorted(fused.items(),
                                                   key=lambda kv: (-kv[1], kv[0]))]
                    mrrs.append(mrr_of(ranked, g))
                row[fop] = round(float(np.mean(mrrs)), 3) if mrrs else None
            rows[nkind] = row
        results[ds] = {"n_queries": 10, "mrr": rows}
        lines.append(f"\n## {ds}\n")
        lines.append("| Normalization | CombSUM | CombMNZ | Linear(\u03b1=0.3) |")
        lines.append("|---------------|--------:|--------:|--------:|")
        for nk in NORMALIZERS:
            r = rows[nk]
            fmt = lambda v: "\u2014" if v is None else f"{v:.3f}"
            lines.append(f"| {nk} | {fmt(r['combsum'])} | {fmt(r['combmnz'])} | {fmt(r['linear'])} |")

    out_md = OUT_DIR / "calibration_baselines.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "calibration_baselines.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
