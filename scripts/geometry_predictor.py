"""Query-level score-geometry -> delta-MRR regression (SIGIR-Final-Tasks Item 2).

Regresses per-query fusion gain `ΔMRR_q = RR_CombSUM,q - RR_RRF,q` on query-level
score-geometry features, with emphasis on TOP-K relevance-conditioned margins
(Reviews §6), and classifies winning queries into Type A/B/C/D (Reviews §7).

Reuses the same SF+SPLADE component traces as counterfactual_magnitude.py:
  docs/papers/Journal A/appendix_alpha/<ds>_comp_{1.0,0.0}[_n100].json
  <run>/query_gold.json

Output: docs/papers/Journal A/appendix_stats/geometry_predictor_n{N}.{json,md}
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

DATASETS = {
    "n100": {
        "hotpotqa": PROJ / "outputs/hotpotqa_benchmark/runs/run_20260824_032535/query_gold.json",
        "musique":  PROJ / "outputs/musique_benchmark/runs/run_20260824_033236/query_gold.json",
        "nq_rear":  PROJ / "outputs/nq_rear_benchmark/runs/run_20260824_033353/query_gold.json",
    },
    "n10": {
        "hotpotqa": PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/query_gold.json",
        "musique":  PROJ / "outputs/musique_benchmark/runs/run_20260822_191925/query_gold.json",
        "scifact":  PROJ / "outputs/scifact_benchmark/runs/run_20260822_191507/query_gold.json",
        "2wikimultihopqa": PROJ / "outputs/2wikimultihopqa_benchmark/runs/run_20260822_100044/query_gold.json",
    },
}
SEED = 42
B = 10000


def comp_path(ds, n):
    if n == 100:
        return ALPHA_DIR / f"{ds}_comp_1.0_n100.json", ALPHA_DIR / f"{ds}_comp_0.0_n100.json"
    return ALPHA_DIR / f"{ds}_comp_1.0.json", ALPHA_DIR / f"{ds}_comp_0.0.json"


def load_components(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(q["results"]) for q in raw]


def load_gold(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = [raw[str(i)] if str(i) in raw else raw[i] for i in range(len(raw))]
        return [set(v) for v in items]
    out = []
    for q in raw:
        g = q.get("gold") or q.get("gold_ids") or q.get("answer_docs")
        if isinstance(g, str):
            g = [g]
        out.append(set(g or []))
    return out


def ranks_from_scores(scores):
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {d: i + 1 for i, (d, _) in enumerate(order)}


def fuse(sf, sp, operator):
    docs = sorted(set(sf) | set(sp))
    sfv = {d: sf.get(d, 0.0) for d in docs}
    spv = {d: sp.get(d, 0.0) for d in docs}
    if operator == "borda":
        fused = fo.fuse(operator, sfv, spv, n_docs=len(docs))
    else:
        fused = fo.fuse(operator, sfv, spv, k=60)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def mrr_of(ranked, gold):
    for rank, d in enumerate(ranked, 1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def gold_rank_in(ranked, gold):
    for rank, d in enumerate(ranked, 1):
        if d in gold:
            return rank
    return None


def kendall_tau_scores(a, b):
    """Kendall tau over aligned doc scores (uses continuous scores, not ranks)."""
    docs = list(set(a) & set(b))
    n = len(docs)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[docs[i]] - a[docs[j]]
            db = b[docs[i]] - b[docs[j]]
            if da * db > 0:
                conc += 1
            elif da * db < 0:
                disc += 1
    denom = conc + disc
    return 1.0 if denom == 0 else (conc - disc) / denom


def geometry_features(sf, sp, gold):
    """Per-query geometry feature vector (Reviews §6 + §7)."""
    docs = sorted(set(sf) | set(sp))
    feats = {}

    def stats(dct):
        vals = sorted(dct.values(), reverse=True)
        arr = np.array(list(dct.values()), dtype=float)
        return vals, arr

    sv, sa = stats(sf)
    pv, pa = stats(sp)
    # within-signal top gaps
    feats["sf_d12"] = (sv[0] - sv[1]) if len(sv) > 1 else 0.0
    feats["sf_d15"] = (sv[0] - sv[4]) if len(sv) > 4 else 0.0
    feats["sp_d12"] = (pv[0] - pv[1]) if len(pv) > 1 else 0.0
    feats["sp_d15"] = (pv[0] - pv[4]) if len(pv) > 4 else 0.0
    feats["sf_sigma"] = float(np.std(sa)) if len(sa) > 1 else 0.0
    feats["sp_sigma"] = float(np.std(pa)) if len(pa) > 1 else 0.0
    # rank agreement (Kendall tau on score orders)
    feats["tau_signal"] = kendall_tau_scores(sf, sp)
    # top-1 mass fraction (kappa)
    tot = sum(sa) + sum(pa)
    feats["kappa"] = float((sa[0] + pv[0]) / tot) if tot > 0 else 0.0

    # cross-signal gold margin
    g = next(iter(gold)) if gold else None
    if g is not None and g in sf and g in sp:
        feats["cross_gold_margin"] = float(sf[g] - sp[g])
        feats["gold_rank_sf"] = ranks_from_scores(sf).get(g)
        feats["gold_rank_sp"] = ranks_from_scores(sp).get(g)
    else:
        feats["cross_gold_margin"] = 0.0
        feats["gold_rank_sf"] = None
        feats["gold_rank_sp"] = None

    # top-k relevance-conditioned margins: gold score minus kth doc in gold's own signal
    def gold_topk(sig, k):
        if g is None or g not in sig:
            return 0.0
        svals = sorted(sig.values(), reverse=True)
        if len(svals) < k:
            return 0.0
        return float(sig[g] - svals[k - 1])

    feats["gold_d13_sf"] = gold_topk(sf, 3)
    feats["gold_d15_sf"] = gold_topk(sf, 5)
    feats["gold_d13_sp"] = gold_topk(sp, 3)
    feats["gold_d15_sp"] = gold_topk(sp, 5)

    # joint margin (§7.5 def): mean over signals of (gold-bestnongold)/max|score|
    def joint(sig):
        if g is None or g not in sig:
            return 0.0
        mx = max(abs(v) for v in sig.values()) or 1.0
        others = [v for d, v in sig.items() if d != g]
        best_non = max(others) if others else 0.0
        return (sig[g] - best_non) / mx

    feats["joint_margin"] = 0.5 * (joint(sf) + joint(sp))
    return feats


def classify_type(rank_combsum, rank_rrf):
    if rank_combsum is None:
        return "C"  # gold not retrieved -> no change
    if rank_rrf is None:
        return "A"  # RRF missed gold entirely, CombSUM found it
    if rank_combsum < rank_rrf:
        # CombSUM promoted gold
        if rank_combsum == 1 and rank_rrf >= 3:
            return "A"
        if rank_combsum <= 2 and rank_rrf >= 5:
            return "A"
        if rank_rrf - rank_combsum >= 2:
            return "B"
        return "B" if rank_combsum < rank_rrf else "C"
    if rank_combsum > rank_rrf:
        return "D"  # CombSUM hurt
    return "C"


def ols_bootstrap(y, X, B=10000, seed=42):
    """Standardized OLS: y ~ X (X already standardized). Returns betas + bootstrap CI."""
    rng = np.random.default_rng(seed)
    n, k = X.shape
    Xb = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ beta
    # bootstrap
    boots = np.zeros((B, k + 1))
    idx = rng.integers(0, n, size=(B, n))
    for b in range(B):
        yi = y[idx[b]]
        Xi = Xb[idx[b]]
        try:
            bb, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
            boots[b] = bb
        except np.linalg.LinAlgError:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots).any(axis=1)]
    ci = np.percentile(boots, [2.5, 97.5], axis=0)
    return {
        "intercept": float(beta[0]),
        "betas": [float(x) for x in beta[1:]],
        "ci_low": [float(x) for x in ci[0, 1:]],
        "ci_high": [float(x) for x in ci[1, 1:]],
        "r2": float(1 - np.var(resid) / np.var(y)) if np.var(y) > 0 else 0.0,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, choices=[10, 100])
    args = ap.parse_args()
    N = args.n
    results = {"sample_size": N, "datasets": {}, "feature_names": []}
    ds_map = DATASETS[f"n{N}"]

    # feature matrix assembled per dataset, then regression on the discriminating set
    all_rows = []

    for ds, gold_path in ds_map.items():
        print(f"=== {ds} (n={N}) ===")
        c1, c0 = comp_path(ds, N)
        if not (c1.exists() and c0.exists()):
            print(f"  SKIP: missing {c1.name}/{c0.name}")
            continue
        sf_list = load_components(c1)
        sp_list = load_components(c0)
        gold_list = load_gold(gold_path)
        nq = min(len(sf_list), len(sp_list), len(gold_list))
        rows = []
        for qi in range(nq):
            sf, sp = sf_list[qi], sp_list[qi]
            gold = gold_list[qi]
            r_cs = fuse(sf, sp, "combsum")
            r_rrf = fuse(sf, sp, "rrf")
            d_mrr = mrr_of(r_cs, gold) - mrr_of(r_rrf, gold)
            feats = geometry_features(sf, sp, gold)
            feats["delta_mrr"] = d_mrr
            feats["rank_combsum"] = gold_rank_in(r_cs, gold)
            feats["rank_rrf"] = gold_rank_in(r_rrf, gold)
            feats["type"] = classify_type(feats["rank_combsum"], feats["rank_rrf"])
            rows.append(feats)
        results["datasets"][ds] = {"n_queries": nq, "rows": rows}
        all_rows.extend(rows)
        print(f"  n={nq} mean delta_MRR={np.mean([r['delta_mrr'] for r in rows]):.4f} "
              f"types={_count_types(rows)}")

    # regression on standardized features across all datasets (pooled) and per dataset
    feature_keys = [
        "gold_d15_sf", "gold_d15_sp", "cross_gold_margin", "joint_margin",
        "tau_signal", "sf_d15", "sp_d15", "kappa",
    ]
    results["feature_names"] = feature_keys

    def build_matrix(rows):
        y = np.array([r["delta_mrr"] for r in rows], dtype=float)
        X = np.array([[r[k] for k in feature_keys] for r in rows], dtype=float)
        # standardize columns
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xz = (X - mu) / sd
        return y, Xz

    # pooled regression
    if all_rows:
        y, Xz = build_matrix(all_rows)
        reg = ols_bootstrap(y, Xz, B=B)
        results["pooled_regression"] = reg
        print(f"\nPOOLED regression R2={reg['r2']:.3f}")
        for name, b, lo, hi in zip(feature_keys, reg["betas"], reg["ci_low"], reg["ci_high"]):
            print(f"  {name:18s} beta={b:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")

    # per-dataset regression
    for ds, d in results["datasets"].items():
        y, Xz = build_matrix(d["rows"])
        if len(y) < 5:
            d["regression"] = None
            continue
        d["regression"] = ols_bootstrap(y, Xz, B=B)
        # Type A vs C margin comparison
        a_rows = [r for r in d["rows"] if r["type"] == "A"]
        c_rows = [r for r in d["rows"] if r["type"] == "C"]
        d["typeA_count"] = len(a_rows)
        d["typeC_count"] = len(c_rows)
        d["typeA_mean_joint"] = float(np.mean([r["joint_margin"] for r in a_rows])) if a_rows else None
        d["typeC_mean_joint"] = float(np.mean([r["joint_margin"] for r in c_rows])) if c_rows else None

    # per-dataset regression summary print
    for ds, d in results["datasets"].items():
        reg = d.get("regression")
        if reg:
            print(f"\n{ds} regression R2={reg['r2']:.3f} "
                  f"gold_d15_sf beta={reg['betas'][0]:+.4f} "
                  f"joint_margin beta={reg['betas'][3]:+.4f}")
            print(f"  TypeA={d['typeA_count']} TypeC={d['typeC_count']} "
                  f"A_joint={d['typeA_mean_joint']} C_joint={d['typeC_mean_joint']}")

    OUT = OUT_DIR / f"geometry_predictor_n{N}.json"
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_md(results, OUT.with_suffix(".md"))
    print(f"\nWROTE {OUT.name} and .md")


def _count_types(rows):
    from collections import Counter
    c = Counter(r["type"] for r in rows)
    return dict(c)


def _write_md(results, md_path):
    lines = [f"# Geometry Predictor — n={results['sample_size']}", ""]
    lines.append("## Pooled regression (ΔMRR_q ~ standardized geometry features)")
    reg = results.get("pooled_regression")
    if reg:
        lines.append("")
        lines.append("| feature | β | CI_low | CI_high |")
        lines.append("|---|---:|---:|---:|")
        for name, b, lo, hi in zip(results["feature_names"], reg["betas"],
                                   reg["ci_low"], reg["ci_high"]):
            lines.append(f"| {name} | {b:+.4f} | {lo:+.4f} | {hi:+.4f} |")
        lines.append(f"| R² | {reg['r2']:.3f} | | |")
    lines.append("")
    lines.append("## Per-dataset")
    for ds, d in results["datasets"].items():
        lines.append(f"### {ds} (n={d['n_queries']})")
        reg = d.get("regression")
        if reg:
            lines.append(f"- R²={reg['r2']:.3f}; gold_d15_sf β={reg['betas'][0]:+.4f} "
                         f"(CI [{reg['ci_low'][0]:+.4f},{reg['ci_high'][0]:+.4f}]); "
                         f"joint_margin β={reg['betas'][3]:+.4f}")
        lines.append(f"- Type A={d.get('typeA_count')} C={d.get('typeC_count')} "
                     f"A_joint={d.get('typeA_mean_joint')} C_joint={d.get('typeC_mean_joint')}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
