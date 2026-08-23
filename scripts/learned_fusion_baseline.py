"""Learned fusion baseline (Final-Improvements Item 20).

Answers: "Can the proposed diagnostic framework be beaten by simply learning
the fusion weights?" (reviewer request, section 6.6.2 of the manuscript).

Design (leakage-safe, fixed before coding):
  Data     real per-query component scores (maxnorm SF / maxnorm SPLADE)
           from the alpha-sweep endpoint runs; gold from query_gold.json.
           Same pools as §7.5.
  Features per document: [s_A, s_B] raw + maxnorm variants (4 features).
  Model    sklearn LogisticRegression("is gold", class_weight=balanced);
           standardized centroid-difference fallback if sklearn is absent.
  Split    leave-one-query-out CV — each held-out query is scored by a model
           trained only on other queries' documents. With n=10 queries this
           avoids both leakage and tiny-test noise.
  Metric   MRR on identical folds as the rrf / combsum baselines.

Usage:  .venv/Scripts/python scripts/learned_fusion_baseline.py
Output: appendix_stats/learned_fusion_baseline.md
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


def load(path):
    return [dict(q["results"]) for q in json.loads(path.read_text(encoding="utf-8"))]


def mrr(ranked, gold):
    for i, d in enumerate(ranked, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ranked(fused):
    return [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def features_for_query(scores_a, scores_b):
    """Feature matrix over the union pool: raw + maxnorm signal scores."""
    docs = sorted(set(scores_a) | set(scores_b))
    maxa = max(scores_a.values()) or 1.0
    maxb = max(scores_b.values()) or 1.0
    X = np.array([[scores_a.get(d, 0.0), scores_b.get(d, 0.0),
                   scores_a.get(d, 0.0) / maxa, scores_b.get(d, 0.0) / maxb]
                  for d in docs])
    return docs, X


def train_score(X_train, y_train, X_eval):
    """Logistic regression with centroid-difference fallback."""
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        return clf.decision_function(X_eval)
    except ImportError:
        sd = X_train.std(0) + 1e-9
        w = (X_train[y_train == 1].mean(0) - X_train[y_train == 0].mean(0)) / sd
        return (X_eval - X_train.mean(0)) @ w


def eval_dataset(ds, sf_path, sp_path, gold_path):
    sf, sp = load(sf_path), load(sp_path)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    n_q = len(sf)

    mrr_rrf, mrr_cs, mrr_lr = [], [], []
    for qi in range(n_q):
        g = set(gold.get(str(qi), []))
        if not g:
            continue
        mrr_rrf.append(mrr(ranked(fo.fuse("rrf", sf[qi], sp[qi], alpha=0.3, k=60)), g))
        mrr_cs.append(mrr(ranked(fo.fuse("combsum", sf[qi], sp[qi], alpha=0.3, k=60)), g))

        # LOQO training set: all documents from every other gold-bearing query
        train_X, train_y = [], []
        for qj in range(n_q):
            if qj == qi:
                continue
            gj = set(gold.get(str(qj), []))
            if not gj:
                continue
            docs_j, X_j = features_for_query(sf[qj], sp[qj])
            y_j = [1 if d in gj else 0 for d in docs_j]
            if sum(y_j):
                train_X.append(X_j)
                train_y.extend(y_j)
        if not train_X:
            mrr_lr.append(0.0)
            continue

        docs_i, X_i = features_for_query(sf[qi], sp[qi])
        try:
            score = train_score(np.vstack(train_X), np.array(train_y), X_i)
        except Exception as e:
            print(f"  model failure {ds} q{qi}: {e}")
            score = np.zeros(len(docs_i))
        mrr_lr.append(mrr([docs_i[i] for i in np.argsort(-score)], g))

    r3 = lambda xs: round(float(np.mean(xs)), 3)
    return {"n": len(mrr_rrf), "rrf": r3(mrr_rrf),
            "combsum": r3(mrr_cs), "learned": r3(mrr_lr)}


def main():
    lines = [
        "# Learned Fusion Baseline vs Fixed Operators (Item 20)\n",
        "Logistic regression over [s_A, s_B, s_A_norm, s_B_norm]; "
        "leave-one-query-out CV (train on other queries' documents, score the "
        "held-out query). Identical pools/golds as §7.5.\n",
        "| Dataset | n | rrf | combsum | learned (LOQO-CV) |",
        "|---------|--:|----:|--------:|------------------:|",
    ]
    for ds, (sf_path, sp_path, gold_path) in DATASETS.items():
        res = eval_dataset(ds, sf_path, sp_path, gold_path)
        lines.append(f"| {ds} | {res['n']} | {res['rrf']} | {res['combsum']} | {res['learned']} |")
        print(ds, res)

    out_md = OUT_DIR / "learned_fusion_baseline.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
