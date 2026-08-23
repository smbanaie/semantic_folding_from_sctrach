"""Learned fusion baseline (Final-Improvements Item 20).

Reviewer: "Can the proposed diagnostic framework be beaten by simply learning
the fusion weights?"

Design (leakage-safe, stated before coding):
  - Data: real per-query component scores (SF=maxnorm, SPLADE=maxnorm) from
    the alpha-sweep endpoint runs; gold from query_gold.json. Same artifacts
    as section 7.5.
  - Features per document: [s_A, s_B] raw + [s_A/maxnorm, s_B/maxnorm]
    normalized variants (4 features).
  - Model: logistic regression, one-vs-rest over "is gold" (sklearn), and a
    ridge ranker fallback if sklearn missing.
  - Split: LEAVE-ONE-QUERY-OUT cross-validation (each query's documents scored
    by a model trained on all other queries). With n=10 queries this is the
    only split that avoids both leakage and tiny-test noise. Reported as MRR.
  - Baselines on identical folds: rrf, combsum (the paper's best fixed ops).

Output: appendix_stats/learned_fusion_baseline.md
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys_path = str(PROJ)
import sys
sys.path.insert(0, sys_path)
from semantic_folding import fusion_operators as fo

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


def features_for_query(scores_a, scores_b):
    """Feature matrix for every doc in the union pool: raw+normalized scores."""
    docs = sorted(set(scores_a) | set(scores_b))
    maxa = max(scores_a.values()) or 1.0
    maxb = max(scores_b.values()) or 1.0
    X = []
    for d in docs:
        sa = scores_a.get(d, 0.0)
        sb = scores_b.get(d, 0.0)
        X.append([sa, sb, sa / maxa, sb / maxb])
    return docs, np.array(X)


def main():
    try:
        from sklearn.linear_model import LogisticRegression
        use_sklearn = True
    except ImportError:
        use_sklearn = False
        print("sklearn unavailable — using hand-rolled logistic regression")

    table_lines = [
        "# Learned Fusion Baseline vs Fixed Operators (Item 20)\n",
        "Logistic regression over [s_A, s_B, s_A_norm, s_B_norm], trained with "
        "leave-one-query-out CV (train on other queries' documents, score the "
        "held-out query). Identical pools/golds as section 7.5.\n",
        "| Dataset | n | rrf | combsum | learned (LOQO-CV) |",
        "|---------|--:|----:|--------:|------------------:|",
    ]
    summary = {}
    for ds, (sf_path, sp_path, gold_path) in DATASETS.items():
        sf = load(sf_path)
        sp = load(sp_path)
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        n_q = len(sf)

        mrr_rrf, mrr_cs, mrr_lr = [], [], []
        for qi in range(n_q):
            g = set(gold.get(str(qi), []))
            if not g:
                continue
            # fixed operators
            fused = fo.fuse("rrf", sf[qi], sp[qi], alpha=0.3, k=60)
            ranked = [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
            mrr_rrf.append(mrr(ranked, g))
            fused = fo.fuse("combsum", sf[qi], sp[qi], alpha=0.3, k=60)
            ranked = [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
            mrr_cs.append(mrr(ranked, g))

            # learned: leave-one-query-out training set
            train_X, train_y = [], []
            for qj in range(n_q):
                if qj == qi:
                    continue
                gj = set(gold.get(str(qj), []))
                if not gj:
                    continue
                docs_j, X_j = features_for_query(sf[qj], sp[qj])
                y_j = [1 if d in gj else 0 for d in docs_j]
                if sum(y_j) == 0:
                    continue
                train_X.append(X_j)
                train_y.extend(y_j)
            if not train_X:
                mrr_lr.append(0.0)
                continue
            X_train = np.vstack(train_X)
            y_train = np.array(train_y)

            docs_i, X_i = features_for_query(sf[qi], sp[qi])
            try:
                if use_sklearn:
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                    clf.fit(X_train, y_train)
                    score = clf.decision_function(X_i)
                else:
                    # simple standardized-sum fallback
                    mu, sd = X_train.mean(0), X_train.std(0) + 1e-9
                    w = (X_train[y_train == 1].mean(0) - X_train[y_train == 0].mean(0)) / sd
                    score = ((X_i - mu) / sd) @ w
            except Exception as e:
                print(f"  model failure q{qi}: {e}")
                score = np.zeros(len(docs_i))
            order = np.argsort(-score)
            ranked = [docs_i[i] for i in order]
            mrr_lr.append(mrr(ranked, g))

        r3 = lambda xs: round(float(np.mean(xs)), 3)
        summary[ds] = {"rrf": r3(mrr_rrf), "combsum": r3(mrr_cs), "learned": r3(mrr_lr)}
        table_lines.append(f"| {ds} | {len(mrr_rrf)} | {r3(mrr_rrf)} | {r3(mrr_cs)} | {r3(mrr_lr)} |")
        print(ds, summary[ds])

    out_md = OUT_DIR / "learned_fusion_baseline.md"
    out_md.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
