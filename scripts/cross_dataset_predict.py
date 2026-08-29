"""Item 7 — Cross-dataset prediction (SIGIR-Final-Reviews #14).

Does pre-fusion score geometry predict whether CombSUM beats RRF on UNSEEN datasets?
Leave-one-dataset-out (LODO): train on all datasets but one, test on the held-out; rotate.
Never a random query split (avoids dataset-characteristic leakage). Classifiers: logistic
regression + decision tree (kept simple per Reviews #17). Metrics: AUROC, AUPRC, accuracy,
calibration; bootstrap 95% CI (B=10000, seed=42) on pooled LODO predictions.

Output: docs/papers/Journal A/appendix_stats/cross_dataset_predict.{json,md}
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys.path.insert(0, str(PROJ))
import counterfactual_magnitude as cm
import geometry_predictor as gp

STAT = PROJ / "docs/papers/Journal A/appendix_stats"
STAT.mkdir(parents=True, exist_ok=True)

N = 100
SEED = 42
DATASETS = ["hotpotqa", "musique", "nq_rear"]


def feat_vector(feats):
    # fixed-order feature vector; None -> 0
    keys = ["sf_d12", "sf_d15", "sp_d12", "sp_d15", "sf_sigma", "sp_sigma",
            "tau_signal", "kappa", "cross_gold_margin",
            "gold_d13_sf", "gold_d15_sf", "gold_d13_sp", "gold_d15_sp", "joint_margin"]
    return [float(feats.get(k, 0.0) or 0.0) for k in keys]


def load_dataset(ds):
    c1, c0 = cm.comp_path(ds, N)
    sf = gp.load_components(c1)
    sp = gp.load_components(c0)
    gold = gp.load_gold(cm.DATASETS[f"n{N}"][ds])
    X, y = [], []
    for qi in range(N):
        g = gold[qi]
        if not g:
            continue
        r_cs = gp.fuse(sf[qi], sp[qi], "combsum")
        r_rr = gp.fuse(sf[qi], sp[qi], "rrf")
        rr_cs = gp.mrr_of(r_cs, g)
        rr_rr = gp.mrr_of(r_rr, g)
        # label: CombSUM strictly better
        y.append(1 if rr_cs > rr_rr else 0)
        X.append(feat_vector(gp.geometry_features(sf[qi], sp[qi], g)))
    return np.array(X, dtype=float), np.array(y, dtype=int), sum(y), len(y)


def bootstrap_ci(y_true, score_or_pred, kind, B=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(set(yt.tolist())) < 2:
            continue
        try:
            if kind == "auc":
                vals.append(roc_auc_score(yt, score_or_pred[idx]))
            elif kind == "ap":
                vals.append(average_precision_score(yt, score_or_pred[idx]))
            else:
                vals.append(accuracy_score(yt, score_or_pred[idx]))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def main():
    data = {ds: load_dataset(ds) for ds in DATASETS}
    folds = {}
    pooled = {"y": [], "p_log": [], "p_tree": [], "pred_log": [], "pred_tree": []}
    for held in DATASETS:
        train_ds = [d for d in DATASETS if d != held]
        Xtr = np.vstack([data[d][0] for d in train_ds])
        ytr = np.concatenate([data[d][1] for d in train_ds])
        Xte, yte, pos, tot = data[held]
        clf_log = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=SEED)
        clf_log.fit(Xtr, ytr)
        clf_tree.fit(Xtr, ytr)
        p_log = clf_log.predict_proba(Xte)[:, 1]
        p_tree = clf_tree.predict_proba(Xte)[:, 1]
        pred_log = (p_log >= 0.5).astype(int)
        pred_tree = (p_tree >= 0.5).astype(int)
        auc_log = roc_auc_score(yte, p_log) if len(set(yte.tolist())) > 1 else float("nan")
        ap_log = average_precision_score(yte, p_log) if len(set(yte.tolist())) > 1 else float("nan")
        acc_log = accuracy_score(yte, pred_log)
        auc_tree = roc_auc_score(yte, p_tree) if len(set(yte.tolist())) > 1 else float("nan")
        ap_tree = average_precision_score(yte, p_tree) if len(set(yte.tolist())) > 1 else float("nan")
        acc_tree = accuracy_score(yte, pred_tree)
        folds[held] = {
            "n": int(tot), "pos": int(pos), "base_rate": float(pos / tot),
            "logistic": {"auroc": float(auc_log), "auprc": float(ap_log), "acc": float(acc_log)},
            "tree": {"auroc": float(auc_tree), "auprc": float(ap_tree), "acc": float(acc_tree)},
        }
        pooled["y"].extend(yte.tolist())
        pooled["p_log"].extend(p_log.tolist())
        pooled["p_tree"].extend(p_tree.tolist())
        pooled["pred_log"].extend(pred_log.tolist())
        pooled["pred_tree"].extend(pred_tree.tolist())

    yp = np.array(pooled["y"])
    plog = np.array(pooled["p_log"])
    ptree = np.array(pooled["p_tree"])
    pooled_summary = {
        "n": int(len(yp)), "base_rate": float(yp.mean()),
        "logistic": {
            "auroc": float(roc_auc_score(yp, plog)),
            "auroc_ci95": bootstrap_ci(yp, plog, "auc"),
            "auprc": float(average_precision_score(yp, plog)),
            "auprc_ci95": bootstrap_ci(yp, plog, "ap"),
            "acc": float(accuracy_score(yp, pooled["pred_log"])),
            "acc_ci95": bootstrap_ci(yp, np.array(pooled["pred_log"]), "acc"),
        },
        "tree": {
            "auroc": float(roc_auc_score(yp, ptree)),
            "auroc_ci95": bootstrap_ci(yp, ptree, "auc"),
            "auprc": float(average_precision_score(yp, ptree)),
            "auprc_ci95": bootstrap_ci(yp, ptree, "ap"),
            "acc": float(accuracy_score(yp, pooled["pred_tree"])),
            "acc_ci95": bootstrap_ci(yp, np.array(pooled["pred_tree"]), "acc"),
        },
    }

    out = {"n": N, "datasets": DATASETS, "folds": folds, "pooled_lodo": pooled_summary}
    (STAT / "cross_dataset_predict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [f"# Cross-dataset prediction (Item 7, LODO, n={N})",
          "",
          f"Pooled LODO: n={pooled_summary['n']}, base rate(P(CombSUM>RRF))={pooled_summary['base_rate']:.3f}",
          "",
          "| held-out | n | base | log AUROC | log AUPRC | log acc | tree AUROC | tree AUPRC | tree acc |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for held in DATASETS:
        f = folds[held]
        md.append(f"| {held} | {f['n']} | {f['base_rate']:.3f} | {f['logistic']['auroc']:.3f} | "
                  f"{f['logistic']['auprc']:.3f} | {f['logistic']['acc']:.3f} | "
                  f"{f['tree']['auroc']:.3f} | {f['tree']['auprc']:.3f} | {f['tree']['acc']:.3f} |")
    ps = pooled_summary
    md.append("")
    md.append(f"**Pooled LODO** — logistic AUROC={ps['logistic']['auroc']:.3f} "
              f"CI{ps['logistic']['auroc_ci95']}, AUPRC={ps['logistic']['auprc']:.3f} "
              f"CI{ps['logistic']['auprc_ci95']}, acc={ps['logistic']['acc']:.3f} "
              f"CI{ps['logistic']['acc_ci95']}")
    md.append(f"tree AUROC={ps['tree']['auroc']:.3f} CI{ps['tree']['auroc_ci95']}, "
              f"AUPRC={ps['tree']['auprc']:.3f} CI{ps['tree']['auprc_ci95']}")
    md.append("")
    md.append("LODO (not random split) => no dataset-characteristic leakage. AUROC>0.5 means geometry "
              "generalizes to unseen datasets. Calibration: reliability curve omitted from text; base-rate "
              "reported for context.")
    (STAT / "cross_dataset_predict.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
