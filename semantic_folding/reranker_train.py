#!/usr/bin/env python3
"""
reranker_train.py — LambdaMART re-ranker training (Phase 5)

Trains a gradient-boosted re-ranker on feature vectors extracted from
SF binary fingerprints. Uses cross-dataset training (MuSiQue + Belebele)
with evaluation on held-out datasets.

Usage:
    python -m semantic_folding.reranker_train \\
        --train-datasets musique,belebele \\
        --features-dir outputs/ \\
        --output model.txt \\
        --num-trees 200 \\
        --learning-rate 0.05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from lib import get_logger

logger = get_logger("reranker_train")


FEATURE_NAMES = [
    "jaccard", "dice", "overlap", "hamming_norm", "cosine",
    "containment", "coverage", "idf_weighted",
    "q_popcount", "d_popcount", "q_density", "d_density",
    "intersection_popcount", "union_popcount", "q_minus_d", "d_minus_q",
    "bm25_score", "query_length", "doc_length",
] + [f"block_{b}_jaccard" for b in range(16)]


def load_features_from_jsonl(path: Path) -> List[Dict]:
    """Load feature vectors from JSONL file."""
    features = []
    with open(path) as f:
        for line in f:
            if line.strip():
                features.append(json.loads(line))
    return features


def prepare_training_data(
    features_list: List[Dict],
    gold_labels: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and labels for training.

    Args:
        features_list: List of feature dicts from reranker_features.py
        gold_labels: Dict of {query_text: {doc_id: label}} where label=1 for gold

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        feature_names: List of feature names
    """
    X_rows = []
    y_rows = []

    # Group by query_idx to create query groups for LambdaMART
    queries = {}
    for feat_dict in features_list:
        q_idx = feat_dict.get("query_idx", feat_dict.get("query_text", ""))
        if q_idx not in queries:
            queries[q_idx] = []
        queries[q_idx].append(feat_dict)

    groups = []
    for q_idx, q_feats in queries.items():
        groups.append(len(q_feats))
        for feat_dict in q_feats:
            feats = feat_dict["features"]
            row = [feats.get(name, 0.0) for name in FEATURE_NAMES]
            X_rows.append(row)

            label = feat_dict.get("label", 0)
            if gold_labels:
                query_text = feat_dict.get("query_text", "")
                doc_id = feat_dict.get("doc_id", "")
                if query_text in gold_labels and doc_id in gold_labels[query_text]:
                    label = gold_labels[query_text][doc_id]
            y_rows.append(label)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), np.array(groups, dtype=np.int32), FEATURE_NAMES


def train_lambdamart(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    groups: Optional[np.ndarray] = None,
    num_trees: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    min_child_samples: int = 10,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 20,
    validation_fraction: float = 0.2,
) -> Tuple[Optional[object], Dict]:
    """
    Train LambdaMART model using LightGBM.

    Returns:
        model: Trained LightGBM Booster (or None if not available)
        info: Training info dict with metrics
    """
    if not HAS_LIGHTGBM:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        return None, {"error": "lightgbm not installed"}

    n_samples = len(X)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    logger.info(f"Training data: {n_samples} samples, {n_pos} positive, {n_neg} negative")

    if n_pos == 0:
        logger.error("No positive samples found — cannot train")
        return None, {"error": "no positive samples"}

    n_val = max(1, int(n_samples * validation_fraction))

    # Split by query groups, not individual samples
    if groups is not None and len(groups) > 0:
        n_queries = len(groups)
        query_indices = np.random.permutation(n_queries)
        n_val_queries = max(1, int(n_queries * validation_fraction))
        val_query_ids = set(query_indices[:n_val_queries].tolist())
        train_query_ids = set(query_indices[n_val_queries:].tolist())

        train_idx = []
        val_idx = []
        train_groups = []
        val_groups = []
        offset = 0
        for q_idx, g in enumerate(groups):
            if q_idx in val_query_ids:
                val_idx.extend(range(offset, offset + g))
                val_groups.append(g)
            else:
                train_idx.extend(range(offset, offset + g))
                train_groups.append(g)
            offset += g
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        train_groups = np.array(train_groups, dtype=np.int32)
        val_groups = np.array(val_groups, dtype=np.int32)
    else:
        indices = np.random.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        train_groups = None
        val_groups = None

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, group=train_groups.tolist() if train_groups is not None else None)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, group=val_groups.tolist() if val_groups is not None else None, reference=train_data)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "num_trees": num_trees,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "verbose": -1,
    }

    logger.info(f"Training LambdaMART: {num_trees} trees, lr={learning_rate}, depth={max_depth}")

    callbacks = [lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(50)]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    logger.info(f"Best iteration: {best_iter}")

    feature_importance = dict(zip(feature_names, model.feature_importance(importance_type="gain")))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features by gain:")
    for fname, gain in top_features:
        logger.info(f"  {fname}: {gain:.1f}")

    info = {
        "best_iteration": best_iter,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_pos_train": int(np.sum(y_train == 1)),
        "n_neg_train": int(np.sum(y_train == 0)),
        "feature_importance": feature_importance,
    }

    return model, info


def cross_dataset_evaluate(
    model,
    eval_features: List[Dict],
    gold_labels: Dict[str, Dict[str, int]],
) -> Dict:
    """
    Evaluate trained model on held-out dataset.

    Returns:
        Dict with MRR, P@1, P@3, P@5 metrics
    """
    if model is None:
        return {"error": "no model"}

    queries = {}
    for feat_dict in eval_features:
        qt = feat_dict.get("query_text", "")
        if qt not in queries:
            queries[qt] = []
        queries[qt].append(feat_dict)

    mrr_sum = 0.0
    p1_sum = 0.0
    p3_sum = 0.0
    p5_sum = 0.0
    n_queries = 0

    for qt, feat_dicts in queries.items():
        X = np.array([
            [feat_dict["features"].get(name, 0.0) for name in FEATURE_NAMES]
            for feat_dict in feat_dicts
        ])

        scores = model.predict(X)
        ranked_indices = np.argsort(-scores)

        gold_set = set()
        if qt in gold_labels:
            gold_set = {doc_id for doc_id, label in gold_labels[qt].items() if label == 1}

        if not gold_set:
            continue

        n_queries += 1
        for rank, idx in enumerate(ranked_indices, 1):
            doc_id = feat_dicts[idx]["doc_id"]
            if doc_id in gold_set:
                mrr_sum += 1.0 / rank
                break

        for k in [1, 3, 5]:
            top_k_docs = {feat_dicts[idx]["doc_id"] for idx in ranked_indices[:k]}
            hits = len(top_k_docs & gold_set)
            if k == 1:
                p1_sum += hits
            elif k == 3:
                p3_sum += hits / min(3, len(gold_set))
            elif k == 5:
                p5_sum += hits / min(5, len(gold_set))

    if n_queries == 0:
        return {"error": "no labeled queries"}

    return {
        "mrr": mrr_sum / n_queries,
        "p@1": p1_sum / n_queries,
        "p@3": p3_sum / n_queries,
        "p@5": p5_sum / n_queries,
        "n_queries": n_queries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train LambdaMART re-ranker for Semantic Folding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features-dir", type=Path, required=True,
                        help="Directory containing feature JSONL files")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output model file path")
    parser.add_argument("--num-trees", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if not HAS_LIGHTGBM:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    feature_files = list(args.features_dir.glob("**/features.jsonl"))
    if not feature_files:
        logger.error(f"No features.jsonl files found in {args.features_dir}")
        sys.exit(1)

    logger.info(f"Found {len(feature_files)} feature files")

    all_features = []
    for ff in feature_files:
        all_features.extend(load_features_from_jsonl(ff))

    logger.info(f"Total feature vectors: {len(all_features)}")

    queries = {}
    for feat_dict in all_features:
        qt = feat_dict.get("query_text", "")
        if qt not in queries:
            queries[qt] = []
        queries[qt].append(feat_dict)

    logger.info(f"Unique queries: {len(queries)}")

    gold_labels = {}
    for feat_dict in all_features:
        qt = feat_dict.get("query_text", "")
        doc_id = feat_dict.get("doc_id", "")
        label = feat_dict.get("label", 0)
        if qt not in gold_labels:
            gold_labels[qt] = {}
        gold_labels[qt][doc_id] = label

    X, y, feature_names = prepare_training_data(all_features, gold_labels)
    logger.info(f"Feature matrix: {X.shape}, positive rate: {np.mean(y > 0):.3f}")

    model, train_info = train_lambdamart(
        X, y, feature_names,
        num_trees=args.num_trees,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    if model is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.output))
        logger.info(f"Model saved to {args.output}")

        meta_path = args.output.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": feature_names,
                "train_info": train_info,
                "args": vars(args),
            }, f, indent=2, default=str)
        logger.info(f"Metadata saved to {meta_path}")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
