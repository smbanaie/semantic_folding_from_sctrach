#!/usr/bin/env python3
"""
Train LambdaMART re-ranker on labeled feature data.

Usage:
    .venv\Scripts\python -m semantic_folding.tools.train_model \
        --features outputs/belebele_benchmark/training_features.jsonl \
        --output outputs/belebele_benchmark/lambdamart_model.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from lib import get_logger
from reranker_train import FEATURE_NAMES, train_lambdamart, prepare_training_data

logger = get_logger("train_model")


def load_features(path: Path) -> List[Dict]:
    """Load feature vectors from JSONL file."""
    features = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                features.append(json.loads(line))
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Train LambdaMART re-ranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", type=Path, required=True,
                        help="Training features JSONL (from generate_training_data)")
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

    if not HAS_LIGHTGBM:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    np.random.seed(args.seed)

    # Load features
    logger.info(f"Loading features from {args.features}")
    features = load_features(args.features)
    logger.info(f"  {len(features)} feature vectors")

    # Prepare data
    X, y, groups, _ = prepare_training_data(features, gold_labels=None)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    logger.info(f"  {n_pos} positive, {n_neg} negative samples")

    if n_pos == 0:
        logger.error("No positive samples found. Check gold labels.")
        sys.exit(1)

    # Train
    model, info = train_lambdamart(
        X, y, FEATURE_NAMES,
        groups=groups,
        num_trees=args.num_trees,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    if model is None:
        logger.error("Training failed")
        sys.exit(1)

    # Save model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    logger.info(f"Model saved to {args.output}")

    # Save metadata
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "train_info": info,
            "args": vars(args),
        }, f, indent=2, default=str)
    logger.info(f"Metadata saved to {meta_path}")

    # Print summary
    logger.info(f"\nTraining complete:")
    logger.info(f"  Best iteration: {info.get('best_iteration', 'N/A')}")
    logger.info(f"  Top features by gain:")
    fi = info.get("feature_importance", {})
    top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    for fname, gain in top:
        logger.info(f"    {fname}: {gain:.1f}")


if __name__ == "__main__":
    main()
