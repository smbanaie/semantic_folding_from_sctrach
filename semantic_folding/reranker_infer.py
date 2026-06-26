#!/usr/bin/env python3
"""
reranker_infer.py — LambdaMART inference for re-ranking (Phase 5)

Loads a trained LambdaMART model and re-ranks candidates from SF retrieval.

Usage:
    python -m semantic_folding.reranker_infer \\
        --model model.txt \\
        --features candidates.jsonl \\
        --output reranked.jsonl
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

logger = get_logger("reranker_infer")

FEATURE_NAMES = [
    "jaccard", "dice", "overlap", "hamming_norm", "cosine",
    "containment", "coverage", "idf_weighted",
    "q_popcount", "d_popcount", "q_density", "d_density",
    "intersection_popcount", "union_popcount", "q_minus_d", "d_minus_q",
    "bm25_score", "query_length", "doc_length",
] + [f"block_{b}_jaccard" for b in range(16)]


def load_model(model_path: Path):
    """Load trained LightGBM model."""
    if not HAS_LIGHTGBM:
        logger.error("lightgbm not installed")
        sys.exit(1)
    return lgb.Booster(model_file=str(model_path))


def rerank_candidates(
    model,
    candidates: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Re-rank candidates using trained model.

    Args:
        model: LightGBM Booster
        candidates: List of feature dicts (from reranker_features.py)
        top_k: Number of top results to return

    Returns:
        Re-ranked list of dicts with 'rerank_score' field
    """
    if not candidates:
        return []

    X = np.array([
        [c["features"].get(name, 0.0) for name in FEATURE_NAMES]
        for c in candidates
    ])

    scores = model.predict(X)
    ranked_indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, 1):
        result = dict(candidates[idx])
        result["rerank_score"] = float(scores[idx])
        result["rerank_rank"] = rank
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Re-rank candidates using trained LambdaMART model",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model = load_model(args.model)
    logger.info(f"Loaded model from {args.model}")

    candidates = []
    with open(args.features) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    logger.info(f"Loaded {len(candidates)} candidates")

    results = rerank_candidates(model, candidates, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Wrote {len(results)} re-ranked results to {args.output}")


if __name__ == "__main__":
    main()
