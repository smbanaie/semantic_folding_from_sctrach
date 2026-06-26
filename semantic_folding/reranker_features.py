#!/usr/bin/env python3
"""
reranker_features.py — Feature extraction for learned re-ranking (Phase 4)

Extracts 33+ features per (query, document) pair from binary fingerprints.
These features serve as input to LambdaMART or other learned re-rankers.

Features extracted:
  - Binary similarity metrics (5): Jaccard, Dice, overlap, Hamming, cosine
  - Asymmetric features (3): containment, coverage, IDF-weighted intersection
  - Bit-density features (8): popcounts, densities, intersection, union, mismatch
  - Block histogram features (16): per-block Jaccard for 16 blocks of 256 bits
  - Auxiliary features (3): BM25 score, query length, doc length

Usage:
    python -m semantic_folding.reranker_features \\
        --dataset belebele \\
        --jsonl data/belebele/converted/belebele.jsonl \\
        --max-queries 50 \\
        --output features.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from lib import (
    load_document_fingerprints,
    load_phrase_fingerprints_sparse,
    normalize_phrase,
    expand_phrases,
    get_logger,
)

logger = get_logger("reranker_features")


def extract_features(
    query_fp: csr_matrix,
    doc_fp: csr_matrix,
    idf_weights: Optional[np.ndarray] = None,
    bm25_score: float = 0.0,
    block_size: int = 256,
    query_length: int = 0,
    doc_length: int = 0,
) -> Dict[str, float]:
    """
    Extract feature vector for a (query, document) pair.

    Args:
        query_fp: Query fingerprint (1 x N sparse matrix)
        doc_fp: Document fingerprint (1 x N sparse matrix)
        idf_weights: IDF weights array (optional)
        bm25_score: BM25 score for this pair (optional)
        block_size: Size of each block for block histogram features
        query_length: Number of tokens in original query
        doc_length: Number of tokens in original document

    Returns:
        Dictionary of feature name -> feature value
    """
    features = {}

    q_indices = set(query_fp.indices) if hasattr(query_fp, 'indices') else set(np.nonzero(query_fp)[0])
    d_indices = set(doc_fp.indices) if hasattr(doc_fp, 'indices') else set(np.nonzero(doc_fp)[0])

    q_popcount = len(q_indices)
    d_popcount = len(d_indices)
    intersection = len(q_indices & d_indices)
    union = len(q_indices | d_indices)

    # Binary similarity metrics
    features['jaccard'] = intersection / union if union > 0 else 0.0
    features['dice'] = (2.0 * intersection / (q_popcount + d_popcount)) if (q_popcount + d_popcount) > 0 else 0.0
    features['overlap'] = (intersection / min(q_popcount, d_popcount)) if min(q_popcount, d_popcount) > 0 else 0.0
    n_bits = query_fp.shape[1] if hasattr(query_fp, 'shape') else len(query_fp)
    features['hamming_norm'] = 1.0 - (len(q_indices ^ d_indices) / n_bits) if n_bits > 0 else 0.0

    # Cosine similarity (binary vectors)
    q_dense = query_fp.toarray().flatten() if hasattr(query_fp, 'toarray') else query_fp
    d_dense = doc_fp.toarray().flatten() if hasattr(doc_fp, 'toarray') else doc_fp
    q_norm = np.linalg.norm(q_dense)
    d_norm = np.linalg.norm(d_dense)
    features['cosine'] = (np.dot(q_dense, d_dense) / (q_norm * d_norm)) if (q_norm > 0 and d_norm > 0) else 0.0

    # Asymmetric features
    features['containment'] = intersection / q_popcount if q_popcount > 0 else 0.0
    features['coverage'] = intersection / d_popcount if d_popcount > 0 else 0.0

    # IDF-weighted intersection
    if idf_weights is not None:
        matched = q_indices & d_indices
        intersection_weight = sum(idf_weights[j] for j in matched if j < len(idf_weights))
        query_weight = sum(idf_weights[j] for j in q_indices if j < len(idf_weights))
        features['idf_weighted'] = intersection_weight / query_weight if query_weight > 0 else 0.0
    else:
        features['idf_weighted'] = features['containment']

    # Bit-density features
    n_bits = query_fp.shape[1] if hasattr(query_fp, 'shape') else len(query_fp)
    features['q_popcount'] = float(q_popcount)
    features['d_popcount'] = float(d_popcount)
    features['q_density'] = q_popcount / n_bits if n_bits > 0 else 0.0
    features['d_density'] = d_popcount / n_bits if n_bits > 0 else 0.0
    features['intersection_popcount'] = float(intersection)
    features['union_popcount'] = float(union)
    features['q_minus_d'] = float(len(q_indices - d_indices))
    features['d_minus_q'] = float(len(d_indices - q_indices))

    # Block histogram features (16 blocks of 256 bits each for 4096-bit fingerprint)
    n_blocks = n_bits // block_size
    if n_blocks > 0:
        for b in range(min(n_blocks, 16)):
            start = b * block_size
            end = start + block_size
            q_block = q_indices & set(range(start, end))
            d_block = d_indices & set(range(start, end))
            block_inter = len(q_block & d_block)
            block_union = len(q_block | d_block)
            features[f'block_{b}_jaccard'] = block_inter / block_union if block_union > 0 else 0.0
    else:
        for b in range(16):
            features[f'block_{b}_jaccard'] = 0.0

    # Auxiliary features
    features['bm25_score'] = bm25_score
    features['query_length'] = float(query_length)
    features['doc_length'] = float(doc_length)

    return features


def extract_features_batch(
    query_fp: csr_matrix,
    doc_fps: Dict[str, csr_matrix],
    idf_weights: Optional[np.ndarray] = None,
    bm25_scores: Optional[Dict[str, float]] = None,
    block_size: int = 256,
    query_length: int = 0,
    doc_lengths: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """
    Extract features for all documents against a single query.

    Returns:
        List of dicts with keys: doc_id, features, label (0 if unknown)
    """
    results = []
    for doc_id, doc_fp in doc_fps.items():
        bm25 = bm25_scores.get(doc_id, 0.0) if bm25_scores else 0.0
        doc_len = doc_lengths.get(doc_id, 0) if doc_lengths else 0

        feats = extract_features(
            query_fp, doc_fp,
            idf_weights=idf_weights,
            bm25_score=bm25,
            block_size=block_size,
            query_length=query_length,
            doc_length=doc_len,
        )
        results.append({
            "doc_id": doc_id,
            "features": feats,
            "label": 0,
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for learned re-ranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fingerprints", dest="phrase_fp_dir", type=Path, required=True)
    parser.add_argument("--doc-fingerprints", dest="doc_fp_dir", type=Path, required=True)
    parser.add_argument("--idf-weights", dest="idf_weights", type=Path, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--query-file", dest="query_file", type=Path, default=None)
    parser.add_argument("--corpus", dest="corpus_path", type=Path, default=None)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--grid-size", dest="grid_size", type=int, default=64)
    parser.add_argument("--top-k", dest="top_k", type=int, default=10)
    parser.add_argument("--keep-verbs", dest="remove_verbs", action="store_false", default=False)
    parser.add_argument("--no-spacy", dest="no_spacy", action="store_true", default=False)
    parser.add_argument("--no-filter-generic", dest="filter_generic", action="store_false", default=True)
    parser.add_argument("--min-word-length", dest="min_word_length", type=int, default=3)
    args = parser.parse_args()

    logger.info("Loading fingerprints...")
    phrase_fps, meta = load_phrase_fingerprints_sparse(args.phrase_fp_dir, args.grid_size)
    doc_fps = load_document_fingerprints(args.doc_fp_dir)
    logger.info(f"  {len(phrase_fps)} phrases, {len(doc_fps)} documents")

    idf_weights = None
    if args.idf_weights and args.idf_weights.exists():
        with open(args.idf_weights) as f:
            idf_dict = json.load(f)
        max_idx = max(int(k) for k in idf_dict.keys()) + 1 if idf_dict else 0
        idf_weights = np.zeros(max_idx)
        for k, v in idf_dict.items():
            idf_weights[int(k)] = v
        logger.info(f"  Loaded IDF weights: {len(idf_dict)} terms")

    queries = []
    if args.query:
        queries = [args.query]
    elif args.query_file:
        with open(args.query_file) as f:
            queries = [line.strip() for line in f if line.strip()]

    if not queries:
        logger.error("No queries provided")
        sys.exit(1)

    all_features = []
    for q_idx, query_text in enumerate(queries):
        logger.info(f"  [{q_idx}] Processing: {query_text[:80]}...")

        phrases = []
        tokens = query_text.lower().split()
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 5, len(tokens) + 1)):
                phrase = " ".join(tokens[i:j])
                norm = normalize_phrase(phrase)
                if norm and norm in phrase_fps:
                    phrases.append(norm)

        if not phrases:
            logger.warning(f"  [{q_idx}] No valid phrases found")
            continue

        from lib import merge_fingerprints
        fps_list = [phrase_fps[p] for p in phrases if p in phrase_fps]
        weights = [1.0] * len(fps_list)
        query_fp = merge_fingerprints(fps_list, weights)

        doc_feats = extract_features_batch(
            query_fp, doc_fps,
            idf_weights=idf_weights,
            query_length=len(tokens),
        )

        for feat_dict in doc_feats:
            feat_dict["query_text"] = query_text
            feat_dict["query_idx"] = q_idx

        all_features.extend(doc_feats)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for feat_dict in all_features:
            f.write(json.dumps(feat_dict) + "\n")

    logger.info(f"Wrote {len(all_features)} feature vectors to {args.output_path}")


if __name__ == "__main__":
    main()
