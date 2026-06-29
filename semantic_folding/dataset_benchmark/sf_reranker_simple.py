"""
SF Re-ranking - SIMPLIFIED WORKING VERSION

Computes query→document similarity using pre-computed fingerprints.
No complex imports needed.

Approach:
1. Build query fingerprint by summing phrase fingerprints for matched phrases
2. Normalize query fingerprint
3. Compute cosine similarity with candidate document fingerprints
4. Interpolate with BM25 scores
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantic_folding.lib import get_logger

logger = get_logger("sf_reranker_simple")


def build_query_fingerprint_simple(
    query: str,
    phrase_fingerprints: Dict[str, sparse.csr_matrix],
    grid_size: int = 64,
) -> sparse.csr_matrix:
    """
    Simplified query fingerprint: sum of matched phrase fingerprints.
    
    Matches query terms to phrase vocabulary (exact match or substring).
    phrase_fingerprints should be a dict mapping phrase text → sparse vector.
    """
    dim = grid_size * grid_size
    query_fp = sparse.csr_matrix((1, dim))
    
    if len(phrase_fingerprints) == 0:
        logger.warning("No phrase fingerprints provided")
        return query_fp
    
    query_terms = query.lower().split()
    matched_phrases = []
    
    # Try exact match first
    for term in query_terms:
        if term in phrase_fingerprints:
            query_fp = query_fp + phrase_fingerprints[term]
            matched_phrases.append(term)
    
    # If no exact matches, try substring matching
    if len(matched_phrases) == 0:
        for term in query_terms:
            for phrase, fp in phrase_fingerprints.items():
                if term in phrase or phrase in term:
                    query_fp = query_fp + fp
                    matched_phrases.append(phrase)
                    break  # Only add first match per term
    
    # Normalize
    if query_fp.nnz > 0:
        norm = sparse.linalg.norm(query_fp)
        if norm > 0:
            query_fp = query_fp / norm
    
    if len(matched_phrases) == 0:
        logger.warning(f"No phrase matches for query: {query[:50]}")
        logger.debug(f"  Query terms: {query_terms}")
        logger.debug(f"  Phrase vocab sample: {list(phrase_fingerprints.keys())[:10]}")
    else:
        logger.debug(f"Matched {len(matched_phrases)} phrases: {matched_phrases[:5]}")
    
    return query_fp


def load_phrase_fingerprints_from_npz(
    npz_path: Path,
    meta_path: Path,
    grid_size: int = 64,
) -> Dict[str, sparse.csr_matrix]:
    """
    Load phrase fingerprints from .npz file.
    
    The .npz file contains a matrix of shape (n_phrases, grid_size^2).
    The meta.json file maps phrase text → row index.
    """
    import numpy as np
    
    # Load fingerprint matrix
    data = np.load(npz_path)
    fingerprints = data['fingerprints']  # Shape: (n_phrases, grid_size^2)
    
    # Load metadata (phrase → index mapping)
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Build phrase → fingerprint dict
    phrase_fps = {}
    
    # Try different possible metadata formats
    if 'phrase_to_idx' in meta:
        # Format: {"phrase_to_idx": {"term1": 0, "term2": 1, ...}}
        phrase_to_idx = meta['phrase_to_idx']
        for phrase, idx in phrase_to_idx.items():
            vec = sparse.csr_matrix(fingerprints[idx])
            phrase_fps[phrase] = vec
    elif 'phrase_to_row' in meta:
        # Format: {"phrase_to_row": {"term1": 0, "term2": 1, ...}}
        phrase_to_row = meta['phrase_to_row']
        for phrase, idx in phrase_to_row.items():
            vec = sparse.csr_matrix(fingerprints[idx])
            phrase_fps[phrase] = vec
    elif 'phrases' in meta:
        # Format: {"phrases": ["term1", "term2", ...]}
        phrases = meta['phrases']
        for idx, phrase in enumerate(phrases):
            vec = sparse.csr_matrix(fingerprints[idx])
            phrase_fps[phrase] = vec
    else:
        # Try to infer from metadata keys
        logger.warning(f"Unknown metadata format, keys: {list(meta.keys())}")
        # Assume rows are in order of some list
        # This is a guess - need to check actual format
    
    logger.info(f"Loaded {len(phrase_fps)} phrase fingerprints from {npz_path}")
    return phrase_fps


def rerank_candidates_with_sf_simple(
    query: str,
    candidates: List[Tuple[str, float]],
    doc_fingerprints: Dict[str, sparse.csr_matrix],
    phrase_fingerprints: Dict[str, sparse.csr_matrix],
    grid_size: int = 64,
    top_k: int = 10,
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Re-rank BM25 candidates using SF similarity.
    
    Parameters
    ----------
    alpha : float
        Interpolation weight: final = alpha * bm25 + (1-alpha) * sf
        alpha=0.5 means equal weight to both
    """
    t0 = time.time()
    
    # Build query fingerprint
    query_fp = build_query_fingerprint_simple(query, phrase_fingerprints, grid_size)
    
    if query_fp.nnz == 0:
        logger.warning(f"Query fingerprint is empty, returning BM25 order")
        return candidates[:top_k]
    
    # Score candidates
    candidate_scores = []
    
    # Normalize BM25 scores to [0, 1]
    bm25_scores = [score for _, score in candidates]
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    min_bm25 = min(bm25_scores) if bm25_scores else 0.0
    bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
    
    for doc_id, bm25_score in candidates:
        if doc_id not in doc_fingerprints:
            continue
        
        doc_fp = doc_fingerprints[doc_id]
        
        # Compute cosine similarity
        sim = cosine_similarity(query_fp, doc_fp)[0, 0]
        
        # Normalize BM25 score to [0, 1]
        bm25_norm = (bm25_score - min_bm25) / bm25_range
        
        # Interpolate: final = alpha * bm25_norm + (1-alpha) * sf
        combined_score = alpha * bm25_norm + (1 - alpha) * sim
        
        candidate_scores.append((doc_id, combined_score))
    
    # Sort by combined score
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    elapsed = time.time() - t0
    logger.debug(f"SF re-ranking took {elapsed:.3f}s for {len(candidates)} candidates")
    
    return candidate_scores[:top_k]


if __name__ == "__main__":
    # Quick test
    print("SF Reranker Simple - Testing...")
    
    # Create dummy data
    grid_size = 16
    dim = grid_size * grid_size
    
    phrase_fps = {
        "test": sparse.csr_matrix([[1, 0, 1, 0] + [0] * (dim - 4)]),
        "query": sparse.csr_matrix([[0, 1, 0, 1] + [0] * (dim - 4)]),
    }
    
    doc_fps = {
        "doc_001": sparse.csr_matrix([[1, 1, 0, 0] + [0] * (dim - 4)]),
        "doc_002": sparse.csr_matrix([[0, 0, 1, 1] + [0] * (dim - 4)]),
    }
    
    candidates = [("doc_001", 0.8), ("doc_002", 0.7)]
    
    reranked = rerank_candidates_with_sf_simple(
        "test query",
        candidates,
        doc_fps,
        phrase_fps,
        grid_size=16,
    )
    
    print(f"Re-ranked: {reranked}")
    print("Done!")
