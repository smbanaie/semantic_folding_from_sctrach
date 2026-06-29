"""
Standalone SF re-ranking module for two-stage neuro-lexical pipeline.

This avoids modifying the complex query_processor.py by implementing
a focused SF re-ranking function that scores ONLY candidate documents.

Usage:
    from sf_reranker import rerank_candidates_with_sf
    
    # BM25 top-K candidates
    candidates = [("doc_001", 0.8), ("doc_002", 0.7), ...]
    
    # Re-rank using SF
    reranked = rerank_candidates_with_sf(
        query="test query",
        candidates=candidates,
        doc_fingerprints=doc_fingerprints,
        phrase_fingerprints=phrase_fingerprints,
        grid_size=64,
    )
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import get_logger

logger = get_logger("sf_reranker")


def load_document_fingerprints(doc_fp_dir: Path, grid_size: int = 64) -> Dict[str, sparse.csr_matrix]:
    """Load document fingerprints from JSON files."""
    doc_fingerprints = {}
    dim = grid_size * grid_size
    
    for fp_file in doc_fp_dir.glob("*.json"):
        with open(fp_file) as f:
            data = json.load(f)
        doc_id = data.get("doc_id", fp_file.stem)
        active_bits = data.get("active_bits", [])
        
        # Convert to sparse vector
        vec = sparse.csr_matrix((1, dim))
        if active_bits:
            vec[0, active_bits] = 1.0
        doc_fingerprints[doc_id] = vec
    
    logger.info(f"Loaded {len(doc_fingerprints)} document fingerprints from {doc_fp_dir}")
    return doc_fingerprints


def load_phrase_fingerprints(phrase_fp_dir: Path, grid_size: int = 64) -> Dict[str, sparse.csr_matrix]:
    """Load phrase fingerprints from JSON files."""
    phrase_fingerprints = {}
    dim = grid_size * grid_size
    
    for fp_file in phrase_fp_dir.glob("*.json"):
        with open(fp_file) as f:
            data = json.load(f)
        phrase_text = data.get("phrase", fp_file.stem)
        active_bits = data.get("active_bits", [])
        
        # Convert to sparse vector
        vec = sparse.csr_matrix((1, dim))
        if active_bits:
            vec[0, active_bits] = 1.0
        phrase_fingerprints[phrase_text] = vec
    
    logger.info(f"Loaded {len(phrase_fingerprints)} phrase fingerprints from {phrase_fp_dir}")
    return phrase_fingerprints


def construct_simple_query_fingerprint(
    query: str,
    phrase_fingerprints: Dict[str, sparse.csr_matrix],
    grid_size: int = 64,
) -> sparse.csr_matrix:
    """
    Simplified query fingerprint construction.
    
    TODO: Replace with actual phrase_extractor + construct_query_fingerprint
    from query_processor.py for production use.
    """
    dim = grid_size * grid_size
    query_fp = sparse.csr_matrix((1, dim))
    
    # Simple approach: use query terms to find matching phrase fingerprints
    query_terms = query.lower().split()
    
    for term in query_terms:
        if term in phrase_fingerprints:
            query_fp = query_fp + phrase_fingerprints[term]
    
    # Normalize
    if query_fp.nnz > 0:
        query_fp = query_fp / sparse.linalg.norm(query_fp)
    
    return query_fp


def rerank_candidates_with_sf(
    query: str,
    candidates: List[Tuple[str, float]],
    doc_fingerprints: Dict[str, sparse.csr_matrix],
    phrase_fingerprints: Dict[str, sparse.csr_matrix],
    grid_size: int = 64,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Re-rank BM25 candidates using Semantic Folding.
    
    Parameters
    ----------
    query : str
        Query text
    candidates : List[Tuple[str, float]]
        BM25 results: [(doc_id, bm25_score), ...]
    doc_fingerprints : Dict[str, sparse.csr_matrix]
        Pre-computed document fingerprints
    phrase_fingerprints : Dict[str, sparse.csr_matrix]
        Pre-computed phrase fingerprints
    grid_size : int
        Grid size (default: 64)
    top_k : int
        Number of re-ranked results to return
    
    Returns
    -------
    List[Tuple[str, float]]
        Re-ranked results: [(doc_id, sf_score), ...]
    """
    # Construct query fingerprint
    query_fp = construct_simple_query_fingerprint(query, phrase_fingerprints, grid_size)
    
    if query_fp.nnz == 0:
        logger.warning(f"Query fingerprint is empty for: {query[:50]}")
        return candidates[:top_k]  # Return BM25 order if SF fails
    
    # Score only candidate documents
    candidate_scores = []
    for doc_id, bm25_score in candidates:
        if doc_id not in doc_fingerprints:
            continue
        
        doc_fp = doc_fingerprints[doc_id]
        
        # Compute cosine similarity
        sim = cosine_similarity(query_fp, doc_fp)[0, 0]
        
        # Combine with BM25 score (simple interpolation)
        # TODO: Make interpolation weight configurable
        combined_score = 0.5 * sim + 0.5 * bm25_score
        
        candidate_scores.append((doc_id, combined_score))
    
    # Sort by combined score
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    return candidate_scores[:top_k]


def rerank_candidates_with_sf_full(
    query: str,
    candidates: List[Tuple[str, float]],
    doc_fingerprints: Dict[str, sparse.csr_matrix],
    phrase_fingerprints: Dict[str, sparse.csr_matrix],
    idf_weights: Dict[str, float] = None,
    grid_size: int = 64,
    top_k: int = 10,
    spreading_steps: int = 1,
    spreading_decay: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Full SF re-ranking with spreading activation.
    
    This is a more complete implementation that mirrors query_processor.py
    but operates only on candidates.
    """
    from semantic_folding.query_processor import (
        extract_query_phrases,
        construct_query_fingerprint,
        apply_spreading,
        rank_documents,
    )
    
    # TODO: Implement full pipeline
    # For now, fall back to simple version
    logger.warning("Full SF re-ranking not yet implemented, using simple version")
    return rerank_candidates_with_sf(
        query, candidates, doc_fingerprints, phrase_fingerprints, grid_size, top_k
    )


if __name__ == "__main__":
    # Test
    print("SF Reranker Module")
    print("This module provides rerank_candidates_with_sf() for two-stage retrieval")
