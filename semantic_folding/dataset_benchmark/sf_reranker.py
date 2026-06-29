"""
SF re-ranking module - FIXED to use actual query processing.

Instead of naive phrase matching, this uses the real query_processor.py
pipeline: phrase_extractor → construct_query_fingerprint → rank_documents
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
from semantic_folding.query_processor import (
    extract_query_phrases,
    construct_query_fingerprint,
    apply_spreading,
    rank_documents,
    process_query,
)

logger = get_logger("sf_reranker")


def rerank_candidates_with_sf(
    query: str,
    candidates: List[Tuple[str, float]],
    doc_fingerprints: Dict[str, sparse.csr_matrix],
    phrase_fingerprints: Dict[str, sparse.csr_matrix],
    grid_size: int = 64,
    top_k: int = 10,
    idf_weights: Dict[str, float] = None,
    spreading_steps: int = 1,
    spreading_decay: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Re-rank BM25 candidates using actual SF pipeline.
    
    Uses process_query() from query_processor.py but filters to candidates only.
    """
    t0 = time.time()
    
    # Load query processor args (simplified)
    import argparse
    args = argparse.Namespace(
        no_spacy=False,
        remove_verbs=False,
        filter_generic=True,
        min_word_length=3,
        weighting="uniform",
        normalization="l2",
        spreading_steps=spreading_steps,
        spreading_decay=spreading_decay,
        normalize_after_spreading=False,
        top_k=top_k,
        min_similarity=0.0,
        use_batch=False,
        phrase_fp_dir=None,
        grid_size=grid_size,
        doc_norm="sqrt_nnz",
        sim_metric="cosine",
        geometric=False,
    )
    
    # Get phrase vocab
    phrase_vocab = set(phrase_fingerprints.keys())
    
    if len(phrase_vocab) == 0:
        logger.warning("No phrase fingerprints available, returning BM25 order")
        return candidates[:top_k]
    
    # Process query (simplified - doesn't actually rank documents)
    # TODO: Modify process_query() to accept candidate list
    try:
        # Extract query phrases
        matched_phrases, missing_phrases, all_expanded = extract_query_phrases(
            query, phrase_vocab,
            use_spacy=not args.no_spacy,
            remove_verbs=args.remove_verbs,
            filter_generic=args.filter_generic,
            min_word_length=args.min_word_length,
        )
        
        if len(matched_phrases) == 0:
            logger.warning(f"No matched phrases for query: {query[:50]}")
            return candidates[:top_k]
        
        # Construct query fingerprint
        query_fp, query_metadata = construct_query_fingerprint(
            matched_phrases, phrase_fingerprints, phrase_vocab,
            weighting=args.weighting,
            idf_weights=idf_weights,
            grid_size=grid_size,
        )
        
        # Apply spreading
        if args.spreading_steps > 0:
            query_fp, spreading_metadata = apply_spreading(
                query_fp, grid_size, args.spreading_steps, args.spreading_decay,
                normalize_after=args.normalize_after_spreading,
            )
        
        # Score ONLY candidate documents
        candidate_scores = []
        for doc_id, bm25_score in candidates:
            if doc_id not in doc_fingerprints:
                continue
            
            doc_fp = doc_fingerprints[doc_id]
            
            # Compute similarity
            from semantic_folding.query_processor import compute_weighted_overlap
            score = compute_weighted_overlap(
                query_fp, doc_fp,
                doc_norm=args.doc_norm,
                sim_metric=args.sim_metric,
                grid_size=grid_size,
            )
            
            candidate_scores.append((doc_id, score))
        
        # Sort by SF score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        elapsed = time.time() - t0
        logger.debug(f"SF re-ranking took {elapsed:.3f}s for {len(candidates)} candidates")
        
        return candidate_scores[:top_k]
        
    except Exception as e:
        logger.error(f"SF re-ranking failed: {e}")
        import traceback
        traceback.print_exc()
        return candidates[:top_k]  # Fall back to BM25 order


if __name__ == "__main__":
    print("SF Reranker Module (Fixed)")
    print("Uses actual query_processor.py pipeline")
