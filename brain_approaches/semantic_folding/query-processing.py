#!/usr/bin/env python3
"""
Query Processing for Semantic Folding Pipeline

This module processes user queries by:
1. Extracting and normalizing query phrases
2. Loading phrase fingerprints from sparse cache
3. Constructing query fingerprint with weighting/normalization
4. Applying spreading for improved recall
5. Computing similarity against document fingerprints
6. Ranking and returning top-k results

Usage:
    python query_processing.py \
        --query "machine learning algorithms" \
        --phrase_fps data/phrase_fingerprints_sparse.json \
        --doc_fps data/doc_fingerprints_sparse.json \
        --top_k 10 \
        --weighting idf \
        --normalization l2 \
        --spreading_radius 1
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from scipy.sparse import csr_matrix, lil_matrix
from loguru import logger
import re
from phrase_extractor import extract_raw_phrases_spacy, extract_raw_phrases_fallback, SPACY_AVAILABLE

from lib import (
    normalize_phrase,
    load_phrase_fingerprints_sparse,
    compute_cosine_similarity,
    normalize_fingerprint,
    get_zorder_neighbors,
    batch_compute_similarities,
    is_valid_phrase_structure
)


def extract_query_phrases(
    query: str,
    phrase_vocab: Set[str],
    use_spacy: bool = True,
    remove_verbs: bool = True
) -> List[str]:
    """
    Extract phrases from query using the SAME pipeline as phrase_extractor.py
    """
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    
    # Use the SAME extraction logic as phrase_extractor
    if use_spacy and SPACY_AVAILABLE:
        raw_phrases = extract_raw_phrases_spacy(query)
    else:
        raw_phrases = extract_raw_phrases_fallback(query, max_ngram=4)
    
    # Normalize and validate EXACTLY like phrase_extractor
    matched_phrases = []
    for phrase in raw_phrases:
        norm = normalize_phrase(phrase, remove_verbs=remove_verbs)
        
        if norm and norm in phrase_vocab:
            tokens = word_tokenize(norm)
            tagged = pos_tag(tokens)
            
            if is_valid_phrase_structure(tagged):
                matched_phrases.append(norm)
    
    logger.info(f"Extracted {len(matched_phrases)} phrases from query: {matched_phrases}")
    return matched_phrases



def construct_query_fingerprint(
    query_phrases: List[str],
    phrase_fingerprints: Dict[str, csr_matrix],
    weighting: str = 'uniform',
    idf_weights: Optional[Dict[str, float]] = None,
    normalization: str = 'l2'
) -> Tuple[Optional[csr_matrix], Dict[str, any]]:
    """
    Construct query fingerprint from phrase fingerprints.
    
    Returns:
        Tuple of (fingerprint, metadata)
    """
    if not query_phrases:
        logger.warning("No query phrases provided")
        return None, {'error': 'no_phrases'}
    
    # Get grid size from first fingerprint
    grid_size_sq = list(phrase_fingerprints.values())[0].shape[1]
    grid_size = int(np.sqrt(grid_size_sq))
    
    # Initialize accumulator
    query_fp = lil_matrix((1, grid_size_sq))
    
    # Track phrase contributions
    phrase_weights_used = {}
    missing_phrases = []
    
    # Aggregate phrase fingerprints
    for phrase in query_phrases:
        if phrase not in phrase_fingerprints:
            logger.warning(f"Phrase not in fingerprints: '{phrase}'")
            missing_phrases.append(phrase)
            continue
        
        phrase_fp = phrase_fingerprints[phrase]
        
        # Apply weighting
        if weighting == 'idf' and idf_weights:
            weight = idf_weights.get(phrase, 1.0)
        elif weighting == 'frequency':
            weight = query_phrases.count(phrase)
        else:  # uniform
            weight = 1.0
        
        query_fp += weight * phrase_fp
        phrase_weights_used[phrase] = weight
    
    if query_fp.nnz == 0:
        logger.error("Query fingerprint is empty after aggregation")
        return None, {
            'error': 'empty_fingerprint',
            'missing_phrases': missing_phrases
        }
    
    query_fp = query_fp.tocsr()
    
    # Store pre-normalization stats
    pre_norm_nnz = query_fp.nnz
    
    # Normalize
    if normalization and normalization != 'none':
        query_fp = normalize_fingerprint(query_fp, method=normalization)
    
    metadata = {
        'num_phrases': len(query_phrases),
        'num_matched': len(phrase_weights_used),
        'num_missing': len(missing_phrases),
        'missing_phrases': missing_phrases,
        'phrase_weights': phrase_weights_used,
        'active_bits_pre_norm': pre_norm_nnz,
        'active_bits': query_fp.nnz,
        'sparsity': query_fp.nnz / grid_size_sq,
        'weighting': weighting,
        'normalization': normalization
    }
    
    logger.success(f"Constructed query fingerprint: {query_fp.nnz} active bits from {len(phrase_weights_used)} phrases")
    return query_fp, metadata


def apply_spreading(
    fingerprint: csr_matrix,
    grid_size: int,
    radius: int = 1,
    decay: float = 0.5,
    normalize_after: bool = True
) -> Tuple[csr_matrix, Dict[str, any]]:
    """
    Apply spreading to fingerprint for improved recall.
    
    Returns:
        Tuple of (spread fingerprint, metadata)
    """
    if radius == 0:
        return fingerprint, {'spreading_applied': False}
    
    original_nnz = fingerprint.nnz
    
    # Convert to dense for spreading
    dense_fp = fingerprint.toarray().reshape(grid_size, grid_size)
    spread_fp = dense_fp.copy()
    
    # Get active coordinates
    active_coords = np.argwhere(dense_fp > 0)
    
    # Apply spreading
    for y, x in active_coords:
        value = dense_fp[y, x]
        neighbors = get_zorder_neighbors(x, y, grid_size, radius)
        
        for nx, ny in neighbors:
            # Apply decay based on distance
            dist = max(abs(nx - x), abs(ny - y))
            spread_value = value * (decay ** dist)
            spread_fp[ny, nx] += spread_value
    
    # Convert back to sparse
    spread_fp_flat = spread_fp.reshape(1, -1)
    result = csr_matrix(spread_fp_flat)
    
    # Optional normalization after spreading
    if normalize_after:
        result = normalize_fingerprint(result, method='l2')
    
    metadata = {
        'spreading_applied': True,
        'radius': radius,
        'decay': decay,
        'active_bits_before': original_nnz,
        'active_bits_after': result.nnz,
        'bits_added': result.nnz - original_nnz
    }
    
    logger.info(f"Applied spreading: {original_nnz} → {result.nnz} active bits (+{result.nnz - original_nnz})")
    return result, metadata


def load_document_fingerprints(
    doc_fps_path: Path
) -> Tuple[Dict[str, csr_matrix], Dict[str, any]]:
    """
    Load document fingerprints from sparse JSON cache.
    
    Returns:
        Tuple of (fingerprints dict, metadata)
    """
    logger.info(f"Loading document fingerprints from {doc_fps_path}")
    
    with open(doc_fps_path, 'r') as f:
        data = json.load(f)
    
    grid_size = data['metadata']['grid_size']
    grid_size_sq = grid_size * grid_size
    
    doc_fingerprints = {}
    for doc_id, fp_data in data['fingerprints'].items():
        indices = fp_data['indices']
        values = fp_data['values']
        
        # Reconstruct sparse matrix
        fp = lil_matrix((1, grid_size_sq))
        for idx, val in zip(indices, values):
            fp[0, idx] = val
        
        doc_fingerprints[doc_id] = fp.tocsr()
    
    metadata = data['metadata']
    logger.success(f"Loaded {len(doc_fingerprints)} document fingerprints (grid_size={grid_size})")
    return doc_fingerprints, metadata


def rank_documents(
    query_fp: csr_matrix,
    doc_fingerprints: Dict[str, csr_matrix],
    top_k: int = 10,
    min_similarity: float = 0.0,
    use_batch: bool = True
) -> Tuple[List[Tuple[str, float]], Dict[str, any]]:
    """
    Rank documents by similarity to query.
    
    Returns:
        Tuple of (ranked results, metadata)
    """
    logger.info(f"Ranking {len(doc_fingerprints)} documents")
    
    if use_batch and len(doc_fingerprints) > 100:
        # Use batch processing for large document sets
        doc_ids = list(doc_fingerprints.keys())
        doc_fps = [doc_fingerprints[doc_id] for doc_id in doc_ids]
        
        similarities = batch_compute_similarities(query_fp, doc_fps)
        scores = list(zip(doc_ids, similarities))
    else:
        # Individual similarity computation
        scores = []
        for doc_id, doc_fp in doc_fingerprints.items():
            similarity = compute_cosine_similarity(query_fp, doc_fp)
            scores.append((doc_id, similarity))
    
    # Filter by minimum similarity
    if min_similarity > 0:
        scores = [(doc_id, sim) for doc_id, sim in scores if sim >= min_similarity]
    
    # Sort by similarity (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Compute statistics
    all_similarities = [s for _, s in scores]
    metadata = {
        'total_documents': len(doc_fingerprints),
        'documents_above_threshold': len(scores),
        'mean_similarity': float(np.mean(all_similarities)) if all_similarities else 0.0,
        'max_similarity': float(np.max(all_similarities)) if all_similarities else 0.0,
        'min_similarity_threshold': min_similarity
    }
    
    return scores[:top_k], metadata


def display_results(
    results: List[Tuple[str, float]],
    query: str,
    query_metadata: Dict[str, any],
    ranking_metadata: Dict[str, any],
    doc_metadata: Optional[Dict[str, Dict[str, any]]] = None,
    verbose: bool = False
) -> None:
    """Display query results in a formatted manner."""
    
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    if verbose:
        print("\nQuery Analysis:")
        print(f"  Phrases matched: {query_metadata.get('num_matched', 0)}/{query_metadata.get('num_phrases', 0)}")
        print(f"  Active bits: {query_metadata.get('active_bits', 0)}")
        print(f"  Sparsity: {query_metadata.get('sparsity', 0):.4f}")
        
        if query_metadata.get('missing_phrases'):
            print(f"  Missing phrases: {', '.join(query_metadata['missing_phrases'])}")
        
        print(f"\nCorpus Statistics:")
        print(f"  Total documents: {ranking_metadata.get('total_documents', 0)}")
        print(f"  Mean similarity: {ranking_metadata.get('mean_similarity', 0):.4f}")
        print(f"  Max similarity: {ranking_metadata.get('max_similarity', 0):.4f}")
    
    print(f"\nTop {len(results)} Results:")
    print("-"*80)
    
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank:2d}. {doc_id:50s} | Score: {score:.4f}")
        
        if verbose and doc_metadata and doc_id in doc_metadata:
            meta = doc_metadata[doc_id]
            if 'matched_phrases' in meta:
                print(f"    Matched phrases: {meta['matched_phrases']}")
            if 'coverage' in meta:
                print(f"    Coverage: {meta['coverage']:.3f}")
    
    print("="*80)


def process_query(
    query: str,
    phrase_fingerprints: Dict[str, csr_matrix],
    doc_fingerprints: Dict[str, csr_matrix],
    args: argparse.Namespace,
    idf_weights: Optional[Dict[str, float]] = None
) -> Tuple[List[Tuple[str, float]], Dict[str, any]]:
    """
    Process a single query end-to-end.
    
    Returns:
        Tuple of (results, combined_metadata)
    """
    phrase_vocab = set(phrase_fingerprints.keys())
    
    # Extract query phrases
    query_phrases = extract_query_phrases(
        query,
        phrase_vocab,
        max_ngram=args.max_ngram,
        min_word_length=args.min_word_length
    )
    
    if not query_phrases:
        logger.error("No valid phrases found in query")
        return [], {'error': 'no_phrases_extracted'}
    
    # Construct query fingerprint
    norm = None if args.normalization == 'none' else args.normalization
    query_fp, query_metadata = construct_query_fingerprint(
        query_phrases,
        phrase_fingerprints,
        weighting=args.weighting,
        idf_weights=idf_weights,
        normalization=norm
    )
    
    if query_fp is None:
        logger.error("Failed to construct query fingerprint")
        return [], query_metadata
    
    # Apply spreading
    spreading_metadata = {}
    if args.spreading_radius > 0:
        grid_size = int(np.sqrt(query_fp.shape[1]))
        query_fp, spreading_metadata = apply_spreading(
            query_fp,
            grid_size,
            radius=args.spreading_radius,
            decay=args.spreading_decay,
            normalize_after=args.normalize_after_spreading
        )
    
    # Rank documents
    results, ranking_metadata = rank_documents(
        query_fp,
        doc_fingerprints,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        use_batch=args.use_batch
    )
    
    # Combine metadata
    combined_metadata = {
        'query': query,
        'query_construction': query_metadata,
        'spreading': spreading_metadata,
        'ranking': ranking_metadata
    }
    
    return results, combined_metadata


def main():
    parser = argparse.ArgumentParser(description='Process queries against document fingerprints')
    
    # Required arguments
    parser.add_argument('--query', type=str, help='Query string (or use --query_file)')
    parser.add_argument('--phrases', type=Path, help='phrases.txt path')
    parser.add_argument('--phrase_fingerprints', type=Path, required=True, help='Phrase fingerprints ')
    parser.add_argument('--doc_fingerprints', type=Path, required=True, help='Document fingerprints (sparse JSON)')
    
    # Optional inputs
    parser.add_argument('--idf_weights', type=Path, help='IDF weights JSON (optional)')
    
    # Query processing parameters
    parser.add_argument('--max_ngram', type=int, default=3, help='Maximum n-gram size for phrase extraction')
    parser.add_argument('--min_word_length', type=int, default=2, help='Minimum word length')
    parser.add_argument('--weighting', type=str, default='uniform', choices=['uniform', 'frequency', 'idf'])
    parser.add_argument('--normalization', type=str, default='l2', choices=['l2', 'l1', 'binary', 'none'])
    
    # Spreading parameters
    parser.add_argument('--spreading_radius', type=int, default=1, help='Spreading radius (0 to disable)')
    parser.add_argument('--spreading_decay', type=float, default=0.5, help='Spreading decay factor')
    parser.add_argument('--normalize_after_spreading', action='store_true', help='Normalize after spreading')
    
    # Ranking parameters
    parser.add_argument('--top_k', type=int, default=10, help='Number of results to return')
    parser.add_argument('--min_similarity', type=float, default=0.0, help='Minimum similarity threshold')
    parser.add_argument('--use_batch', action='store_true', default=True, help='Use batch similarity computation')
    
    # Output options
    parser.add_argument('--output_json', type=Path, help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.query and not args.query_file:
        parser.error("Either --query or --query_file must be provided")
    
    if not args.phrase_fps.exists():
        logger.error(f"Phrase fingerprints file not found: {args.phrase_fps}")
        return
    
    if not args.doc_fps.exists():
        logger.error(f"Document fingerprints file not found: {args.doc_fps}")
        return
    
    # Load phrase fingerprints
    phrase_fingerprints = load_phrase_fingerprints_sparse(args.phrase_fps)
    
    # Load IDF weights if needed
    idf_weights = None
    if args.weighting == 'idf':
        if args.idf_weights and args.idf_weights.exists():
            with open(args.idf_weights, 'r') as f:
                idf_weights = json.load(f)
            logger.info(f"Loaded IDF weights for {len(idf_weights)} phrases")
        else:
            logger.warning("IDF weighting requested but no weights file provided, using uniform weights")
            args.weighting = 'uniform'
    
    # Load document fingerprints
    doc_fingerprints, doc_metadata = load_document_fingerprints(args.doc_fps)
    
    # Process queries
    all_results = []
    
    if args.query:
        # Single query
        results, metadata = process_query(
            args.query,
            phrase_fingerprints,
            doc_fingerprints,
            args,
            idf_weights
        )
        
        display_results(
            results,
            args.query,
            metadata['query_construction'],
            metadata['ranking'],
            verbose=args.verbose
        )
        
        all_results.append({
            'query': args.query,
            'results': [(doc_id, float(score)) for doc_id, score in results],
            'metadata': metadata
        })
    
    elif args.query_file:
        # Multiple queries from file
        with open(args.query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(queries)} queries from {args.query_file}")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\n[{i}/{len(queries)}] Processing: {query}")
            
            results, metadata = process_query(
                query,
                phrase_fingerprints,
                doc_fingerprints,
                args,
                idf_weights
            )
            
            display_results(
                results,
                query,
                metadata['query_construction'],
                metadata['ranking'],
                verbose=args.verbose
            )
            
            all_results.append({
                'query': query,
                'results': [(doc_id, float(score)) for doc_id, score in results],
                'metadata': metadata
            })
    
    # Save results to JSON if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.success(f"Saved results to {args.output_json}")


if __name__ == '__main__':
    main()
