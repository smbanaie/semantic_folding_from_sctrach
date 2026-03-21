#!/usr/bin/env python3
"""
Term-Context Matrix Builder for Semantic Folding Pipeline

Constructs sparse term-context co-occurrence matrix from corpus and phrase inventory,
applying TF-IDF normalization to reduce high-frequency term dominance.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set
from nltk import pos_tag, word_tokenize

from loguru import logger
from tqdm import tqdm

# Import from centralized library
from lib import (
    load_contexts,
    load_phrases,
    find_phrase_occurrences,
    normalize_phrase,
    is_valid_phrase_structure
)

# Sparse matrix dependencies
try:
    import scipy.sparse
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available. Install with: pip install scipy numpy")
    SCIPY_AVAILABLE = False


def apply_tf_idf_normalization(
    matrix: scipy.sparse.lil_matrix,
    num_contexts: int
) -> scipy.sparse.lil_matrix:
    """
    Apply TF-IDF weighting to term-context matrix.
    
    TF-IDF reduces the dominance of high-frequency terms by weighting
    each term by its inverse document frequency:
    
        TF-IDF(t,d) = TF(t,d) × log(N / DF(t))
    
    where:
        TF(t,d) = raw count of term t in document d
        DF(t) = number of documents containing term t
        N = total number of documents
    
    Args:
        matrix: Sparse LIL matrix (contexts × phrases)
        num_contexts: Total number of contexts
    
    Returns:
        TF-IDF weighted sparse matrix
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy unavailable, skipping TF-IDF normalization")
        return matrix
    
    num_phrases = matrix.shape[1]
    
    # Convert to CSC for efficient column operations
    matrix_csc = matrix.tocsc()
    
    # Calculate document frequency (DF) for each phrase
    # DF(t) = number of contexts where phrase t appears (non-zero entries)
    df = np.diff(matrix_csc.indptr)
    
    # Calculate inverse document frequency (IDF)
    # Use standard IDF formula with smoothing: log(N / DF)
    # Add 1 to DF to avoid log(0) for phrases appearing in all documents
    idf = np.log(num_contexts / (df + 1))
    
    # Apply IDF weighting to each column (phrase)
    # Multiply each column by its IDF value
    idf_diag = scipy.sparse.diags(idf, format='csc')
    normalized_matrix = matrix_csc @ idf_diag
    
    # Convert back to LIL format for consistency
    normalized_matrix = normalized_matrix.tolil()
    
    logger.info(f"TF-IDF applied: {matrix.nnz} → {normalized_matrix.nnz} non-zero entries")
    logger.info(f"IDF range: [{idf.min():.4f}, {idf.max():.4f}]")
    
    return normalized_matrix


def normalize_and_validate_phrases(
    phrases: List[Tuple[str, int]],
    remove_verbs: bool = True
) -> List[Tuple[str, int]]:
    """
    Normalize and validate phrases for matrix construction.
    
    Ensures phrases match the normalization applied during extraction,
    maintaining consistency across the pipeline.
    
    Args:
        phrases: List of (phrase, frequency) tuples
        remove_verbs: Remove verbal elements during normalization
    
    Returns:
        List of (normalized_phrase, frequency) tuples
    """
    normalized_phrases = []
    skipped = 0
    
    for phrase, freq in phrases:
        # Normalize phrase using lib.py function
        normalized = normalize_phrase(phrase, remove_verbs=remove_verbs)
        
        # Skip empty phrases after normalization
        if not normalized or not normalized.strip():
            skipped += 1
            logger.warning(f"Skipped {phrase} (Normalized:{normalized}) - invalid phrases after normalization")
            continue
        
        # Validate phrase structure
        # POS-tag the normalized phrase before structural validation
        tagged = pos_tag(word_tokenize(normalized))
        if not is_valid_phrase_structure(tagged):
            skipped += 1
            logger.warning(f"Skipped {phrase} (Normalized:{normalized}) - wrong phrases structure")
            continue
        
        normalized_phrases.append((normalized, freq))
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} invalid phrases after normalization")
    
    logger.info(f"Normalized phrases: {len(phrases)} → {len(normalized_phrases)}")
    
    return normalized_phrases


def build_phrase_index(phrases: List[Tuple[str, int]]) -> Dict[str, int]:
    """
    Build efficient phrase lookup index.
    
    Args:
        phrases: List of (phrase, frequency) tuples
    
    Returns:
        Dictionary mapping phrase to matrix column index
    """
    return {phrase: idx for idx, (phrase, _) in enumerate(phrases)}


def count_phrase_in_context(
    context_text: str,
    phrases: List[str],
    phrase_to_idx: Dict[str, int],
    use_word_boundaries: bool = True
) -> Dict[int, int]:
    """
    Count all phrase occurrences in a single context.
    
    Args:
        context_text: Normalized context text
        phrases: List of normalized phrases to search for
        phrase_to_idx: Phrase to index mapping
        use_word_boundaries: Use word boundary detection
    
    Returns:
        Dictionary mapping phrase index to occurrence count
    """
    counts = {}
    
    for phrase in phrases:
        count = find_phrase_occurrences(
            context_text,
            phrase,
            use_word_boundaries=use_word_boundaries
        )
        
        if count > 0:
            phrase_idx = phrase_to_idx[phrase]
            counts[phrase_idx] = count
    
    return counts


def build_term_context_matrix(
    phrases: List[Tuple[str, int]],
    contexts: List[Tuple[str, str]],
    normalize_tfidf: bool = True,
    use_word_boundaries: bool = True,
    remove_verbs: bool = True
) -> scipy.sparse.lil_matrix:
    """
    Construct sparse term-context co-occurrence matrix.
    
    Args:
        phrases: List of (phrase, frequency) tuples
        contexts: List of (context_id, context_text) tuples
        normalize_tfidf: Apply TF-IDF normalization
        use_word_boundaries: Use word boundary detection for phrase matching
        remove_verbs: Remove verbs during phrase normalization
    
    Returns:
        Sparse matrix (contexts × phrases) with co-occurrence counts
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy required for sparse matrix operations. "
                         "Install with: pip install scipy numpy")
    
    # Normalize and validate phrases
    logger.info("Normalizing and validating phrases...")
    phrases = normalize_and_validate_phrases(phrases, remove_verbs=remove_verbs)
    
    if not phrases:
        raise ValueError("No valid phrases after normalization")
    
    num_contexts = len(contexts)
    num_phrases = len(phrases)
    
    logger.info(f"Building matrix: {num_contexts} contexts × {num_phrases} phrases")
    
    # Initialize sparse matrix (LIL format for efficient construction)
    matrix = scipy.sparse.lil_matrix((num_contexts, num_phrases), dtype=np.float32)
    
    # Build phrase lookup index
    phrase_list = [p[0] for p in phrases]
    phrase_to_idx = build_phrase_index(phrases)
    
    # Track statistics
    total_matches = 0
    contexts_with_matches = 0
    
    # Build co-occurrence matrix
    with tqdm(total=num_contexts, desc="Building matrix") as pbar:
        for context_idx, (context_id, context_text) in enumerate(contexts):
            # Normalize context text using lib.py function
            # This ensures consistency with phrase extraction
            normalized_text = normalize_phrase(context_text, remove_verbs=remove_verbs)
            
            # Count phrase occurrences in this context
            phrase_counts = count_phrase_in_context(
                normalized_text,
                phrase_list,
                phrase_to_idx,
                use_word_boundaries=use_word_boundaries
            )
            
            # Update matrix
            if phrase_counts:
                contexts_with_matches += 1
                for phrase_idx, count in phrase_counts.items():
                    matrix[context_idx, phrase_idx] = count
                    total_matches += count
            
            pbar.update(1)
            
            # Log progress periodically
            if (context_idx + 1) % 1000 == 0:
                nnz = matrix.nnz
                density = nnz / (num_contexts * num_phrases) * 100
                logger.info(f"Progress: {context_idx + 1}/{num_contexts} contexts, "
                          f"{nnz:,} non-zero entries ({density:.4f}% density), "
                          f"{total_matches:,} total matches")
    
    logger.info(f"Contexts with matches: {contexts_with_matches}/{num_contexts} "
               f"({contexts_with_matches/num_contexts*100:.2f}%)")
    logger.info(f"Total phrase matches: {total_matches:,}")
    
    # Apply TF-IDF normalization
    if normalize_tfidf:
        logger.info("Applying TF-IDF normalization...")
        matrix = apply_tf_idf_normalization(matrix, num_contexts)
    
    return matrix

def save_matrix(
    matrix: scipy.sparse.lil_matrix,
    contexts: List[Tuple[str, str]],
    phrases: List[Tuple[str, int]],
    output_path: Path
) -> None:
    """
    Save sparse matrix in compressed NPZ format with metadata.

    The matrix is transposed from (num_contexts × num_phrases) to
    (num_phrases × num_contexts) before saving so that downstream steps
    (semantic_space.py, phrase_fingerprints.py) receive it in the expected
    orientation: rows = phrases, columns = contexts.

    Args:
        matrix: Sparse co-occurrence matrix (num_contexts × num_phrases)
        contexts: List of (context_id, context_text) tuples
        phrases: List of (phrase, frequency) tuples
        output_path: Output file path (.npz)
    """
    logger.info(f"Saving matrix to: {output_path}")

    # Convert to CSR then transpose to (num_phrases × num_contexts)
    # Downstream contract: rows = phrases, columns = contexts
    csr_matrix_ctx_phrase = matrix.tocsr()                  # (num_contexts, num_phrases)
    csr_matrix_phrase_ctx = csr_matrix_ctx_phrase.T.tocsr() # (num_phrases,  num_contexts)

    logger.info(
        f"Matrix transposed for saving: "
        f"{csr_matrix_ctx_phrase.shape} → {csr_matrix_phrase_ctx.shape} "
        f"(phrases × contexts)"
    )

    # Save transposed matrix data
    np.savez_compressed(
        output_path,
        data   = csr_matrix_phrase_ctx.data,
        indices= csr_matrix_phrase_ctx.indices,
        indptr = csr_matrix_phrase_ctx.indptr,
        shape  = csr_matrix_phrase_ctx.shape   # (num_phrases, num_contexts)
    )

    # Save metadata — num_contexts / num_phrases reflect the LOGICAL counts,
    # not the saved matrix orientation, so downstream can validate correctly.
    import json
    metadata = {
        'num_contexts'       : len(contexts),
        'num_phrases'        : len(phrases),
        'nnz'                : int(csr_matrix_phrase_ctx.nnz),
        'density'            : float(
            csr_matrix_phrase_ctx.nnz / (len(phrases) * len(contexts))
        ),
        'matrix_shape'       : list(csr_matrix_phrase_ctx.shape),  # [num_phrases, num_contexts]
        'matrix_orientation' : 'phrases x contexts',               # explicit contract tag
        'context_ids'        : [ctx[0] for ctx in contexts],
        'phrases'            : [p[0]   for p   in phrases],
        'phrase_frequencies' : [p[1]   for p   in phrases],
    }

    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.success(
        f"Saved (phrases × contexts): {csr_matrix_phrase_ctx.shape}, "
        f"{csr_matrix_phrase_ctx.nnz:,} non-zero entries"
    )
    logger.info(f"Metadata saved to: {metadata_path}")


def log_statistics(
    matrix: scipy.sparse.lil_matrix,
    contexts: List[Tuple[str, str]],
    phrases: List[Tuple[str, int]]
) -> None:
    """Log comprehensive matrix statistics.
    
    Expects matrix in (num_contexts × num_phrases) orientation,
    i.e. BEFORE the transpose applied during save_matrix().
    """
    num_contexts, num_phrases = matrix.shape

    # Guard: confirm orientation matches expectations
    assert num_contexts == len(contexts), (
        f"Shape mismatch: matrix has {num_contexts} rows "
        f"but {len(contexts)} contexts were passed."
    )
    assert num_phrases == len(phrases), (
        f"Shape mismatch: matrix has {num_phrases} cols "
        f"but {len(phrases)} phrases were passed."
    )
    """Log comprehensive matrix statistics"""
    num_contexts, num_phrases = matrix.shape
    nnz = matrix.nnz
    density = nnz / (num_contexts * num_phrases) * 100
    
    logger.info("Matrix Statistics:")
    logger.info(f"  Shape: {num_contexts} contexts × {num_phrases} phrases")
    logger.info(f"  Non-zero entries: {nnz:,}")
    logger.info(f"  Density: {density:.6f}%")
    logger.info(f"  Sparsity: {100 - density:.6f}%")
    
    # Memory estimate (CSR format)
    memory_mb = (nnz * 8 + (num_contexts + 1) * 4 + nnz * 4) / (1024 * 1024)
    logger.info(f"  Estimated memory (CSR): ~{memory_mb:.2f} MB")
    
    # Convert to CSR for efficient row/column operations
    matrix_csr = matrix.tocsr()
    
    # Context statistics
    context_counts = np.array(matrix_csr.sum(axis=1)).flatten()
    non_empty_contexts = np.count_nonzero(context_counts)
    logger.info(f"  Non-empty contexts: {non_empty_contexts}/{num_contexts} "
               f"({non_empty_contexts/num_contexts*100:.2f}%)")
    logger.info(f"  Avg phrases per context: {context_counts.mean():.2f}")
    
    # Phrase statistics
    phrase_counts = np.array(matrix_csr.sum(axis=0)).flatten()
    non_empty_phrases = np.count_nonzero(phrase_counts)
    logger.info(f"  Non-empty phrases: {non_empty_phrases}/{num_phrases} "
               f"({non_empty_phrases/num_phrases*100:.2f}%)")
    logger.info(f"  Avg contexts per phrase: {phrase_counts.mean():.2f}")
    
    # Top phrases
    if non_empty_phrases > 0:
        top_indices = phrase_counts.argsort()[-5:][::-1]
        logger.info("  Top 5 phrases by occurrence:")
        for idx in top_indices:
            if phrase_counts[idx] > 0:
                logger.info(f"    '{phrases[idx][0]}': {phrase_counts[idx]:.1f} occurrences")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Build term-context co-occurrence matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python term_context.py --phrases phrases.txt --corpus corpus.txt --output matrix.npz
  python term_context.py --phrases phrases.txt --corpus corpus.txt --output matrix.npz --no-tfidf
  python term_context.py --phrases phrases.txt --corpus corpus.txt --output matrix.npz --min-freq 3
        """
    )
    parser.add_argument("--phrases", required=True, 
                       help="Path to phrases file (phrase:frequency format)")
    parser.add_argument("--corpus", required=True, 
                       help="Path to corpus file (context_id|||context_text format)")
    parser.add_argument("--output", required=True, 
                       help="Output matrix file (.npz)")
    parser.add_argument("--min-freq", type=int, default=0,
                       help="Minimum phrase frequency threshold (default: 0)")
    parser.add_argument("--no-tfidf", action="store_true",
                       help="Skip TF-IDF normalization")
    parser.add_argument("--no-word-boundaries", action="store_true",
                       help="Disable word boundary detection in phrase matching")
    parser.add_argument("--keep-verbs", action="store_true",
                       help="Keep verbal elements during normalization")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Term-Context Matrix Construction")
    logger.info("=" * 60)
    logger.info(f"Phrases: {args.phrases}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Min frequency: {args.min_freq}")
    logger.info(f"TF-IDF: {not args.no_tfidf}")
    logger.info(f"Word boundaries: {not args.no_word_boundaries}")
    logger.info(f"Remove verbs: {not args.keep_verbs}")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading phrases...")
    phrases = load_phrases(Path(args.phrases), min_freq=args.min_freq)
    
    logger.info("Loading contexts...")
    contexts = load_contexts(Path(args.corpus))
    
    # Build matrix
    matrix = build_term_context_matrix(
        phrases,
        contexts,
        normalize_tfidf=not args.no_tfidf,
        use_word_boundaries=not args.no_word_boundaries,
        remove_verbs=not args.keep_verbs
    )
    
    # Log statistics
    log_statistics(matrix, contexts, phrases)
    
    # Save matrix
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_matrix(matrix, contexts, phrases, output_path)
    
    logger.success("=" * 60)
    logger.success("Matrix construction completed successfully")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
