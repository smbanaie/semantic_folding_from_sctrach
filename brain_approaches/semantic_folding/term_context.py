#!/usr/bin/env python3
"""
Modernized Term-Context Matrix Builder for Semantic Folding Pipeline

Builds sparse term-context co-occurrence matrix from corpus and phrases,
with memory-efficient processing and visualization capabilities.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import loguru
from loguru import logger
from tqdm import tqdm

# Try to import scipy for sparse matrices
try:
    import scipy.sparse
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available. Install with: pip install scipy")
    SCIPY_AVAILABLE = False


def load_phrases(phrases_path: Path) -> List[str]:
    """Load phrases from file"""
    logger.info(f"Loading phrases from: {phrases_path}")

    phrases = []
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                phrase = line.split(':', 1)[0].strip()
                if phrase:
                    phrases.append(phrase)

    logger.success(f"Loaded {len(phrases)} phrases")
    return phrases


def load_contexts(corpus_path: Path) -> List[Tuple[str, str]]:
    """Load contexts from corpus file"""
    logger.info(f"Loading contexts from: {corpus_path}")

    contexts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue

            context_id, context_text = line.split(',', 1)
            context_id = context_id.strip()
            context_text = context_text.strip()

            if context_id and context_text:
                contexts.append((context_id, context_text))

    logger.success(f"Loaded {len(contexts)} contexts")
    return contexts


def apply_tf_idf_normalization(matrix: scipy.sparse.lil_matrix,
                             phrases: List[str],
                             contexts: List[Tuple[str, str]]) -> scipy.sparse.lil_matrix:
    """Apply TF-IDF normalization to reduce high-frequency word dominance"""
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available, skipping TF-IDF normalization")
        return matrix

    num_contexts, num_phrases = matrix.shape

    # Convert to CSR for efficient operations
    matrix_csr = matrix.tocsr()

    # Calculate document frequency (DF) for each phrase
    df = np.array([np.count_nonzero(matrix_csr[:, i].toarray()) for i in range(num_phrases)])

    # Calculate inverse document frequency (IDF)
    # Add 1 to avoid division by zero, and smooth IDF
    idf = np.log((num_contexts + 1) / (df + 1)) + 1

    # Apply TF-IDF weighting
    # TF is already the raw counts in our matrix
    # Multiply each column by its IDF weight
    for i in range(num_phrases):
        if idf[i] > 0:
            matrix_csr[:, i] = matrix_csr[:, i].multiply(idf[i])

    # Convert back to LIL format for compatibility
    normalized_matrix = matrix_csr.tolil()

    # Log some statistics about the normalization
    original_nnz = matrix.nnz
    normalized_nnz = normalized_matrix.nnz
    logger.info(f"TF-IDF normalization applied: {original_nnz} -> {normalized_nnz} non-zero entries")

    return normalized_matrix


def build_term_context_matrix_sparse(phrases: List[str],
                                   contexts: List[Tuple[str, str]],
                                   chunk_size: int = 1000,
                                   normalize: bool = True) -> scipy.sparse.lil_matrix:
    """Build sparse term-context matrix efficiently"""
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy not available for sparse matrix operations")

    num_contexts = len(contexts)
    num_phrases = len(phrases)

    logger.info(f"Building sparse matrix: {num_contexts} contexts × {num_phrases} phrases")

    # Use LIL matrix for efficient construction
    matrix = scipy.sparse.lil_matrix((num_contexts, num_phrases), dtype=np.int32)

    # Create phrase to index mapping for faster lookup
    phrase_to_idx = {phrase: idx for idx, phrase in enumerate(phrases)}

    # Process contexts in chunks to show progress
    with tqdm(total=num_contexts, desc="Building matrix") as pbar:
        for context_idx, (context_id, context_text) in enumerate(contexts):
            # Count phrase occurrences in this context
            context_counts = defaultdict(int)
            for phrase in phrases:
                count = context_text.count(phrase)
                if count > 0:
                    context_counts[phrase] = count

            # Update matrix
            for phrase, count in context_counts.items():
                phrase_idx = phrase_to_idx[phrase]
                matrix[context_idx, phrase_idx] = count

            # Update progress
            pbar.update(1)

            # Log progress every 1000 contexts
            if (context_idx + 1) % 1000 == 0:
                nnz = matrix.nnz
                density = nnz / (num_contexts * num_phrases) * 100
                logger.info(f"Processed {context_idx + 1}/{num_contexts} contexts, "
                          f"{nnz} non-zero entries ({density:.4f}% density)")

    # Apply TF-IDF normalization to reduce high-frequency word dominance
    if normalize:
        logger.info("Applying TF-IDF normalization to reduce high-frequency word dominance...")
        matrix = apply_tf_idf_normalization(matrix, phrases, contexts)

    return matrix


def build_term_context_matrix_dense(phrases: List[str],
                                   contexts: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """Fallback dense matrix builder when scipy not available"""
    logger.warning("Using dense matrix (scipy not available)")

    term_context_matrix = defaultdict(lambda: defaultdict(int))

    num_contexts = len(contexts)
    with tqdm(total=num_contexts, desc="Building matrix") as pbar:
        for context_id, context_text in contexts:
            for phrase in phrases:
                count = context_text.count(phrase)
                if count > 0:
                    term_context_matrix[context_id][phrase] = count
            pbar.update(1)

    return term_context_matrix


def save_sparse_matrix(matrix: scipy.sparse.lil_matrix,
                      contexts: List[Tuple[str, str]],
                      phrases: List[str],
                      output_path: Path) -> None:
    """Save sparse matrix in NPZ format"""
    logger.info(f"Saving sparse matrix to: {output_path}")

    # Convert to CSR for efficient storage
    csr_matrix = matrix.tocsr()

    # Save matrix
    np.savez_compressed(output_path, data=csr_matrix.data, indices=csr_matrix.indices,
                       indptr=csr_matrix.indptr, shape=csr_matrix.shape)

    # Save metadata
    metadata = {
        'num_contexts': len(contexts),
        'num_phrases': len(phrases),
        'nnz': csr_matrix.nnz,
        'density': csr_matrix.nnz / (len(contexts) * len(phrases)),
        'context_ids': [ctx[0] for ctx in contexts],
        'phrases': phrases
    }

    metadata_path = output_path.with_suffix('.json')
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Saved sparse matrix: {csr_matrix.shape}, {csr_matrix.nnz} non-zero entries")


def save_dense_matrix(matrix: Dict[str, Dict[str, int]],
                     contexts: List[Tuple[str, str]],
                     phrases: List[str],
                     output_path: Path) -> None:
    """Save dense matrix as CSV (not recommended for large matrices)"""
    logger.warning("Saving dense matrix as CSV (may be very large)")

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Context ID'] + phrases)

        # Data rows
        context_ids = [ctx[0] for ctx in contexts]
        for context_id in context_ids:
            row = [matrix[context_id].get(phrase, 0) for phrase in phrases]
            writer.writerow([context_id] + row)

    logger.success(f"Saved dense matrix to: {output_path}")


def create_sparsity_visualization(matrix: scipy.sparse.spmatrix,
                                 output_dir: Path,
                                 sample_size: int = 100) -> None:
    """Create sparsity heatmap visualization"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Sample a subset for visualization
        if matrix.shape[0] > sample_size or matrix.shape[1] > sample_size:
            sample_matrix = matrix[:sample_size, :sample_size].toarray()
        else:
            sample_matrix = matrix.toarray()

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot sparsity pattern
        plt.spy(sample_matrix, markersize=1, aspect='auto')
        plt.xlabel('Phrases')
        plt.ylabel('Contexts')
        plt.title(f'Matrix Sparsity Pattern (Sample {sample_matrix.shape[0]}×{sample_matrix.shape[1]})')

        # Save plot
        viz_path = output_dir / "matrix_sparsity.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.success(f"Created sparsity visualization: {viz_path}")

    except ImportError:
        logger.warning("matplotlib not available for visualization")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")


def log_matrix_statistics(matrix, contexts: List[Tuple[str, str]], phrases: List[str]) -> None:
    """Log comprehensive matrix statistics"""
    logger.info("Term-Context Matrix Statistics:")

    if SCIPY_AVAILABLE and hasattr(matrix, 'shape') and hasattr(matrix, 'nnz'):
        # Sparse matrix
        num_contexts, num_phrases = matrix.shape
        nnz = matrix.nnz
        density = nnz / (num_contexts * num_phrases) * 100

        logger.info(f"  Matrix shape: {num_contexts} × {num_phrases}")
        logger.info(f"  Non-zero entries: {nnz:,}")
        logger.info(f"  Density: {density:.6f}%")
        logger.info(f"  Sparsity: {100 - density:.6f}%")

        # Memory usage estimate
        memory_mb = (nnz * 4) / (1024 * 1024)  # 4 bytes per int32
        logger.info(f"  Estimated memory: {memory_mb:.2f} MB")

        # Phrase frequency statistics
        phrase_frequencies = np.array(matrix.sum(axis=0)).flatten()

    else:
        # Dense matrix stats
        num_contexts = len(contexts)
        num_phrases = len(phrases)
        total_cells = num_contexts * num_phrases
        filled_cells = sum(len(context_data) for context_data in matrix.values())
        density = filled_cells / total_cells * 100

        logger.info(f"  Matrix shape: {num_contexts} × {num_phrases}")
        logger.info(f"  Filled cells: {filled_cells:,}")
        logger.info(f"  Density: {density:.6f}%")

        # Phrase frequency statistics
        phrase_frequencies = []
        for phrase in phrases:
            freq = sum(matrix.get(ctx[0], {}).get(phrase, 0) for ctx in contexts)
            phrase_frequencies.append(freq)

    if len(phrase_frequencies) > 0:
        if hasattr(phrase_frequencies, 'argmax'):  # numpy array
            max_freq_idx = phrase_frequencies.argmax()
            min_freq_idx = phrase_frequencies.argmin()
            max_freq = int(phrase_frequencies[max_freq_idx])
            min_freq = int(phrase_frequencies[min_freq_idx])
        else:  # regular list
            max_freq = max(phrase_frequencies)
            min_freq = min(phrase_frequencies)
            max_freq_idx = phrase_frequencies.index(max_freq)
            min_freq_idx = phrase_frequencies.index(min_freq)

        logger.info(f"  Most frequent phrase: {phrases[max_freq_idx]} "
                   f"({max_freq} occurrences)")
        logger.info(f"  Least frequent phrase: {phrases[min_freq_idx]} "
                   f"({min_freq} occurrences)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Build term-context matrix")
    parser.add_argument("--phrases_path", required=True, help="Path to phrases.txt file")
    parser.add_argument("--corpus_path", required=True, help="Path to corpus.txt file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Processing chunk size")
    parser.add_argument("--no_normalization", action="store_true", help="Skip TF-IDF normalization")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    logger.info("Starting term-context matrix construction...")
    logger.info(f"Phrases: {args.phrases_path}")
    logger.info(f"Corpus: {args.corpus_path}")
    logger.info(f"Output: {args.output_dir}")

    # Load data
    phrases = load_phrases(Path(args.phrases_path))
    contexts = load_contexts(Path(args.corpus_path))

    # Build matrix
    logger.info("Building term-context matrix...")
    normalize = not args.no_normalization
    if normalize:
        logger.info("TF-IDF normalization enabled (reduces high-frequency word dominance)")
    else:
        logger.info("TF-IDF normalization disabled")

    if SCIPY_AVAILABLE:
        matrix = build_term_context_matrix_sparse(phrases, contexts, args.chunk_size, normalize)
    else:
        matrix = build_term_context_matrix_dense(phrases, contexts)

    # Log statistics
    log_matrix_statistics(matrix, contexts, phrases)

    # Save matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if SCIPY_AVAILABLE and hasattr(matrix, 'nnz'):
            matrix_path = output_dir / "term_context_matrix.npz"
            save_sparse_matrix(matrix, contexts, phrases, matrix_path)
        else:
            matrix_path = output_dir / "term_context_matrix.csv"
            save_dense_matrix(matrix, contexts, phrases, matrix_path)
    except Exception as e:
        logger.error(f"Failed to save matrix: {e}")
        # Fallback: try to save as dense CSV
        try:
            matrix_path = output_dir / "term_context_matrix.csv"
            # Convert sparse to dense for CSV if needed
            if SCIPY_AVAILABLE and hasattr(matrix, 'toarray'):
                dense_matrix = {}
                for i, context_id in enumerate([ctx[0] for ctx in contexts]):
                    row_data = matrix[i, :].toarray().flatten()
                    dense_matrix[context_id] = [int(x) for x in row_data]
                save_dense_matrix(dense_matrix, contexts, phrases, matrix_path)
                logger.success(f"Matrix saved as dense CSV fallback: {matrix_path}")
            else:
                save_dense_matrix(matrix, contexts, phrases, matrix_path)
                logger.success(f"Matrix saved as dense CSV: {matrix_path}")
        except Exception as e2:
            logger.error(f"Fallback save also failed: {e2}")
            logger.success(f"Matrix built successfully but save failed. Shape: {len(contexts)} x {len(phrases)}")

    # Create visualization
    if not args.no_visualization and SCIPY_AVAILABLE and hasattr(matrix, 'nnz'):
        try:
            create_sparsity_visualization(matrix, output_dir)
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")

    logger.success("Term-context matrix construction completed")


if __name__ == "__main__":
    main()