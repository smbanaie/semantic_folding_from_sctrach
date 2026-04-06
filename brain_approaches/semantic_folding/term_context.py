#!/usr/bin/env python3
r"""
Term-Context Matrix Builder (Architectural Bypass Edition)
==========================================================

Pipeline step: **term-context-matrix**

Constructs a sparse term-context co-occurrence matrix from the pre-validated 
vocabulary and context mapping generated in Step 1. 

By leveraging the `phrase_to_contexts.json` bipartite graph, this module 
bypasses the $\mathcal{O}(C \times V)$ text-matching bottleneck entirely, 
operating in pure $\mathcal{O}(N)$ time (where $N$ is the number of mapped 
phrase-context pairs).

Output directory layout
-----------------------
    <output_dir>/
    ├── term_context_matrix.npz      ← scipy sparse matrix (Phrases × Contexts)
    ├── term_context_matrix.json     ← metadata / vocab / context IDs / integer maps
    └── idf_weights.json             ← per-phrase IDF floats (if TF-IDF enabled)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Scipy Import
# ---------------------------------------------------------------------------
try:
    import scipy.sparse
    SCIPY_AVAILABLE = True
except ImportError:
    logger.error("scipy is required for sparse matrix operations. Install with: pip install scipy numpy")
    SCIPY_AVAILABLE = False
    exit(1)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_corpus_ids(corpus_path: Path) -> List[str]:
    """
    Scans the corpus to extract the ordered list of Context IDs.

    This bypasses loading the raw text, extracting only the IDs to establish 
    the topological dimensions (columns) of the matrix.

    Parameters
    ----------
    corpus_path : Path
        Path to the raw corpus file (CSV format expected: `id,text`).

    Returns
    -------
    List[str]
        Ordered list of context ID strings.
    """
    context_ids = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or ',' not in line:
                continue
            ctx_id, _ = line.split(',', 1)
            context_ids.append(ctx_id.strip())
    return context_ids


def load_vocabulary(vocab_path: Path) -> List[Tuple[str, int]]:
    """
    Loads the ordered vocabulary extracted in Step 1.

    Parameters
    ----------
    vocab_path : Path
        Path to `vocabulary.csv`. Expected format is `phrase,frequency`.

    Returns
    -------
    List[Tuple[str, int]]
        List of tuples containing (phrase_string, total_frequency).
    """
    phrases = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                phrase = row[0].strip()
                freq = int(row[1])
                phrases.append((phrase, freq))
    return phrases


# ---------------------------------------------------------------------------
# TF-IDF normalization
# ---------------------------------------------------------------------------

def apply_tf_idf_normalization(
    matrix: scipy.sparse.csr_matrix,
    num_contexts: int,
) -> Tuple[scipy.sparse.csr_matrix, np.ndarray]:
    r"""
    Apply TF-IDF weighting to a (Phrases × Contexts) sparse matrix.

    The weighting formula used is:
    $$TF-IDF(t, d) = TF(t, d) \times \log\left(\frac{N}{DF(t) + 1}\right)$$
    Where $N$ is the total number of contexts, and $DF(t)$ is the number of 
    contexts containing term $t$.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The binary occurrence matrix of shape (num_phrases, num_contexts).
    num_contexts : int
        Total number of documents/contexts in the corpus ($N$).

    Returns
    -------
    Tuple[scipy.sparse.csr_matrix, np.ndarray]
        - The TF-IDF normalized sparse matrix.
        - The 1D numpy array of computed IDF weights.
    """
    # In CSR, np.diff(indptr) gives the number of non-zero elements per row (DF)
    df: np.ndarray = np.diff(matrix.indptr)

    # Smoothed IDF: \log(N / (DF + 1))
    idf: np.ndarray = np.log(num_contexts / (df + 1))

    # Scale each row by its IDF value via left-multiplication with a diagonal matrix
    idf_diag = scipy.sparse.diags(idf, format="csr")
    normalized_matrix = idf_diag @ matrix

    logger.info(
        f"TF-IDF applied: {matrix.nnz:,} → {normalized_matrix.nnz:,} non-zero entries"
    )
    logger.info(f"IDF range: [{idf.min():.4f}, {idf.max():.4f}]")

    return normalized_matrix, idf


# ---------------------------------------------------------------------------
# Core Matrix Builder (The Bypass)
# ---------------------------------------------------------------------------

def build_term_context_matrix(
    phrases: List[Tuple[str, int]],
    context_ids: List[str],
    phrase_mapping: Dict[str, List[str]],
    normalize_tfidf: bool = True,
) -> Tuple[scipy.sparse.csr_matrix, Optional[np.ndarray]]:
    r"""
    Constructs the term-context matrix natively in (Phrases × Contexts) format.
    
    This function utilizes $\mathcal{O}(1)$ dictionary lookups to map the predefined 
    bipartite graph directly into the sparse matrix structure, bypassing all 
    expensive string matching and NLP overhead.

    Parameters
    ----------
    phrases : List[Tuple[str, int]]
        The loaded vocabulary.
    context_ids : List[str]
        The loaded list of corpus context IDs.
    phrase_mapping : Dict[str, List[str]]
        The `phrase_to_contexts.json` mapping from Step 1.
    normalize_tfidf : bool, default=True
        Whether to apply TF-IDF weighting to the raw binary matrix.

    Returns
    -------
    Tuple[scipy.sparse.csr_matrix, Optional[np.ndarray]]
        The populated sparse CSR matrix, and optionally the IDF array.
    """
    num_phrases = len(phrases)
    num_contexts = len(context_ids)
    
    logger.info(f"Allocating matrix dimensions: {num_phrases:,} phrases × {num_contexts:,} contexts")

    # Fast O(1) lookups
    ctx_id_to_idx = {cid: idx for idx, cid in enumerate(context_ids)}
    phrase_to_idx = {p[0]: idx for idx, p in enumerate(phrases)}

    # LIL format is highly efficient for targeted row/col insertions
    matrix = scipy.sparse.lil_matrix((num_phrases, num_contexts), dtype=np.float32)

    total_occurrences = 0

    for phrase, mapped_contexts in phrase_mapping.items():
        if phrase not in phrase_to_idx:
            continue  
            
        row_idx = phrase_to_idx[phrase]
        
        for ctx_id in mapped_contexts:
            if ctx_id in ctx_id_to_idx:
                col_idx = ctx_id_to_idx[ctx_id]
                matrix[row_idx, col_idx] = 1.0 
                total_occurrences += 1

    logger.info(f"Populated matrix with {total_occurrences:,} explicit semantic mappings")

    # Convert to CSR for mathematical operations and saving
    matrix_csr = matrix.tocsr()

    idf_array: Optional[np.ndarray] = None
    if normalize_tfidf:
        logger.info("Applying TF-IDF normalization...")
        matrix_csr, idf_array = apply_tf_idf_normalization(matrix_csr, num_contexts)

    return matrix_csr, idf_array


# ---------------------------------------------------------------------------
# Output Writer
# ---------------------------------------------------------------------------

def save_outputs(
    matrix: scipy.sparse.csr_matrix,
    context_ids: List[str],
    phrases: List[Tuple[str, int]],
    phrase_mapping: Dict[str, List[str]],
    idf_array: Optional[np.ndarray],
    output_dir: Path,
) -> None:
    """
    Persists the (Phrases × Contexts) matrix and metadata downstream.

    Critically, this function maps the raw string context IDs into integer 
    column indices and saves them in `term_context_matrix.json` under the 
    `phrase_contexts` key, which is strictly required by the downstream 
    Step 4 (`phrase_fingerprints.py`).

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The populated matrix.
    context_ids : List[str]
        Ordered corpus context IDs.
    phrases : List[Tuple[str, int]]
        Ordered vocabulary.
    phrase_mapping : Dict[str, List[str]]
        The raw Step 1 mapping of `Phrase -> List[Context_ID]`.
    idf_array : Optional[np.ndarray]
        IDF weights to save, if applicable.
    output_dir : Path
        Directory to save all artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = output_dir / "term_context_matrix.npz"
    meta_path = output_dir / "term_context_matrix.json"
    idf_path  = output_dir / "idf_weights.json"

    # 1. Sparse Matrix (Standard Scipy Save)
    scipy.sparse.save_npz(npz_path, matrix)
    logger.success(f"Matrix written      → {npz_path}  (shape={matrix.shape}, nnz={matrix.nnz:,})")

    # 2. Metadata JSON
    # Remap context IDs to numerical matrix indices for downstream algorithms (Step 4)
    ctx_id_to_idx = {cid: idx for idx, cid in enumerate(context_ids)}
    numeric_phrase_contexts: Dict[str, List[int]] = {}
    
    for phrase_tuple in phrases:
        phrase = phrase_tuple[0]
        if phrase in phrase_mapping:
            # Convert string context IDs to integer matrix column indices
            numeric_phrase_contexts[phrase] = [
                ctx_id_to_idx[cid] for cid in phrase_mapping[phrase] if cid in ctx_id_to_idx
            ]

    metadata = {
        "num_phrases"        : len(phrases),
        "num_contexts"       : len(context_ids),
        "nnz"                : int(matrix.nnz),
        "density"            : float(matrix.nnz / max(len(phrases) * len(context_ids), 1)),
        "matrix_shape"       : list(matrix.shape),   # [num_phrases, num_contexts]
        "matrix_orientation" : "phrases x contexts",
        "context_ids"        : context_ids,
        "phrases"            : [p[0] for p in phrases],
        "phrase_frequencies" : [p[1] for p in phrases],
        "phrase_contexts"    : numeric_phrase_contexts, # Critically needed by Step 4
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.success(f"Metadata written    → {meta_path}")

    # 3. IDF Weights
    if idf_array is not None and len(idf_array) > 0:
        idf_dict = {
            phrase[0]: float(idf_val) for phrase, idf_val in zip(phrases, idf_array)
        }
        with open(idf_path, "w", encoding="utf-8") as fh:
            json.dump(idf_dict, fh, indent=2, ensure_ascii=False)
        logger.success(f"IDF weights written → {idf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build term-context matrix using the \mathcal{O}(1) Architectural Bypass.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--vocab", required=True, type=Path,
        help="Path to vocabulary.csv generated by Step 1",
    )
    parser.add_argument(
        "--mapping", required=True, type=Path,
        help="Path to phrase_to_contexts.json generated by Step 1",
    )
    parser.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to corpus file (to establish Context ID order/columns)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output DIRECTORY — all artefacts are written here",
    )
    parser.add_argument(
        "--no-tfidf", action="store_true",
        help="Disable TF-IDF normalization (raw binary occurrences are saved)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Term-Context Matrix Builder (Architectural Bypass)")
    logger.info("=" * 60)

    # 1. Load Artefacts
    logger.info(f"Loading vocabulary from {args.vocab}...")
    phrases = load_vocabulary(args.vocab)
    
    logger.info(f"Loading corpus IDs from {args.corpus}...")
    context_ids = load_corpus_ids(args.corpus)
    
    logger.info(f"Loading context mapping from {args.mapping}...")
    with open(args.mapping, 'r', encoding='utf-8') as f:
        phrase_mapping = json.load(f)

    # 2. Build Matrix
    matrix, idf_array = build_term_context_matrix(
        phrases=phrases,
        context_ids=context_ids,
        phrase_mapping=phrase_mapping,
        normalize_tfidf=not args.no_tfidf
    )

    # 3. Save Outputs
    save_outputs(
        matrix=matrix,
        context_ids=context_ids,
        phrases=phrases,
        phrase_mapping=phrase_mapping,
        idf_array=idf_array,
        output_dir=args.output_dir
    )

    logger.success("=" * 60)
    logger.success("Matrix construction completed successfully.")
    logger.success("=" * 60)

if __name__ == "__main__":
    main()
