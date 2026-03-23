#!/usr/bin/env python3
"""
Term-Context Matrix Builder for Semantic Folding Pipeline

Constructs sparse term-context co-occurrence matrix from corpus and phrase inventory,
applying TF-IDF normalization to reduce high-frequency term dominance.

Output directory layout (mirrors Step 4 / Step 5 convention):

    <output_dir>/
    ├── term_context_matrix.npz      ← sparse matrix  (phrases × contexts)
    ├── term_context_matrix.json     ← metadata / vocab / context IDs
    └── idf_weights.json             ← per-phrase IDF floats  (only if TF-IDF enabled)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from nltk import pos_tag, word_tokenize
from tqdm import tqdm

# Import from centralized library
from lib import (
    find_phrase_occurrences,
    is_valid_phrase_structure,
    load_contexts,
    load_phrases,
    normalize_phrase,
)

# ---------------------------------------------------------------------------
# Optional scipy import
# ---------------------------------------------------------------------------
try:
    import scipy.sparse
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available. Install with: pip install scipy numpy")
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# TF-IDF normalization
# ---------------------------------------------------------------------------

def apply_tf_idf_normalization(
    matrix: "scipy.sparse.lil_matrix",
    num_contexts: int,
) -> Tuple["scipy.sparse.lil_matrix", np.ndarray]:
    """
    Apply TF-IDF weighting to a term-context matrix.

    TF-IDF reduces the dominance of high-frequency terms by weighting
    each term by its inverse document frequency:

        TF-IDF(t, d) = TF(t, d) × log(N / (DF(t) + 1))

    where:
        TF(t, d)  = raw count of term t in context d
        DF(t)     = number of contexts that contain term t
        N         = total number of contexts

    The +1 smoothing in the denominator prevents log(0) when a phrase
    appears in every context.

    Args:
        matrix:       Sparse LIL matrix  (num_contexts × num_phrases).
        num_contexts: Total number of contexts (= N in the formula above).

    Returns:
        Tuple of:
          - normalized_matrix : TF-IDF weighted LIL matrix, same shape.
          - idf_array         : 1-D numpy array of IDF values, shape (num_phrases,).
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy unavailable — skipping TF-IDF normalization")
        return matrix, np.array([])

    # Work in CSC format: one column = one phrase → efficient column-wise ops
    matrix_csc = matrix.tocsc()

    # DF(t) = number of non-zero entries in column t
    # np.diff(indptr) gives the count of stored values per column in CSC
    df: np.ndarray = np.diff(matrix_csc.indptr)

    # Smoothed IDF: log(N / (DF + 1))
    # Using (DF + 1) instead of DF avoids log(0) for ubiquitous phrases
    idf: np.ndarray = np.log(num_contexts / (df + 1))

    # Scale each column by its IDF value via a diagonal matrix multiply
    # Result: each non-zero entry becomes  TF(t,d) * IDF(t)
    idf_diag = scipy.sparse.diags(idf, format="csc")
    normalized_matrix = (matrix_csc @ idf_diag).tolil()

    logger.info(
        f"TF-IDF applied: {matrix.nnz:,} → {normalized_matrix.nnz:,} non-zero entries"
    )
    logger.info(f"IDF range: [{idf.min():.4f}, {idf.max():.4f}]")

    return normalized_matrix, idf


# ---------------------------------------------------------------------------
# Phrase normalization & validation
# ---------------------------------------------------------------------------

def normalize_and_validate_phrases(
    phrases: List[Tuple[str, int]],
    remove_verbs: bool = True,
) -> List[Tuple[str, int]]:
    """
    Normalize and validate phrases before matrix construction.

    Applies the same normalization used during corpus extraction so that
    phrase strings match what will be found during context scanning.

    Steps per phrase:
      1. Normalize via ``lib.normalize_phrase`` (lowercasing, verb removal, …).
      2. Reject if the result is empty or whitespace-only.
      3. POS-tag the normalized form and reject if it fails structural validation.

    Args:
        phrases:      List of (phrase_text, frequency) tuples from the phrase file.
        remove_verbs: If True, verbal tokens are stripped during normalization.

    Returns:
        Filtered list of (normalized_phrase_text, frequency) tuples.
        Phrases that fail either check are dropped and logged as warnings.
    """
    normalized_phrases: List[Tuple[str, int]] = []
    skipped = 0

    for phrase, freq in phrases:
        # Step 1 — normalize
        normalized = normalize_phrase(phrase, remove_verbs=remove_verbs)

        # Step 2 — reject empty results
        if not normalized or not normalized.strip():
            skipped += 1
            logger.warning(
                f"Skipped '{phrase}' (normalized → '{normalized}') "
                "— empty after normalization"
            )
            continue

        # Step 3 — POS-based structural validation
        tagged = pos_tag(word_tokenize(normalized))
        if not is_valid_phrase_structure(tagged):
            skipped += 1
            logger.warning(
                f"Skipped '{phrase}' (normalized → '{normalized}') "
                "— invalid POS structure"
            )
            continue

        normalized_phrases.append((normalized, freq))

    if skipped > 0:
        logger.warning(f"Skipped {skipped} phrases during normalization/validation")

    logger.info(
        f"Phrase normalization: {len(phrases):,} input → "
        f"{len(normalized_phrases):,} valid"
    )

    return normalized_phrases


# ---------------------------------------------------------------------------
# Phrase index
# ---------------------------------------------------------------------------

def build_phrase_index(phrases: List[Tuple[str, int]]) -> Dict[str, int]:
    """
    Build a phrase-text → column-index lookup dictionary.

    Args:
        phrases: List of (phrase_text, frequency) tuples.

    Returns:
        Dict mapping each phrase string to its integer column index in the
        co-occurrence matrix.
    """
    return {phrase: idx for idx, (phrase, _) in enumerate(phrases)}


# ---------------------------------------------------------------------------
# Per-context phrase counting
# ---------------------------------------------------------------------------

def count_phrase_in_context(
    context_text: str,
    phrases: List[str],
    phrase_to_idx: Dict[str, int],
    use_word_boundaries: bool = True,
) -> Dict[int, int]:
    """
    Count occurrences of every phrase inside a single context string.

    Only phrases with at least one occurrence are included in the result
    to keep the returned dict sparse (avoids storing thousands of zeros).

    Args:
        context_text:        Pre-normalized context string to search within.
        phrases:             Ordered list of phrase strings to search for.
        phrase_to_idx:       Mapping from phrase string to matrix column index.
        use_word_boundaries: If True, matches are restricted to word boundaries.

    Returns:
        Dict of {column_index: occurrence_count} for phrases found at least once.
    """
    counts: Dict[int, int] = {}

    for phrase in phrases:
        count = find_phrase_occurrences(
            context_text,
            phrase,
            use_word_boundaries=use_word_boundaries,
        )
        if count > 0:
            counts[phrase_to_idx[phrase]] = count

    return counts


# ---------------------------------------------------------------------------
# Core matrix builder
# ---------------------------------------------------------------------------

def build_term_context_matrix(
    phrases: List[Tuple[str, int]],
    contexts: List[Tuple[str, str]],
    normalize_tfidf: bool = True,
    use_word_boundaries: bool = True,
    remove_verbs: bool = True,
) -> Tuple["scipy.sparse.lil_matrix", Optional[np.ndarray], List[Tuple[str, int]]]:
    """
    Construct a sparse term-context co-occurrence matrix.

    The matrix is built in **(num_contexts × num_phrases)** orientation
    and is transposed to **(num_phrases × num_contexts)** only at save time
    (see ``save_outputs``).

    Processing steps:
      1. Normalize and validate the phrase vocabulary.
      2. For each context, normalize the text and count phrase occurrences.
      3. Optionally apply column-wise TF-IDF weighting.

    Args:
        phrases:             Raw (phrase_text, frequency) list from the phrase file.
        contexts:            List of (context_id, context_text) tuples.
        normalize_tfidf:     Apply TF-IDF normalization after counting.
        use_word_boundaries: Use word-boundary matching during phrase search.
        remove_verbs:        Strip verbal tokens during normalization.

    Returns:
        Tuple of three values:
          - matrix       : Sparse LIL matrix, shape (num_contexts, num_phrases).
          - idf_array    : 1-D numpy array of IDF values (num_phrases,),
                           or ``None`` when ``normalize_tfidf=False``.
          - final_phrases: Normalized & validated phrase list used as the
                           column vocabulary (may be shorter than input).
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for sparse matrix operations. "
            "Install with: pip install scipy numpy"
        )

    # ── Step 1: phrase vocabulary ────────────────────────────────────────────
    logger.info("Normalizing and validating phrases...")
    final_phrases = normalize_and_validate_phrases(phrases, remove_verbs=remove_verbs)

    if not final_phrases:
        raise ValueError("No valid phrases remained after normalization/validation.")

    num_contexts = len(contexts)
    num_phrases  = len(final_phrases)
    logger.info(f"Matrix dimensions: {num_contexts:,} contexts × {num_phrases:,} phrases")

    # ── Step 2: allocate sparse matrix ───────────────────────────────────────
    # LIL (List of Lists) format is the most efficient for incremental row fills
    matrix = scipy.sparse.lil_matrix((num_contexts, num_phrases), dtype=np.float32)

    phrase_list   = [p[0] for p in final_phrases]
    phrase_to_idx = build_phrase_index(final_phrases)

    total_matches         = 0
    contexts_with_matches = 0

    # ── Step 3: fill co-occurrence counts ────────────────────────────────────
    with tqdm(total=num_contexts, desc="Building matrix") as pbar:
        for context_idx, (context_id, context_text) in enumerate(contexts):

            # Normalize context text to match phrase normalization
            normalized_text = normalize_phrase(context_text, remove_verbs=remove_verbs)

            phrase_counts = count_phrase_in_context(
                normalized_text,
                phrase_list,
                phrase_to_idx,
                use_word_boundaries=use_word_boundaries,
            )

            if phrase_counts:
                contexts_with_matches += 1
                for phrase_idx, count in phrase_counts.items():
                    matrix[context_idx, phrase_idx] = count
                    total_matches += count

            pbar.update(1)

            # Periodic density log (every 1 000 contexts)
            if (context_idx + 1) % 1_000 == 0:
                nnz     = matrix.nnz
                density = nnz / (num_contexts * num_phrases) * 100
                logger.info(
                    f"Progress: {context_idx + 1:,}/{num_contexts:,} contexts | "
                    f"{nnz:,} non-zero ({density:.4f}% density) | "
                    f"{total_matches:,} total matches"
                )

    logger.info(
        f"Contexts with ≥1 match: {contexts_with_matches:,}/{num_contexts:,} "
        f"({contexts_with_matches / num_contexts * 100:.2f}%)"
    )
    logger.info(f"Total phrase matches: {total_matches:,}")

    # ── Step 4: optional TF-IDF normalization ────────────────────────────────
    idf_array: Optional[np.ndarray] = None
    if normalize_tfidf:
        logger.info("Applying TF-IDF normalization...")
        matrix, idf_array = apply_tf_idf_normalization(matrix, num_contexts)

    return matrix, idf_array, final_phrases


# ---------------------------------------------------------------------------
# Save all outputs into the output directory
# ---------------------------------------------------------------------------
def save_outputs(
    matrix:       "scipy.sparse.lil_matrix",
    contexts:     List[Tuple[str, str]],
    final_phrases: List[Tuple[str, int]],
    idf_array:    Optional[np.ndarray],
    output_dir:   Path,
) -> None:
    """
    Persist all artefacts produced by the term-context step into *output_dir*.

    Output layout::

        <output_dir>/
        ├── term_context_matrix.npz    — sparse matrix (phrases × contexts)
        ├── term_context_matrix.json   — metadata / vocab / context IDs
        └── idf_weights.json           — per-phrase IDF floats (TF-IDF only)

    The matrix is **transposed** from the internal (num_contexts × num_phrases)
    orientation to **(num_phrases × num_contexts)** before saving, matching the
    downstream contract expected by ``semantic_space.py`` and
    ``phrase_fingerprints.py``.

    Args:
        matrix:        Sparse LIL matrix in (num_contexts × num_phrases) order.
        contexts:      List of (context_id, context_text) tuples.
        final_phrases: Normalized phrase vocabulary used as column labels.
        idf_array:     1-D IDF weight array (num_phrases,), or None.
        output_dir:    Directory that will receive all output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = output_dir / "term_context_matrix.npz"
    meta_path = output_dir / "term_context_matrix.json"
    idf_path  = output_dir / "idf_weights.json"

    # ── 1. Sparse matrix ─────────────────────────────────────────────────────
    # Internal orientation  : (num_contexts, num_phrases)
    # Saved orientation     : (num_phrases,  num_contexts)  — downstream contract
    csr_ctx_phrase  = matrix.tocsr()                    # (num_contexts, num_phrases)
    csr_phrase_ctx  = csr_ctx_phrase.T.tocsr()          # (num_phrases,  num_contexts)

    logger.info(
        f"Transposing matrix for save: "
        f"{csr_ctx_phrase.shape} → {csr_phrase_ctx.shape}  (phrases × contexts)"
    )

    np.savez_compressed(
        npz_path,
        data    = csr_phrase_ctx.data,
        indices = csr_phrase_ctx.indices,
        indptr  = csr_phrase_ctx.indptr,
        shape   = csr_phrase_ctx.shape,   # (num_phrases, num_contexts)
    )
    logger.success(
        f"Matrix written      → {npz_path}  "
        f"shape={csr_phrase_ctx.shape}  nnz={csr_phrase_ctx.nnz:,}"
    )

    # ── 2. Extract phrase-context mappings ───────────────────────────────────
    # Build a dict mapping each phrase to the list of context indices where it appears
    phrase_context_map: Dict[str, List[int]] = {}
    for phrase_idx in range(csr_phrase_ctx.shape[0]):
        row = csr_phrase_ctx.getrow(phrase_idx)
        # Get context indices where this phrase has non-zero weight
        context_indices = row.indices.tolist()
        phrase_text = final_phrases[phrase_idx][0]
        phrase_context_map[phrase_text] = context_indices

    logger.info(
        f"Extracted phrase-context mappings for {len(phrase_context_map):,} phrases"
    )

    # ── 3. Metadata JSON ─────────────────────────────────────────────────────
    metadata = {
        "num_contexts"       : len(contexts),
        "num_phrases"        : len(final_phrases),
        "nnz"                : int(csr_phrase_ctx.nnz),
        "density"            : float(
            csr_phrase_ctx.nnz / max(len(final_phrases) * len(contexts), 1)
        ),
        "matrix_shape"       : list(csr_phrase_ctx.shape),   # [num_phrases, num_contexts]
        "matrix_orientation" : "phrases x contexts",          # downstream contract tag
        "context_ids"        : [ctx[0] for ctx in contexts],
        "phrases"            : [p[0]   for p   in final_phrases],
        "phrase_frequencies" : [p[1]   for p   in final_phrases],
        "phrase_contexts"    : phrase_context_map,            # phrase → [context_idx, ...]
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.success(f"Metadata written    → {meta_path}")

    # ── 4. IDF weights JSON ───────────────────────────────────────────────────
    if idf_array is not None and len(idf_array) > 0:
        # Build {phrase_string: idf_float} — json.dump cannot serialize np.float32
        idf_dict = {
            phrase_str: float(idf_val)
            for phrase_str, idf_val in zip(
                [p[0] for p in final_phrases],   # ← phrase strings, not (str, int) tuples
                idf_array,
            )
        }
        with open(idf_path, "w", encoding="utf-8") as fh:
            json.dump(idf_dict, fh, indent=2, ensure_ascii=False)
        logger.success(
            f"IDF weights written → {idf_path}  ({len(idf_dict):,} phrases)"
        )
    else:
        logger.info("TF-IDF disabled — idf_weights.json not written")

# ---------------------------------------------------------------------------
# Statistics logger
# ---------------------------------------------------------------------------

def log_statistics(
    matrix:       "scipy.sparse.lil_matrix",
    contexts:     List[Tuple[str, str]],
    final_phrases: List[Tuple[str, int]],
) -> None:
    """
    Log comprehensive statistics about the co-occurrence matrix.

    Expects the matrix in **(num_contexts × num_phrases)** orientation —
    i.e. the internal representation *before* the transpose applied during
    ``save_outputs()``.

    Args:
        matrix:        Sparse LIL matrix (num_contexts × num_phrases).
        contexts:      Context list — used only for count validation.
        final_phrases: Phrase vocabulary — used for count validation and labels.
    """
    num_contexts, num_phrases = matrix.shape

    # Guard: ensure the caller is not accidentally passing a transposed matrix
    assert num_contexts == len(contexts), (
        f"Shape mismatch: matrix has {num_contexts} rows "
        f"but {len(contexts)} contexts were supplied."
    )
    assert num_phrases == len(final_phrases), (
        f"Shape mismatch: matrix has {num_phrases} cols "
        f"but {len(final_phrases)} phrases were supplied."
    )

    nnz     = matrix.nnz
    density = nnz / (num_contexts * num_phrases) * 100

    logger.info("Matrix Statistics:")
    logger.info(f"  Shape:    {num_contexts:,} contexts × {num_phrases:,} phrases")
    logger.info(f"  Non-zero: {nnz:,}")
    logger.info(f"  Density:  {density:.6f}%")
    logger.info(f"  Sparsity: {100 - density:.6f}%")

    # Approximate memory footprint in CSR format:
    #   data array   : nnz × 8 bytes (float64 worst case)
    #   indices array: nnz × 4 bytes (int32)
    #   indptr array : (num_contexts + 1) × 4 bytes
    memory_mb = (nnz * 8 + nnz * 4 + (num_contexts + 1) * 4) / (1024 ** 2)
    logger.info(f"  Est. memory (CSR): ~{memory_mb:.2f} MB")

    matrix_csr = matrix.tocsr()

    # ── Context-level stats ───────────────────────────────────────────────────
    context_counts     = np.array(matrix_csr.sum(axis=1)).flatten()
    non_empty_contexts = int(np.count_nonzero(context_counts))
    logger.info(
        f"  Non-empty contexts:    {non_empty_contexts:,}/{num_contexts:,} "
        f"({non_empty_contexts / num_contexts * 100:.2f}%)"
    )
    logger.info(f"  Avg phrases/context:   {context_counts.mean():.2f}")

    # ── Phrase-level stats ────────────────────────────────────────────────────
    phrase_counts      = np.array(matrix_csr.sum(axis=0)).flatten()
    non_empty_phrases  = int(np.count_nonzero(phrase_counts))
    logger.info(
        f"  Non-empty phrases:     {non_empty_phrases:,}/{num_phrases:,} "
        f"({non_empty_phrases / num_phrases * 100:.2f}%)"
    )
    logger.info(f"  Avg contexts/phrase:   {phrase_counts.mean():.2f}")

    # ── Top-5 phrases by total weighted occurrence ────────────────────────────
    if non_empty_phrases > 0:
        top_indices = phrase_counts.argsort()[-5:][::-1]
        logger.info("  Top 5 phrases by occurrence:")
        for idx in top_indices:
            if phrase_counts[idx] > 0:
                logger.info(
                    f"    '{final_phrases[idx][0]}': {phrase_counts[idx]:.2f}"
                )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, run the pipeline, and save all outputs."""
    parser = argparse.ArgumentParser(
        description="Build term-context co-occurrence matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run — outputs go to ./output/tc_matrix/
  python term_context.py --phrases phrases.txt --corpus corpus.txt --output ./output/tc_matrix

  # Skip TF-IDF (no idf_weights.json will be written)
  python term_context.py --phrases phrases.txt --corpus corpus.txt --output ./output/tc_matrix --no-tfidf

  # Raise minimum phrase frequency to reduce vocabulary size
  python term_context.py --phrases phrases.txt --corpus corpus.txt --output ./output/tc_matrix --min-freq 3
        """,
    )

    # ── Required arguments ────────────────────────────────────────────────────
    parser.add_argument(
        "--phrases", required=True,
        help="Path to phrases file  (one 'phrase:frequency' entry per line)",
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Path to corpus file  (one 'context_id|||context_text' entry per line)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output DIRECTORY — all artefacts are written here",
    )

    # ── Optional arguments ────────────────────────────────────────────────────
    parser.add_argument(
        "--min-freq", type=int, default=0,
        help="Minimum phrase frequency to include (default: 0 = keep all)",
    )
    parser.add_argument(
        "--no-tfidf", action="store_true",
        help="Disable TF-IDF normalization (raw co-occurrence counts are saved)",
    )
    parser.add_argument(
        "--no-word-boundaries", action="store_true",
        help="Disable word-boundary enforcement during phrase matching",
    )
    parser.add_argument(
        "--keep-verbs", action="store_true",
        help="Retain verbal tokens during phrase normalization",
    )

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Term-Context Matrix Construction")
    logger.info("=" * 60)
    logger.info(f"Phrases file:    {args.phrases}")
    logger.info(f"Corpus file:     {args.corpus}")
    logger.info(f"Output dir:      {args.output}")
    logger.info(f"Min frequency:   {args.min_freq}")
    logger.info(f"TF-IDF:          {not args.no_tfidf}")
    logger.info(f"Word boundaries: {not args.no_word_boundaries}")
    logger.info(f"Remove verbs:    {not args.keep_verbs}")
    logger.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading phrases...")
    raw_phrases = load_phrases(Path(args.phrases), min_freq=args.min_freq)
    logger.info(f"Loaded {len(raw_phrases):,} phrases")

    logger.info("Loading contexts...")
    contexts = load_contexts(Path(args.corpus))
    logger.info(f"Loaded {len(contexts):,} contexts")

    # ── Build ─────────────────────────────────────────────────────────────────
    matrix, idf_array, final_phrases = build_term_context_matrix(
        raw_phrases,
        contexts,
        normalize_tfidf     = not args.no_tfidf,
        use_word_boundaries = not args.no_word_boundaries,
        remove_verbs        = not args.keep_verbs,
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    # Pass final_phrases (post-normalization vocabulary), NOT raw_phrases
    log_statistics(matrix, contexts, final_phrases)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    save_outputs(matrix, contexts, final_phrases, idf_array, output_dir)

    logger.success("=" * 60)
    logger.success("Matrix construction completed successfully")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
