#!/usr/bin/env python3
"""
doc_fingerprints.py — Step 5 of the Semantic Folding Pipeline

Aggregates phrase-level sparse fingerprints (Step 4) into document-level
Sparse Distributed Representations (SDRs) using TF-IDF weighted union,
then sparsifies via Morton (Z-order) curve thresholding.

Pipeline position
-----------------
Step 1  phrase_extractor.py   → vocabulary.csv
Step 2  term_context.py       → term_context_matrix.*, idf_weights.json
Step 3  semantic_space.py     → context_coordinates.json
Step 4  phrase_fingerprints.py→ phrase_fingerprints/
Step 5  doc_fingerprints.py   → doc_fingerprints/          ← THIS FILE
Step 6  query_processing.py   → query results

Consistency guarantee
---------------------
Every phrase extracted from a document in this step goes through the
**identical** normalization + expansion path used in Step 1
(phrase_extractor.py → process_corpus_with_expansion).  Specifically:

    raw text
        └─ extract_and_normalize_phrases()   # spaCy / NLTK + normalize_phrase()
                └─ expand_phrases()          # sub-phrase generation
                        └─ vocab filter      # keep only known phrase_fps keys

This ensures that a document containing "deep neural network" activates
fingerprints for "deep neural", "neural network", "neural", "network", etc.,
exactly mirroring how the vocabulary was built in Step 1.

Usage
-----
    python doc_fingerprints.py \
        --corpus      data/corpus.txt \
        --fingerprints outputs/run/phrase_fingerprints \
        --idf-weights outputs/run/term_context/idf_weights.json \
        --output-dir  outputs/run/doc_fingerprints \
        --grid-size   16 \
        --top-percent 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
from phrase_extractor import (
    extract_raw_phrases_spacy, 
    extract_raw_phrases_fallback, 
    SPACY_AVAILABLE
)

from lib import (
    compute_fingerprint_diversity,
    expand_phrases,
    is_valid_phrase_structure,
    load_contexts_dict,
    normalize_fingerprint,
    sparsify_fingerprint,
    normalize_phrase,  
)



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_outputs(
    fingerprints  : np.ndarray,
    doc_index_map : Dict[str, int],
    stats         : dict,
    output_dir    : Path,
) -> None:
    """
    Persist Step-5 outputs to ``output_dir``.

    Three files are written:

    ``doc_fingerprints.npz``
        Compressed NumPy archive containing one key ``"fingerprints"``.
        Shape: ``(n_docs, grid_size²)``, dtype ``float32``.
        Row *i* is the SDR for the document whose id maps to *i* in
        ``doc_fingerprints_meta.json``.

    ``doc_fingerprints_meta.json``
        JSON object mapping each ``doc_id`` (str) to its row index (int)
        inside the ``.npz`` matrix.  Needed by downstream steps to look
        up a specific document's fingerprint without loading the full matrix.

    ``doc_fingerprints_stats.json``
        JSON object with run-level statistics (document counts, sparsity,
        grid parameters, etc.) for provenance and debugging.

    Parameters
    ----------
    fingerprints : np.ndarray, shape (n_docs, grid_size²)
        Dense float32 matrix of (optionally normalised) document SDRs.
    doc_index_map : Dict[str, int]
        Mapping ``doc_id → row_index`` into *fingerprints*.
    stats : dict
        Scalar statistics produced by :func:`build_doc_fingerprints`.
    output_dir : Path
        Destination directory; created (including parents) if absent.

    Raises
    ------
    OSError
        Re-raised from ``numpy.savez_compressed`` or ``open()`` on I/O
        failure so the caller can handle it with a meaningful error message.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path   = output_dir / "doc_fingerprints.npz"
    meta_path  = output_dir / "doc_fingerprints_meta.json"
    stats_path = output_dir / "doc_fingerprints_stats.json"

    np.savez_compressed(str(npz_path), fingerprints=fingerprints)
    logger.info("Fingerprint matrix written → %s  shape=%s", npz_path, fingerprints.shape)

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(doc_index_map, fh, ensure_ascii=False, indent=2)
    logger.info("Doc-index map written     → %s  (%d entries)", meta_path, len(doc_index_map))

    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    logger.info("Run statistics written    → %s", stats_path)


# ---------------------------------------------------------------------------
# Per-document phrase extractor
# ---------------------------------------------------------------------------

def extract_phrases_from_doc(
    text            : str,
    phrase_vocab    : Set[str],
    use_spacy       : bool = True,
    remove_verbs    : bool = False,
    filter_generic  : bool = True,
    min_word_length : int  = 3,
) -> List[str]:
    """
    Extract vocabulary-matched phrases from a single document string.

    This function is the Step-5 counterpart of
    ``phrase_extractor.process_corpus_with_expansion`` and must follow the
    **same three-stage pipeline** to guarantee that phrase representations
    are identical at index time (Step 1) and at fingerprint-build time
    (Step 5).
    """
    # ── Stage 1: extraction + normalisation ──────────────────────────────────
        # ── Stage 1: extraction + normalisation ──────────────────────────────────
    if use_spacy:
        raw_phrases = extract_raw_phrases_spacy(text)
    else:
        raw_phrases = extract_raw_phrases_fallback(text)

    candidates: Set[str] = set()
    for phrase in raw_phrases:
        norm = normalize_phrase(phrase, remove_verbs=remove_verbs)
        if norm:
            candidates.add(norm)


    if not candidates:
        logger.debug("No phrases extracted from text snippet: %r...", text[:80])
        return []

    # ── Stage 2: sub-phrase expansion ────────────────────────────────────────
    expanded: List[str] = expand_phrases(
        list(candidates),
        context_text    = text,             
        filter_generic  = filter_generic,
        min_word_length = min_word_length,
    )

    # ── Stage 3: vocabulary filter ────────────────────────────────────────────
    matched: List[str] = [p for p in expanded if p in phrase_vocab]

    if not matched:
        logger.debug("No vocabulary matches in text snippet: %r...", text[:80])

    return matched


# ---------------------------------------------------------------------------
# Single-document fingerprint builder
# ---------------------------------------------------------------------------

def build_document_fingerprint(
    doc_text            : str,
    phrase_fingerprints : np.ndarray,
    phrase_to_row       : Dict[str, int],
    idf_weights         : Dict[str, float],
    grid_size           : int,
    use_spacy           : bool = True,
    remove_verbs        : bool = True,
    filter_generic      : bool = True,
    min_word_length     : int  = 3,
) -> Optional[csr_matrix]:
    r"""
    Build a raw (un-sparsified) TF-IDF weighted fingerprint for one document.

    The function accumulates phrase fingerprint vectors into a single
    document-level vector using a weighted sum:

    .. math::

        \mathbf{f}_{\text{doc}} = \sum_{p \in P(d)} \text{tf}(p, d)
        \cdot \text{idf}(p) \cdot \mathbf{f}_p
    """
    n = grid_size * grid_size
    
    # ── Optimized Accumulator: Use dense numpy array instead of lil_matrix ────
    acc = np.zeros(n, dtype=np.float32)

    # ── Stage 1–3 via extract_phrases_from_doc (with expansion) ──────────────
    matched_phrases = extract_phrases_from_doc(
        text            = doc_text,
        phrase_vocab    = set(phrase_to_row.keys()),
        use_spacy       = use_spacy,
        remove_verbs    = remove_verbs,
        filter_generic  = filter_generic,
        min_word_length = min_word_length,
    )

    if not matched_phrases:
        return None

    # ── TF count from matched list (duplicates = expansion-path boosts) ───────
    tf: Dict[str, int] = {}
    for phrase in matched_phrases:
        tf[phrase] = tf.get(phrase, 0) + 1

    # ── Weighted accumulation ─────────────────────────────────────────────────
    hits = 0
    for phrase, term_freq in tf.items():
        if phrase not in phrase_to_row:
            continue

        row_index = phrase_to_row[phrase]
        vec = phrase_fingerprints[row_index]
        weight = term_freq * idf_weights.get(phrase, 1.0)
        
        acc += weight * vec
        hits += 1

    if hits == 0:
        return None

    # Cast to sparse matrix of shape (1, n) only once after all accumulations
    return csr_matrix(acc.reshape(1, -1))


# ---------------------------------------------------------------------------
# SDR sparsifier
# ---------------------------------------------------------------------------

def sparsify_to_sdr(
    fingerprint : csr_matrix,
    top_percent : float,
    grid_size   : int,
) -> csr_matrix:
    """
    Sparsify a weighted fingerprint to a fixed-density SDR.
    """
    top_k = max(1, int(round(top_percent * grid_size * grid_size)))
    return sparsify_fingerprint(
        fingerprint,
        top_k      = top_k,
        use_zorder = True,
        grid_size  = grid_size,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_doc_fingerprints(
    corpus_path       : Path,
    fingerprints_path : Path,
    idf_weights_path  : Optional[Path],
    grid_size         : int   = 16,
    top_percent       : float = 0.1,
    normalize         : bool  = True,
    normalize_method  : str   = "l2",
    use_spacy         : bool  = True,
    remove_verbs      : bool  = True,
    filter_generic    : bool  = True,
    min_word_length   : int   = 3,
    compute_diversity : bool  = False,
    diversity_sample  : int   = 100,
) -> Tuple[np.ndarray, Dict[str, int], dict]:
    """
    Full Step-5 pipeline: build document SDRs from phrase fingerprints.
    """

    # ── 1. Load Phrase Fingerprints (Step 4) ──────────────────────────────────
    npz_path = fingerprints_path / "phrase_fingerprints.npz"
    meta_path = fingerprints_path / "phrase_fingerprints_meta.json"
    
    if not npz_path.exists() or not meta_path.exists():
        logger.error(f"Missing Step 4 outputs in directory: {fingerprints_path}")
        sys.exit(1)

    logger.info("Loading phrase fingerprints from %s ...", npz_path)
    data = np.load(str(npz_path))
    phrase_fingerprints = data["fingerprints"]  # dense array

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
        # Handle dict wrapping if present
        phrase_to_row = meta_data.get("phrase_to_row", meta_data)

    logger.info("  %d phrase fingerprints loaded. Grid size: %d", len(phrase_to_row), phrase_fingerprints.shape[1])

    # ── 2. Load IDF Weights (Step 2) ──────────────────────────────────────────
    idf_weights = {}
    if idf_weights_path and idf_weights_path.exists():
        logger.info("Loading global IDF weights from %s ...", idf_weights_path)
        with open(idf_weights_path, 'r', encoding='utf-8') as f:
            idf_weights = json.load(f)
    else:
        logger.warning("No IDF weights provided or found. Defaulting to TF-only accumulation.")

    # ── 3. Corpus ─────────────────────────────────────────────────────────────
    logger.info("Loading corpus from %s ...", corpus_path)
    contexts = load_contexts_dict(corpus_path)
    logger.info("  %d documents loaded", len(contexts))

    # ── 4. Per-document build loop ────────────────────────────────────────────
    top_k_bits = max(1, int(round(top_percent * grid_size * grid_size)))
    logger.info(
        "Building document fingerprints "
        "(grid=%d, top_percent=%.3f → top_k=%d bits) ...",
        grid_size, top_percent, top_k_bits,
    )

    if use_spacy and not SPACY_AVAILABLE:
        logger.warning(
            "spaCy requested but unavailable — falling back to NLTK. "
            "Ensure this matches the setting used in Step 1."
        )

    sparse_fps    : Dict[str, csr_matrix] = {}
    doc_index_map : Dict[str, int]        = {}
    active_bits   : List[int]             = []
    skipped = 0

    for doc_id, doc_text in contexts.items():

        # 4a. Raw TF-IDF weighted fingerprint
        raw_fp = build_document_fingerprint(
            doc_text            = doc_text,
            phrase_fingerprints = phrase_fingerprints,
            phrase_to_row       = phrase_to_row,
            idf_weights         = idf_weights,
            grid_size           = grid_size,
            use_spacy           = use_spacy,
            remove_verbs        = remove_verbs,
            filter_generic      = filter_generic,
            min_word_length     = min_word_length,
        )

        if raw_fp is None:
            skipped += 1
            logger.debug("  doc %s — no matching phrases, skipped", doc_id)
            continue

        # 4b. Sparsify to SDR
        sparse_fp = sparsify_to_sdr(
            raw_fp,
            top_percent = top_percent,
            grid_size   = grid_size,
        )

        # 4c. Optional normalisation
        if normalize:
            sparse_fp = normalize_fingerprint(sparse_fp, method=normalize_method)

        row_idx               = len(sparse_fps)
        sparse_fps[doc_id]    = sparse_fp
        doc_index_map[doc_id] = row_idx
        active_bits.append(sparse_fp.nnz)

        logger.debug(
            "  doc %-20s  nnz=%4d  density=%.4f",
            doc_id, sparse_fp.nnz, sparse_fp.nnz / (grid_size * grid_size),
        )

    logger.info(
        "Built %d document fingerprints  (%d skipped — no vocabulary match)",
        len(sparse_fps), skipped,
    )

    # ── 5. Optional diversity report ─────────────────────────────────────────
    if compute_diversity and sparse_fps:
        logger.info(
            "Computing fingerprint diversity (sample=%d) ...", diversity_sample
        )
        diversity = compute_fingerprint_diversity(
            sparse_fps, sample_size=diversity_sample
        )
        for metric, value in sorted(diversity.items()):
            logger.info("  %-30s = %.6f", metric, value)

    # ── 6. Stack to dense matrix ──────────────────────────────────────────────
    fp_matrix = (
        np.vstack(
            [sparse_fps[d].toarray().astype(np.float32) for d in sparse_fps]
        )
        if sparse_fps
        else np.zeros((0, grid_size * grid_size), dtype=np.float32)
    )

    stats = {
        "total_documents"    : len(contexts),
        "fingerprinted_docs" : len(sparse_fps),
        "skipped_docs"       : skipped,
        "skip_rate_pct"      : (
            round(skipped / len(contexts) * 100, 2) if contexts else 0.0
        ),
        "vector_size"        : grid_size * grid_size,
        "avg_active_bits"    : (
            round(float(np.mean(active_bits)), 2) if active_bits else 0.0
        ),
        "grid_size"          : grid_size,
        "top_percent"        : top_percent,
    }

    return fp_matrix, doc_index_map, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5 — Build document-level Sparse Distributed "
            "Representations (SDRs) from phrase fingerprints (Step 4)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required I/O ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="CSV corpus file (context_id,context_text). Same file as Step 1.",
    )
    parser.add_argument(
        "--fingerprints", type=Path, required=True,
        help="Step 4 output directory (contains phrase_fingerprints.npz + _meta.json).",
    )
    parser.add_argument(
        "--idf-weights", type=Path, required=False, dest="idf_weights",
        help="Path to idf_weights.json produced by Step 2.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, dest="output",
        help="Directory into which doc fingerprint outputs are written.",
    )

    # ── Grid / sparsity ───────────────────────────────────────────────────────
    parser.add_argument(
        "--grid-size", type=int, default=16, dest="grid_size",
        help="Side length of the square semantic grid. Must match Steps 3 & 4.",
    )
    parser.add_argument(
        "--top-percent", type=float, default=0.1, dest="top_percent",
        help="Fraction of grid cells kept active per document SDR.",
    )

    # ── Normalisation ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--normalize", action="store_true", default=True,
        help="Normalise each SDR after sparsification (default: on).",
    )
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false",
        help="Disable SDR normalisation.",
    )
    parser.add_argument(
        "--normalize-method", type=str, default="l2",
        choices=["l1", "l2", "max"], dest="normalize_method",
        help="Normalisation strategy.",
    )

    # ── Phrase extraction flags (must mirror Step 1 settings) ────────────────
    parser.add_argument(
        "--no-spacy", action="store_true", default=False,
        help="Force NLTK fallback extraction (use if Step 1 used --no-spacy).",
    )
    parser.add_argument(
        "--no-verbs", dest="keep_verbs", action="store_false",
        help="Remove verb forms during normalisation.",
    )

    parser.add_argument(
        "--no-filter-generic", dest="filter_generic", action="store_false",
        default=True,
        help="Keep generic single words during expansion (mirrors Step 1 flag).",
    )
    parser.add_argument(
        "--min-word-length", type=int, default=3, dest="min_word_length",
        help="Minimum character length for single-word tokens kept after expansion.",
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--compute-diversity", action="store_true", default=False,
        help="Compute and log pairwise fingerprint diversity statistics.",
    )
    parser.add_argument(
        "--diversity-sample", type=int, default=100, dest="diversity_sample",
        help="Number of documents sampled for diversity computation.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Semantic Folding — Step 5: Document Fingerprints")
    logger.info("=" * 60)
    logger.info("  corpus          : %s", args.corpus)
    logger.info("  fingerprints    : %s", args.fingerprints)
    logger.info("  idf_weights     : %s", args.idf_weights)
    logger.info("  output_dir      : %s", args.output)
    logger.info("  grid_size       : %d  (%d bits)", args.grid_size, args.grid_size ** 2)
    logger.info("  top_percent     : %.4f  (%.1f%%)", args.top_percent, args.top_percent * 100)
    logger.info("  normalize       : %s (%s)", args.normalize, args.normalize_method)
    logger.info("  use_spacy       : %s", not args.no_spacy)
    logger.info("  keep_verbs      : %s", args.keep_verbs)
    logger.info("  filter_generic  : %s", args.filter_generic)
    logger.info("  min_word_length : %d", args.min_word_length)
    logger.info("=" * 60)

    fp_matrix, doc_index_map, stats = build_doc_fingerprints(
        corpus_path       = args.corpus,
        fingerprints_path = args.fingerprints,
        idf_weights_path  = args.idf_weights,
        grid_size         = args.grid_size,
        top_percent       = args.top_percent,
        normalize         = args.normalize,
        normalize_method  = args.normalize_method,
        use_spacy         = not args.no_spacy,
        remove_verbs      = not args.keep_verbs,
        filter_generic    = args.filter_generic,
        min_word_length   = args.min_word_length,
        compute_diversity = args.compute_diversity,
        diversity_sample  = args.diversity_sample,
    )

    try:
        write_outputs(fp_matrix, doc_index_map, stats, args.output)
    except OSError as exc:
        logger.error("Failed to write outputs: %s", exc)
        sys.exit(4)

    logger.info("Step 5 complete — outputs written to: %s", args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
