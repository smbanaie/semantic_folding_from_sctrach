#!/usr/bin/env python3
"""
doc_fingerprints.py — Step 5 of the Semantic Folding Pipeline

Aggregates phrase-level sparse fingerprints (Step 4) into document-level
Sparse Distributed Representations (SDRs) using TF-IDF weighted union,
then sparsifies via Morton (Z-order) curve thresholding.

Usage:
    python doc_fingerprints.py \
        --corpus      data/corpus.txt \
        --phrases     outputs/run/phrases.txt \
        --fingerprints outputs/run/phrase_fingerprints \
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
from phrase_extractor import extract_and_normalize_phrases, SPACY_AVAILABLE

from lib import (
    compute_fingerprint_diversity,
    compute_idf_weights,
    export_fingerprints_to_numpy,
    load_contexts_dict,
    load_phrases,
    normalize_fingerprint,
    sparsify_fingerprint,
    load_phrase_fingerprints_sparse,
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
# Output writer  (mirrors phrase_fingerprints.py pattern)
# ---------------------------------------------------------------------------

def write_outputs(
    fingerprints  : np.ndarray,
    doc_index_map : Dict[str, int],
    stats         : dict,
    output_dir    : Path,
) -> None:
    """
    Write three files to ``output_dir``:

    - ``doc_fingerprints.npz``       — compressed matrix, shape (n_docs, grid_size²)
    - ``doc_fingerprints_meta.json`` — doc_id → row-index mapping
    - ``doc_fingerprints_stats.json``— run statistics
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
    text       : str,
    phrase_fps : Dict[str, np.ndarray],
    use_spacy  : bool = True,
) -> List[str]:
    """
    Extract normalized phrases from document text that exist in the
    phrase fingerprint vocabulary (phrase_fps keys).

    Uses the same pipeline as Step 1 (phrase_extractor.py) to guarantee
    identical normalization.

    Returns
    -------
    List[str]
        Vocabulary-matched phrases (duplicates preserved for TF counting).
    """
    candidates: Set[str] = extract_and_normalize_phrases(
        text,
        use_spacy    = use_spacy,
        remove_verbs = True,
    )
    matched = [p for p in candidates if p in phrase_fps]
    if not matched:
        logger.debug("No vocabulary matches in text snippet: %r...", text[:80])
    return matched


# ---------------------------------------------------------------------------
# Coordinate-set → csr_matrix converter
# ---------------------------------------------------------------------------

def coords_to_csr(
    coords    : Set[Tuple[int, int]],
    grid_size : int,
) -> csr_matrix:
    """Convert (row, col) coordinate set → (1, grid_size²) csr_matrix."""
    n   = grid_size * grid_size
    mat = lil_matrix((1, n), dtype=np.float32)
    for (r, c) in coords:
        if 0 <= r < grid_size and 0 <= c < grid_size:
            mat[0, r * grid_size + c] = 1.0
    return mat.tocsr()


# ---------------------------------------------------------------------------
# Single-document fingerprint builder  (called inside the corpus loop)
# ---------------------------------------------------------------------------

def build_document_fingerprint(
    doc_text            : str,
    phrase_fingerprints : Dict[str, csr_matrix],
    idf_weights         : Dict[str, float],
    grid_size           : int,
    use_spacy           : bool = True,
    remove_verbs        : bool = True,
) -> Optional[csr_matrix]:
    """
    Build a raw (un-sparsified) TF-IDF weighted fingerprint for one document.

    Steps
    -----
    1. Extract phrases present in the vocabulary.
    2. Accumulate weighted sum:  fp += idf(phrase) * phrase_vector
    3. Return None if no vocabulary phrase was found.

    Returns
    -------
    csr_matrix of shape (1, grid_size²), or None.
    """
    n      = grid_size * grid_size
    acc    = lil_matrix((1, n), dtype=np.float32)
    hits   = 0

    matched_phrases = extract_phrases_from_doc(
        doc_text, phrase_fingerprints, use_spacy=use_spacy
    )

    for phrase in matched_phrases:
        vec = phrase_fingerprints.get(phrase)
        if vec is None:
            continue
        weight = idf_weights.get(phrase, 1.0)

        # vec may be ndarray (1-D) or csr_matrix (1, n)
        if isinstance(vec, np.ndarray):
            flat = vec.flatten()[:n]
            acc[0, : len(flat)] += weight * flat
        else:
            acc += weight * vec

        hits += 1

    if hits == 0:
        return None

    return acc.tocsr()


# ---------------------------------------------------------------------------
# SDR sparsifier  (thin wrapper around lib.sparsify_fingerprint)
# ---------------------------------------------------------------------------

def sparsify_to_sdr(
    fingerprint : csr_matrix,
    top_percent : float,
    grid_size   : int,
) -> csr_matrix:
    """Keep only the top ``top_percent`` fraction of bits (Morton ordering)."""
    top_k = max(1, int(round(top_percent * grid_size * grid_size)))
    return sparsify_fingerprint(
        fingerprint,
        top_k      = top_k,
        use_zorder = True,
        grid_size  = grid_size,
    )


# ---------------------------------------------------------------------------
# Main pipeline  (build only — saving is handled by write_outputs)
# ---------------------------------------------------------------------------

def build_doc_fingerprints(
    corpus_path       : Path,
    phrases_path      : Path,
    fingerprints_path : Path,
    grid_size         : int   = 16,
    top_percent       : float = 0.1,
    min_freq          : int   = 1,
    normalize         : bool  = True,
    normalize_method  : str   = "l2",
    use_spacy         : bool  = True,
    remove_verbs      : bool  = True,
    compute_diversity : bool  = False,
    diversity_sample  : int   = 100,
) -> Tuple[np.ndarray, Dict[str, int], dict]:
    """
    Full Step-5 pipeline (build only — saving handled by write_outputs).

    Returns
    -------
    fingerprint_matrix : np.ndarray, shape (n_docs, grid_size²)
    doc_index_map      : Dict[str, int]  —  doc_id → row index
    stats              : dict
    """

    # 1. Phrase inventory
    logger.info("Loading phrase inventory from %s ...", phrases_path)
    phrase_tuples = load_phrases(phrases_path, min_freq=min_freq)
    phrase_set    = {p for p, _ in phrase_tuples}
    logger.info("  %d phrases loaded (min_freq=%d)", len(phrase_set), min_freq)

    # 2. Phrase fingerprints from Step 4 output directory
    logger.info("Loading phrase fingerprints from %s ...", fingerprints_path)
    phrase_fingerprints = load_phrase_fingerprints_sparse(
        fingerprints_dir = fingerprints_path,
        grid_size        = grid_size,
    )
    phrase_fingerprints = {
        p: v for p, v in phrase_fingerprints.items() if p in phrase_set
    }
    logger.info("  %d phrase fingerprints after inventory filter.", len(phrase_fingerprints))

    if not phrase_fingerprints:
        logger.error(
            "No phrase fingerprints remain after inventory filter. "
            "Check that --phrases and --fingerprints are from the same pipeline run."
        )
        sys.exit(1)

    # 3. Corpus
    logger.info("Loading corpus from %s ...", corpus_path)
    contexts = load_contexts_dict(corpus_path)
    logger.info("  %d documents loaded", len(contexts))

    # 4. IDF weights
    logger.info("Computing IDF weights ...")
    idf_weights = compute_idf_weights(
        list(phrase_fingerprints.keys()),
        list(contexts.values()),
    )

    # 5 + 6 + 7.  Build → sparsify → normalise
    top_k_bits = max(1, int(round(top_percent * grid_size * grid_size)))
    logger.info(
        "Building document fingerprints (grid=%d, top_percent=%.3f → top_k=%d bits) ...",
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

        # ── single-document fingerprint ──────────────────────────────
        raw_fp = build_document_fingerprint(          # ← correct function name
            doc_text            = doc_text,
            phrase_fingerprints = phrase_fingerprints,
            idf_weights         = idf_weights,
            grid_size           = grid_size,
            use_spacy           = use_spacy,
            remove_verbs        = remove_verbs,
        )
        if raw_fp is None:
            skipped += 1
            logger.debug("  doc %s — no matching phrases, skipped", doc_id)
            continue

        sparse_fp = sparsify_to_sdr(raw_fp, top_percent=top_percent, grid_size=grid_size)

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

    # 8. Optional diversity report
    if compute_diversity and sparse_fps:
        logger.info("Computing fingerprint diversity (sample=%d) ...", diversity_sample)
        diversity = compute_fingerprint_diversity(sparse_fps, sample_size=diversity_sample)
        for metric, value in sorted(diversity.items()):
            logger.info("  %-30s = %.6f", metric, value)

    # 9. Convert sparse dict → dense matrix
    fp_matrix = (
        np.vstack([sparse_fps[d].toarray().astype(np.float32) for d in sparse_fps])
        if sparse_fps
        else np.zeros((0, grid_size * grid_size), dtype=np.float32)
    )

    stats = {
        "total_documents"    : len(contexts),
        "fingerprinted_docs" : len(sparse_fps),
        "skipped_docs"       : skipped,
        "skip_rate_pct"      : round(skipped / len(contexts) * 100, 2) if contexts else 0.0,
        "vector_size"        : grid_size * grid_size,
        "avg_active_bits"    : round(float(np.mean(active_bits)), 2) if active_bits else 0.0,
        "grid_size"          : grid_size,
        "top_percent"        : top_percent,
    }

    return fp_matrix, doc_index_map, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5 — Build document fingerprints from phrase SDRs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--corpus",       type=Path, required=True)
    parser.add_argument("--phrases",      type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True,
        help="Step 4 output directory (contains phrase_fingerprints.npz + _meta.json).")

    # mirrors phrase_fingerprints.py exactly
    parser.add_argument(
        "--output-dir", type=Path, required=True, dest="output",
        help="Directory into which doc fingerprint outputs are written.",
    )

    parser.add_argument("--grid-size",         type=int,   default=16,   dest="grid_size")
    parser.add_argument("--top-percent",       type=float, default=0.1,  dest="top_percent")
    parser.add_argument("--min-freq",          type=int,   default=1,    dest="min_freq")

    parser.add_argument("--normalize",         action="store_true",  default=True)
    parser.add_argument("--no-normalize",      dest="normalize",     action="store_false")
    parser.add_argument("--normalize-method",  type=str, default="l2",
                        choices=["l1", "l2", "max"])

    parser.add_argument("--no-spacy",          action="store_true",  default=False)
    parser.add_argument("--remove-verbs",      action="store_true",  default=True)
    parser.add_argument("--keep-verbs",        dest="remove_verbs",  action="store_false")

    parser.add_argument("--compute-diversity", action="store_true",  default=False)
    parser.add_argument("--diversity-sample",  type=int,   default=100)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Semantic Folding — Step 5: Document Fingerprints")
    logger.info("=" * 60)
    logger.info("  corpus       : %s", args.corpus)
    logger.info("  phrases      : %s", args.phrases)
    logger.info("  fingerprints : %s", args.fingerprints)
    logger.info("  output_dir   : %s", args.output)
    logger.info("  grid_size    : %d  (%d bits)", args.grid_size, args.grid_size ** 2)
    logger.info("  top_percent  : %.4f  (%.1f%%)", args.top_percent, args.top_percent * 100)
    logger.info("  normalize    : %s (%s)", args.normalize, args.normalize_method)
    logger.info("=" * 60)

    fp_matrix, doc_index_map, stats = build_doc_fingerprints(
        corpus_path       = args.corpus,
        phrases_path      = args.phrases,
        fingerprints_path = args.fingerprints,
        grid_size         = args.grid_size,
        top_percent       = args.top_percent,
        min_freq          = args.min_freq,
        normalize         = args.normalize,
        normalize_method  = args.normalize_method,
        use_spacy         = not args.no_spacy,
        remove_verbs      = args.remove_verbs,
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
