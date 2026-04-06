#!/usr/bin/env python3
"""
query_processing.py — Step 6 of the Semantic Folding Pipeline

Processes user queries by extracting phrases, constructing a sparse
query fingerprint, optionally applying Z-order spreading, and ranking
documents by cosine similarity against Step-5 document fingerprints.

Pipeline position
-----------------
Step 1  phrase_extractor.py    → phrases.txt
Step 2  term_context.py        → phrase_context_matrix.*
Step 3  semantic_space.py      → grid layout
Step 4  phrase_fingerprints.py → phrase_fingerprints/
Step 5  doc_fingerprints.py    → doc_fingerprints/
Step 6  query_processing.py    → ranked results          ← THIS FILE

Consistency guarantee
---------------------
Query phrase extraction follows the **identical** three-stage pipeline
used in Steps 1 and 5:

    raw query text
        └─ extract_raw_phrases_*()     # spaCy noun chunks / NLTK n-grams
                └─ normalize_phrase()  # lowercase → stopwords → lemmatize
                        └─ expand_phrases()   # sub-phrase generation
                                └─ vocab filter (keys of phrase_fingerprints)

The ``--remove-verbs``, ``--no-spacy``, ``--no-filter-generic``, and
``--min-word-length`` flags **must** be set identically to the values used
in Step 1 (``phrase_extractor.py``) so that query normalisation produces
the same token sequences as those stored in the vocabulary.

Usage
-----
    python query_processing.py \\
        --query "machine learning algorithms" \\
        --phrase-fp-dir  outputs/run/phrase_fingerprints/ \\
        --doc-fp-dir     outputs/run/doc_fingerprints/ \\
        --grid-size      16 \\
        --top-k          10 \\
        --weighting      idf \\
        --normalization  l2 \\
        --spreading-steps 1
"""

from __future__ import annotations
from collections import Counter
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from loguru import logger

from phrase_extractor import (
    extract_raw_phrases_spacy,
    extract_raw_phrases_fallback,
    debug_phrase_extraction_pipeline,
    SPACY_AVAILABLE,
)
from lib import (
    normalize_phrase,
    expand_phrases,
    load_phrase_fingerprints_sparse,
    compute_cosine_similarity,
    normalize_fingerprint,
    get_zorder_neighbors,
    batch_compute_similarities,
    is_valid_phrase_structure,
    load_document_fingerprints,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_vector_size(phrase_fingerprints: dict) -> int:
    """
    Return the flat fingerprint dimension from the first dictionary entry.

    Handles both dense ``np.ndarray`` and sparse ``csr_matrix`` values so
    the function is robust to whichever storage format Step 4 produced.

    Parameters
    ----------
    phrase_fingerprints : dict
        Non-empty mapping of ``{phrase: vector}`` as returned by
        ``lib.load_phrase_fingerprints_sparse``.

    Returns
    -------
    int
        Number of elements in the flat (1-D) fingerprint vector,
        i.e. ``grid_size * grid_size``.
    """
    first = next(iter(phrase_fingerprints.values()))
    if hasattr(first, "toarray"):
        return first.toarray().ravel().shape[0]
    return np.asarray(first).ravel().shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# Phrase extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_query_phrases(
    query           : str,
    phrase_vocab    : Set[str],
    use_spacy       : bool = True,
    remove_verbs    : bool = True,
    filter_generic  : bool = True,
    min_word_length : int  = 3,
) -> List[str]:
    """
    Extract vocabulary-matched phrases from a raw query string.

    This function is the Step-6 counterpart of
    ``phrase_extractor.process_corpus_with_expansion`` and applies the
    **same three-stage pipeline** used in Steps 1 and 5 to guarantee that
    query phrases map to the same vocabulary entries built during indexing:

    Stage 1 — Extraction + normalisation
        ``extract_raw_phrases_spacy`` (or the NLTK fallback) detects noun
        chunks, named entities, and compound nouns.  Each candidate is
        piped through ``lib.normalize_phrase`` (lowercase → stop-word
        removal → optional verb removal → lemmatisation → POS validation).

    Stage 2 — Sub-phrase expansion
        ``lib.expand_phrases`` generates all meaningful sub-phrases from
        each normalised phrase (bigrams and trigrams from longer phrases,
        individual tokens from shorter ones), mirroring the expansion
        logic in ``process_corpus_with_expansion``.

        **This stage is critical for recall.**  Without it a query
        containing "deep neural network" would not activate the
        fingerprints for "neural network" or "neural", even though those
        sub-phrases exist in the vocabulary built during Step 1.

    Stage 3 — Vocabulary filter
        Only phrases present as keys in ``phrase_vocab`` are retained,
        exactly mirroring the ``min_freq`` filter applied in Step 1.

    Parameters
    ----------
    query : str
        Raw query string entered by the user.
    phrase_vocab : Set[str]
        Complete set of normalised phrase strings from the loaded phrase
        fingerprint dictionary (``set(phrase_fingerprints.keys())``).
    use_spacy : bool, optional
        When ``True`` (default), attempt spaCy noun-chunk extraction.
        **Must match the flag used during Step 1.**
    remove_verbs : bool, optional
        Strip verb tokens before lemmatisation (default: ``True``).
        **Must match the flag used during Step 1** — the corpus default
        is ``True``, so this parameter default has been corrected from the
        original ``False`` to avoid silent vocabulary mismatches.
    filter_generic : bool, optional
        Remove generic single-word tokens during expansion (default:
        ``True``).  **Must match the flag used during Step 1.**
    min_word_length : int, optional
        Minimum character length for single-word tokens kept after
        expansion (default: ``3``).  **Must match the flag used during
        Step 1.**

    Returns
    -------
    List[str]
        Vocabulary-matched, normalised phrase strings.  Duplicates are
        preserved so that term-frequency weighting can be applied by the
        caller.  Returns an empty list when no phrases match the
        vocabulary.

    Notes
    -----
    - If spaCy is requested but unavailable a WARNING is emitted and the
      NLTK fallback is used automatically.
    - The function is stateless and safe to call from parallel workers.

    Examples
    --------
    Suppose the vocabulary contains ``{"neural network", "deep neural",
    "neural", "network"}``.

    >>> phrases = extract_query_phrases(
    ...     "The model uses a deep neural network.",
    ...     phrase_vocab=vocab,
    ...     use_spacy=True,
    ...     remove_verbs=True,
    ... )
    >>> sorted(set(phrases))
    ['deep neural', 'neural', 'neural network', 'network']
    """
    # ── Stage 1: extraction + normalisation ──────────────────────────────────
    if use_spacy and SPACY_AVAILABLE:
        raw_phrases = extract_raw_phrases_spacy(query)
    else:
        if use_spacy and not SPACY_AVAILABLE:
            logger.warning(
                f"spaCy requested but unavailable — using NLTK fallback. "
                f"Verify this matches the setting used in Step 1."
            )
        raw_phrases = extract_raw_phrases_fallback(query, max_ngram=4)

    candidates: List[str] = []
    for phrase in raw_phrases:
        norm = normalize_phrase(phrase, remove_verbs=remove_verbs)
        if norm:
            candidates.append(norm)

    if not candidates:
        logger.debug(f"No candidates after normalisation for query: {query!r}")
        return []

    # ── Stage 2: sub-phrase expansion ────────────────────────────────────────
    expanded: List[str] = expand_phrases(
        candidates,
        context_text    = query,            # <-- UPDATED: Pass raw query text for validation
        filter_generic  = filter_generic,
        min_word_length = min_word_length,
    )

    # ── Stage 3: vocabulary filter ────────────────────────────────────────────
    matched: List[str] = [p for p in expanded if p in phrase_vocab]

    logger.info(
        f"Query phrase extraction: {len(candidates)} raw → "
        f"{len(expanded)} expanded → {len(matched)} vocab hits"
    )
    if matched:
        logger.debug(f"Matched phrases: {matched}")

        debug_report = debug_phrase_extraction_pipeline(
        query,
        use_spacy=use_spacy,
        remove_verbs=remove_verbs,
        filter_generic=filter_generic,
        min_word_length=min_word_length,
        vocab=phrase_vocab,
        )   
    
        logger.debug("="*60)
        logger.debug("PHRASE EXTRACTION PIPELINE TRACE")
        logger.debug(f"Query: {query}")
        logger.debug("-"*60)
        
        for stage_name, stage_data in debug_report["stages"].items():
            logger.debug(f"\n{stage_name.upper()}:")
            logger.debug(f"  Count: {stage_data['count']}")
            if stage_data['count'] <= 20:  # Only show if reasonable
                logger.debug(f"  Phrases: {stage_data['phrases']}")
            
            if 'dropped' in stage_data and stage_data['dropped'] > 0:
                logger.debug(f"  ⚠ Dropped: {stage_data['dropped']} phrases")
            
            if 'out_of_vocab' in stage_data and stage_data['out_of_vocab']:
                logger.debug(f"  ⚠ Out-of-vocab: {stage_data['out_of_vocab']}")
        
        logger.debug("\nLOSSES BY STAGE:")
        for stage, losses in debug_report["losses"].items():
            if losses:
                logger.debug(f"  {stage}: {len(losses)} phrases")
                for loss in losses[:5]:  # Show first 5
                    logger.debug(f"    - {loss}")
        
        logger.debug("\nSUMMARY:")
        for key, val in debug_report["summary"].items():
            logger.debug(f"  {key}: {val}")
        logger.debug("="*60)
    

    return matched

# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint construction
# ─────────────────────────────────────────────────────────────────────────────


def construct_query_fingerprint(
    query_phrases       : List[str],
    phrase_fingerprints : Dict[str, csr_matrix],
    weighting           : str                    = "uniform",
    idf_weights         : Optional[Dict[str, float]] = None,
    normalization       : str                    = "l2",
) -> Tuple[Optional[csr_matrix], Dict]:
    """
    Aggregate per-phrase fingerprints into a single query fingerprint vector.

    Each phrase in ``query_phrases`` that is found in
    ``phrase_fingerprints`` is weighted and accumulated into a dense array 
    before being converted to a sparse row vector. The aggregated vector 
    is optionally normalised before return.

    The function is intentionally conservative: it returns
    ``(None, metadata)`` in every failure mode rather than raising,
    allowing the caller to decide whether to skip or abort.

    Parameters
    ----------
    query_phrases : List[str]
        Ordered list of normalised phrase strings from
        :func:`extract_query_phrases`.  An empty list causes an immediate
        ``(None, {"error": "no_phrases"})`` return.
    phrase_fingerprints : Dict[str, csr_matrix]
        Mapping of ``{phrase: fingerprint_vector}`` from Step 4.  Each
        value must be a ``(1, grid_size²)`` sparse row vector.
    weighting : str, optional
        Scalar weight assigned to each phrase contribution:

        ``"uniform"``
            Every unique phrase contributes weight ``1.0``, regardless of frequency.
        ``"frequency"``
            Weight equals the term frequency (TF) of the phrase in
            ``query_phrases`` (promotes repeated terms).
        ``"idf"``
            Weight is the term frequency multiplied by the IDF score from ``idf_weights`` 
            (falls back to TF * 1.0 for unknown phrases). Requires ``idf_weights``;
            degrades to TF-only if ``idf_weights`` is ``None``.

    idf_weights : Dict[str, float] or None, optional
        ``{phrase: idf_score}`` mapping.  Only consulted when
        ``weighting="idf"``.
    normalization : str, optional
        Normalisation applied to the aggregated vector before return.
        Supported values: ``"l2"`` (default), ``"l1"``, ``"binary"``,
        ``"none"``.

    Returns
    -------
    fingerprint : csr_matrix or None
        Sparse ``(1, grid_size²)`` query vector.  ``None`` on failure.
    metadata : Dict
        Always present; contains an ``"error"`` key on failure or
        detailed statistics on success (see docstring body for full
        key list).

    Notes
    -----
    - *Refactored:* Accumulation is performed on a dense `np.zeros` float32 array 
      for O(1) additions, completely bypassing the `lil_matrix` bottleneck. 
      It is converted to `csr_matrix` only at the end.
    - *Refactored:* Uses `collections.Counter` to group unique phrases. This fixes an O(N²) 
      performance bottleneck from `list.count()` and resolves a logical bug where iterating 
      over duplicate phrases effectively squared the term frequency contributions.
    - When using IDF weighting, the weighted values are preserved as floats
      rather than binarized, allowing rare terms to contribute more strongly
      to similarity scores.
    """
    if not query_phrases:
        logger.warning("No query phrases provided to construct_query_fingerprint")
        return None, {"error": "no_phrases"}

    if not phrase_fingerprints:
        logger.error("Phrase fingerprints dictionary is empty")
        return None, {"error": "empty_phrase_vocabulary"}

    grid_size_sq = _infer_vector_size(phrase_fingerprints)
    
    # 1. Accumulate into a dense array instead of lil_matrix for massive speedup
    acc = np.zeros(grid_size_sq, dtype=np.float32)
    
    phrase_weights_used: Dict[str, float] = {}
    missing_phrases:     List[str]        = []
    
    # 2. Pre-compute Term Frequency using Counter (Fixes O(N^2) and squaring bug)
    phrase_counts = Counter(query_phrases)

    for phrase, tf in phrase_counts.items():
        if phrase not in phrase_fingerprints:
            logger.warning(f"Phrase not in fingerprints vocabulary: '{phrase}'")
            # Store missing phrase but avoid appending duplicates
            missing_phrases.append(phrase)
            continue

        phrase_fp = phrase_fingerprints[phrase]

        # 3. Apply exact mathematical weights once per unique phrase
        if weighting == "idf" and idf_weights:
            base_weight = float(idf_weights.get(phrase, 1.0))
            weight = base_weight * float(tf)
        elif weighting == "frequency":
            weight = float(tf)
        else: # "uniform"
            weight = 1.0

        # High-speed dense accumulation
        if hasattr(phrase_fp, "toarray"):
            fp_array = phrase_fp.toarray().ravel()
        else:
            fp_array = np.asarray(phrase_fp).ravel()
            
        acc += weight * fp_array
        phrase_weights_used[phrase] = weight

    # Check if accumulator is empty (all values are 0.0)
    if not np.any(acc):
        if missing_phrases:
            logger.error(
                f"All {len(missing_phrases)} unique query phrase(s) are "
                f"out-of-vocabulary: {missing_phrases}"
            )
        else:
            logger.error("Query fingerprint is empty after aggregation")
        return None, {
            "error":           "empty_fingerprint",
            "missing_phrases": missing_phrases,
        }

    # Convert dense accumulator to CSR matrix
    acc_csr = csr_matrix(acc.reshape(1, -1))
    pre_norm_nnz = acc_csr.nnz

    # Apply normalization but keep as float values (don't binarize)
    if normalization and normalization != "none":
        acc_csr = normalize_fingerprint(acc_csr, method=normalization)

    # Calculate sparsity based on non-zero elements
    post_norm_nnz = acc_csr.nnz

    metadata = {
        "num_phrases":          len(query_phrases),
        "num_matched":          len(phrase_weights_used),
        "num_missing":          len(missing_phrases),
        "missing_phrases":      missing_phrases,
        "phrase_weights":       phrase_weights_used,
        "active_bits_pre_norm": pre_norm_nnz,
        "active_bits":          post_norm_nnz,
        "sparsity":             post_norm_nnz / grid_size_sq,
        "weighting":            weighting,
        "normalization":        normalization,
    }

    logger.success(
        f"Query fingerprint: {post_norm_nnz} active elements from "
        f"{len(phrase_weights_used)} phrases (weighted)"
    )
    return acc_csr, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Spreading
# ─────────────────────────────────────────────────────────────────────────────

def apply_spreading(
    fingerprint     : csr_matrix,
    grid_size       : int,
    radius          : int   = 1,
    decay           : float = 0.5,
    normalize_after : bool  = True,
) -> Tuple[csr_matrix, Dict]:
    r"""
    Apply Z-order neighbourhood spreading to a query fingerprint.

    Spreading propagates activation from each active bit outward to its
    Z-order spatial neighbours, attenuated by a decay factor per unit of
    Chebyshev distance.  This soft-expands the query's representational
    footprint so that documents sharing *nearby* rather than *identical*
    grid coordinates can still contribute to similarity.

    Parameters
    ----------
    fingerprint : csr_matrix
        Sparse ``(1, grid_size²)`` query fingerprint.
    grid_size : int
        Side length of the square Z-order grid.
    radius : int, optional
        Maximum Chebyshev distance to which activation is spread.
        ``0`` is a no-op (returns the input unchanged).  Default: ``1``.
    decay : float, optional
        Multiplicative attenuation per unit Chebyshev distance.  Must be
        in ``(0.0, 1.0]``.  Default: ``0.5``.
    normalize_after : bool, optional
        When ``True`` (default), L2-normalise the spread fingerprint
        before return to keep it on the same unit sphere as document
        fingerprints.

    Returns
    -------
    result : csr_matrix
        Spread (and optionally normalised) ``(1, grid_size²)`` fingerprint.
    metadata : Dict
        ``spreading_applied`` — ``False`` when ``radius=0``.
        ``radius``, ``decay``, ``active_bits_before``,
        ``active_bits_after``, ``bits_added``.

    Notes
    -----
    - The sparsity guard (Issue 4 in the analysis) has been corrected to
      use ``fingerprint.shape[1]`` (number of cells) rather than
      ``fingerprint.shape[0]`` (number of rows, always 1) as the
      denominator.  The threshold is also lowered to ``0.02`` so that
      short queries on a 16×16 grid are not unconditionally suppressed;
      after fixing the expansion stage (Issue 1) a typical short query
      will have $\geq 3$ matched phrases, yielding sparsity
      $\approx 0.035$, which exceeds the new threshold.
    - Neighbour coordinates are obtained via ``lib.get_zorder_neighbors``,
      which respects grid boundaries.
    - The intermediate computation uses a dense ``(grid_size, grid_size)``
      NumPy array; for grids larger than 64×64 consider a sparse
      spreading implementation.
    """
    if radius == 0:
        return fingerprint, {"spreading_applied": False}

    # ── Corrected sparsity guard (Issue 4) ───────────────────────────────────
    n_cells  = fingerprint.shape[1]            # grid_size²  (was shape[0] = 1)
    sparsity = fingerprint.nnz / n_cells       # true bit-density ratio

    if sparsity < 0.02:                        # threshold lowered from 0.08
        logger.warning(
            f"Fingerprint very sparse ({sparsity:.4f}, {fingerprint.nnz} bits "
            f"/ {n_cells} cells) — spreading skipped to avoid score collapse."
        )
        return fingerprint, {
            "spreading_applied": False,
            "reason":            "sparsity_too_low",
            "sparsity":          sparsity,
        }

    original_nnz = fingerprint.nnz

    dense_fp      = fingerprint.toarray().reshape(grid_size, grid_size)
    spread_fp     = dense_fp.copy()
    active_coords = np.argwhere(dense_fp > 0)

    for y, x in active_coords:
        value     = dense_fp[y, x]
        neighbors = get_zorder_neighbors(x, y, grid_size, radius)
        for nx, ny in neighbors:
            dist = max(abs(nx - x), abs(ny - y))
            spread_fp[ny, nx] += value * (decay ** dist)

    result = csr_matrix(spread_fp.reshape(1, -1))

    if normalize_after:
        result = normalize_fingerprint(result, method="l2")

    metadata = {
        "spreading_applied":  True,
        "radius":             radius,
        "decay":              decay,
        "active_bits_before": original_nnz,
        "active_bits_after":  result.nnz,
        "bits_added":         result.nnz - original_nnz,
    }

    logger.info(
        f"Spreading: {original_nnz} → {result.nnz} active bits "
        f"(+{result.nnz - original_nnz}) | radius={radius} decay={decay:.2f}"
    )
    return result, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_documents(
    query_fp        : csr_matrix,
    doc_fingerprints: Dict[str, csr_matrix],
    top_k           : int   = 10,
    min_similarity  : float = 0.0,
    use_batch       : bool  = True,
    **kwargs,
) -> Tuple[List[Tuple[str, float]], Dict]:
    """
    Rank documents by weighted overlap with the query fingerprint.

    Instead of cosine similarity (which loses IDF signal when comparing
    float query vectors against binary document vectors), this computes
    a weighted overlap score: for each active bit in a document, sum the
    corresponding weight from the query vector.

    This preserves the IDF weighting from the query — rare terms contribute
    more strongly to the score than common terms.

    Parameters
    ----------
    query_fp : csr_matrix
        Query fingerprint vector (1, grid_size²), potentially float-weighted.
    doc_fingerprints : Dict[str, csr_matrix]
        Mapping of {doc_id: fingerprint_vector}.
    top_k : int
        Number of top results to return.
    min_similarity : float, optional
        Minimum score threshold; documents below this are excluded.
    use_batch : bool, optional
        Accepted for API compatibility; ignored (dot-product path is always
        used regardless of corpus size).
    **kwargs
        Catches any additional legacy keyword arguments without error.

    Returns
    -------
    results : List[Tuple[str, float]]
        Ranked list of (doc_id, score) tuples, sorted descending by score,
        truncated to top_k.
    metadata : Dict
        ``total_documents``, ``documents_above_threshold``,
        ``mean_similarity``, ``max_similarity``.
    """
    if query_fp is None or query_fp.nnz == 0:
        logger.warning("Query fingerprint is empty")
        return [], {
            "total_documents":           0,
            "documents_above_threshold": 0,
            "mean_similarity":           0.0,
            "max_similarity":            0.0,
        }

    if not doc_fingerprints:
        logger.warning("No document fingerprints provided")
        return [], {
            "total_documents":           0,
            "documents_above_threshold": 0,
            "mean_similarity":           0.0,
            "max_similarity":            0.0,
        }

    all_scores: List[Tuple[str, float]] = []

    for doc_id, doc_fp in doc_fingerprints.items():
        if doc_fp.nnz == 0:
            continue

        # Weighted overlap: dot product sums query weights at positions where
        # the document is active.  Rare (high-IDF) query terms contribute more.
        score = float(query_fp.dot(doc_fp.T).toarray()[0, 0])

        # Length-normalise to avoid bias toward documents with more active bits.
        score = score / np.sqrt(doc_fp.nnz)

        all_scores.append((doc_id, score))

    if not all_scores:
        return [], {
            "total_documents":           len(doc_fingerprints),
            "documents_above_threshold": 0,
            "mean_similarity":           0.0,
            "max_similarity":            0.0,
        }

    raw_scores = [s for _, s in all_scores]
    mean_sim   = float(np.mean(raw_scores))
    max_sim    = float(np.max(raw_scores))

    # Apply threshold filter
    filtered = [(doc_id, score) for doc_id, score in all_scores
                if score >= min_similarity]

    # Sort descending
    filtered.sort(key=lambda x: x[1], reverse=True)

    metadata = {
        "total_documents":           len(doc_fingerprints),
        "documents_above_threshold": len(filtered),
        "mean_similarity":           mean_sim,
        "max_similarity":            max_sim,
    }

    return filtered[:top_k], metadata


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def display_results(
    results          : List[Tuple[str, float]],
    query            : str,
    query_metadata   : Dict,
    ranking_metadata : Dict,
    doc_metadata     : Optional[Dict[str, Dict]] = None,
    verbose          : bool = False,
) -> None:
    """
    Print ranked results to stdout in a human-readable tabular format.

    Parameters
    ----------
    results : List[Tuple[str, float]]
        Ranked ``(doc_id, score)`` pairs.
    query : str
        Original raw query string (used as section heading).
    query_metadata : Dict
        Metadata from :func:`construct_query_fingerprint`.
    ranking_metadata : Dict
        Metadata from :func:`rank_documents`.
    doc_metadata : Dict or None, optional
        Optional per-document annotations for verbose mode.
    verbose : bool, optional
        Print full diagnostic blocks when ``True`` (default: ``False``).
    """
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    if verbose:
        print("\nQuery Analysis:")
        print(
            f"  Phrases matched : "
            f"{query_metadata.get('num_matched', 0)}/"
            f"{query_metadata.get('num_phrases', 0)}"
        )
        print(f"  Active bits     : {query_metadata.get('active_bits', 0)}")
        print(f"  Sparsity        : {query_metadata.get('sparsity', 0):.4f}")

        if query_metadata.get("missing_phrases"):
            print(
                f"  Missing phrases : "
                f"{', '.join(query_metadata['missing_phrases'])}"
            )

        print("\nCorpus Statistics:")
        print(f"  Total documents : {ranking_metadata.get('total_documents', 0)}")
        print(f"  Mean similarity : {ranking_metadata.get('mean_similarity', 0):.4f}")
        print(f"  Max similarity  : {ranking_metadata.get('max_similarity', 0):.4f}")

    print(f"\nTop {len(results)} Results:")
    print("-" * 80)

    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank:2d}. {doc_id:50s} | Score: {score:.4f}")

        if verbose and doc_metadata and doc_id in doc_metadata:
            meta = doc_metadata[doc_id]
            if "matched_phrases" in meta:
                print(f"    Matched phrases : {meta['matched_phrases']}")
            if "coverage" in meta:
                print(f"    Coverage        : {meta['coverage']:.3f}")

    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end single query processor
# ─────────────────────────────────────────────────────────────────────────────

def process_query(
    query               : str,
    phrase_fingerprints : Dict[str, csr_matrix],
    doc_fingerprints    : Dict[str, csr_matrix],
    args                : argparse.Namespace,
    idf_weights         : Optional[Dict[str, float]] = None,
) -> Tuple[List[Tuple[str, float]], Dict]:
    """
    Execute the full query pipeline for a single query string.

    Orchestrates phrase extraction → fingerprint construction → optional
    spreading → document ranking in a single call.  All intermediate
    metadata is collected and returned alongside ranked results.

    Pipeline stages
    ---------------
    1. :func:`extract_query_phrases` — three-stage extraction with
       expansion (corrected to match Steps 1 & 5).
    2. :func:`construct_query_fingerprint` — TF/IDF/uniform weighted
       phrase vector aggregation.
    3. :func:`apply_spreading` — optional Z-order neighbourhood spreading.
    4. :func:`rank_documents` — weighted overlap ranking.

    Parameters
    ----------
    query : str
        Raw query string.
    phrase_fingerprints : Dict[str, csr_matrix]
        Pre-loaded phrase fingerprint dictionary (Step 4 output).
    doc_fingerprints : Dict[str, csr_matrix]
        Pre-loaded document fingerprint dictionary (Step 5 output).
    args : argparse.Namespace
        Parsed CLI arguments.  Consumed attributes:

        ``args.normalization``            — ``"l2"`` | ``"l1"`` | ``"binary"`` | ``"none"``.
        ``args.weighting``                — ``"uniform"`` | ``"frequency"`` | ``"idf"``.
        ``args.spreading_steps``          — int, default ``0``.
        ``args.spreading_decay``          — float, default ``0.5``.
        ``args.normalize_after_spreading``— bool, default ``False``.
        ``args.top_k``                    — int, default ``10``.
        ``args.min_similarity``           — float, default ``0.0``.
        ``args.use_batch``                — bool, default ``True``.
        ``args.no_spacy``                 — bool, default ``False``.
        ``args.remove_verbs``             — bool, default ``True``.
        ``args.filter_generic``           — bool, default ``True``.
        ``args.min_word_length``          — int, default ``3``.

    idf_weights : Dict[str, float] or None, optional
        IDF weight map; only used when ``args.weighting == "idf"``.

    Returns
    -------
    results : List[Tuple[str, float]]
        Ranked ``(doc_id, score)`` pairs, truncated to ``args.top_k``.
    combined_metadata : Dict
        Keys: ``"query"``, ``"query_construction"``, ``"spreading"``,
        ``"ranking"``, and ``"error"`` (only on failure).

    Notes
    -----
    - Returns ``([], metadata)`` rather than raising on failure so that a
      multi-query batch loop can continue past a single bad query.
    - ``args.normalization == "none"`` is normalised to ``None`` before
      being forwarded for API compatibility.
    - The ``use_spacy`` flag is now correctly derived from ``args.no_spacy``
      (Issue 3 fix) instead of being hardcoded to ``True``.
    - ``remove_verbs`` now defaults to ``True`` (Issue 2 fix) consistent
      with Step 1 and Step 5 defaults.
    """
    phrase_vocab = set(phrase_fingerprints.keys())

    # ── Step 1: phrase extraction with expansion ──────────────────────────────
    use_spacy       = not getattr(args, "no_spacy",        False)
    remove_verbs    = getattr(args, "remove_verbs",        True)
    filter_generic  = getattr(args, "filter_generic",      True)
    min_word_length = getattr(args, "min_word_length",     3)

    query_phrases = extract_query_phrases(
        query,
        phrase_vocab,
        use_spacy       = use_spacy,
        remove_verbs    = remove_verbs,
        filter_generic  = filter_generic,
        min_word_length = min_word_length,
    )

    if not query_phrases:
        logger.error(f"No valid phrases found in query: {query!r}")
        return [], {
            "query":              query,
            "error":              "no_phrases_extracted",
            "query_construction": {},
            "spreading":          {},
            "ranking": {
                "total_documents":           0,
                "documents_above_threshold": 0,
            },
        }

    # ── Step 2: build query fingerprint ───────────────────────────────────────
    norm       = None if args.normalization == "none" else args.normalization
    query_fp, query_metadata = construct_query_fingerprint(
        query_phrases,
        phrase_fingerprints,
        weighting     = args.weighting,
        idf_weights   = idf_weights,
        normalization = norm,
    )

    if query_fp is None:
        error_type = query_metadata.get("error", "unknown")
        logger.error(f"Failed to construct query fingerprint: {error_type}")
        return [], {
            "query":              query,
            "error":              error_type,
            "query_construction": query_metadata,
            "spreading":          {},
            "ranking": {
                "total_documents":           0,
                "documents_above_threshold": 0,
            },
        }

    # ── Step 3: spreading (optional) ──────────────────────────────────────────
    spreading_metadata: Dict = {}
    spreading_steps = getattr(args, "spreading_steps", 0)
    if spreading_steps > 0:
        grid_size = int(np.sqrt(query_fp.shape[1]))
        query_fp, spreading_metadata = apply_spreading(
            query_fp,
            grid_size,
            radius          = spreading_steps,
            decay           = getattr(args, "spreading_decay",           0.5),
            normalize_after = getattr(args, "normalize_after_spreading", False),
        )

    # ── Step 4: rank documents ────────────────────────────────────────────────
    results, ranking_metadata = rank_documents(
        query_fp,
        doc_fingerprints,
        top_k          = getattr(args, "top_k",          10),
        min_similarity = getattr(args, "min_similarity", 0.0),
        use_batch      = getattr(args, "use_batch",      True),
    )

    combined_metadata = {
        "query":              query,
        "query_construction": query_metadata,
        "spreading":          spreading_metadata,
        "ranking":            ranking_metadata,
    }

    return results, combined_metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for the Step-6 CLI.

    All phrase-extraction flags (``--no-spacy``, ``--remove-verbs`` /
    ``--keep-verbs``, ``--no-filter-generic``, ``--min-word-length``)
    **must** be set to the same values used in Step 1
    (``phrase_extractor.py``) to guarantee consistent phrase
    representations across the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace consumed by :func:`main`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 6 — Process queries against document fingerprints "
            "using the Semantic Folding pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required I/O ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query string to process.",
    )
    parser.add_argument(
        "--phrase-fp-dir", dest="phrase_fp_dir", type=Path, required=True,
        help="Step 4 phrase fingerprint directory.",
    )
    parser.add_argument(
        "--doc-fp-dir", dest="doc_fp_dir", type=Path, required=True,
        help="Step 5 document fingerprint directory.",
    )

    # ── Optional inputs ───────────────────────────────────────────────────────
    parser.add_argument(
        "--idf", dest="idf_weights", type=Path, default=None,
        help="IDF weights JSON file (required when --weighting idf).",
    )
    parser.add_argument(
        "--query-file", dest="query_file", type=Path, default=None,
        help="Text file with one query per line (alternative to --query).",
    )

    # ── Grid parameters ───────────────────────────────────────────────────────
    parser.add_argument(
        "--grid-size", dest="grid_size", type=int, default=16,
        help="Side length of the N×N semantic grid. Must match Steps 3–5.",
    )

    # ── Phrase extraction flags (must mirror Step 1 settings) ─────────────────
    parser.add_argument(
        "--no-spacy", dest="no_spacy", action="store_true", default=False,
        help="Force NLTK fallback extraction (use if Step 1 used --no-spacy).",
    )
    parser.add_argument(
        "--remove-verbs", dest="remove_verbs", action="store_true", default=True,
        help="Strip verb tokens before lemmatisation (default: on, mirrors Step 1).",
    )
    parser.add_argument(
        "--keep-verbs", dest="remove_verbs", action="store_false",
        help="Keep verb forms (use only if Step 1 used --keep-verbs).",
    )
    parser.add_argument(
        "--no-filter-generic", dest="filter_generic", action="store_false",
        default=True,
        help="Keep generic single words during expansion (mirrors Step 1 flag).",
    )
    parser.add_argument(
        "--min-word-length", dest="min_word_length", type=int, default=3,
        help="Minimum token character length kept after expansion.",
    )

    # ── Weighting / normalisation ──────────────────────────────────────────────
    parser.add_argument(
        "--weighting", type=str, default="uniform",
        choices=["uniform", "frequency", "idf"],
        help="Phrase weighting strategy for fingerprint aggregation.",
    )
    parser.add_argument(
        "--normalization", type=str, default="l2",
        choices=["l2", "l1", "binary", "none"],
        help="Query fingerprint normalisation method.",
    )

    # ── Spreading ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--spreading-steps", dest="spreading_steps", type=int, default=1,
        help="Spreading radius in Z-order grid (0 to disable).",
    )
    parser.add_argument(
        "--spreading-decay", dest="spreading_decay", type=float, default=0.5,
        help="Decay factor per step during spreading.",
    )
    parser.add_argument(
        "--normalize-after-spreading", dest="normalize_after_spreading",
        action="store_true", default=False,
        help="L2-normalise fingerprint after spreading.",
    )

    # ── Ranking ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--top-k", dest="top_k", type=int, default=10,
        help="Maximum number of results to return per query.",
    )
    parser.add_argument(
        "--min-similarity", dest="min_similarity", type=float, default=0.0,
        help="Minimum score threshold for results.",
    )
    parser.add_argument(
        "--use-batch", dest="use_batch", action="store_true", default=True,
        help="Accepted for compatibility; dot-product ranking is always used.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", dest="output_json", type=Path, default=None,
        help="Save all query results to this JSON file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed query analysis and corpus statistics.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point for Step 6 of the Semantic Folding pipeline.

    Loads phrase and document fingerprints, optionally loads IDF weights,
    collects queries from ``--query`` and/or ``--query-file``, processes
    each query via :func:`process_query`, displays results, and optionally
    saves all results to a JSON file.

    Exit codes
    ----------
    0   — Success (all queries processed; some may have had no results).
    1   — Fatal error (missing input directories or empty fingerprint dicts).
    """
    args = parse_args()

    # ── Validate mutually dependent arguments ─────────────────────────────────
    if not args.query and not args.query_file:
        logger.error("Either --query or --query-file must be provided.")
        sys.exit(1)

    if args.weighting == "idf" and not args.idf_weights:
        logger.warning(
            "IDF weighting requested but --idf not provided → "
            "falling back to uniform."
        )
        args.weighting = "uniform"

    # ── Validate input directories ────────────────────────────────────────────
    if not args.phrase_fp_dir.exists():
        logger.error(f"Phrase fingerprint dir not found: {args.phrase_fp_dir}")
        sys.exit(1)

    if not args.doc_fp_dir.exists():
        logger.error(f"Document fingerprint dir not found: {args.doc_fp_dir}")
        sys.exit(1)

    # ── Load phrase fingerprints ───────────────────────────────────────────────
    try:
        phrase_fingerprints = load_phrase_fingerprints_sparse(
            args.phrase_fp_dir, args.grid_size
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Failed to load phrase fingerprints: {exc}")
        sys.exit(1)

    if not phrase_fingerprints:
        logger.error(
            "Phrase fingerprints dict is empty — check Step 4 output."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(phrase_fingerprints)} phrase fingerprints.")

    # ── Load IDF weights ───────────────────────────────────────────────────────
    idf_weights: Optional[Dict[str, float]] = None
    if args.weighting == "idf":
        if args.idf_weights and args.idf_weights.exists():
            try:
                with open(args.idf_weights, "r", encoding="utf-8") as fh:
                    idf_weights = json.load(fh)
                logger.info(f"Loaded IDF weights for {len(idf_weights)} phrases.")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    f"Failed to read IDF weights ({exc}) → falling back to uniform."
                )
                args.weighting = "uniform"
        else:
            logger.warning(
                f"IDF weights file not found: {args.idf_weights} "
                f"→ falling back to uniform."
            )
            args.weighting = "uniform"

    # ── Load document fingerprints ─────────────────────────────────────────────
    try:
        doc_fingerprints, doc_metadata = load_document_fingerprints(args.doc_fp_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Failed to load document fingerprints: {exc}")
        sys.exit(1)

    if not doc_fingerprints:
        logger.error(
            "Document fingerprints dict is empty — check Step 5 output."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(doc_fingerprints)} document fingerprints.")

    # ── Collect queries ────────────────────────────────────────────────────────
    queries: List[str] = []

    if args.query:
        queries.append(args.query.strip())

    if args.query_file:
        if not args.query_file.exists():
            logger.error(f"Query file not found: {args.query_file}")
            sys.exit(1)
        try:
            with open(args.query_file, "r", encoding="utf-8") as fh:
                file_queries = [ln.strip() for ln in fh if ln.strip()]
            queries.extend(file_queries)
            logger.info(
                f"Loaded {len(file_queries)} queries from {args.query_file}."
            )
        except OSError as exc:
            logger.error(f"Could not read query file: {exc}")
            sys.exit(1)

    if not queries:
        logger.error("No queries to process.")
        sys.exit(1)

    # ── Process queries ────────────────────────────────────────────────────────
    all_results = []

    for i, query in enumerate(queries, 1):
        logger.info(f"[{i}/{len(queries)}] Processing: {query!r}")

        results, metadata = process_query(
            query,
            phrase_fingerprints,
            doc_fingerprints,
            args,
            idf_weights,
        )

        if "error" in metadata:
            logger.error(
                f"Query [{i}] failed — {metadata['error']}: {query!r}"
            )
            if args.verbose:
                missing = (
                    metadata
                    .get("query_construction", {})
                    .get("missing_phrases", [])
                )
                if missing:
                    logger.debug(f"  OOV phrases: {missing}")
        else:
            display_results(
                results,
                query,
                metadata["query_construction"],
                metadata["ranking"],
                doc_metadata = doc_metadata,
                verbose      = args.verbose,
            )

        all_results.append({
            "query":    query,
            "results":  [(doc_id, float(score)) for doc_id, score in results],
            "metadata": metadata,
        })

    # ── Save output ────────────────────────────────────────────────────────────
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(args.output_json, "w", encoding="utf-8") as fh:
                json.dump(all_results, fh, indent=2, ensure_ascii=False)
            logger.success(f"Results saved → {args.output_json}")
        except OSError as exc:
            logger.error(f"Failed to write output file: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
