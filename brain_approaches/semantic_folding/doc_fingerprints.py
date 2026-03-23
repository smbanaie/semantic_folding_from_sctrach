#!/usr/bin/env python3
"""
doc_fingerprints.py — Step 5 of the Semantic Folding Pipeline

Aggregates phrase-level sparse fingerprints (Step 4) into document-level
Sparse Distributed Representations (SDRs) using TF-IDF weighted union,
then sparsifies via Morton (Z-order) curve thresholding.

Pipeline position
-----------------
Step 1  phrase_extractor.py   → phrases.txt
Step 2  term_context.py       → phrase_context_matrix.*
Step 3  semantic_space.py     → grid layout
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
    python doc_fingerprints.py \\
        --corpus      data/corpus.txt \\
        --phrases     outputs/run/phrases.txt \\
        --fingerprints outputs/run/phrase_fingerprints \\
        --output-dir  outputs/run/doc_fingerprints \\
        --grid-size   16 \\
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
    expand_phrases,
    export_fingerprints_to_numpy,
    is_valid_phrase_structure,
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
    phrase_fps      : Dict[str, np.ndarray],
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
    (Step 5):

    Stage 1 — Extraction + normalisation
        ``extract_and_normalize_phrases()`` (from ``phrase_extractor.py``)
        applies spaCy noun-chunk / NER / compound-noun detection (or the
        NLTK n-gram fallback), then pipes each candidate through
        ``lib.normalize_phrase()`` (lowercase → stop-word removal →
        optional verb removal → lemmatisation → POS validation).

    Stage 2 — Sub-phrase expansion
        ``lib.expand_phrases()`` generates all meaningful sub-phrases from
        each normalised phrase (bigrams and trigrams from longer phrases,
        individual tokens from shorter ones), mirroring the expansion
        logic in ``process_corpus_with_expansion``.  Without this stage a
        document containing "deep neural network" would *not* activate the
        fingerprints for "neural network" or "neural", even though those
        sub-phrases exist in the vocabulary built during Step 1.

    Stage 3 — Vocabulary filter
        Only phrases present as keys in *phrase_fps* are retained.
        This is equivalent to the ``min_freq`` filter applied during
        Step 1: sub-phrases that did not survive frequency pruning are
        silently dropped here too.

    Parameters
    ----------
    text : str
        Raw document text (the second field of a corpus CSV line).
    phrase_fps : Dict[str, np.ndarray]
        Phrase → fingerprint mapping loaded from Step 4 output.
        Acts as the vocabulary: only phrases in this dict are returned.
    use_spacy : bool, optional
        Forward to ``extract_and_normalize_phrases``.  Must match the
        flag used during Step 1 to ensure identical tokenisation.
        Default: ``True``.
    remove_verbs : bool, optional
        Forward to ``extract_and_normalize_phrases`` and then to
        ``lib.normalize_phrase``.  When ``True`` (default), all verb
        forms are stripped before lemmatisation, consistent with the
        Step-1 default.
    filter_generic : bool, optional
        Forward to ``lib.expand_phrases``.  When ``True`` (default),
        generic single-word tokens (stop words, very short words,
        numerics) are excluded from the expansion output, consistent
        with the Step-1 default.
    min_word_length : int, optional
        Minimum character length for single-word tokens kept after
        expansion.  Forward to ``lib.expand_phrases``.  Default: ``3``.

    Returns
    -------
    List[str]
        Vocabulary-matched, normalised phrase strings.
        **Duplicates are preserved** so that the caller can build a
        term-frequency count (a phrase appearing via multiple expansion
        paths is counted multiple times).

    Notes
    -----
    - If neither extraction nor expansion yields any vocabulary match the
      function returns an empty list and emits a DEBUG-level log line.
    - The function is intentionally *stateless*: it does not cache
      results or modify any shared data structure, making it safe to
      call from parallel workers in future.

    Examples
    --------
    Suppose the vocabulary contains ``{"neural network": fp1,
    "deep neural": fp2, "neural": fp3}``.

    >>> phrases = extract_phrases_from_doc(
    ...     "The model uses a deep neural network.",
    ...     phrase_fps=vocab,
    ...     use_spacy=True,
    ...     remove_verbs=True,
    ... )
    >>> sorted(set(phrases))
    ['deep neural', 'neural', 'neural network']
    """
    # ── Stage 1: extraction + normalisation ──────────────────────────────────
    candidates: Set[str] = extract_and_normalize_phrases(
        text,
        use_spacy    = use_spacy,
        remove_verbs = remove_verbs,
    )

    if not candidates:
        logger.debug("No phrases extracted from text snippet: %r...", text[:80])
        return []

    # ── Stage 2: sub-phrase expansion ────────────────────────────────────────
    # expand_phrases() returns a *sorted deduplicated* list, which is fine
    # here because we perform vocab filtering next and duplicates from
    # different expansion paths are intentionally preserved via the list
    # comprehension below.
    expanded: List[str] = expand_phrases(
        list(candidates),
        filter_generic  = filter_generic,
        min_word_length = min_word_length,
    )

    # ── Stage 3: vocabulary filter ────────────────────────────────────────────
    matched: List[str] = [p for p in expanded if p in phrase_fps]

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
    """
    Convert a set of 2-D grid coordinates to a flat ``(1, grid_size²)`` CSR matrix.

    Each ``(row, col)`` pair is mapped to the linear index
    ``row * grid_size + col`` and the corresponding entry is set to ``1.0``.
    Coordinates outside ``[0, grid_size)`` in either dimension are silently
    ignored to guard against off-by-one errors from upstream grid mapping.

    Parameters
    ----------
    coords : Set[Tuple[int, int]]
        Collection of ``(row, col)`` pairs representing active cells on the
        semantic grid.  Duplicate coordinates are collapsed to a single
        activation (lil_matrix assignment is idempotent for the same index).
    grid_size : int
        Side length of the square grid (e.g. ``16`` for a 16 × 16 grid).
        The output vector length is ``grid_size * grid_size``.

    Returns
    -------
    csr_matrix
        Shape ``(1, grid_size²)``, dtype ``float32``.  Active cells contain
        ``1.0``; all other entries are structurally zero (sparse).

    Examples
    --------
    >>> m = coords_to_csr({(0, 0), (1, 2)}, grid_size=4)
    >>> m.shape
    (1, 16)
    >>> m.nnz
    2
    """
    n   = grid_size * grid_size
    mat = lil_matrix((1, n), dtype=np.float32)
    for (r, c) in coords:
        if 0 <= r < grid_size and 0 <= c < grid_size:
            mat[0, r * grid_size + c] = 1.0
    return mat.tocsr()


# ---------------------------------------------------------------------------
# Single-document fingerprint builder
# ---------------------------------------------------------------------------

def build_document_fingerprint(
    doc_text            : str,
    phrase_fingerprints : Dict[str, csr_matrix],
    idf_weights         : Dict[str, float],
    grid_size           : int,
    use_spacy           : bool = True,
    remove_verbs        : bool = True,
    filter_generic      : bool = True,
    min_word_length     : int  = 3,
) -> Optional[csr_matrix]:
    """
    Build a raw (un-sparsified) TF-IDF weighted fingerprint for one document.

    The function accumulates phrase fingerprint vectors into a single
    document-level vector using a weighted sum:

    .. math::

        \\mathbf{f}_{\\text{doc}} = \\sum_{p \\in P(d)} \\text{tf}(p, d)
        \\cdot \\text{idf}(p) \\cdot \\mathbf{f}_p

    where $P(d)$ is the multiset of vocabulary-matched phrases extracted
    from document $d$, $\\mathbf{f}_p$ is the phrase fingerprint vector
    from Step 4, and $\\text{tf}(p, d)$ is the within-document term
    frequency derived from phrase expansion (a phrase that is reachable
    via multiple expansion paths contributes proportionally more weight).

    Steps
    -----
    1. **Phrase extraction** — delegate to :func:`extract_phrases_from_doc`
       which applies the three-stage pipeline (normalisation → expansion →
       vocab filter) identical to Step 1.
    2. **TF counting** — count occurrences of each phrase in the matched
       list (duplicates from expansion paths act as natural TF boosts).
    3. **Weighted accumulation** — for each phrase add
       ``tf * idf(phrase) * phrase_vector`` to the accumulator.
    4. **Return** — return the accumulator as a CSR matrix, or ``None``
       if no vocabulary phrase was found in the document.

    Parameters
    ----------
    doc_text : str
        Raw document text (the value field from the corpus CSV).
    phrase_fingerprints : Dict[str, csr_matrix]
        Phrase → SDR vector mapping from Step 4.  Keys are normalised
        phrase strings; values are ``(1, grid_size²)`` CSR matrices or
        1-D NumPy arrays.
    idf_weights : Dict[str, float]
        Phrase → IDF score mapping computed by
        ``lib.compute_idf_weights``.  Phrases absent from this dict
        receive a default weight of ``1.0``.
    grid_size : int
        Side length of the square semantic grid.  Determines the output
        vector length ``grid_size²``.
    use_spacy : bool, optional
        Forwarded to :func:`extract_phrases_from_doc`.  Must match the
        setting used in Step 1.  Default: ``True``.
    remove_verbs : bool, optional
        Forwarded to :func:`extract_phrases_from_doc` and ultimately to
        ``lib.normalize_phrase``.  Default: ``True`` (consistent with
        Step-1 default).
    filter_generic : bool, optional
        Forwarded to :func:`extract_phrases_from_doc` →
        ``lib.expand_phrases``.  Default: ``True``.
    min_word_length : int, optional
        Forwarded to :func:`extract_phrases_from_doc` →
        ``lib.expand_phrases``.  Default: ``3``.

    Returns
    -------
    csr_matrix or None
        ``(1, grid_size²)`` float32 CSR matrix containing the weighted
        sum of matched phrase vectors, or ``None`` if no vocabulary
        phrase could be extracted from *doc_text*.

    Notes
    -----
    - The returned matrix is *not* sparsified or normalised; those steps
      are applied by :func:`sparsify_to_sdr` and
      ``lib.normalize_fingerprint`` in the main pipeline loop.
    - Both ``csr_matrix`` and 1-D ``np.ndarray`` phrase vectors are
      handled transparently so the function is compatible with either
      storage format produced by Step 4.
    """
    n   = grid_size * grid_size
    acc = lil_matrix((1, n), dtype=np.float32)

    # ── Stage 1–3 via extract_phrases_from_doc (with expansion) ──────────────
    matched_phrases = extract_phrases_from_doc(
        text            = doc_text,
        phrase_fps      = phrase_fingerprints,
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
        vec = phrase_fingerprints.get(phrase)
        if vec is None:
            continue

        weight = term_freq * idf_weights.get(phrase, 1.0)

        if isinstance(vec, np.ndarray):
            flat = vec.flatten()[:n]
            acc[0, : len(flat)] += weight * flat
        else:
            # csr_matrix path
            acc += weight * vec

        hits += 1

    if hits == 0:
        return None

    return acc.tocsr()


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

    Retains only the top ``top_percent`` fraction of cells (by activation
    value) using Morton (Z-order) curve ordering as a tie-breaking
    criterion, then zeros the rest.  This is a thin wrapper around
    ``lib.sparsify_fingerprint`` that translates the percentage parameter
    into an absolute bit count.

    The Morton ordering ensures that spatially adjacent cells on the 2-D
    semantic grid are treated as neighbours during tie-breaking, preserving
    the topographic structure of the grid layout produced in Step 3.

    Parameters
    ----------
    fingerprint : csr_matrix
        Raw ``(1, grid_size²)`` weighted fingerprint produced by
        :func:`build_document_fingerprint`.
    top_percent : float
        Fraction of grid cells to keep active, e.g. ``0.1`` keeps the
        top 10 % of cells.  The absolute count is computed as
        ``max(1, round(top_percent * grid_size²))``.
    grid_size : int
        Side length of the square semantic grid.  Used to derive the
        absolute ``top_k`` threshold and passed to ``sparsify_fingerprint``
        for Morton index computation.

    Returns
    -------
    csr_matrix
        Sparsified ``(1, grid_size²)`` SDR.  The number of non-zero
        entries is at most ``top_k`` (may be fewer if the input itself
        has fewer active cells).

    Examples
    --------
    >>> sdr = sparsify_to_sdr(raw_fp, top_percent=0.1, grid_size=16)
    >>> sdr.nnz <= int(round(0.1 * 16 * 16))
    True
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
    phrases_path      : Path,
    fingerprints_path : Path,
    grid_size         : int   = 16,
    top_percent       : float = 0.1,
    min_freq          : int   = 1,
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

    This is the top-level orchestrator for Step 5.  It loads all required
    inputs, iterates over the corpus, builds one SDR per document, and
    returns the results as a dense matrix together with metadata.  Saving
    is intentionally delegated to :func:`write_outputs` so the function
    remains unit-testable without touching the filesystem.

    Pipeline stages inside this function
    -------------------------------------
    1. **Load phrase inventory** from *phrases_path* (Step 1 output).
    2. **Load phrase fingerprints** from *fingerprints_path* (Step 4 output)
       and filter to the loaded inventory.
    3. **Load corpus** (``context_id,context_text`` CSV) via
       ``lib.load_contexts_dict``.
    4. **Compute IDF weights** for all vocabulary phrases over the corpus.
    5. **Per-document loop**:
       a. :func:`build_document_fingerprint` → raw TF-IDF weighted vector.
       b. :func:`sparsify_to_sdr` → fixed-density SDR (Morton ordering).
       c. ``lib.normalize_fingerprint`` → optional L1/L2/max normalisation.
    6. **Diversity report** (optional) via ``lib.compute_fingerprint_diversity``.
    7. **Stack** sparse dict into a dense ``(n_docs, grid_size²)`` matrix.

    Parameters
    ----------
    corpus_path : Path
        CSV corpus file: one line per document, format
        ``context_id,context_text``.  Must be the same file used in Step 1.
    phrases_path : Path
        ``phrases.txt`` produced by Step 1 (``phrase_extractor.py``).
        Format: ``phrase:count`` per line.
    fingerprints_path : Path
        Directory produced by Step 4 (``phrase_fingerprints.py``).
        Must contain ``phrase_fingerprints.npz`` and
        ``phrase_fingerprints_meta.json``.
    grid_size : int, optional
        Side length of the square semantic grid (default: ``16``).
        Must match the value used in Steps 3 and 4.
    top_percent : float, optional
        Fraction of grid cells kept active per document SDR (default:
        ``0.1``, i.e. 10 %).  Must be in ``(0, 1]``.
    min_freq : int, optional
        Minimum phrase frequency threshold applied when loading the
        phrase inventory (default: ``1``).  Raise to ``2`` or higher to
        discard rare phrases.
    normalize : bool, optional
        If ``True`` (default), normalise each SDR after sparsification
        using *normalize_method*.
    normalize_method : str, optional
        Normalisation strategy passed to ``lib.normalize_fingerprint``.
        One of ``"l1"``, ``"l2"`` (default), ``"max"``.
    use_spacy : bool, optional
        Use spaCy for phrase extraction (default: ``True``).  Set to
        ``False`` to force the NLTK n-gram fallback.  **Must match the
        setting used in Step 1.**
    remove_verbs : bool, optional
        Strip verb forms during normalisation (default: ``True``).
        **Must match the setting used in Step 1.**
    filter_generic : bool, optional
        Remove generic single words during expansion (default: ``True``).
        **Must match the setting used in Step 1.**
    min_word_length : int, optional
        Minimum character length for single-word tokens kept after
        expansion (default: ``3``).  **Must match the setting used in
        Step 1.**
    compute_diversity : bool, optional
        If ``True``, compute and log pairwise fingerprint diversity
        statistics after the main loop (default: ``False``).  Can be
        slow for large corpora.
    diversity_sample : int, optional
        Maximum number of documents sampled for the diversity computation
        (default: ``100``).

    Returns
    -------
    fingerprint_matrix : np.ndarray
        Shape ``(n_docs, grid_size²)``, dtype ``float32``.  Each row is
        the SDR for the corresponding document.
    doc_index_map : Dict[str, int]
        ``doc_id → row_index`` mapping into *fingerprint_matrix*.
    stats : dict
        Run-level statistics with keys:

        - ``total_documents``    — documents in the corpus
        - ``fingerprinted_docs`` — documents with ≥1 vocabulary phrase
        - ``skipped_docs``       — documents with no vocabulary match
        - ``skip_rate_pct``      — ``skipped_docs / total_documents * 100``
        - ``vector_size``        — ``grid_size²``
        - ``avg_active_bits``    — mean ``nnz`` across SDRs
        - ``grid_size``          — the *grid_size* parameter
        - ``top_percent``        — the *top_percent* parameter

    Raises
    ------
    SystemExit(1)
        If no phrase fingerprints remain after the inventory filter
        (mismatched pipeline run artefacts).

    Notes
    -----
    - Documents that yield no vocabulary-matched phrases after extraction
      and expansion are skipped entirely (not assigned a row in the
      output matrix).  The ``skipped_docs`` statistic tracks these.
    - The function logs a WARNING if spaCy is requested but unavailable,
      prompting the user to verify consistency with Step 1.
    """

    # ── 1. Phrase inventory ───────────────────────────────────────────────────
    logger.info("Loading phrase inventory from %s ...", phrases_path)
    phrase_tuples = load_phrases(phrases_path, min_freq=min_freq)
    phrase_set    = {p for p, _ in phrase_tuples}
    logger.info("  %d phrases loaded (min_freq=%d)", len(phrase_set), min_freq)

    # ── 2. Phrase fingerprints (Step 4) ───────────────────────────────────────
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

    # ── 3. Corpus ─────────────────────────────────────────────────────────────
    logger.info("Loading corpus from %s ...", corpus_path)
    contexts = load_contexts_dict(corpus_path)
    logger.info("  %d documents loaded", len(contexts))

    # ── 4. IDF weights ────────────────────────────────────────────────────────
    logger.info("Computing IDF weights ...")
    idf_weights = compute_idf_weights(
        list(phrase_fingerprints.keys()),
        list(contexts.values()),
    )

    # ── 5. Per-document build loop ────────────────────────────────────────────
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

        # 5a. Raw TF-IDF weighted fingerprint
        raw_fp = build_document_fingerprint(
            doc_text            = doc_text,
            phrase_fingerprints = phrase_fingerprints,
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

        # 5b. Sparsify to SDR
        sparse_fp = sparsify_to_sdr(
            raw_fp,
            top_percent = top_percent,
            grid_size   = grid_size,
        )

        # 5c. Optional normalisation
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

    # ── 6. Optional diversity report ─────────────────────────────────────────
    if compute_diversity and sparse_fps:
        logger.info(
            "Computing fingerprint diversity (sample=%d) ...", diversity_sample
        )
        diversity = compute_fingerprint_diversity(
            sparse_fps, sample_size=diversity_sample
        )
        for metric, value in sorted(diversity.items()):
            logger.info("  %-30s = %.6f", metric, value)

    # ── 7. Stack to dense matrix ──────────────────────────────────────────────
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
    """
    Parse and return command-line arguments for the Step-5 CLI.

    All parameters that affect phrase extraction (``--no-spacy``,
    ``--keep-verbs``, ``--no-filter-generic``, ``--min-word-length``)
    **must** be set to the same values used when running Step 1
    (``phrase_extractor.py``) to guarantee consistent phrase
    representations across the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace consumed by :func:`main`.
    """
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
        "--phrases", type=Path, required=True,
        help="phrases.txt produced by Step 1 (phrase_extractor.py).",
    )
    parser.add_argument(
        "--fingerprints", type=Path, required=True,
        help="Step 4 output directory (contains phrase_fingerprints.npz + _meta.json).",
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
    parser.add_argument(
        "--min-freq", type=int, default=1, dest="min_freq",
        help="Minimum phrase frequency when loading the phrase inventory.",
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
        "--keep-verbs", action="store_true", default=True,
        help="keep verb forms during normalisation (default: on, mirrors Step 1).",
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
    """
    CLI entry point for Step 5 of the Semantic Folding pipeline.

    Parses arguments, runs :func:`build_doc_fingerprints`, and writes
    outputs via :func:`write_outputs`.  Exits with code 4 on I/O failure.
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Semantic Folding — Step 5: Document Fingerprints")
    logger.info("=" * 60)
    logger.info("  corpus          : %s", args.corpus)
    logger.info("  phrases         : %s", args.phrases)
    logger.info("  fingerprints    : %s", args.fingerprints)
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
        phrases_path      = args.phrases,
        fingerprints_path = args.fingerprints,
        grid_size         = args.grid_size,
        top_percent       = args.top_percent,
        min_freq          = args.min_freq,
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
