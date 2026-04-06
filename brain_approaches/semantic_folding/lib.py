"""
lib.py - Core Utilities for Semantic Folding Pipeline

This module provides essential utilities for the semantic folding pipeline, including:
- Text normalization and lemmatization with POS-aware processing
- Phrase expansion and filtering strategies
- File I/O operations for phrases, contexts, and fingerprints
- Sparse fingerprint representation handling

The module ensures consistency across all pipeline stages by providing
centralized implementations of common operations like phrase normalization,
word boundary detection, and fingerprint loading.

Key Design Principles:
- Cached lemmatization for performance (@lru_cache)
- POS-aware text processing for semantic accuracy
- Sparse representation support for memory efficiency
- Consistent normalization across all pipeline stages

Author: [Your Name]
Date: 2026-03-18
"""

import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy.sparse import hstack, csr_matrix, lil_matrix
from rich import print
from loguru import logger
import numpy as np
from functools import lru_cache
import json, os


nltk.data.path.insert(0, 'C:\\nltk_data')
os.environ['NLTK_DATA'] = r'C:\nltk_data'

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
en_stop_words = set(stopwords.words('english'))

# ============================================================================
# CORE NLP UTILITIES
# ============================================================================

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Convert Penn Treebank POS tag to WordNet POS tag.
    
    WordNet uses a simplified POS tag set (NOUN, VERB, ADJ, ADV) while
    Penn Treebank uses a more granular set. This mapping enables accurate
    lemmatization by providing the correct POS context.
    
    Args:
        treebank_tag: Penn Treebank POS tag (e.g., 'NN', 'VBD', 'JJ')
    
    Returns:
        WordNet POS tag (wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV)
        Defaults to wordnet.NOUN for unrecognized tags
    
    Examples:
        >>> get_wordnet_pos('NN')
        'n'  # wordnet.NOUN
        >>> get_wordnet_pos('VBD')
        'v'  # wordnet.VERB
        >>> get_wordnet_pos('JJ')
        'a'  # wordnet.ADJ
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


@lru_cache(maxsize=10000)
def lemmatize_token(word: str, pos_tag: str) -> str:
    """
    Lemmatize a single token with POS-aware processing and caching.
    
    Lemmatization reduces words to their base form (lemma) while considering
    their part-of-speech. Caching significantly improves performance for
    repeated tokens across large corpora.
    
    Args:
        word: Input word to lemmatize
        pos_tag: Penn Treebank POS tag for the word
    
    Returns:
        Lemmatized form of the word in lowercase
    
    Examples:
        >>> lemmatize_token('running', 'VBG')
        'run'
        >>> lemmatize_token('better', 'JJR')
        'good'
        >>> lemmatize_token('mice', 'NNS')
        'mouse'
    
    Note:
        The @lru_cache decorator caches up to 10,000 unique (word, pos_tag)
        pairs, providing substantial speedup for corpus-level processing.
    """
    pos = get_wordnet_pos(pos_tag)
    return lemmatizer.lemmatize(word.lower(), pos=pos)


def is_generic_word(word: str, min_length: int = 3) -> bool:
    """
    Determine if a single word is too generic to carry semantic meaning.
    
    Generic words are filtered out during phrase expansion to maintain
    semantic quality. A word is considered generic if it meets any of:
    - Too short (< min_length characters)
    - Common stop word (articles, prepositions, etc.)
    - Pure numeric string
    
    Args:
        word: Input word to evaluate
        min_length: Minimum character length threshold (default: 3)
    
    Returns:
        True if word is generic and should be filtered, False otherwise
    
    Examples:
        >>> is_generic_word('the')
        True  # stop word
        >>> is_generic_word('ai')
        True  # too short (< 3 chars)
        >>> is_generic_word('123')
        True  # numeric
        >>> is_generic_word('algorithm')
        False  # meaningful content word
    """
    if len(word) < min_length:
        return True
    if word in en_stop_words:
        return True
    if word.isdigit():
        return True
    return False


def is_valid_phrase_structure(tagged_tokens: List[Tuple[str, str]]) -> bool:
    """
    Validate phrase structure based on POS tag patterns.
    
    Valid phrases must contain meaningful content (nouns/adjectives) and
    avoid degenerate patterns like pure verb phrases or stop word sequences.
    
    Valid patterns include:
    - Noun phrases: 'machine learning', 'neural network'
    - Adjective phrases: 'deep', 'convolutional'
    - Mixed noun-adjective: 'artificial intelligence', 'semantic space'
    
    Invalid patterns include:
    - Pure verb phrases: 'is running', 'has been'
    - Pure stop words: 'the of', 'in a'
    
    Args:
        tagged_tokens: List of (word, POS_tag) tuples from pos_tag()
    
    Returns:
        True if phrase structure is valid, False otherwise
    
    Examples:
        >>> is_valid_phrase_structure([('neural', 'JJ'), ('network', 'NN')])
        True
        >>> is_valid_phrase_structure([('is', 'VBZ'), ('running', 'VBG')])
        False  # pure verb phrase
    """
    if not tagged_tokens:
        return False
    
    pos_tags = [tag for _, tag in tagged_tokens]
    
    # Reject pure verb phrases
    if all(tag.startswith('V') for tag in pos_tags):
        return False
    
    # Require at least one noun or adjective
    has_content = any(tag.startswith(('N', 'J')) for tag in pos_tags)
    return has_content


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_phrase(text: str, remove_verbs: bool = True) -> Optional[str]:
    """
    Normalize a phrase with consistent lemmatization and filtering.
    
    This is the core normalization function used across all pipeline stages
    to ensure consistent phrase representation. The normalization process:
    
    1. Lowercase and clean punctuation (preserving hyphens)
    2. Tokenize into words
    3. POS tag for context-aware processing
    4. Filter stop words and optionally verbs
    5. Lemmatize remaining tokens
    6. Validate final phrase structure
    
    Args:
        text: Input phrase or text segment
        remove_verbs: If True, filter out all verb forms (default: True)
                     Set to False for context text processing
    
    Returns:
        Normalized phrase string, or None if phrase becomes invalid
        after filtering
    
    Examples:
        >>> normalize_phrase('Machine Learning Algorithms')
        'machine learning algorithm'
        >>> normalize_phrase('The cats are running quickly')
        'cat quickly'  # verbs removed, lemmatized
        >>> normalize_phrase('is the', remove_verbs=True)
        None  # becomes empty after filtering
    
    Note:
        This function should be used consistently across:
        - Phrase extraction (phrase_extractor.py)
        - Context processing (term_context.py)
        - Fingerprint generation (phrase_fingerprints.py, doc_fingerprints.py)
        - Query processing (query_processing.py)
    """
    # Clean text
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)  # Keep hyphens for compound words
    
    # Tokenize and tag
    tokens = word_tokenize(text)
    if not tokens:
        return None
    
    tagged_tokens = pos_tag(tokens)
    
    # Filter and lemmatize
    # Filter and lemmatize
    processed = []
    valid_tagged_tokens = [] # ADD THIS to safely track the tags of kept words
    
    for word, tag in tagged_tokens:
        # Skip stop words
        if word in en_stop_words:
            continue
        
        # Skip verbs if requested
        if remove_verbs and tag.startswith('V'):
            continue
        
        # Skip non-alphabetic tokens
        if not word.isalpha():
            continue
        
        # Lemmatize
        lemma = lemmatize_token(word, tag)
        processed.append(lemma)
        valid_tagged_tokens.append((lemma, tag)) # Track the lemma and its tag
    
    if not processed:
        return None
    
    # Validate structure using our safely tracked list
    if not is_valid_phrase_structure(valid_tagged_tokens):
        return None
    
    return ' '.join(processed)



# ============================================================================
# PHRASE EXPANSION
# ============================================================================
from typing import List, Optional, Set
# Assuming these helpers are in the same lib.py file
# from .utils import normalize_phrase, is_generic_word 

def expand_phrases(phrases: List[str], 
                   context_text: str,
                   filter_generic: bool = True,
                   min_word_length: int = 3) -> List[str]:
    """
    Expand multi-word phrases and validate them against the source context.
    
    This function generates all meaningful sub-phrases from a list of longer 
    phrases and, crucially, verifies that each sub-phrase actually exists as a 
    substring within the provided `context_text`. This prevents the creation of 
    "phantom phrases"—combinations that are syntactically possible but do not 
    empirically appear in the source document.

    This validation step is critical for building a high-quality, grounded
    vocabulary for the semantic folding pipeline.
    
    Expansion rules by phrase length (applied after normalization):
    - 2-word phrases → add individual words (if valid)
    - 3-word phrases → add all 2-word combinations + individual words (if valid)
    - 4+ word phrases → add all 3-word, 2-word, and 1-word combinations (if valid)
    
    Args:
        phrases: List of input phrases to expand, extracted from a single context.
        context_text: The raw source text of the context from which `phrases` 
                      were extracted. Used to validate sub-phrase existence.
        filter_generic: If True, remove generic single words (default: True).
        min_word_length: Minimum character length for single words (default: 3).
    
    Returns:
        A sorted list of unique, normalized, and validated phrases that exist
        within the source context_text.
    
    Examples:
        >>> context = "The field of machine learning includes deep neural networks."
        >>> expand_phrases(['deep neural networks'], context)
        ['deep neural', 'deep neural network', 'neural network'] 
        # Note: 'deep', 'neural', 'networks' might be filtered or not present
        # depending on normalization and other rules.

        >>> context = "We must change negative thought patterns."
        >>> expand_phrases(['change negative thought patterns'], context)
        ['change negative', 'change negative thought', 'negative thought', 
         'negative thought pattern', 'thought pattern']
         # Assumes 'patterns' lemmatizes to 'pattern'.
    """
    expanded_and_validated = set()
    lower_context = context_text.lower()
    
    for phrase in phrases:
        # Normalize the full phrase first
        normalized = normalize_phrase(phrase, remove_verbs=True)
        if not normalized:
            continue
        
        # A temporary set to hold all combinatorial candidates for this one phrase
        candidates = set()
        candidates.add(normalized) # The full phrase is always a candidate
        
        words = normalized.split()
        n = len(words)
        
        # Generate all possible n-gram candidates combinatorially
        if n == 2:
            candidates.update(words)
        elif n == 3:
            for i in range(n - 1): candidates.add(' '.join(words[i:i+2])) # Bigrams
            candidates.update(words) # Unigrams
        elif n >= 4:
            for i in range(n - 2): candidates.add(' '.join(words[i:i+3])) # Trigrams
            for i in range(n - 1): candidates.add(' '.join(words[i:i+2])) # Bigrams
            candidates.update(words) # Unigrams
            
        # --- Validation Step ---
        # Only add candidates that pass all filters and exist in the source text
        for candidate in candidates:
            # Filter 1: Is it a generic single word?
            if ' ' not in candidate: # It's a unigram
                if filter_generic and is_generic_word(candidate, min_word_length):
                    continue
            
            # Filter 2 (Crucial): Does it actually exist in the raw context?
            if candidate in lower_context:
                expanded_and_validated.add(candidate)
                
    return sorted(list(expanded_and_validated))


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def load_phrases(phrases_path: Path, min_freq: int = 0) -> List[Tuple[str, int]]:
    """
    Load phrases with frequencies from phrase inventory file.
    
    Expected file format (one phrase per line):
        phrase_text:frequency
    
    Example:
        machine learning:150
        neural network:89
        deep learning:203
    
    Args:
        phrases_path: Path to phrases file
        min_freq: Minimum frequency threshold (default: 0, no filtering)
    
    Returns:
        List of (phrase, frequency) tuples for phrases meeting threshold
    
    Raises:
        FileNotFoundError: If phrases_path does not exist
        ValueError: If file format is invalid
    
    Note:
        Phrases are NOT normalized during loading. Normalization should
        be applied separately using normalize_phrase() when needed.
    """
    logger.info(f"Loading phrases from: {phrases_path}")
    
    phrases = []
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                phrase, freq_str = line.split(':', 1)
                phrase = phrase.strip()
                try:
                    freq = int(freq_str.strip())
                    if freq >= min_freq and phrase:
                        phrases.append((phrase, freq))
                except ValueError:
                    logger.warning(f"Invalid frequency for phrase: '{line}'")
                    continue
    
    logger.success(f"Loaded {len(phrases)} phrases from: {phrases_path}")
    return phrases


def find_phrase_occurrences(text: str, phrase: str, 
                           use_word_boundaries: bool = True) -> int:
    """
    Count phrase occurrences in text with proper word boundary detection.
    
    Word boundary detection ensures accurate matching by preventing
    false positives from substring matches (e.g., 'cat' should not
    match 'concatenate').
    
    Args:
        text: Input text to search
        phrase: Phrase to search for
        use_word_boundaries: If True, only match complete words (default: True)
    
    Returns:
        Number of occurrences found
    
    Examples:
        >>> find_phrase_occurrences('the cat and the cats', 'cat', True)
        1  # matches 'cat' but not 'cats'
        >>> find_phrase_occurrences('the cat and the cats', 'cat', False)
        2  # matches both 'cat' and 'cats' (substring)
    
    Note:
        Always use word boundaries (use_word_boundaries=True) for accurate
        phrase matching in semantic contexts.
    """
    import re
    
    if use_word_boundaries:
        # Escape special regex characters in phrase
        escaped_phrase = re.escape(phrase)
        # Use word boundaries for accurate matching
        pattern = r'\b' + escaped_phrase + r'\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(matches)
    else:
        # Fallback to simple substring matching
        return text.lower().count(phrase.lower())


def load_contexts(corpus_path: Path) -> List[Tuple[str, str]]:
    """
    Load contexts from corpus file with normalization.
    
    Expected file format (CSV):
        context_id,context_text
    
    Example:
        ctx_0,Machine learning is a subset of artificial intelligence
        ctx_1,Neural networks are inspired by biological neurons
    
    Args:
        corpus_path: Path to corpus file
    
    Returns:
        List of (context_id, normalized_context_text) tuples
    
    Note:
        Context text is normalized using normalize_phrase(remove_verbs=False)
        to preserve verbs, which can be important for context understanding.
    """
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
            
            # Normalize context text (keep verbs for context)
            normalized_text = normalize_phrase(context_text, remove_verbs=False)
            if normalized_text:
                contexts.append((context_id, normalized_text))
    
    logger.success(f"Loaded {len(contexts)} contexts from: {corpus_path}")
    return contexts


def load_contexts_dict(corpus_path: Path) -> Dict[str, str]:
    """
    Load context texts as dictionary mapping context_id to text.
    
    Expected file format (CSV):
        context_id,context_text
    
    Args:
        corpus_path: Path to corpus file
    
    Returns:
        Dictionary mapping context_id -> context_text (not normalized)
    
    Note:
        Unlike load_contexts(), this function does NOT normalize text.
        Use this when you need the original context text.
    """
    logger.info(f"Loading context texts from: {corpus_path}")
    
    contexts = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            
            context_id, context_text = line.split(',', 1)
            contexts[context_id.strip()] = context_text.strip()
    
    logger.success(f"Loaded {len(contexts)} context texts from: {corpus_path}")
    return contexts


# ============================================================================
# FINGERPRINT LOADING UTILITIES
# ============================================================================
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fingerprint Loaders
#  Used by: query_processing.py (Step 6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_fingerprint_matrix(
    npz_path    : Path,
    index_path  : Path,
    npz_key     : str,
    label       : str,                    # "phrase" or "document" — for log messages
    grid_size   : Optional[int] = None,   # if given, column count is validated
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Shared low-level loader for any fingerprint .npz + index-map pair.

    Parameters
    ----------
    npz_path:
        Path to the .npz file containing the dense float32 matrix.
    index_path:
        Path to the JSON file containing {entity_string: row_index}.
    npz_key:
        Key inside the .npz archive that holds the matrix (e.g. "fingerprints").
    label:
        Human-readable entity type used in log/error messages.
    grid_size:
        If provided, validates that matrix columns == grid_size².

    Returns
    -------
    matrix     : np.ndarray  — shape (n_entities, vector_size)
    index_map  : Dict[str, int]

    Raises
    ------
    FileNotFoundError  — if either file is missing.
    KeyError           — if npz_key is absent from the archive.
    ValueError         — if grid_size is given and column count mismatches.
    """
    # ── Validate files exist ─────────────────────────────────────────────────
    for p in (npz_path, index_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {label} fingerprint file not found: {p}"
            )

    # ── Load matrix ──────────────────────────────────────────────────────────
    logger.info(f"Loading {label} fingerprint matrix from: {npz_path}")
    archive = np.load(str(npz_path))

    if npz_key not in archive:
        raise KeyError(
            f"Key '{npz_key}' not found in {npz_path.name}. "
            f"Available keys: {list(archive.keys())}"
        )

    matrix: np.ndarray = archive[npz_key]          # (n_entities, vector_size)
    n_entities, vector_size = matrix.shape
    logger.info(
        f"{label.capitalize()} matrix shape: {matrix.shape} "
        f"(n={n_entities}, vec={vector_size})"
    )

    # ── Optional column-count validation ─────────────────────────────────────
    if grid_size is not None:
        expected = grid_size * grid_size
        if vector_size != expected:
            raise ValueError(
                f"{label.capitalize()} matrix has {vector_size} columns but "
                f"grid_size={grid_size} implies {expected} columns. "
                f"Did you pass the correct --grid-size?"
            )

    # ── Load entity → row-index map ──────────────────────────────────────────
    logger.info(f"Loading {label} index map from: {index_path}")
    with open(index_path, "r", encoding="utf-8") as fh:
        index_map: Dict[str, int] = json.load(fh)

    # ── Row-count sanity check ────────────────────────────────────────────────
    if len(index_map) != n_entities:
        logger.warning(
            f"{label.capitalize()} index map has {len(index_map)} entries "
            f"but matrix has {n_entities} rows — possible misalignment."
        )

    return matrix, index_map

# ─────────────────────────────────────────────────────────────────────────────

def load_phrase_fingerprints_sparse(
    fingerprints_dir : Path,
    grid_size        : int,
) -> Dict[str, np.ndarray]:
    """
    Load phrase fingerprints produced by Step 4 (phrase_fingerprints.py).

    Expected files in fingerprints_dir:
        phrase_fingerprints.npz        — dense float32 matrix,
                                         key "fingerprints",
                                         shape (n_phrases, grid_size²)
        phrase_fingerprints_meta.json  — {phrase_string: row_index}

    Parameters
    ----------
    fingerprints_dir:
        Step 4 output directory (e.g. outputs/run/phrase_fingerprints/).
    grid_size:
        Grid side-length; used to validate matrix column count.

    Returns
    -------
    Dict[str, np.ndarray]
        phrase_string  →  float32 vector of length grid_size².

    Raises
    ------
    FileNotFoundError  — if either expected file is missing.
    ValueError         — if column count != grid_size²,
                         or if index_map references out-of-bound rows.
    """
    fingerprints_dir = Path(fingerprints_dir)

    matrix, index_map = _load_fingerprint_matrix(
        npz_path   = fingerprints_dir / "phrase_fingerprints.npz",
        index_path = fingerprints_dir / "phrase_fingerprints_meta.json",
        npz_key    = "fingerprints",
        label      = "phrase",
        grid_size  = grid_size,
    )

    n_rows = matrix.shape[0]
    n_keys = len(index_map)

    # ── alignment audit ──────────────────────────────────────────────────────
    if n_keys != n_rows:
        logger.warning(
            f"index_map has {n_keys} entries but matrix has {n_rows} rows "
            f"— index map and matrix may be misaligned. "
            f"Only mapped entries will be used."
        )

    # ── out-of-bound row check ────────────────────────────────────────────────
    bad_phrases = {
        phrase: idx
        for phrase, idx in index_map.items()
        if idx < 0 or idx >= n_rows
    }
    if bad_phrases:
        raise ValueError(
            f"index_map contains {len(bad_phrases)} out-of-bound row "
            f"reference(s) for matrix with {n_rows} rows. "
            f"Examples: { {k: v for k, v in list(bad_phrases.items())[:5]} }"
        )

    # ── build output dict — only rows that are mapped ─────────────────────────
    phrase_fps: Dict[str, np.ndarray] = {
        phrase: matrix[idx].astype(np.float32)
        for phrase, idx in index_map.items()
    }

    # ── report unmapped rows (matrix rows with no phrase key) ─────────────────
    mapped_row_indices = set(index_map.values())
    unmapped_rows = [i for i in range(n_rows) if i not in mapped_row_indices]
    if unmapped_rows:
        logger.warning(
            f"{len(unmapped_rows)} matrix row(s) have no corresponding phrase "
            f"key in the index map and will be ignored. "
            f"First few unmapped row indices: {unmapped_rows[:10]}"
        )

    logger.success(
        f"Loaded {len(phrase_fps)} phrase fingerprints "
        f"(grid_size={grid_size}, vector_size={grid_size**2}, "
        f"matrix_rows={n_rows}, mapped={len(phrase_fps)}, "
        f"unmapped_rows={len(unmapped_rows)})."
    )
    return phrase_fps


# ─────────────────────────────────────────────────────────────────────────────

def load_document_fingerprints(
    doc_fp_dir : Path,
) -> Tuple[Dict[str, "csr_matrix"], Dict]:
    """
    Load document fingerprints produced by Step 5 (doc_fingerprints.py).

    Expected files in doc_fp_dir:
        doc_fingerprints.npz         — dense float32 matrix,
                                       key "fingerprints",
                                       shape (n_docs, grid_size²)
        doc_fingerprints_meta.json   — {doc_id: row_index}
        doc_fingerprints_stats.json  — run statistics including grid_size

    Parameters
    ----------
    doc_fp_dir:
        Step 5 output directory (e.g. outputs/run/doc_fingerprints/).

    Returns
    -------
    doc_fingerprints : Dict[str, csr_matrix]
        doc_id  →  sparse row-vector of length grid_size².
    combined_metadata : Dict
        All fields from stats.json plus "grid_size" and "num_docs".

    Raises
    ------
    FileNotFoundError  — if any of the three expected files is missing.
    """
    from scipy.sparse import csr_matrix          # local import — scipy optional

    doc_fp_dir = Path(doc_fp_dir)

    stats_path = doc_fp_dir / "doc_fingerprints_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {stats_path}\n"
            f"Make sure Step 5 completed successfully."
        )

    # ── Read grid_size from stats so _load_fingerprint_matrix can validate ───
    with open(stats_path, "r", encoding="utf-8") as fh:
        stats: Dict = json.load(fh)
    grid_size: int = int(stats.get("grid_size", 16))

    matrix, index_map = _load_fingerprint_matrix(
        npz_path   = doc_fp_dir / "doc_fingerprints.npz",
        index_path = doc_fp_dir / "doc_fingerprints_meta.json",
        npz_key    = "fingerprints",
        label      = "document",
        grid_size  = grid_size,
    )

    # ── Build doc_id → sparse row-vector ─────────────────────────────────────
    doc_fingerprints: Dict[str, "csr_matrix"] = {
        doc_id: csr_matrix(matrix[row_idx].reshape(1, -1))
        for doc_id, row_idx in index_map.items()
    }

    combined_metadata = {
        **stats,
        "grid_size" : grid_size,
        "num_docs"  : len(doc_fingerprints),
    }

    logger.success(
        f"Loaded {len(doc_fingerprints)} document fingerprints "
        f"(grid_size={grid_size})."
    )
    return doc_fingerprints, combined_metadata


def load_phrase_fingerprints_sparse(
    fingerprints_dir : Path,
    grid_size        : int,
) -> Dict[str, np.ndarray]:
    """
    Load phrase fingerprints produced by Step 4 (phrase_fingerprints.py).

    Step 4 writes two files into its output directory:
        phrase_fingerprints.npz        — dense float32 matrix, key "fingerprints",
                                         shape (n_phrases, grid_size * grid_size)
        phrase_fingerprints_meta.json  — { phrase_string: row_index, ... }

    Parameters
    ----------
    fingerprints_dir:
        The Step 4 output DIRECTORY  (e.g. outputs/run/phrase_fingerprints/).
        Both expected files must exist inside it.
    grid_size:
        Grid side length used as a sanity check against the matrix shape.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping  phrase_string  →  float32 vector of length grid_size².

    Raises
    ------
    FileNotFoundError
        If either expected file is missing.
    ValueError
        If the matrix column count does not match grid_size².
    """
    npz_path  = fingerprints_dir / "phrase_fingerprints.npz"
    meta_path = fingerprints_dir / "phrase_fingerprints_meta.json"

    # ── Validate files exist ─────────────────────────────────────────────────
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Fingerprint matrix not found: {npz_path}\n"
            f"Expected Step 4 output inside: {fingerprints_dir}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Phrase index map not found: {meta_path}\n"
            f"Expected Step 4 output inside: {fingerprints_dir}"
        )

    # ── Load matrix ──────────────────────────────────────────────────────────
    logger.info(f"Loading fingerprint matrix from: {npz_path}")
    data   = np.load(str(npz_path))
    matrix = data["fingerprints"]                    # shape (n_phrases, grid_size²)

    expected_cols = grid_size * grid_size
    if matrix.shape[1] != expected_cols:
        raise ValueError(
            f"Matrix has {matrix.shape[1]} columns but "
            f"grid_size={grid_size} implies {expected_cols} columns. "
            f"Did you pass the correct --grid-size?"
        )

    logger.info(
        f"Matrix shape: {matrix.shape} "
        f"(n_phrases={matrix.shape[0]}, vector_size={matrix.shape[1]})"
    )

    # ── Load phrase → row-index map ──────────────────────────────────────────
    logger.info(f"Loading phrase index map from: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        token_map: Dict[str, int] = json.load(fh)

    if len(token_map) != matrix.shape[0]:
        logger.warning(
            f"token_map has {len(token_map)} entries but matrix has "
            f"{matrix.shape[0]} rows — index map and matrix may be misaligned."
        )

    # ── Build phrase → vector dict ───────────────────────────────────────────
    phrase_fps: Dict[str, np.ndarray] = {
        phrase: matrix[idx].astype(np.float32)
        for phrase, idx in token_map.items()
    }

    logger.success(f"Loaded {len(phrase_fps)} phrase fingerprints.")
    return phrase_fps



def load_fingerprint_cache(
    cache_path: Path,
    grid_size: int
) -> Dict[str, csr_matrix]:
    """
    Load document fingerprints from cache file into sparse matrix format.
    
    Expected file format (JSON):
        {
            "doc_id_1": {
                "coordinates": [[x1, y1], [x2, y2], ...],
                "values": [v1, v2, ...]
            },
            "doc_id_2": { ... }
        }
    
    The sparse CSR (Compressed Sparse Row) format is optimal for:
    - Memory efficiency with high-dimensional sparse data
    - Fast row slicing and matrix-vector operations
    - Efficient similarity computations
    
    Args:
        cache_path: Path to fingerprint cache JSON file
        grid_size: Size of the semantic grid (determines matrix dimensions)
    
    Returns:
        Dictionary mapping doc_id -> csr_matrix of shape (1, grid_size²)
    
    Example:
        >>> cache = load_fingerprint_cache(Path('doc_fps.json'), 128)
        >>> cache['doc_1'].shape
        (1, 16384)  # 128 * 128
        >>> cache['doc_1'].nnz
        47  # number of active bits
    
    Note:
        Each fingerprint is stored as a row vector (1, grid_size²) for
        compatibility with similarity computation functions.
    """
    logger.info(f"Loading fingerprint cache from: {cache_path}")
    
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    fingerprints = {}
    total_dims = grid_size * grid_size
    
    for doc_id, fp_data in cache_data.items():
        coords = fp_data.get('coordinates', [])
        values = fp_data.get('values', [])
        
        if not coords:
            logger.warning(f"Empty fingerprint for document: '{doc_id}'")
            continue
        
        # Convert 2D coordinates to 1D indices
        indices = [x * grid_size + y for x, y in coords]
        
        # Create sparse matrix (row vector)
        row = np.zeros(1, dtype=int)
        col = np.array(indices, dtype=int)
        data = np.array(values, dtype=float)
        
        # Build CSR matrix
        sparse_fp = csr_matrix((data, (row, col)), shape=(1, total_dims))
        fingerprints[doc_id] = sparse_fp
    
    logger.success(f"Loaded {len(fingerprints)} document fingerprints from: {cache_path}")
    return fingerprints


def save_fingerprint_cache(
    fingerprints: Dict[str, csr_matrix],
    cache_path: Path,
    grid_size: int
) -> None:
    """
    Save document fingerprints to cache file in JSON format.
    
    Converts sparse CSR matrices to JSON-serializable format with
    explicit coordinate and value storage.
    
    Args:
        fingerprints: Dictionary mapping doc_id -> csr_matrix
        cache_path: Path to output cache JSON file
        grid_size: Size of the semantic grid
    
    Example:
        >>> fps = {'doc_1': csr_matrix(...), 'doc_2': csr_matrix(...)}
        >>> save_fingerprint_cache(fps, Path('cache.json'), 128)
    
    Note:
        The cache file can be loaded back using load_fingerprint_cache()
        for fast retrieval without recomputation.
    """
    logger.info(f"Saving fingerprint cache to: {cache_path}")
    
    cache_data = {}
    
    for doc_id, sparse_fp in fingerprints.items():
        # Convert sparse matrix to coordinates and values
        sparse_fp = sparse_fp.tocoo()  # Convert to COO for easy iteration
        
        coords = []
        values = []
        
        for i, j, v in zip(sparse_fp.row, sparse_fp.col, sparse_fp.data):
            # Convert 1D index back to 2D coordinates
            x = j // grid_size
            y = j % grid_size
            coords.append([int(x), int(y)])
            values.append(float(v))
        
        cache_data[doc_id] = {
            'coordinates': coords,
            'values': values
        }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.success(f"Saved {len(cache_data)} fingerprints to: {cache_path}")


# ============================================================================
# COORDINATE UTILITIES
# ============================================================================

def load_context_coordinates(coords_path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Load context coordinates from semantic space mapping file.
    
    Expected file format (CSV):
        context_id,x,y
        ctx_0,45,67
        ctx_1,23,89
    
    These coordinates represent the position of each context in the
    discretized semantic space grid, generated by semantic_space.py.
    
    Args:
        coords_path: Path to context coordinates CSV file
    
    Returns:
        Dictionary mapping context_id -> (x, y) grid coordinates
    
    Example:
        >>> coords = load_context_coordinates(Path('context_coords.csv'))
        >>> coords['ctx_0']
        (45, 67)
    
    Note:
        This file is generated by semantic_space.py and is required for
        phrase fingerprint generation in phrase_fingerprints.py.
    """
    logger.info(f"Loading context coordinates from: {coords_path}")
    
    coordinates = {}
    
    with open(coords_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 3:
                logger.warning(f"Invalid coordinate line format: '{line}'")
                continue
            
            context_id, x_str, y_str = parts
            context_id = context_id.strip()
            
            try:
                x = int(x_str.strip())
                y = int(y_str.strip())
                coordinates[context_id] = (x, y)
            except ValueError:
                logger.warning(
                    f"Invalid coordinates for '{context_id}': "
                    f"x='{x_str.strip()}', y='{y_str.strip()}'"
                )
                continue
    
    logger.success(f"Loaded coordinates for {len(coordinates)} contexts from: {coords_path}")
    return coordinates

# ============================================================================
# IDF COMPUTATION
# ============================================================================

def compute_idf_weights(
    phrases: List[str],
    contexts: List[str]
) -> Dict[str, float]:
    """
    Compute IDF (Inverse Document Frequency) weights for phrases.
    
    IDF weights measure the discriminative power of phrases across contexts.
    Rare phrases receive higher weights, while common phrases receive lower
    weights, following the formula:
    
        IDF(phrase) = log(N / df(phrase))
    
    where N is the total number of contexts and df(phrase) is the number
    of contexts containing the phrase.
    
    Args:
        phrases: List of phrases to compute IDF for
        contexts: List of context texts
    
    Returns:
        Dictionary mapping phrase -> IDF weight
    
    Example:
        >>> phrases = ['machine learning', 'the', 'neural network']
        >>> contexts = ['machine learning is...', 'the neural network...']
        >>> idf = compute_idf_weights(phrases, contexts)
        >>> idf['machine learning'] > idf['the']
        True  # 'machine learning' is more discriminative
    
    Note:
        IDF weights are used in doc_fingerprints.py and query_processing.py
        to emphasize discriminative phrases in document representations.
    """
    logger.info(f"Computing IDF weights for {len(phrases)} phrases across {len(contexts)} contexts")
    
    # Count document frequency for each phrase
    df = defaultdict(int)
    
    for context in contexts:
        context_lower = context.lower()
        seen_phrases = set()
        
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower not in seen_phrases:
                if find_phrase_occurrences(context_lower, phrase_lower, use_word_boundaries=True) > 0:
                    df[phrase_lower] += 1
                    seen_phrases.add(phrase_lower)
    
    # Compute IDF weights
    N = len(contexts)
    idf_weights = {}
    
    for phrase in phrases:
        phrase_lower = phrase.lower()
        doc_freq = df.get(phrase_lower, 0)
        
        if doc_freq > 0:
            idf_weights[phrase] = np.log(N / doc_freq)
        else:
            # Assign maximum IDF for phrases not found in any context
            idf_weights[phrase] = np.log(N)
    
    logger.success(f"Computed IDF weights for {len(idf_weights)} phrases")
    return idf_weights


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

def compute_cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.
    
    Formula:
        cos(θ) = (A · B) / (||A|| × ||B||)
    
    Args:
        vec1: First vector (numpy array or sparse matrix)
        vec2: Second vector (numpy array or sparse matrix)
    
    Returns:
        Cosine similarity score in range [-1, 1]
    
    Examples:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([1, 0, 1, 0])
        >>> compute_cosine_similarity(v1, v2)
        1.0  # identical vectors
        
        >>> v3 = np.array([1, 0, 0, 0])
        >>> v4 = np.array([0, 1, 0, 0])
        >>> compute_cosine_similarity(v3, v4)
        0.0  # orthogonal vectors
    
    Note:
        Handles both dense numpy arrays and sparse scipy matrices.
        Returns 0.0 if either vector has zero magnitude.
    """
    # Convert sparse matrices to dense if needed
    if hasattr(vec1, 'toarray'):
        vec1 = vec1.toarray().flatten()
    if hasattr(vec2, 'toarray'):
        vec2 = vec2.toarray().flatten()
    
    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)


def compute_jaccard_similarity(
    set1: Set,
    set2: Set
) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Jaccard similarity measures the overlap between two sets as the ratio
    of intersection to union:
    
        J(A, B) = |A ∩ B| / |A ∪ B|
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        Jaccard similarity score in range [0, 1]
    
    Examples:
        >>> s1 = {1, 2, 3, 4}
        >>> s2 = {3, 4, 5, 6}
        >>> compute_jaccard_similarity(s1, s2)
        0.333...  # 2 common / 6 total
        
        >>> s3 = {1, 2, 3}
        >>> s4 = {1, 2, 3}
        >>> compute_jaccard_similarity(s3, s4)
        1.0  # identical sets
    
    Note:
        Returns 0.0 if both sets are empty.
        Useful for comparing sparse fingerprint coordinate sets.
    """
    if not set1 and not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

# ============================================================================
# Z-ORDER CURVE UTILITIES
# ============================================================================

def xy_to_morton(x: int, y: int) -> int:
    """
    Convert 2D coordinates to Morton code (Z-order curve index).
    
    Morton codes interleave the binary representations of x and y coordinates,
    creating a space-filling curve that preserves spatial locality. Points
    close in 2D space tend to have similar Morton codes.
    
    Algorithm:
        For x=5 (binary: 101) and y=3 (binary: 011):
        Interleave: y1 x1 y0 x0 y2 x2 → 100111 = 39
    
    Args:
        x: X coordinate (non-negative integer)
        y: Y coordinate (non-negative integer)
    
    Returns:
        Morton code (Z-order index)
    
    Examples:
        >>> xy_to_morton(0, 0)
        0
        >>> xy_to_morton(1, 0)
        1
        >>> xy_to_morton(0, 1)
        2
        >>> xy_to_morton(1, 1)
        3
        >>> xy_to_morton(5, 3)
        39
    
    Note:
        Used in phrase_fingerprints.py and doc_fingerprints.py for
        Z-order curve thresholding, which preserves spatial structure
        when selecting top-k bits.
    """
    def part1by1(n: int) -> int:
        """Spread bits of n by inserting a 0 between each bit"""
        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n
    
    return (part1by1(y) << 1) + part1by1(x)


def morton_to_xy(morton: int) -> Tuple[int, int]:
    """
    Convert Morton code back to 2D coordinates.
    
    Inverse operation of xy_to_morton(), extracting the interleaved
    x and y coordinates from the Morton code.
    
    Args:
        morton: Morton code (Z-order index)
    
    Returns:
        Tuple of (x, y) coordinates
    
    Examples:
        >>> morton_to_xy(0)
        (0, 0)
        >>> morton_to_xy(1)
        (1, 0)
        >>> morton_to_xy(2)
        (0, 1)
        >>> morton_to_xy(3)
        (1, 1)
        >>> morton_to_xy(39)
        (5, 3)
    
    Note:
        Useful for debugging and visualization of Z-order traversal.
    """
    def compact1by1(n: int) -> int:
        """Extract every other bit"""
        n &= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0F0F0F0F
        n = (n ^ (n >> 4)) & 0x00FF00FF
        n = (n ^ (n >> 8)) & 0x0000FFFF
        return n
    
    x = compact1by1(morton)
    y = compact1by1(morton >> 1)
    return (x, y)


def get_zorder_neighbors(
    x: int,
    y: int,
    grid_size: int,
    radius: int = 1
) -> List[Tuple[int, int]]:
    """
    Get neighboring coordinates within a given radius in Z-order space.
    
    Returns all valid grid coordinates within Manhattan distance 'radius'
    from the given point, useful for spreading activation in semantic space.
    
    Args:
        x: Center X coordinate
        y: Center Y coordinate
        grid_size: Size of the grid (for boundary checking)
        radius: Manhattan distance radius (default: 1)
    
    Returns:
        List of (x, y) coordinate tuples within radius
    
    Examples:
        >>> get_zorder_neighbors(5, 5, 10, radius=1)
        [(4, 5), (6, 5), (5, 4), (5, 6), (4, 4), (4, 6), (6, 4), (6, 6)]
        
        >>> get_zorder_neighbors(0, 0, 10, radius=1)
        [(1, 0), (0, 1), (1, 1)]  # boundary-aware
    
    Note:
        Used in query_processing.py for spreading query fingerprints
        to improve recall by activating nearby semantic regions.
    """
    neighbors = []
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            # Skip center point
            if dx == 0 and dy == 0:
                continue
            
            nx = x + dx
            ny = y + dy
            
            # Check boundaries
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbors.append((nx, ny))
    
    return neighbors


# ============================================================================
# FINGERPRINT MANIPULATION
# ============================================================================

def normalize_fingerprint(
    fingerprint: csr_matrix,
    method: str = 'l2'
) -> csr_matrix:
    """
    Normalize a sparse fingerprint vector.
    
    Normalization methods:
    - 'l2': L2 normalization (unit vector), preserves direction
    - 'l1': L1 normalization (sum to 1), preserves relative magnitudes
    - 'binary': Binarize (all non-zero values → 1)
    
    Args:
        fingerprint: Sparse fingerprint matrix (shape: 1 × D)
        method: Normalization method ('l2', 'l1', or 'binary')
    
    Returns:
        Normalized sparse fingerprint matrix
    
    Examples:
        >>> fp = csr_matrix([[1, 2, 0, 3]])
        >>> normalize_fingerprint(fp, 'l2')
        # Returns unit vector with same direction
        
        >>> normalize_fingerprint(fp, 'binary')
        # Returns [[1, 1, 0, 1]]
    
    Raises:
        ValueError: If method is not recognized
    
    Note:
        L2 normalization is standard for cosine similarity computation.
        Binary normalization is useful for pure overlap-based matching.
    """
    if method == 'l2':
        # L2 normalization
        norm = np.sqrt(fingerprint.multiply(fingerprint).sum())
        if norm > 0:
            return fingerprint / norm
        return fingerprint
    
    elif method == 'l1':
        # L1 normalization
        norm = np.abs(fingerprint).sum()
        if norm > 0:
            return fingerprint / norm
        return fingerprint
    
    elif method == 'binary':
        # Binarize
        fp_copy = fingerprint.copy()
        fp_copy.data = np.ones_like(fp_copy.data)
        return fp_copy
    
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def merge_fingerprints(
    fingerprints: List[csr_matrix],
    weights: Optional[List[float]] = None
) -> csr_matrix:
    """
    Merge multiple fingerprints with optional weighting.
    
    Combines multiple sparse fingerprints into a single representation
    by weighted summation. Useful for:
    - Combining phrase fingerprints into document fingerprints
    - Merging multi-query representations
    - Creating composite semantic representations
    
    Args:
        fingerprints: List of sparse fingerprint matrices (same shape)
        weights: Optional list of weights (default: uniform weighting)
    
    Returns:
        Merged sparse fingerprint matrix
    
    Examples:
        >>> fp1 = csr_matrix([[1, 0, 1, 0]])
        >>> fp2 = csr_matrix([[0, 1, 1, 0]])
        >>> merge_fingerprints([fp1, fp2])
        # Returns [[1, 1, 2, 0]]
        
        >>> merge_fingerprints([fp1, fp2], weights=[0.7, 0.3])
        # Returns weighted combination
    
    Raises:
        ValueError: If fingerprints have different shapes
        ValueError: If weights length doesn't match fingerprints length
    
    Note:
        All fingerprints must have the same shape.
        Result is NOT automatically normalized.
    """
    if not fingerprints:
        raise ValueError("Cannot merge empty fingerprint list")
    
    # Validate shapes
    shape = fingerprints[0].shape
    for fp in fingerprints[1:]:
        if fp.shape != shape:
            raise ValueError(f"Shape mismatch: {fp.shape} != {shape}")
    
    # Set uniform weights if not provided
    if weights is None:
        weights = [1.0] * len(fingerprints)
    
    if len(weights) != len(fingerprints):
        raise ValueError(
            f"Weights length {len(weights)} != fingerprints length {len(fingerprints)}"
        )
    
    # Weighted sum
    merged = weights[0] * fingerprints[0]
    for w, fp in zip(weights[1:], fingerprints[1:]):
        merged = merged + w * fp
    
    return merged


def sparsify_fingerprint(
    fingerprint: csr_matrix,
    top_k: int,
    use_zorder: bool = False,
    grid_size: Optional[int] = None,
) -> csr_matrix:
    """
    Sparsify a fingerprint by keeping only the top-k active bits.
    
    Reduces fingerprint density by retaining only the highest-value
    entries, which corresponds to the most strongly activated semantic
    regions. Two selection strategies are supported:
    
    - Standard: Select top-k by value (highest activation first)
    - Z-order:  Select top-k by Morton code order (spatially coherent)
    
    Args:
        fingerprint: Input sparse fingerprint matrix (shape: 1 × D)
        top_k: Number of bits to retain
        use_zorder: If True, use Z-order curve ordering (default: False)
        grid_size: Required when use_zorder=True for coordinate conversion
    
    Returns:
        Sparsified fingerprint with at most top_k non-zero entries
    
    Examples:
        >>> fp = csr_matrix([[0.1, 0.9, 0.0, 0.5, 0.3]])
        >>> sparsify_fingerprint(fp, top_k=2).toarray()
        array([[0. , 0.9, 0. , 0.5, 0. ]])
    
    Raises:
        ValueError: If use_zorder=True but grid_size is not provided
    
    Note:
        Z-order sparsification preserves spatial coherence in the
        semantic grid, which can improve retrieval quality.
    """
    if use_zorder and grid_size is None:
        raise ValueError("grid_size is required when use_zorder=True")

    # Convert to dense for processing
    dense = fingerprint.toarray().flatten()
    nonzero_indices = np.nonzero(dense)[0]

    if len(nonzero_indices) <= top_k:
        return fingerprint  # Already sparse enough

    if use_zorder:
        # Sort nonzero indices by Morton code (Z-order)
        morton_codes = [
            (xy_to_morton(int(idx // grid_size), int(idx % grid_size)), idx)
            for idx in nonzero_indices
        ]
        morton_codes.sort(key=lambda x: x[0])
        selected_indices = [idx for _, idx in morton_codes[:top_k]]
    else:
        # Sort by activation value (descending), keep top_k
        sorted_indices = nonzero_indices[np.argsort(dense[nonzero_indices])[::-1]]
        selected_indices = sorted_indices[:top_k]

    # Build new sparse matrix with only selected indices
    new_dense = np.zeros_like(dense)
    new_dense[selected_indices] = dense[selected_indices]

    return csr_matrix(new_dense.reshape(1, -1))

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_fingerprint(
    fingerprint: csr_matrix,
    grid_size: int,
    min_active: int = 1,
    max_active: Optional[int] = None
) -> bool:
    """
    Validate fingerprint properties.
    
    Checks:
    - Correct shape (1 × grid_size²)
    - Minimum number of active bits
    - Maximum number of active bits (if specified)
    - All values are non-negative
    
    Args:
        fingerprint: Sparse fingerprint matrix
        grid_size: Expected grid size
        min_active: Minimum number of active bits (default: 1)
        max_active: Maximum number of active bits (optional)
    
    Returns:
        True if fingerprint is valid, False otherwise
    
    Examples:
        >>> fp = csr_matrix([[1, 0, 1, 0]])
        >>> validate_fingerprint(fp, grid_size=2, min_active=1)
        True
        
        >>> validate_fingerprint(fp, grid_size=2, min_active=5)
        False  # not enough active bits
    
    Note:
        Use this for quality control in fingerprint generation pipelines.
    """
    expected_dims = grid_size * grid_size
    
    # Check shape
    if fingerprint.shape != (1, expected_dims):
        logger.warning(f"Invalid shape: {fingerprint.shape}, expected (1, {expected_dims})")
        return False
    
    # Check number of active bits
    n_active = fingerprint.nnz
    
    if n_active < min_active:
        logger.warning(f"Too few active bits: {n_active} < {min_active}")
        return False
    
    if max_active is not None and n_active > max_active:
        logger.warning(f"Too many active bits: {n_active} > {max_active}")
        return False
    
    # Check for negative values
    if hasattr(fingerprint, 'data'):
        if np.any(fingerprint.data < 0):
            logger.warning("Fingerprint contains negative values")
            return False
    
    return True


def compute_fingerprint_stats(
    fingerprints: Dict[str, csr_matrix]
) -> Dict[str, float]:
    """
    Compute statistics for a collection of fingerprints.
    
    Computed metrics:
    - Mean sparsity (percentage of zero values)
    - Mean number of active bits
    - Standard deviation of active bits
    - Min/max active bits
    
    Args:
        fingerprints: Dictionary mapping ID -> fingerprint matrix
    
    Returns:
        Dictionary of statistics
    
    Example:
        >>> fps = {'doc1': csr_matrix(...), 'doc2': csr_matrix(...)}
        >>> stats = compute_fingerprint_stats(fps)
        >>> print(stats['mean_active_bits'])
        47.3
    
    Note:
        Useful for quality assessment and hyperparameter tuning.
    """
    if not fingerprints:
        return {}
    
    active_bits = [fp.nnz for fp in fingerprints.values()]
    total_dims = list(fingerprints.values())[0].shape[1]
    
    stats = {
        'n_fingerprints': len(fingerprints),
        'total_dimensions': total_dims,
        'mean_active_bits': np.mean(active_bits),
        'std_active_bits': np.std(active_bits),
        'min_active_bits': np.min(active_bits),
        'max_active_bits': np.max(active_bits),
        'mean_sparsity': 1.0 - (np.mean(active_bits) / total_dims)
    }
    
    return stats


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Ensure NLTK data is available
def _ensure_nltk_data():
    """Download required NLTK data if not present"""
    required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else f'corpora/{data_name}')
        except LookupError:
            logger.info(f"Downloading NLTK data: {data_name}")
            nltk.download(data_name, quiet=True)

# Initialize on module import
# _ensure_nltk_data()

def batch_compute_similarities(
    query_fp: csr_matrix,
    doc_fps: List[csr_matrix]
) -> np.ndarray:
    """
    Compute cosine similarities between query and multiple documents efficiently.
    
    Args:
        query_fp: Query fingerprint (1 × N sparse matrix)
        doc_fps: List of document fingerprints (each 1 × N sparse matrix)
        
    Returns:
        Array of similarity scores, one per document
    """
    from scipy.sparse import vstack
    
    # Stack documents into (num_docs, N) matrix
    doc_matrix = vstack(doc_fps)
    
    # Convert query to dense for computation
    query_dense = query_fp.toarray().flatten()
    query_norm = np.linalg.norm(query_dense)
    
    if query_norm == 0:
        return np.zeros(len(doc_fps))
    
    # Compute dot products: (num_docs, N) @ (N,) → (num_docs,)
    dot_products = doc_matrix.dot(query_dense)
    
    # Compute document norms: sqrt of sum of squares per row
    doc_norms = np.sqrt(np.array(doc_matrix.multiply(doc_matrix).sum(axis=1)).flatten())
    
    # Avoid division by zero
    doc_norms[doc_norms == 0] = 1e-10
    
    # Cosine similarity: dot / (norm_q * norm_d)
    similarities = dot_products / (query_norm * doc_norms)
    
    return similarities


def get_fingerprint_overlap(
    fp1: csr_matrix,
    fp2: csr_matrix
) -> Tuple[int, int, int]:
    """
    Compute overlap statistics between two fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Tuple of (intersection_size, fp1_only, fp2_only)
    """
    # Get active indices
    indices1 = set(fp1.indices)
    indices2 = set(fp2.indices)
    
    intersection = len(indices1 & indices2)
    fp1_only = len(indices1 - indices2)
    fp2_only = len(indices2 - indices1)
    
    return intersection, fp1_only, fp2_only


def visualize_fingerprint(
    fingerprint: csr_matrix,
    grid_size: int,
    title: str = "Fingerprint",
    output_path: Optional[Path] = None
) -> None:
    """
    Create heatmap visualization of fingerprint.
    
    Args:
        fingerprint: Sparse fingerprint matrix (1 × N)
        grid_size: Grid dimension
        title: Plot title
        output_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert to dense 2D
        dense_fp = fingerprint.toarray().reshape(grid_size, grid_size)
        
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            dense_fp,
            annot=False,
            cmap='YlOrRd',
            cbar=True,
            square=True,
            cbar_kws={'label': 'Activation'}
        )
        
        plt.title(title, fontsize=12, pad=10)
        plt.xlabel('Grid X', fontsize=10)
        plt.ylabel('Grid Y', fontsize=10)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.warning("Matplotlib not available for visualization")
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")


def export_fingerprints_to_numpy(
    fingerprints: Dict[str, csr_matrix],
    output_path: Path,
    grid_size: int
) -> None:
    """
    Export fingerprints to dense numpy format for analysis.
    
    Args:
        fingerprints: Dictionary of sparse fingerprints
        output_path: Output .npz file path
        grid_size: Grid dimension
    """
    dense_fps = {}
    for key, fp in fingerprints.items():
        dense_fps[key] = fp.toarray().reshape(grid_size, grid_size)
    
    np.savez_compressed(output_path, **dense_fps)
    logger.success(f"Exported {len(fingerprints)} fingerprints to {output_path}")


def compute_fingerprint_diversity(
    fingerprints: Dict[str, csr_matrix],
    sample_size: int = 100
) -> Dict[str, float]:
    """
    Compute diversity metrics for a set of fingerprints.
    
    Args:
        fingerprints: Dictionary of fingerprints
        sample_size: Number of pairs to sample for diversity computation
        
    Returns:
        Dictionary of diversity metrics
    """
    import random
    
    if len(fingerprints) < 2:
        return {'avg_similarity': 0.0, 'diversity_score': 1.0}
    
    fp_list = list(fingerprints.values())
    similarities = []
    
    # Sample pairs
    num_samples = min(sample_size, len(fp_list) * (len(fp_list) - 1) // 2)
    
    for _ in range(num_samples):
        i, j = random.sample(range(len(fp_list)), 2)
        sim = compute_cosine_similarity(fp_list[i], fp_list[j])
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    diversity = 1 - avg_sim
    
    return {
        'avg_similarity': float(avg_sim),
        'diversity_score': float(diversity),
        'num_samples': num_samples
    }
