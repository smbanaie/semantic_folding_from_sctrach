"""
Phrase Extractor - First stage of Semantic Folding pipeline
Improved version using lib.py functions consistently
"""

import argparse
from pathlib import Path
from typing import Set, Dict, Tuple
from collections import Counter, defaultdict
from loguru import logger
import sys

# Import lib functions
from lib import (
    expand_phrases,
    normalize_phrase,
    is_valid_phrase_structure,
    is_generic_word
)

# Try to import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.success("spaCy model 'en_core_web_sm' loaded successfully")
    except OSError:
        logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not installed. Using fallback extraction.")
    SPACY_AVAILABLE = False

from nltk.tokenize import word_tokenize
from nltk import pos_tag


# ============================================================================
# PHRASE EXTRACTION
# ============================================================================

def extract_raw_phrases_spacy(text: str) -> Set[str]:
    """Extract raw phrases using spaCy (before normalization)."""
    if not SPACY_AVAILABLE:
        return set()

    phrases = set()

    try:
        doc = nlp(text)

        for sent in doc.sents:
            # Noun chunks
            for chunk in sent.noun_chunks:
                phrase = chunk.text.strip()
                if phrase and len(phrase) > 1:
                    phrases.add(phrase)

            # Named entities
            for ent in sent.ents:
                phrase = ent.text.strip()
                if phrase and len(phrase) > 1:
                    phrases.add(phrase)

            # Compound nouns
            nouns = []
            for token in sent:
                if token.pos_ == "NOUN":
                    nouns.append(token.text)
                else:
                    if len(nouns) >= 2:
                        phrases.add(' '.join(nouns))
                    nouns = []

            if len(nouns) >= 2:
                phrases.add(' '.join(nouns))

    except Exception as e:
        logger.error(f"spaCy extraction failed: {e}")

    return phrases


def extract_raw_phrases_fallback(text: str, max_ngram: int = 4) -> Set[str]:
    """Fallback extraction using n-grams."""
    phrases = set()

    try:
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)

        filtered = [(word, tag) for word, tag in tagged
                   if word.isalpha() and len(word) > 1]

        if not filtered:
            return phrases

        # Extract all n-grams, let lib.py validate them
        for n in range(1, min(max_ngram + 1, len(filtered) + 1)):
            for i in range(len(filtered) - n + 1):
                ngram = filtered[i:i+n]
                words = [w for w, _ in ngram]
                phrase = ' '.join(words)
                phrases.add(phrase)

    except Exception as e:
        logger.error(f"Fallback extraction failed: {e}")

    return phrases


def extract_and_normalize_phrases(text: str,
                                  use_spacy: bool = True,
                                  remove_verbs: bool = True) -> Set[str]:
    """
    Extract and normalize phrases in one pass.
    Ensures consistency with lib.py's normalization.
    """
    if use_spacy and SPACY_AVAILABLE:
        raw_phrases = extract_raw_phrases_spacy(text)
    else:
        raw_phrases = extract_raw_phrases_fallback(text)

    normalized = set()
    for phrase in raw_phrases:
        norm = normalize_phrase(phrase, remove_verbs=remove_verbs)

        if norm:
            tokens = word_tokenize(norm)
            tagged = pos_tag(tokens)

            if is_valid_phrase_structure(tagged):
                normalized.add(norm)

    return normalized


# ============================================================================
# CORE FIX: contiguous subsequence check
# ============================================================================

def is_subphrase(sub_words: list, full_words: list) -> bool:
    """
    Check if sub_words is a contiguous subsequence of full_words.

    Fixes the original bug where `expanded_phrase in original_phrase.split()`
    only matched single tokens, missing multi-word sub-phrases like
    'cultural group' inside 'different cultural group'.
    """
    n, m = len(full_words), len(sub_words)
    if m >= n:
        return False
    return any(full_words[i:i + m] == sub_words for i in range(n - m + 1))


# ============================================================================
# CORPUS PROCESSING WITH PROPER EXPANSION
# ============================================================================

def process_corpus_with_expansion(corpus_path: Path,
                                  use_spacy: bool = True,
                                  keep_verbs: bool = True,
                                  min_freq: int = 2,
                                  filter_generic: bool = True,
                                  min_word_length: int = 3) -> Counter:
    """
    Process corpus with proper expansion logic.
    1. Normalize BEFORE counting
    2. Track which contexts contain each phrase
    3. Expand phrases BEFORE applying min_freq filter
    4. Use contiguous subsequence check for frequency inheritance (bug fix)
    5. Sum-based aggregation instead of max
    6. POS-validate all expanded sub-phrases
    """
    logger.info(f"Processing corpus: {corpus_path}")

    # Track phrase occurrences per context
    phrase_contexts: Dict[str, Set[str]] = defaultdict(set)
    context_count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue

            parts = line.split(',', 1)
            if len(parts) != 2:
                continue

            context_id, context_text = parts
            context_text = context_text.strip()

            if not context_text:
                continue

            phrases = extract_and_normalize_phrases(
                context_text,
                use_spacy=use_spacy,
                remove_verbs=not keep_verbs
            )

            for phrase in phrases:
                phrase_contexts[phrase].add(context_id)

            context_count += 1

            if context_count % 100 == 0:
                logger.info(f"Processed {context_count} contexts, "
                            f"found {len(phrase_contexts)} unique phrases")

    logger.success(f"Extracted {len(phrase_contexts)} unique phrases "
                   f"from {context_count} contexts")

    # Raw context-based frequencies (no min_freq yet)
    phrase_counts = Counter({
        phrase: len(contexts)
        for phrase, contexts in phrase_contexts.items()
    })

    # This preserves sub-phrases of low-frequency parents.
    logger.info("Expanding phrases (before frequency filter)...")
    phrases_to_expand = list(phrase_counts.keys())
    expanded_phrases = expand_phrases(
        phrases_to_expand,
        filter_generic=filter_generic,
        min_word_length=min_word_length
    )

    # ── FIX: sum-based aggregation with contiguous subsequence check ─────────
    expanded_counts: Dict[str, int] = defaultdict(int)

    for original_phrase, original_freq in phrase_counts.items():
        original_words = original_phrase.split()

        # The parent phrase itself
        expanded_counts[original_phrase] += original_freq

        # Every expanded sub-phrase that is a true contiguous sub-sequence
        for sub in expanded_phrases:
            sub_words = sub.split()
            if sub_words == original_words:
                continue  # already handled above
            if is_subphrase(sub_words, original_words):
                # Sum: accumulate across all parents that contain this sub-phrase
                expanded_counts[sub] += original_freq

    # ── POS-validate all expanded sub-phrases ────────────────────────────────
    validated: Counter = Counter()
    for phrase, freq in expanded_counts.items():
        tokens = word_tokenize(phrase)
        tagged = pos_tag(tokens)
        if is_valid_phrase_structure(tagged):
            validated[phrase] = freq

    # ── Apply minimum frequency filter ───────────────────────────────────────
    result = Counter({p: f for p, f in validated.items() if f >= min_freq})

    logger.success(f"Expanded to {len(result)} phrases after validation "
                   f"and min_freq={min_freq} filter")
    return result


# ============================================================================
# STATISTICS AND OUTPUT
# ============================================================================

def print_statistics(phrase_counts: Counter):
    """Print extraction statistics."""
    total_phrases = len(phrase_counts)
    total_occurrences = sum(phrase_counts.values())

    word_counts = Counter()
    for phrase in phrase_counts:
        word_count = len(phrase.split())
        word_counts[word_count] += 1

    generic_count = sum(1 for p in phrase_counts if len(p.split()) == 1 and is_generic_word(p))

    logger.info("=" * 60)
    logger.info("PHRASE EXTRACTION STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total unique phrases: {total_phrases}")
    logger.info(f"Total occurrences: {total_occurrences}")
    logger.info(f"Average frequency: {total_occurrences / total_phrases:.2f}")
    logger.info(f"Generic single words: {generic_count} ({100*generic_count/total_phrases:.1f}%)")
    logger.info("")
    logger.info("Word count distribution:")
    for word_count in sorted(word_counts.keys()):
        count = word_counts[word_count]
        percentage = (count / total_phrases) * 100
        logger.info(f"  {word_count}-word phrases: {count} ({percentage:.1f}%)")
    logger.info("")
    logger.info("Top 20 most frequent phrases:")
    for i, (phrase, count) in enumerate(phrase_counts.most_common(20), 1):
        logger.info(f"  {i:2d}. {phrase:30s} ({count:4d})")
    logger.info("=" * 60)


def save_phrases(phrase_counts: Counter, output_path: Path):
    """Save phrases in format: phrase:count"""
    logger.info(f"Saving phrases to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for phrase, count in phrase_counts.most_common():
            f.write(f"{phrase}:{count}\n")

    logger.success(f"Saved {len(phrase_counts)} phrases")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract meaningful phrases from corpus using lib.py",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--corpus', type=Path, required=True,
                        help='Path to corpus file (format: context_id,context_text)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to output phrases file')
    parser.add_argument('--keep-verbs', action='store_true',
                        help='Force fallback extraction', default=True, dest="keep_verbs")
    parser.add_argument('--no-spacy', action='store_true',
                        help='Force fallback extraction')
    parser.add_argument('--no-filter-generic', action='store_true',
                        help='Keep generic single words')
    parser.add_argument('--min-word-length', type=int, default=3,
                        help='Minimum character length for single words (default: 3)')
    parser.add_argument('--min-freq', type=int, default=2,
                        help='Minimum phrase frequency (default: 2)')
    parser.add_argument('--stats', action='store_true',
                        help='Print detailed statistics')

    args = parser.parse_args()

    if not args.corpus.exists():
        logger.error(f"Corpus file not found: {args.corpus}")
        sys.exit(1)

    logger.info("Configuration:")
    logger.info(f"  Corpus: {args.corpus}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Method: {'Fallback' if args.no_spacy else 'spaCy' if SPACY_AVAILABLE else 'Fallback'}")
    logger.info(f"  Filter generic: {not args.no_filter_generic}")
    logger.info(f"  Min frequency: {args.min_freq}")
    logger.info("")

    phrase_counts = process_corpus_with_expansion(
        corpus_path=args.corpus,
        use_spacy=not args.no_spacy,
        min_freq=args.min_freq,
        filter_generic=not args.no_filter_generic,
        min_word_length=args.min_word_length,
        keep_verbs=args.keep_verbs
    )

    logger.info(f"args.output: {args.output}")
    save_phrases(phrase_counts, args.output)

    if args.stats:
        print_statistics(phrase_counts)

    logger.success("Phrase extraction complete!")


if __name__ == "__main__":
    main()
