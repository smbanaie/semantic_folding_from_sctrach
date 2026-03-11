#!/usr/bin/env python3
"""
Modernized Phrase Extractor for Semantic Folding Pipeline

Extracts noun and verb phrases from corpus using spaCy with proper batching,
progress tracking, and visualization capabilities.
"""

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from lib import expand_phrases, process_lemmatize

import loguru
from loguru import logger
from tqdm import tqdm

# Try to import spaCy, with fallback
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_MODEL_AVAILABLE = True
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        SPACY_MODEL_AVAILABLE = False
        nlp = None
except ImportError:
    logger.warning("spaCy not available. Install with: pip install spacy")
    SPACY_AVAILABLE = False
    SPACY_MODEL_AVAILABLE = False
    nlp = None


def extract_phrases_spacy(text: str, batch_size: int = 500) -> List[str]:
    """Extract phrases using spaCy with proper batching"""
    if not SPACY_AVAILABLE or not SPACY_MODEL_AVAILABLE:
        raise RuntimeError("spaCy not available. Cannot extract phrases.")

    # Process in batches to avoid memory issues
    phrases = []

    # Split text into sentences first (rough approximation)
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        batch_text = '. '.join(batch_sentences)

        doc = nlp(batch_text)

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 1:  # Allow single words for now
                phrases.append(chunk.text.lower().strip())

        # Extract verb phrases (improved logic)
        for token in doc:
            if token.pos_ == "VERB":
                # Get verb with its auxiliaries and particles
                verb_phrase = []
                # Look backwards for auxiliaries
                for child in token.lefts:
                    if child.dep_ in ("aux", "auxpass") and child.pos_ == "AUX":
                        verb_phrase.append(child.text)
                verb_phrase.append(token.text)
                # Look forwards for particles
                for child in token.rights:
                    if child.dep_ == "prt" and child.pos_ == "PART":
                        verb_phrase.append(child.text)
                if len(verb_phrase) > 1:  # Only multi-word verb phrases
                    phrases.append(' '.join(verb_phrase).lower().strip())
    logger.info(phrases)
    return phrases


def extract_phrases_fallback(text: str) -> List[str]:
    """Fallback phrase extraction using simple heuristics - only 1-4 word phrases"""
    logger.warning("Using fallback phrase extraction (spaCy not available)")

    phrases = []
    words = text.lower().split()

    # Extract 1-4 word phrases
    for i in range(len(words)):
        # Single words (if they look like content words)
        if len(words[i]) > 2 and words[i].isalnum():
            phrases.append(words[i])

        # 2-word phrases
        if i < len(words) - 1:
            bigram = f"{words[i]} {words[i+1]}"
            # Only include if both words are reasonable length and alphanumeric
            if (len(words[i]) > 2 and len(words[i+1]) > 2 and
                words[i].isalnum() and words[i+1].isalnum()):
                phrases.append(bigram)

        # 3-word phrases
        if i < len(words) - 2:
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            # Only include if all words are reasonable length and alphanumeric
            if (len(words[i]) > 2 and len(words[i+1]) > 2 and len(words[i+2]) > 2 and
                all(word.isalnum() for word in [words[i], words[i+1], words[i+2]])):
                phrases.append(trigram)

        # 4-word phrases
        if i < len(words) - 3:
            fourgram = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
            # Only include if all words are reasonable length and alphanumeric
            if (len(words[i]) > 2 and len(words[i+1]) > 2 and len(words[i+2]) > 2 and len(words[i+3]) > 2 and
                all(word.isalnum() for word in [words[i], words[i+1], words[i+2], words[i+3]])):
                phrases.append(fourgram)

    logger.info("Extraxt Phrases - Fallback")
    logger.info(phrases)
    return phrases


def filter_and_normalize_phrases(phrase_counts: Counter,
                               min_length: int = 2,
                               max_length: int = 25,
                               min_freq: int = 0,
                               max_words: int = 4) -> Dict[str, int]:
    """Filter and normalize phrases"""
    filtered = {}
    logger.info("=================== filter_and_normalize_phrases ==================")
    for phrase, freq in phrase_counts.items():
        # Skip if frequency too low
        if freq < min_freq:
            continue

        # Normalize
        normalized = phrase.strip().lower()

        normalized = process_lemmatize(normalized)

        # Skip if too short or too long (character length)
        if len(normalized) < min_length or len(normalized) > max_length:
            continue

        # Skip if contains only punctuation or numbers
        if not any(c.isalpha() for c in normalized):
            continue

        # Count words (split by whitespace)
        word_count = len(normalized.split())

        # Skip if too many or too few words (only allow 1-4 word phrases)
        if word_count < 1 or word_count > max_words:
            continue

        filtered[normalized] = freq

    return filtered


def process_corpus_file(corpus_path: Path,
                       batch_size: int = 100,
                       use_spacy: bool = True) -> Counter:
    """Process corpus file and extract phrases"""
    logger.info(f"Processing corpus file: {corpus_path}")

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    total_phrases = Counter()
    total_lines = sum(1 for _ in open(corpus_path, 'r', encoding='utf-8'))

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Processing corpus")):
            # Parse line: format is "idx,title: text"
            logger.info(f"Start Extracting for the Line: {line}")
            if ',' not in line:
                continue

            text_part = line[line.find(",")+1:]

            # Extract just the text content
            title_and_text = text_part
            # if not title_and_text:
            #     title_and_text = text_part
            logger.info(f"Text&Title : {title_and_text}")

            # Extract phrases
            try:
                if use_spacy and SPACY_AVAILABLE and SPACY_MODEL_AVAILABLE:
                    phrases = extract_phrases_spacy(title_and_text, batch_size)
                else:
                    phrases = extract_phrases_fallback(title_and_text)

                phrases = expand_phrases(phrases)
                final_phrases = []
                for p in phrases : 
                    phrase = process_lemmatize(p)
                    if len(phrase) > 1 : 
                        final_phrases.append(phrase)
                total_phrases.update(phrases)
                final_phrases = [ item for item in set(final_phrases)]
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")
                continue

            # Log progress
            if (line_num + 1) % 1000 == 0:
                logger.info(f"Processed {line_num + 1}/{total_lines} lines, {len(total_phrases)} unique phrases so far")

    return total_phrases

def extract_phrases_general(text: str) -> List[str]:
    """
    Extract, expand, and linguistically normalize meaningful phrases from text.

    Steps:
    1. Extract phrases (via spaCy or fallback logic)
    2. Expand multi-word phrases into sub-phrases
    3. Lemmatize and clean (remove verbs and stopwords)
    4. Deduplicate and return
    """
    final_phrases: List[str] = []
    logger.info("--" * 20)

    try:
        if SPACY_AVAILABLE and SPACY_MODEL_AVAILABLE:
            phrases = extract_phrases_spacy(text)
        else:
            phrases = extract_phrases_fallback(text)

        logger.info(f"Before Expansion: {phrases}")
        phrases = expand_phrases(phrases)
        logger.info(f"After Expansion: {phrases}")

        # Lemmatize and filter
        for p in phrases:
            phrase = process_lemmatize(p).strip()
            if len(phrase) > 1:  # ensure non-empty and meaningful
                final_phrases.append(phrase)

        # Deduplicate while preserving clean results
        final_phrases = list(set(final_phrases))

        logger.info(f"Final Phrases: {final_phrases}")
        logger.info("--" * 20)

        return final_phrases

    except Exception as e:
        logger.warning(f"Error processing text: {e}")
        return []


def create_visualization(phrases: Dict[str, int], output_dir: Path) -> None:
    """Create visualization of top phrases"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        sns.set_style("whitegrid")

        # Get top 50 phrases
        top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:50]
        phrase_names = [p[0][:30] + '...' if len(p[0]) > 30 else p[0] for p in top_phrases]
        frequencies = [p[1] for p in top_phrases]

        # Create figure
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(phrase_names)), frequencies)
        plt.yticks(range(len(phrase_names)), phrase_names)
        plt.xlabel('Frequency')
        plt.ylabel('Phrase')
        plt.title('Top 50 Most Frequent Phrases')
        plt.tight_layout()

        # Save plot
        viz_path = output_dir / "phrase_frequencies.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.success(f"Created phrase frequency visualization: {viz_path}")

    except ImportError:
        logger.warning("matplotlib/seaborn not available for visualization")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")


def save_phrases(phrases: Dict[str, int], output_path: Path) -> None:
    """Save phrases to file in format: phrase: frequency"""
    logger.info(f"Saving {len(phrases)} phrases to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for phrase, freq in sorted(phrases.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{phrase}: {freq}\n")

    logger.success(f"Saved phrases to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract phrases from corpus")
    parser.add_argument("--corpus_path", required=True, help="Path to corpus.txt file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for spaCy processing")
    parser.add_argument("--min_freq", type=int, default=0, help="Minimum phrase frequency")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum phrase length")
    parser.add_argument("--max_length", type=int, default=25, help="Maximum phrase length")
    parser.add_argument("--use_spacy", action="store_true", default=True, help="Use spaCy for extraction")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization")
    parser.add_argument("--max_words", type=int, default=4, help="Maximum words per phrase (1-4)")

    args = parser.parse_args()

    # Extract phrases
    logger.info("Starting phrase extraction...")
    logger.info(f"Corpus: {args.corpus_path}")
    logger.info(f"Output: {args.output_dir}")
    phrase_counts = process_corpus_file(
        Path(args.corpus_path),
        batch_size=args.batch_size,
        use_spacy=args.use_spacy
    )

    # Filter and normalize
    logger.info("Filtering and normalizing phrases...")
    filtered_phrases = filter_and_normalize_phrases(
        phrase_counts,
        min_length=args.min_length,
        max_length=args.max_length,
        min_freq=0,
        max_words=args.max_words
    )

    # Log statistics
    logger.info("Phrase extraction statistics:")
    logger.info(f"  Raw phrases extracted: {len(phrase_counts)}")
    logger.info(f"  Filtered phrases: {len(filtered_phrases)}")
    logger.info(f"  Most frequent phrase: {max(filtered_phrases.items(), key=lambda x: x[1])}")

    # Save phrases
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phrases_file = output_dir / "phrases.txt"
    save_phrases(filtered_phrases, phrases_file)

    # Create visualization
    if not args.no_visualization:
        create_visualization(filtered_phrases, output_dir)

    logger.success("Phrase extraction completed")


if __name__ == "__main__":
    main()