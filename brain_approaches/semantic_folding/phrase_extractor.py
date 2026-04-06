r"""
Phrase Extractor - Step 1 of Semantic Folding Pipeline
Date: 1405/01/14 | 2026/04/03

This module constitutes the first stage of the Semantic Folding pipeline. 
Its theoretical objective is to map a raw unstructured text corpus $C$ into a 
finite vocabulary of meaningful semantic features $\mathcal{V}$. 

These extracted phrases act as the fundamental basis vectors (dimensions) 
for the subsequent Semantic Space Mapping (Step 3) and Fingerprint Generation (Step 4). 
By isolating noun chunks, named entities, and structurally valid n-grams, we ensure 
that the resulting topological space captures high-signal semantic anchors while 
discarding low-information topological noise.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set, Dict, Tuple, List, Counter, Optional, Any
from collections import Counter as CounterType, defaultdict
from loguru import logger

# Import lib functions
from lib import (
    expand_phrases,
    normalize_phrase,
    is_valid_phrase_structure,
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

# NLTK is only needed for the fallback
if not SPACY_AVAILABLE:
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
# Add this function to phrase_extractor.py

def debug_phrase_extraction_pipeline(
    text: str,
    use_spacy: bool = True,
    remove_verbs: bool = True,
    filter_generic: bool = True,
    min_word_length: int = 3,
    min_freq: int = 1,
    vocab: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Trace every stage of phrase extraction to identify where terms are lost.
    
    Returns a detailed breakdown showing:
    - Raw extraction output
    - Post-normalization survivors
    - Post-expansion results
    - Vocabulary matches (if vocab provided)
    """
    from collections import defaultdict
    
    debug_info = {
        "input_text": text,
        "stages": {},
        "losses": defaultdict(list),
    }
    
    # Stage 1: Raw extraction
    if use_spacy and SPACY_AVAILABLE:
        raw_phrases = extract_raw_phrases_spacy(text)
        method = "spacy"
    else:
        raw_phrases = extract_raw_phrases_fallback(text, max_ngram=4)
        method = "nltk"
    
    debug_info["stages"]["1_raw_extraction"] = {
        "method": method,
        "count": len(raw_phrases),
        "phrases": raw_phrases,
    }
    
    # Stage 2: Normalization
    normalized = []
    for phrase in raw_phrases:
        norm = normalize_phrase(phrase, remove_verbs=remove_verbs)
        if norm:
            normalized.append(norm)
        else:
            debug_info["losses"]["normalization"].append({
                "original": phrase,
                "reason": "filtered_by_normalize_phrase"
            })
    
    debug_info["stages"]["2_normalized"] = {
        "count": len(normalized),
        "phrases": normalized,
        "dropped": len(raw_phrases) - len(normalized),
    }
    
    # Stage 3: Expansion
    expanded = expand_phrases(
        normalized,
        context_text=text,
        filter_generic=filter_generic,
        min_word_length=min_word_length,
    )
    
    # Track what was added vs dropped
    normalized_set = set(normalized)
    expanded_set = set(expanded)
    added = expanded_set - normalized_set
    dropped = normalized_set - expanded_set
    
    debug_info["stages"]["3_expanded"] = {
        "count": len(expanded),
        "phrases": expanded,
        "added_by_expansion": list(added),
        "dropped_during_expansion": list(dropped),
    }
    
    for phrase in dropped:
        debug_info["losses"]["expansion"].append({
            "phrase": phrase,
            "reason": "filtered_during_subphrase_generation"
        })
    
    # Stage 4: Vocabulary filter (if provided)
    if vocab:
        matched = [p for p in expanded if p in vocab]
        missed = [p for p in expanded if p not in vocab]
        
        debug_info["stages"]["4_vocab_matched"] = {
            "count": len(matched),
            "phrases": matched,
            "out_of_vocab": missed,
        }
        
        for phrase in missed:
            debug_info["losses"]["vocabulary"].append({
                "phrase": phrase,
                "reason": "not_in_training_vocabulary"
            })
    
    # Summary statistics
    debug_info["summary"] = {
        "raw_to_normalized_loss": len(raw_phrases) - len(normalized),
        "normalized_to_expanded_loss": len(normalized) - len(expanded),
        "expanded_to_vocab_loss": len(expanded) - len(matched) if vocab else 0,
        "total_loss_rate": 1 - (len(matched) / len(raw_phrases)) if vocab and raw_phrases else 0,
    }
    
    return debug_info


# ============================================================================
# PHRASE EXTRACTION
# ============================================================================

def extract_raw_phrases_spacy(text: str) -> Set[str]:
    """
    Extract raw linguistic phrases using dependency parsing and NER.
    
    This function leverages spaCy's linguistic model to identify:
    1. Noun Chunks (e.g., 'the continuous representation')
    2. Named Entities (e.g., 'Turing Award')
    3. Compound Nouns identified via sequential 'NOUN' POS tags.
    """
    if not SPACY_AVAILABLE:
        return set()

    phrases = set()
    try:
        doc = nlp(text)

        # 1. Noun chunks
        for chunk in doc.noun_chunks:
            phrases.add(chunk.text.strip())

        # 2. Named entities
        for ent in doc.ents:
            phrases.add(ent.text.strip())
            
    except Exception as e:
        logger.error(f"spaCy extraction failed on a document: {e}")

    return {p for p in phrases if p and len(p) > 1}


def extract_raw_phrases_fallback(text: str, max_ngram: int = 4) -> Set[str]:
    r"""
    Fallback phrase extraction generating structural $n$-grams.
    """
    phrases = set()
    try:
        tokens = word_tokenize(text.lower())
        filtered = [word for word in tokens if word.isalpha() and len(word) > 1]
        
        if not filtered:
            return phrases

        for n in range(1, min(max_ngram + 1, len(filtered) + 1)):
            for i in range(len(filtered) - n + 1):
                phrase = ' '.join(filtered[i:i+n])
                phrases.add(phrase)

    except Exception as e:
        logger.error(f"Fallback extraction failed on a document: {e}")

    return phrases

# ============================================================================
# CORPUS PROCESSING WITH PROPER EXPANSION
# ============================================================================
def process_corpus_with_expansion(
    corpus_path: Path, 
    use_spacy: bool = True,
    min_freq: int = 2,
    filter_generic: bool = True,
    min_word_length: int = 3,
    keep_verbs: bool = True
) -> Tuple[CounterType[str], Dict[str, List[str]]]:
    r"""
    Parse corpus, extract phrases, expand hierarchies, and map to contexts.
    
    This is the core execution loop for Step 1. It scans the corpus document 
    by document, ensuring that extracted and expanded phrases are validated 
    against their local source text. It then captures the precise bipartite 
    mapping between semantic dimensions (phrases) and contexts.
    """
    raw_phrase_contexts: Dict[str, Set[str]] = defaultdict(set)
    
    logger.info(f"Reading, extracting, and locally expanding from corpus: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if (i + 1) % 1000 == 0:
                logger.debug(f"Processed {i+1} lines...")
                
            if not line.strip() or ',' not in line: 
                continue
                
            try:
                ctx_id, text = line.split(',', 1)
                ctx_id = ctx_id.strip()
                text = text.strip().lower() # Normalize context text once for expansion
            except ValueError:
                logger.warning(f"Skipping malformed line {i+1}: {line[:100]}...")
                continue
            
            # Stage 1: Extract raw candidates
            if use_spacy:
                raw_phrases = extract_raw_phrases_spacy(text)
            else:
                raw_phrases = extract_raw_phrases_fallback(text)
            
            # Stage 2: Normalize base phrases
            base_phrases = set()
            for phrase in raw_phrases:
                norm = normalize_phrase(phrase, remove_verbs=not keep_verbs)
                if norm:
                    base_phrases.add(norm)
            
            if not base_phrases:
                continue

            # Stage 3: Expand and validate strictly against THIS specific context
            valid_sub_phrases = expand_phrases(
                base_phrases,
                context_text=text,           
                filter_generic=filter_generic,
                min_word_length=min_word_length
            )
            
            # Stage 4: Map valid phrases directly to the context ID
            for phrase in valid_sub_phrases:
                raw_phrase_contexts[phrase].add(ctx_id)

    logger.info(f"Generated {len(raw_phrase_contexts)} unique candidate phrases. Applying frequency filter...")

    # Stage 5: Sparsity Filtering (Optimized)
    final_vocabulary: CounterType[str] = CounterType()
    final_mapping: Dict[str, List[str]] = {}
    
    for phrase, ctx_set in raw_phrase_contexts.items():
        doc_freq = len(ctx_set)
        
        if doc_freq >= min_freq:
            final_vocabulary[phrase] = doc_freq
            final_mapping[phrase] = sorted(list(ctx_set)) 

    logger.info(f"Final validated vocabulary size: {len(final_vocabulary)} phrases.")
    
    return final_vocabulary, final_mapping

# ============================================================================
# STATISTICS AND OUTPUT
# ============================================================================

def save_phrases(
    phrase_counts: CounterType[str], 
    phrase_to_contexts: Dict[str, List[str]], 
    output_path: Path
):
    r"""
    Persist the extracted vocabulary $\mathcal{V}$ and frequencies to disk.
    Also persist the bipartite graph mapping $phrase \rightarrow [context\_ids]$ to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save the Vocabulary and Frequencies (Corrected format to CSV)
    vocab_csv_path = output_path.with_suffix('.csv')
    logger.info(f"Saving vocabulary and frequencies to: {vocab_csv_path}")
    with open(vocab_csv_path, 'w', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.writer(f)
        for phrase, count in phrase_counts.most_common():
            writer.writerow([phrase, count])

    # 2. Save the Architectural Fix Mapping (for Step 2)
    mapping_path = output_path.parent / "phrase_to_contexts.json"
    logger.info(f"Saving context mapping to: {mapping_path}")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(phrase_to_contexts, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved {len(phrase_counts)} phrases and their mappings.")

def print_statistics(phrase_counts: CounterType[str]):
    """
    [FIXED] Implemented this function to display vocabulary statistics.
    """
    if not phrase_counts:
        logger.warning("No phrases to generate statistics for.")
        return

    print("\n" + "="*50)
    print("      Vocabulary Distributional Statistics")
    print("="*50)
    
    total_phrases = len(phrase_counts)
    total_occurrences = sum(phrase_counts.values())
    
    print(f"  Total unique phrases: {total_phrases:,}")
    print(f"  Total occurrences (sum of freqs): {total_occurrences:,}")
    print(f"  Avg. frequency per phrase: {total_occurrences / total_phrases:.2f}")
    
    # Phrase length distribution
    length_dist = defaultdict(int)
    for phrase in phrase_counts:
        length_dist[len(phrase.split())] += 1
        
    print("\n  Distribution by phrase length (n-grams):")
    for length, count in sorted(length_dist.items()):
        percentage = (count / total_phrases) * 100
        print(f"    - {length}-grams: {count:>7,} phrases ({percentage:.2f}%)")
        
    # Top 10 most common phrases
    print("\n  Top 10 Most Frequent Phrases:")
    for i, (phrase, count) in enumerate(phrase_counts.most_common(10)):
        print(f"    {i+1}. '{phrase}' (count: {count:,})")

    print("="*50 + "\n")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract meaningful phrases (Semantic Vectors) from corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--corpus', type=Path, required=True, help='Path to corpus file (format: context_id,context_text)')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory to save vocabulary.csv and mapping.json')
    parser.add_argument('--keep-verbs', action='store_true', default=False, help='Do not strip verbs during normalization')
    parser.add_argument('--no-spacy', action='store_true', help='Force fallback N-gram extraction')
    parser.add_argument('--no-filter-generic', action='store_true', help='Keep generic single words (lowers signal-to-noise ratio)')
    parser.add_argument('--min-word-length', type=int, default=3, help='Minimum character length $L_{min}$ for single words (default: 3)')
    parser.add_argument('--min-freq', type=int, default=2, help='Sparsity filter threshold $min\\_freq$ (default: 2)')
    parser.add_argument('--stats', action='store_true', help='Print detailed distributional statistics')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.corpus.exists():
        logger.error(f"Corpus file not found: {args.corpus}")
        sys.exit(1)

    logger.info("Configuration:")
    logger.info(f"  Corpus: {args.corpus}")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info(f"  Method: {'Fallback' if args.no_spacy or not SPACY_AVAILABLE else 'spaCy'}")
    logger.info(f"  Filter generic: {not args.no_filter_generic}")
    logger.info(f"  Min frequency threshold: {args.min_freq}")
    logger.info(f"  Keep verbs: {args.keep_verbs}")
    logger.info("")

    phrase_counts, phrase_to_contexts = process_corpus_with_expansion(
        corpus_path=args.corpus,
        use_spacy=not args.no_spacy,
        min_freq=args.min_freq,
        filter_generic=not args.no_filter_generic,
        min_word_length=args.min_word_length,
        keep_verbs=args.keep_verbs
    )
    
    # Define output path for vocabulary file
    output_vocab_path = args.output_dir / 'vocabulary' # Suffix will be added by save_phrases

    save_phrases(phrase_counts, phrase_to_contexts, output_vocab_path)

    if args.stats:
        print_statistics(phrase_counts)

    logger.success("Phrase extraction (Step 1) complete!")

if __name__ == "__main__":
    main()
