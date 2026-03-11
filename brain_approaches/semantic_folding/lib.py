import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
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
from scipy.sparse import hstack
from rich import print
from typing import List
from loguru import logger
import numpy as np
from typing import Optional

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def load_phrases(phrases_path: Path) -> List[str]:
    """Load phrases from file"""
    logger.info(f"Loading phrases from: {phrases_path}")

    phrases = []
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                phrase = line.split(':', 1)[0].strip()
                if phrase:
                    phrases.append(phrase)

    logger.success(f"Loaded {len(phrases)} phrases")
    return phrases

def load_context_fingerprint_cache(context_fingerprints_dir: Path) -> Dict[str, np.ndarray]:
    cache = {}
    for file in context_fingerprints_dir.glob("*.txt"):
        context_id = file.stem
        with open(file, 'r', encoding='utf-8') as f:
            matrix_data = []
            for line in f:
                row = [int(x) for x in line.strip().split('\t')]
                matrix_data.append(row)
        fingerprint = np.array(matrix_data)
        cache[context_id] = fingerprint
    return cache

def load_fingerprint_cache(fingerprints_dir: Path,
                          phrases: List[str],
                          grid_size: int) -> Dict[str, Optional[np.ndarray]]:
    """Load and cache fingerprint matrices"""
    logger.info(f"Loading fingerprint matrices from: {fingerprints_dir}")

    cache = {}
    loaded_count = 0

    for phrase in phrases:
        # Create safe filename (same logic as in phrase_fingerprints.py)
        safe_name = "".join(c for c in phrase if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')[:50]

        fingerprint_file = fingerprints_dir / f"{safe_name}_fingerprint.txt"

        try:
            if fingerprint_file.exists():
                with open(fingerprint_file, 'r', encoding='utf-8') as f:
                    matrix_data = []
                    for line in f:
                        row = [int(x) for x in line.strip().split('\t')]
                        matrix_data.append(row)

                fingerprint = np.array(matrix_data)
                if fingerprint.shape == (grid_size, grid_size):
                    cache[phrase] = fingerprint
                    loaded_count += 1
                else:
                    logger.warning(f"Fingerprint for phrase '{phrase}' has wrong shape: {fingerprint.shape}")
                    cache[phrase] = None
            else:
                cache[phrase] = None

        except Exception as e:
            logger.warning(f"Failed to load fingerprint for phrase '{phrase}': {e}")
            cache[phrase] = None

    logger.success(f"Loaded {loaded_count}/{len(phrases)} fingerprint matrices")
    return cache

# Function to convert POS tags to WordNet format
def get_wordnet_pos(treebank_tag):
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

# Function to lemmatize a list of words
def lemmatize_words(words):
    # POS tagging
    tagged_words = pos_tag(words)
    # Lemmatize each word based on its POS tag
    lemmatized_words = []
    for word, tag in tagged_words:
        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=pos)
        lemmatized_words.append(lemma)
    return lemmatized_words

en_stop_words = set[str](stopwords.words('english'))

def expand_phrases(phrases: List[str]) -> List[str]:
    """
    Expand phrase list:
    - 2-word → add singles
    - 3-word → add all 2-word & 1-word combos
    - 4+ word → add all 3-, 2-, 1-word combos
    Then:
    - Filter out stop words and pure verb phrases.
    """

    expanded = set(phrases)

    # --- Expansion logic ---
    for phrase in phrases:
        words = phrase.split()
        n = len(words)

        if n == 2:
            expanded.update(words)
        elif n == 3:
            expanded.update([' '.join(words[i:i+2]) for i in range(n - 1)])
            expanded.update(words)
        elif n >= 4:
            expanded.update([' '.join(words[i:i+3]) for i in range(n - 2)])
            expanded.update([' '.join(words[i:i+2]) for i in range(n - 1)])
            expanded.update(words)

    # --- NLP cleanup phase ---
    cleaned_phrases = set()
    for ph in expanded:
        tokens = word_tokenize(ph.lower())
        if not tokens:
            continue

        # POS tag and lemmatize
        tagged = pos_tag(tokens)
        lemmas = [
            lemmatizer.lemmatize(w, get_wordnet_pos(t))
            for w, t in tagged
            if w.isalpha() and w not in en_stop_words
        ]

        # Skip short or uninformative phrases
        if not lemmas:
            continue

        # Skip if all are verbs
        pos_tags = [t for _, t in tagged]
        if all(tag.startswith('V') for tag in pos_tags):
            continue

        cleaned_phrases.add(' '.join(lemmas))

    return list(cleaned_phrases)


def remove_edge_stop_words_archive(text: str) -> str:
    words = text.split() 
    if len(words) >=1 :
            edge_stopwords = ['must','and', 'or', 'the', 'a', 'an', 'which', 'it','of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'about','can', 'these', 'this', 'those', 'their','our', 'my', 'has been', 'have been', 'that' ,'the', 'also', 'how', 'are', 'its', 'be', 'they']
            for stop_word in edge_stopwords:
                if text.startswith(stop_word+" ") : 
                    return text.replace(stop_word,"").strip()
                if text.endswith(" "+stop_word) : 
                    return text.replace(stop_word,"").strip()
    elif text in en_stop_words: 
        return ''
    return text
                    
# Preprocessing function for unigrams
def process_lemmatize_archive(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters, but preserve hyphens
    text = re.sub(r'[^\w\s-]', '', text)  # Keep hyphens
    # Tokenize
    text = remove_edge_stop_words_archive(text)

    tokens = word_tokenize(text)
    # Lemmatize tokens
    lemmatized_tokens = lemmatize_words(tokens)
    return ' '.join(lemmatized_tokens)


def remove_edge_stop_words(text: str) -> str:
    """
    Remove stop words that appear at the beginning or end of the phrase,
    preserving meaningful content. Uses token-level NLP logic.
    """
    tokens = word_tokenize(text.lower())
    if not tokens:
        return text

    # Remove stopwords only at beginning and end positions
    start_idx = 0
    end_idx = len(tokens)

    while start_idx < end_idx and tokens[start_idx] in en_stop_words:
        start_idx += 1
    while end_idx > start_idx and tokens[end_idx - 1] in en_stop_words:
        end_idx -= 1

    trimmed_tokens = tokens[start_idx:end_idx]
    return ' '.join(trimmed_tokens)

def process_lemmatize(text: str) -> str:
    """
    Normalize text for feature extraction:
    - Lowercase
    - Remove punctuation/special chars (preserve hyphens)
    - Tokenize
    - Remove stop words
    - Lemmatize tokens to base form
    - Remove verbs (keep nouns, adjectives, adverbs, etc.)
    """
    text = text.lower()
    # Clean punctuation and special characters, preserving hyphens
    text = re.sub(r'[^\w\s-]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # POS tagging for filtering and lemmatization
    tagged_tokens = pos_tag(tokens)

    processed = []
    for word, tag in tagged_tokens:
        # Skip stopwords first
        if word in en_stop_words:
            continue

        # Skip verbs entirely
        if tag.startswith('V'):
            continue

        # Lemmatize remaining tokens by their part of speech
        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=pos)

        # Only keep alphabetic tokens (no numbers, punctuation)
        if lemma.isalpha():
            processed.append(lemma)

    # Remove leading/trailing stopwords at phrase level as extra cleanup
    final_text = remove_edge_stop_words(' '.join(processed))
    return final_text


def load_contexts(corpus_path: Path) -> List[Tuple[str, str]]:
    """Load contexts from corpus file"""
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
            context_text = process_lemmatize(context_text)

            if context_id and context_text:
                contexts.append((context_id, context_text))

    logger.success(f"Loaded {len(contexts)} contexts")
    return contexts


def load_contexts_dict(corpus_path: Path) -> Dict[str, str]:
    """Load context texts from corpus file"""
    logger.info(f"Loading context texts from: {corpus_path}")

    contexts = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue

            context_id, context_text = line.split(',', 1)
            context_id = context_id.strip()
            contexts[context_id] = context_text.strip()

    logger.success(f"Loaded {len(contexts)} context texts")
    return contexts