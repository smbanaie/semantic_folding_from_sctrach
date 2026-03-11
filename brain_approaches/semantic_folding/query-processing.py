import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from lib import load_fingerprint_cache, load_context_fingerprint_cache, load_phrases
from phrase_extractor import extract_phrases_general
from loguru import logger

def generate_query_fingerprint(query_text: str, fingerprint_cache: Dict[str, np.ndarray], phrases: List[str], grid_size: int) -> np.ndarray:
    query_fingerprint = np.zeros((grid_size, grid_size), dtype=np.int32)
    # This is a very basic implementation, you may want to improve it
    query_phrases = extract_phrases_general(query_text)
    logger.info(f"Query Phrases: {query_phrases}")
    for phrase in phrases:
        if phrase in query_phrases:
            query_fingerprint += fingerprint_cache.get(phrase, np.zeros((grid_size, grid_size)))
    return query_fingerprint

def custom_vector_search(query_fingerprint: np.ndarray, context_fingerprint_cache: Dict[str, np.ndarray], grid_size: int, top_k: int = 3) -> List[Tuple[str, float]]:
    similarities = []
    for context_id, fingerprint in context_fingerprint_cache.items():
        similarity = 0
        for i in range(fingerprint.shape[0]):
            for j in range(fingerprint.shape[1]):
                if fingerprint[i, j] > 0:
                    neighborhood = query_fingerprint[max(0, i-1):min(grid_size, i+2), max(0, j-1):min(grid_size, j+2)]
                    if np.any(neighborhood > 0):
                        similarity += 1
        similarities.append((context_id, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Query processing")
    parser.add_argument("--query_text", required=True, help="Query text")
    parser.add_argument("--fingerprints_dir", required=True, help="Directory containing fingerprint matrices")
    parser.add_argument("--doc_fingerprints_dir", required=True, help="Directory containing DOC fingerprint matrices")
    parser.add_argument("--phrases_path", required=True, help="Path to phrases.txt file")
    parser.add_argument("--grid_size", type=int, default=32, help="Grid size (default: 32)")
    parser.add_argument("--top_k", type=int, default=3, help="Top k results (default: 3)")
    args = parser.parse_args()

    phrases = load_phrases(args.phrases_path)

    doc_fingerprints_dir: str =   args.doc_fingerprints_dir
    fingerprints_dir: str =   args.fingerprints_dir
    query_text: str =   args.query_text
    grid_size : int = args.grid_size 

    doc_fingerprint_cache = load_context_fingerprint_cache(Path(doc_fingerprints_dir))
    phrase_fingerprint_cache = load_fingerprint_cache(Path(fingerprints_dir), phrases, grid_size )

    query_fingerprint = generate_query_fingerprint(query_text, phrase_fingerprint_cache, phrases, args.grid_size)

    similarities = custom_vector_search(query_fingerprint, doc_fingerprint_cache, args.top_k)

    print("Top {} similar contexts:".format(args.top_k))
    for context_id, similarity in similarities:
        print(f"Context ID: {context_id}, Similarity: {similarity}")

if __name__ == "__main__":
    main()

    