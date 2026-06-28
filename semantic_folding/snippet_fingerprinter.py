#!/usr/bin/env python3
"""
snippet_fingerprinter.py — Snippet-level fingerprinting for Semantic Folding.

Creates per-snippet fingerprints using sliding windows over document text,
then uses max-pooling to compute document scores from snippet scores.

Usage:
    python -m semantic_folding.snippet_fingerprinter \\
        --corpus corpus.txt \\
        --fingerprints doc_fingerprints/ \\
        --window-size 3 \\
        --stride 2 \\
        --output snippet_fingerprints/
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger

import numpy as np
from scipy.sparse import csr_matrix


def extract_snippets(
    text: str,
    window_size: int = 3,
    stride: int = 2,
) -> List[str]:
    """
    Extract overlapping snippets from text using sentence windowing.
    
    Args:
        text: Input text
        window_size: Number of sentences per snippet
        stride: Step size between snippets
    
    Returns:
        List of snippet strings
    """
    # Split into sentences (simple regex-based)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= window_size:
        return [text]  # Return full text if too short
    
    snippets = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        snippet = ' '.join(sentences[i:i + window_size])
        snippets.append(snippet)
    
    # Add the last snippet if not covered
    if (len(sentences) - window_size) % stride != 0:
        snippet = ' '.join(sentences[-window_size:])
        snippets.append(snippet)
    
    return snippets


def create_snippet_fingerprints(
    doc_fingerprints: Dict[str, csr_matrix],
    doc_texts: Dict[str, str],
    phrase_fingerprints: Dict[str, csr_matrix],
    grid_size: int = 64,
    window_size: int = 3,
    stride: int = 2,
) -> Dict[str, List[Tuple[str, csr_matrix]]]:
    """
    Create snippet-level fingerprints for all documents.
    
    Args:
        doc_fingerprints: Dict mapping doc_id -> document fingerprint
        doc_texts: Dict mapping doc_id -> document text
        phrase_fingerprints: Dict mapping phrase -> phrase fingerprint
        grid_size: Grid size
        window_size: Snippet window size (sentences)
        stride: Stride between snippets
    
    Returns:
        Dict mapping doc_id -> list of (snippet_id, snippet_fingerprint)
    """
    from phrase_extractor import extract_query_phrases
    from lib import expand_phrases, normalize_phrase
    
    snippet_fps = {}
    
    for doc_id, doc_text in doc_texts.items():
        if doc_id not in doc_fingerprints:
            continue
        
        # Extract snippets
        snippets = extract_snippets(doc_text, window_size, stride)
        
        doc_snippets = []
        for i, snippet in enumerate(snippets):
            snippet_id = f"{doc_id}_snippet_{i:04d}"
            
            # Extract phrases from snippet
            snippet_phrases = []
            words = snippet.lower().split()
            for phrase in phrase_fingerprints.keys():
                if phrase in snippet.lower():
                    snippet_phrases.append(phrase)
            
            # Build snippet fingerprint by accumulating phrase fingerprints
            if snippet_phrases:
                acc = np.zeros(grid_size * grid_size, dtype=np.float32)
                for phrase in snippet_phrases:
                    if phrase in phrase_fingerprints:
                        fp = phrase_fingerprints[phrase]
                        if hasattr(fp, 'toarray'):
                            fp_dense = fp.toarray().ravel()
                        else:
                            fp_dense = np.asarray(fp).ravel()
                        acc += fp_dense
                
                snippet_fp = csr_matrix(acc.reshape(1, -1))
                doc_snippets.append((snippet_id, snippet_fp))
        
        snippet_fps[doc_id] = doc_snippets
    
    logger.info(f"Created snippet fingerprints for {len(snippet_fps)} documents")
    return snippet_fps


def max_pool_snippet_scores(
    doc_id: str,
    query_fp: csr_matrix,
    snippet_fps: List[Tuple[str, csr_matrix]],
) -> float:
    """
    Compute document score as max of snippet scores.
    
    Args:
        doc_id: Document ID
        query_fp: Query fingerprint
        snippet_fps: List of (snippet_id, snippet_fingerprint)
    
    Returns:
        Max snippet score
    """
    if not snippet_fps:
        return 0.0
    
    max_score = 0.0
    for snippet_id, snippet_fp in snippet_fps:
        if snippet_fp.nnz == 0:
            continue
        
        # Compute cosine similarity
        query_norm = np.sqrt(query_fp.power(2).sum())
        snippet_norm = np.sqrt(snippet_fp.power(2).sum())
        
        if query_norm < 1e-9 or snippet_norm < 1e-9:
            continue
        
        dot = float(query_fp.dot(snippet_fp.T).toarray()[0, 0])
        score = dot / (query_norm * snippet_norm)
        
        max_score = max(max_score, score)
    
    return max_score


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Snippet fingerprinting for Semantic Folding")
    parser.add_argument("--corpus", required=True, help="Corpus file path")
    parser.add_argument("--fingerprints", required=True, help="Document fingerprints directory")
    parser.add_argument("--phrase-fingerprints", required=True, help="Phrase fingerprints directory")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size")
    parser.add_argument("--window-size", type=int, default=3, help="Snippet window size (sentences)")
    parser.add_argument("--stride", type=int, default=2, help="Stride between snippets")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Load document texts
    doc_texts = {}
    with open(args.corpus, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                comma_idx = line.find(',')
                if comma_idx > 0:
                    doc_id = line[:comma_idx]
                    text = line[comma_idx+1:]
                    doc_texts[doc_id] = text
    
    # Load document fingerprints
    from lib import load_document_fingerprints
    doc_fps, _ = load_document_fingerprints(Path(args.fingerprints))
    
    # Load phrase fingerprints
    from lib import load_phrase_fingerprints_sparse
    phrase_fps = load_phrase_fingerprints_sparse(Path(args.phrase_fingerprints), args.grid_size)
    
    # Create snippet fingerprints
    snippet_fps = create_snippet_fingerprints(
        doc_fps, doc_texts, phrase_fps,
        args.grid_size, args.window_size, args.stride
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save snippet fingerprints
    for doc_id, snippets in snippet_fps.items():
        for snippet_id, snippet_fp in snippets:
            np.save(output_dir / f"{snippet_id}.npy", snippet_fp.toarray())
    
    logger.success(f"Saved snippet fingerprints to {output_dir}")
