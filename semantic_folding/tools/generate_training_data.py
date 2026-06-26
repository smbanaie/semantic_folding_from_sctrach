#!/usr/bin/env python3
"""
Generate labeled training data for LambdaMART re-ranker.

Reads existing benchmark runs (indexed via generic_benchmark.py) and generates
feature vectors with gold labels for each (query, document) pair.

Supports any dataset that has a completed index run with:
  - doc_fingerprints/
  - phrase_fingerprints/
  - term_context_matrix/idf_weights.json
  - corpus.txt
  - query_gold.json
  - query_doc_map.json

Usage:
    .venv\\Scripts\\python -m semantic_folding.tools.generate_training_data \\
        --run-dir outputs/belebele_benchmark/runs/run_20260617_144613 \\
        --jsonl data/belebele/converted/belebele.jsonl \\
        --output outputs/belebele_benchmark/training_features.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import (
    load_document_fingerprints,
    load_phrase_fingerprints_sparse,
    normalize_phrase,
    merge_fingerprints,
    get_logger,
)
from reranker_features import extract_features
from reranker_train import FEATURE_NAMES

logger = get_logger("generate_training_data")


def load_corpus_texts(corpus_path: Path) -> Tuple[List[str], List[str]]:
    """Load corpus texts as (doc_id_list, text_list)."""
    doc_id_list = []
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                comma_idx = line.find(",")
                if comma_idx > 0:
                    doc_id_list.append(line[:comma_idx].strip())
                    texts.append(line[comma_idx + 1:].strip())
    return doc_id_list, texts


def load_idf_dict(idf_path: Path) -> Dict[str, float]:
    """Load IDF weights keyed by phrase text."""
    with open(idf_path, encoding="utf-8") as f:
        return json.load(f)


def extract_query_phrases(
    query_text: str,
    phrase_fps: Dict[str, csr_matrix],
) -> List[str]:
    """Extract and normalize query phrases that exist in the vocabulary."""
    phrases = []
    tokens = query_text.lower().split()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 5, len(tokens) + 1)):
            phrase = " ".join(tokens[i:j])
            norm = normalize_phrase(phrase)
            if norm and norm in phrase_fps:
                phrases.append(norm)
    return list(set(phrases))


def build_query_fingerprint(
    phrases: List[str],
    phrase_fps: Dict[str, csr_matrix],
    idf_dict: Optional[Dict[str, float]] = None,
) -> Optional[csr_matrix]:
    """Build query fingerprint from phrases, weighted by IDF if available."""
    fps_list = []
    weights = []
    for p in phrases:
        if p in phrase_fps:
            fps_list.append(phrase_fps[p])
            if idf_dict is not None and p in idf_dict:
                weights.append(idf_dict[p])
            else:
                weights.append(1.0)

    if not fps_list:
        return None

    return merge_fingerprints(fps_list, weights)


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled training data for LambdaMART re-ranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to indexed benchmark run directory")
    parser.add_argument("--jsonl", type=Path, required=True,
                        help="Original JSONL with query text (used during indexing)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file for training features")
    parser.add_argument("--grid-size", type=int, default=64,
                        help="Grid size (must match the run)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Maximum number of queries to process")
    args = parser.parse_args()

    run_dir = args.run_dir
    logger.info(f"Loading run from {run_dir}")

    # Validate run directory
    required_files = [
        "doc_fingerprints", "phrase_fingerprints",
        "query_gold.json", "query_doc_map.json", "corpus.txt",
    ]
    for rf in required_files:
        path = run_dir / rf
        if not path.exists():
            logger.error(f"Missing required file/dir: {path}")
            sys.exit(1)

    # Load fingerprints
    doc_fps, doc_meta = load_document_fingerprints(run_dir / "doc_fingerprints")
    phrase_fps = load_phrase_fingerprints_sparse(
        run_dir / "phrase_fingerprints", args.grid_size
    )
    doc_id_list = list(doc_fps.keys())
    logger.info(f"  {len(doc_fps)} documents, {len(phrase_fps)} phrases")

    # Load IDF weights (keyed by phrase text)
    idf_path = run_dir / "term_context_matrix" / "idf_weights.json"
    idf_dict = None
    if idf_path.exists():
        idf_dict = load_idf_dict(idf_path)
        logger.info(f"  Loaded IDF weights: {len(idf_dict)} phrases")

    # Load gold labels
    with open(run_dir / "query_gold.json", encoding="utf-8") as f:
        gold_labels = json.load(f)
    logger.info(f"  {len(gold_labels)} queries with gold labels")

    # Load query-doc map (contains candidate doc IDs)
    with open(run_dir / "query_doc_map.json", encoding="utf-8") as f:
        query_doc_map = json.load(f)
    logger.info(f"  {len(query_doc_map)} query-doc mappings")

    # Load original JSONL to get query text (supports both formats)
    query_texts = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Format 1: query_idx + question (from create_query_jsonl)
            if "query_idx" in entry:
                q_idx = entry["query_idx"]
                query_text = entry.get("question", entry.get("query", ""))
            # Format 2: sequential with question field (belebele, etc.)
            else:
                q_idx = entry.get("query_idx", entry.get("idx", len(query_texts)))
                query_text = entry.get("query", entry.get("question", ""))
            if query_text:
                query_texts[q_idx] = query_text
    logger.info(f"  Loaded {len(query_texts)} query texts from JSONL")

    # Load corpus texts for BM25
    corpus_texts = []
    corpus_path = run_dir / "corpus.txt"
    if corpus_path.exists():
        _, texts = load_corpus_texts(corpus_path)
        corpus_texts = texts
        logger.info(f"  Loaded {len(corpus_texts)} corpus texts")

    # Compute BM25 scores for all queries
    bm25_scores_per_query: Dict[int, Dict[str, float]] = {}
    if corpus_texts:
        try:
            from query_processor import BM25Scorer
            bm25 = BM25Scorer(corpus_texts)
            for q_idx_str in gold_labels:
                q_idx = int(q_idx_str)
                query_text = query_texts.get(q_idx, "")
                if query_text:
                    bm25_scores = bm25.score_all(query_text)
                    bm25_scores_per_query[q_idx] = {
                        doc_id_list[i]: score for i, score in bm25_scores
                    }
            logger.info(f"  Computed BM25 scores for {len(bm25_scores_per_query)} queries")
        except Exception as e:
            logger.warning(f"  BM25 computation failed: {e}")

    # Generate features
    all_features = []
    query_count = 0

    for q_idx_str, gold_docs in gold_labels.items():
        q_idx = int(q_idx_str)

        if args.max_queries and query_count >= args.max_queries:
            break

        # Get query text from JSONL entries
        query_text = query_texts.get(q_idx, "")

        if not query_text:
            logger.warning(f"  [{q_idx}] No query text found, skipping")
            continue

        logger.info(f"  [{q_idx}] {query_text[:70]}...")

        # Extract phrases and build query fingerprint
        phrases = extract_query_phrases(query_text, phrase_fps)
        if not phrases:
            logger.warning(f"  [{q_idx}] No valid phrases found in vocabulary")
            continue

        query_fp = build_query_fingerprint(phrases, phrase_fps, idf_dict)
        if query_fp is None:
            logger.warning(f"  [{q_idx}] Failed to build query fingerprint")
            continue

        query_tokens = query_text.split()
        bm25_scores = bm25_scores_per_query.get(q_idx, {})
        gold_set = set(gold_docs) if isinstance(gold_docs, list) else set()

        # Extract features for each document
        for doc_id in doc_id_list:
            doc_fp = doc_fps[doc_id]
            bm25_score = bm25_scores.get(doc_id, 0.0)

            feats = extract_features(
                query_fp, doc_fp,
                idf_weights=None,  # IDF dict is used at phrase level, not bit level
                bm25_score=bm25_score,
                query_length=len(query_tokens),
            )

            label = 1 if doc_id in gold_set else 0

            all_features.append({
                "query_text": query_text,
                "query_idx": q_idx,
                "doc_id": doc_id,
                "features": feats,
                "label": label,
            })

        query_count += 1

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for feat_dict in all_features:
            f.write(json.dumps(feat_dict, default=lambda o: float(o) if hasattr(o, 'item') else o) + "\n")

    # Stats
    n_queries = len(set(f["query_idx"] for f in all_features))
    n_positive = sum(1 for f in all_features if f["label"] == 1)
    n_negative = sum(1 for f in all_features if f["label"] == 0)

    logger.info(f"\nGenerated {len(all_features)} feature vectors:")
    logger.info(f"  Queries: {n_queries}")
    logger.info(f"  Positive (gold): {n_positive}")
    logger.info(f"  Negative: {n_negative}")
    logger.info(f"  Features per vector: {len(FEATURE_NAMES)}")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
