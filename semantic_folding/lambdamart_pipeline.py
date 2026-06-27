#!/usr/bin/env python3
"""
lambdamart_pipeline.py — End-to-end LambdaMART training and evaluation.

Steps:
1. Load fingerprints from an existing benchmark run
2. Extract features for all (query, doc) pairs
3. Label gold documents
4. Train LambdaMART
5. Evaluate with cross-validation

Usage:
    # Train on one dataset, evaluate on same
    python -m semantic_folding.lambdamart_pipeline \\
        --run-dir outputs/belebele_benchmark/runs/run_20260627_094213 \\
        --dataset belebele \\
        --output models/lambdamart_belebele.txt

    # Cross-dataset: train on A, evaluate on B
    python -m semantic_folding.lambdamart_pipeline \\
        --train-run-dir outputs/belebele_benchmark/runs/run_A \\
        --eval-run-dir outputs/nq_rear_benchmark/runs/run_B \\
        --train-dataset belebele \\
        --eval-dataset nq_rear \\
        --output models/lambdamart_cross.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lib import (
    load_document_fingerprints,
    load_phrase_fingerprints_sparse,
    normalize_phrase,
    merge_fingerprints,
    get_logger,
)

logger = get_logger("lambdamart_pipeline")

# Import reranker modules
from reranker_features import extract_features
from reranker_train import (
    prepare_training_data,
    train_lambdamart,
    cross_dataset_evaluate,
    FEATURE_NAMES,
)


def load_run_data(run_dir: Path, dataset: str, max_queries: int = 500):
    """Load all data from a benchmark run directory."""
    logger.info(f"Loading run data from {run_dir}")

    # Load fingerprints
    phrase_fps = load_phrase_fingerprints_sparse(
        run_dir / "phrase_fingerprints", grid_size=64
    )
    doc_fps, _doc_meta = load_document_fingerprints(run_dir / "doc_fingerprints")
    logger.info(f"  {len(phrase_fps)} phrases, {len(doc_fps)} documents")

    # IDF weights (phrase→value dict; reranker needs bit-indexed array, skip for now)
    idf_weights = None

    # Load query-doc mapping and gold labels (keyed by integer index as string)
    with open(run_dir / "query_doc_map.json", encoding="utf-8") as f:
        raw_qdm = json.load(f)
    with open(run_dir / "query_gold.json", encoding="utf-8") as f:
        raw_qg = json.load(f)

    # Load query texts from JSONL
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    jsonl_path = Path(meta.get("source_jsonl", ""))
    query_texts = []
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    query_texts.append(entry.get("question", entry.get("query", "")))
    query_texts = query_texts[:max_queries]

    # Build index→query_text mapping and re-key maps by query text
    query_doc_map = {}
    query_gold = {}
    for idx_str, doc_ids in raw_qdm.items():
        idx = int(idx_str)
        if idx >= len(query_texts):
            continue
        qt = query_texts[idx]
        query_doc_map[qt] = doc_ids

    for idx_str, gold_doc_ids in raw_qg.items():
        idx = int(idx_str)
        if idx >= len(query_texts):
            continue
        qt = query_texts[idx]
        # gold is a list of doc_ids → convert to {doc_id: 1} dict
        query_gold[qt] = {did: 1 for did in gold_doc_ids}

    n_gold = sum(len(v) for v in query_gold.values())
    logger.info(f"  {len(query_texts)} queries, {n_gold} gold docs")

    return {
        "phrase_fps": phrase_fps,
        "doc_fps": doc_fps,
        "idf_weights": idf_weights,
        "query_doc_map": query_doc_map,
        "query_gold": query_gold,
        "query_texts": query_texts,
    }


def build_query_fingerprint(query_text: str, phrase_fps: dict) -> Optional[np.ndarray]:
    """Build a query fingerprint from text."""
    tokens = query_text.lower().split()
    phrases = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 5, len(tokens) + 1)):
            phrase = " ".join(tokens[i:j])
            norm = normalize_phrase(phrase)
            if norm and norm in phrase_fps:
                phrases.append(norm)

    if not phrases:
        return None

    fps_list = [phrase_fps[p] for p in phrases]
    weights = [1.0] * len(fps_list)
    query_fp = merge_fingerprints(fps_list, weights)
    return query_fp


def extract_features_for_dataset(
    run_data: dict,
    dataset_name: str,
    max_queries: int = 500,
) -> List[Dict]:
    """Extract features for all (query, doc) pairs in a dataset."""
    phrase_fps = run_data["phrase_fps"]
    doc_fps = run_data["doc_fps"]
    idf_weights = run_data["idf_weights"]
    query_doc_map = run_data["query_doc_map"]
    query_gold = run_data["query_gold"]
    query_texts = run_data["query_texts"]

    all_features = []
    n_labeled = 0

    for q_idx, query_text in enumerate(query_texts[:max_queries]):
        if q_idx % 10 == 0:
            logger.info(f"  [{q_idx}/{min(len(query_texts), max_queries)}] Extracting features...")

        # Build query fingerprint
        query_fp = build_query_fingerprint(query_text, phrase_fps)
        if query_fp is None:
            logger.warning(f"  [{q_idx}] No valid phrases for query: {query_text[:60]}...")
            continue

        # Get candidate docs for this query
        doc_ids = query_doc_map.get(query_text, list(doc_fps.keys())[:20])

        # Get gold labels
        gold = query_gold.get(query_text, {})
        # Handle both string and int keys
        gold_set = set()
        for k, v in gold.items():
            if v == 1 or v == "1":
                gold_set.add(k)

        query_length = len(query_text.split())

        for doc_id in doc_ids:
            if doc_id not in doc_fps:
                continue

            doc_fp = doc_fps[doc_id]
            doc_length = doc_fp.nnz

            feats = extract_features(
                query_fp, doc_fp,
                idf_weights=idf_weights,
                query_length=query_length,
                doc_length=doc_length,
            )

            label = 1 if doc_id in gold_set else 0
            if label == 1:
                n_labeled += 1

            all_features.append({
                "doc_id": doc_id,
                "query_text": query_text,
                "query_idx": q_idx,
                "features": feats,
                "label": label,
                "dataset": dataset_name,
            })

    logger.info(f"  Extracted {len(all_features)} feature vectors ({n_labeled} positive)")
    return all_features


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end LambdaMART training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Benchmark run directory (contains phrase_fingerprints/, doc_fingerprints/)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., belebele)")
    parser.add_argument("--eval-run-dir", type=Path, default=None,
                        help="Evaluation run directory (for cross-dataset evaluation)")
    parser.add_argument("--eval-dataset", type=str, default=None,
                        help="Evaluation dataset name")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output model path")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--num-trees", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Step 1: Load training data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading training data")
    logger.info("=" * 60)
    train_data = load_run_data(args.run_dir, args.dataset, args.max_queries)

    # Step 2: Extract features
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting features")
    logger.info("=" * 60)
    train_features = extract_features_for_dataset(
        train_data, args.dataset, args.max_queries
    )

    if not train_features:
        logger.error("No features extracted — cannot train")
        sys.exit(1)

    # Step 3: Prepare training data
    logger.info("=" * 60)
    logger.info("STEP 3: Preparing training data")
    logger.info("=" * 60)
    gold_labels = {}
    for feat_dict in train_features:
        qt = feat_dict["query_text"]
        doc_id = feat_dict["doc_id"]
        label = feat_dict["label"]
        if qt not in gold_labels:
            gold_labels[qt] = {}
        gold_labels[qt][doc_id] = label

    X, y, groups, feature_names = prepare_training_data(train_features, gold_labels)
    logger.info(f"  Feature matrix: {X.shape}, positive rate: {np.mean(y > 0):.3f}")

    # Step 4: Train LambdaMART
    logger.info("=" * 60)
    logger.info("STEP 4: Training LambdaMART")
    logger.info("=" * 60)
    t0 = time.time()
    model, train_info = train_lambdamart(
        X, y, feature_names, groups=groups,
        num_trees=args.num_trees,
        learning_rate=args.learning_rate,
    )
    train_time = time.time() - t0
    logger.info(f"  Training completed in {train_time:.1f}s")

    if model is None:
        logger.error("Training failed")
        sys.exit(1)

    # Step 5: Save model
    logger.info("=" * 60)
    logger.info("STEP 5: Saving model")
    logger.info("=" * 60)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    logger.info(f"  Model saved to {args.output}")

    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "feature_names": feature_names,
            "train_info": {k: v for k, v in train_info.items() if k != "feature_importance"},
            "top_features": dict(sorted(
                train_info.get("feature_importance", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
            "args": vars(args),
        }, f, indent=2, default=str)

    # Step 6: Evaluate (same dataset or cross-dataset)
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluation")
    logger.info("=" * 60)

    if args.eval_run_dir and args.eval_dataset:
        logger.info(f"Cross-dataset evaluation: train={args.dataset} -> eval={args.eval_dataset}")
        eval_data = load_run_data(args.eval_run_dir, args.eval_dataset, args.max_queries)
        eval_features = extract_features_for_dataset(
            eval_data, args.eval_dataset, args.max_queries
        )
    else:
        logger.info(f"Same-dataset evaluation on {args.dataset}")
        eval_features = train_features

    eval_gold = {}
    for feat_dict in eval_features:
        qt = feat_dict["query_text"]
        doc_id = feat_dict["doc_id"]
        label = feat_dict["label"]
        if qt not in eval_gold:
            eval_gold[qt] = {}
        eval_gold[qt][doc_id] = label

    metrics = cross_dataset_evaluate(model, eval_features, eval_gold)
    logger.info(f"  Results: {json.dumps(metrics, indent=2)}")

    # Compare with SF-only baseline
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Training samples: {len(train_features)}")
    logger.info(f"  Evaluation queries: {metrics.get('n_queries', 0)}")
    logger.info(f"  LambdaMART MRR: {metrics.get('mrr', 0):.4f}")
    logger.info(f"  LambdaMART P@1: {metrics.get('p@1', 0):.4f}")
    logger.info(f"  Training time: {train_time:.1f}s")
    logger.info(f"  Model: {args.output}")


if __name__ == "__main__":
    main()
