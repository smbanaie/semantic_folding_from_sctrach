# Reranker Training Tools

Tools for generating labeled training data and training LambdaMART re-rankers for the Semantic Folding pipeline.

## Overview

The re-ranking cascade architecture (Phase 5) retrieves top-K candidates via SF, then re-ranks them using a learned LambdaMART model trained on 35 features per (query, document) pair.

## Files

| File | Purpose |
|------|---------|
| `generate_training_data.py` | Extract labeled feature vectors from existing benchmark runs |
| `train_model.py` | Train LambdaMART model on extracted features |
| `evaluate_cascade.py` | Evaluate re-ranking cascade on held-out data |

## Workflow

```bash
# Step 1: Generate training data from an indexed run
.venv\Scripts\python -m semantic_folding.tools.generate_training_data \
  --run-dir outputs/belebele_benchmark/runs/run_20260617_144613 \
  --output outputs/belebele_benchmark/training_features.jsonl

# Step 2: Train LambdaMART model
.venv\Scripts\python -m semantic_folding.tools.train_model \
  --features outputs/belebele_benchmark/training_features.jsonl \
  --output outputs/belebele_benchmark/lambdamart_model.txt

# Step 3: Evaluate cascade
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark benchmark \
  --dataset belebele \
  --run-dir outputs/belebele_benchmark/runs/run_20260617_144613 \
  --jsonl data/belebele/converted/belebele.jsonl \
  --query-start 0 --query-end 50 \
  --rerank --rerank-model outputs/belebele_benchmark/lambdamart_model.txt
```

## Multi-Dataset Training

For cross-dataset training (recommended):

```bash
# Generate features from multiple datasets
.venv\Scripts\python -m semantic_folding.tools.generate_training_data \
  --run-dir outputs/musique_benchmark/runs/run_20260617_022257 \
  --output temp/features_musique.jsonl

.venv\Scripts\python -m semantic_folding.tools.generate_training_data \
  --run-dir outputs/belebele_benchmark/runs/run_20260617_144613 \
  --output temp/features_belebele.jsonl

# Combine features
type temp\features_musique.jsonl temp\features_belebele.jsonl > temp\features_combined.jsonl

# Train on combined data
.venv\Scripts\python -m semantic_folding.tools.train_model \
  --features temp/features_combined.jsonl \
  --output temp/lambdamart_combined.txt
```

## Features (35 total)

| Category | Count | Features |
|----------|-------|----------|
| Binary similarity | 5 | Jaccard, Dice, overlap, Hamming, cosine |
| Asymmetric | 3 | Containment, coverage, IDF-weighted |
| Bit-density | 8 | popcounts, densities, intersection, union, mismatch |
| Block histogram | 16 | Per-block Jaccard (16 blocks of 256 bits) |
| Auxiliary | 3 | BM25 score, query length, doc length |

## Dependencies

- `lightgbm >= 4.0` (for LambdaMART training)
- `scikit-learn` (already a project dependency)
- `numpy`, `scipy` (already project dependencies)
