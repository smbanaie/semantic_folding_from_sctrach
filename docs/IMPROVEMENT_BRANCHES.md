# Improvement Branches Summary

**Date:** 2026-06-13  
**Goal:** Improve semantic folding to beat BM25 on Belebele dataset

---

## Branches Created

| Branch | Improvement | Status | Description |
|--------|-------------|--------|-------------|
| `feature/hybrid-scoring` | Hybrid SF+BM25 | ✅ Ready | Combine semantic folding with BM25 scores |
| `feature/l2-doc-normalization` | L2 Doc Normalization | ✅ Ready | Try L2 instead of sqrt(nnz) for doc scoring |
| `feature/query-expansion` | Query Expansion | ✅ Ready | Add medical synonyms to expand queries |
| `feature/tfidf-reranking` | TF-IDF Re-ranking | ✅ Ready | Post-SF lexical boost with TF-IDF |
| `feature/tsne-perplexity` | t-SNE Perplexity | ✅ Ready | Test perplexity values 10, 30, 50 |

---

## How to Test Each Branch

```bash
# 1. Hybrid scoring
git checkout feature/hybrid-scoring
.venv\Scripts\python semantic_folding\dataset_benchmark\generic_benchmark.py all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl \
  --max-queries 50

# 2. L2 normalization
git checkout feature/l2-doc-normalization
.venv\Scripts\python semantic_folding\dataset_benchmark\generic_benchmark.py all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl \
  --max-queries 50

# 3. Query expansion
git checkout feature/query-expansion
.venv\Scripts\python semantic_folding\dataset_benchmark\generic_benchmark.py all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl \
  --max-queries 50

# 4. TF-IDF re-ranking
git checkout feature/tfidf-reranking
.venv\Scripts\python semantic_folding\dataset_benchmark\generic_benchmark.py all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl \
  --max-queries 50

# 5. t-SNE perplexity testing
git checkout feature/tsne-perplexity
# Edit config to test perplexity=10, 50
```

---

## Baseline Results (for comparison)

| Dataset | SF MRR | BM25 MRR | Target |
|---------|--------|----------|--------|
| Belebele | 0.740 | 0.995 | > 0.995 |
| Belebele | 0.840 | 0.995 | > 0.860 |

---

## Decision Criteria (from RECOMMENDATIONS.md)

- **MRR improvement > 0.01** → adopt
- **MRR within ±0.005, but AP improves > 0.02** → adopt (better ranking)
- **Failure count drops** → adopt even if MRR flat
- **Otherwise** → reject and document

---

## Next Steps

1. Test each branch on Belebele (50 queries)
2. Compare MRR, AP, P@1 results
3. Select best performing branch(es)
4. Merge into main
5. Run full benchmark (100 queries)
6. Update reports.md with results
