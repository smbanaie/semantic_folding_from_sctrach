# HippoRAG2 Reading Comprehension Benchmark Results

**Date**: 2026-06-16
**Method**: Semantic Folding (recommended settings) vs BM25
**Datasets**: 5 HippoRAG2 reading comprehension datasets (50 queries each)

## Configuration (SF Recommended)

| Parameter | Value |
|-----------|-------|
| Grid size | 64 |
| Spreading | radius=1, decay=0.5 |
| Top percent | 0.10 |
| Weighting | IDF |
| Normalization | L2 (sqrt_nnz for docs) |
| Perplexity | 30 |

## Results

### SF vs BM25 Comparison

| Dataset | Type | SF MRR | BM25 MRR | Δ | Winner |
|---------|------|--------|----------|---|--------|
| **PopQA** | Entity lookup | **1.000** | 1.000 | 0% | Tie |
| **NarrativeQA** | Script comprehension | 0.939 | **0.980** | -4.2% | BM25 |
| **2WikiMultihopQA** | Multi-hop compositional | 0.788 | **0.921** | -14.4% | BM25 |
| **HotpotQA** | Multi-hop Wikipedia | 0.726 | **0.869** | -16.4% | BM25 |
| **NQ-REaR** | Factoid retrieval | 0.521 | **0.625** | -16.6% | BM25 |

### Ranking by SF Performance

| Rank | Dataset | SF MRR | BM25 MRR | Gap |
|------|---------|--------|----------|-----|
| 1 | PopQA | 1.000 | 1.000 | 0% |
| 2 | NarrativeQA | 0.939 | 0.980 | -4.2% |
| 3 | 2WikiMultihopQA | 0.788 | 0.921 | -14.4% |
| 4 | HotpotQA | 0.726 | 0.869 | -16.4% |
| 5 | NQ-REaR | 0.521 | 0.625 | -16.6% |

## Key Findings

### 1. BM25 Dominates All Datasets
BM25 outperforms SF on 4/5 datasets (tied on PopQA). The gap ranges from -4.2% (NarrativeQA) to -16.6% (NQ-REaR).

### 2. SF Strongest on NarrativeQA (MRR=0.939)
NarrativeQA requires understanding movie scripts — semantic understanding helps here. SF is only 4.2% behind BM25, much closer than on other datasets.

### 3. SF Weakest on Factoid Retrieval (NQ-REaR, MRR=0.521)
NQ-REaR requires exact term matching — BM25's strength. SF's semantic approach adds noise here.

### 4. Multi-hop Tasks: Large Gap
On HotpotQA (-16.4%) and 2WikiMultihopQA (-14.4%), BM25 significantly outperforms SF. Multi-hop reasoning requires precise term matching across passages.

### 5. BM25 is 100-300x Faster
BM25 runs in ~10s per dataset vs SF's ~1000-2000s. This is because SF requires building fingerprints for the entire corpus.

## Analysis by Task Type

### Single-Hop Tasks
| Dataset | SF MRR | BM25 MRR | Notes |
|---------|--------|----------|-------|
| PopQA | 1.000 | 1.000 | Trivial (2 passages/query) |
| NQ-REaR | 0.521 | 0.625 | BM25 wins by 10.4% |

### Multi-Hop Tasks
| Dataset | SF MRR | BM25 MRR | Notes |
|---------|--------|----------|-------|
| HotpotQA | 0.726 | 0.869 | BM25 wins by 14.3% |
| 2WikiMultihopQA | 0.788 | 0.921 | BM25 wins by 13.3% |

### Comprehension Tasks
| Dataset | SF MRR | BM25 MRR | Notes |
|---------|--------|----------|-------|
| NarrativeQA | 0.939 | 0.980 | BM25 wins by 4.1% (closest) |

## Recommendations

### For Production Use
- **Use BM25** for all reading comprehension tasks
- BM25 is faster (100-300x) and more accurate

### For Research
- **SF's strength is semantic understanding** — it nearly matches BM25 on NarrativeQA
- **SF's weakness is lexical matching** — it struggles on factoid and multi-hop tasks
- **Hybrid SF+BM25** could help on tasks with paraphrased queries

### Future Work
1. Test on MuSiQue (20 candidates, multi-hop)
2. Test on larger candidate pools (50+ passages)
3. Investigate why SF excels on biomedical QA (PubMedQA MRR=0.969)

## Files

- SF results: `temp/hipporag2_all_results.json`
- BM25 results: `temp/hipporag2_bm25_results.json`
- Adapters: `semantic_folding/dataset_benchmark/adapters/`
