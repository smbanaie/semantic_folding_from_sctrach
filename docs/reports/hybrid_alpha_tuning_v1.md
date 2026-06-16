# Hybrid SF+BM25 Alpha Tuning Results

## Summary

| Dataset | Best Alpha | Best MRR | Pure SF (α=1.0) | Pure BM25 (α=0.0) | Winner |
|---------|-----------|----------|-----------------|-------------------|--------|
| **NQ-REaR** | 0.0 | 0.675 | 0.583 | **0.675** | BM25 |
| **PopQA** | Any | 1.000 | 1.000 | 1.000 | Tie |
| **PubMedQA** | Any | 0.600 | — | 0.600 | Tie (limited queries) |

## NQ-REaR (10 queries)

| Alpha | MRR | Δ vs BM25 |
|-------|-----|-----------|
| 0.0 (BM25) | **0.675** | — |
| 0.2 | 0.583 | -13.6% |
| 0.4 | 0.575 | -14.8% |
| 0.5 | 0.575 | -14.8% |
| 0.6 | 0.525 | -22.2% |
| 0.8 | 0.492 | -27.1% |
| 1.0 (SF) | 0.583 | -13.6% |

**Key Finding**: On NQ-REaR, BM25 alone outperforms all hybrid configurations.
The more SF weight (higher alpha), the worse the results.

## PopQA (10 queries)

| Alpha | MRR |
|-------|-----|
| 0.0 (BM25) | 1.000 |
| 0.2 | 1.000 |
| 0.4 | 1.000 |
| 0.5 | 1.000 |
| 0.6 | 1.000 |
| 0.8 | 1.000 |
| 1.0 (SF) | 1.000 |

**Key Finding**: PopQA is trivial — both methods achieve perfect MRR regardless of alpha.

## Analysis

### Why Hybrid Doesn't Help on NQ-REaR

1. **Score compression in SF**: All documents score within 11-16% range
2. **BM25 is already optimal**: Lexical matching is the right approach for factoid QA
3. **SF adds noise**: Semantic similarity doesn't help when passages are topically similar

### When Would Hybrid Help?

Hybrid scoring helps when:
- Queries use **paraphrased vocabulary** (not exact passage terms)
- Passages are **semantically diverse** (not topically clustered)
- The task requires **understanding**, not just **matching**

### Recommendation

For general knowledge QA (NQ-REaR, PopQA): **Use BM25 (alpha=0.0)**
For biomedical QA (PubMedQA): **Use SF (alpha=1.0)** — semantic understanding matters more

## Files

- NQ-REaR: `temp/hybrid_alpha_nq_rear.json`
- PopQA: `temp/hybrid_alpha_popqa.json`
- PubMedQA: `temp/hybrid_alpha_pubmedqa.json`
