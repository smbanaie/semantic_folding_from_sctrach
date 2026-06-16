# Hybrid SF+BM25 Alpha Tuning Results

## Summary

| Dataset | Candidates | Best Alpha | MRR | Pure SF | Pure BM25 | Winner |
|---------|-----------|-----------|-----|---------|-----------|--------|
| **NQ-REaR** | ~10/query | **0.0** | **0.675** | 0.583 | **0.675** | BM25 |
| **PopQA** | 2/query | Any | 1.000 | 1.000 | 1.000 | Tie |
| **PubMedQA** | 3-4/query | Any | 1.000 | 1.000 | 1.000 | Tie |

## NQ-REaR (10 queries) — The Hard Dataset

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

## PopQA (10 queries) — Trivial

| Alpha | MRR |
|-------|-----|
| All | 1.000 |

Both SF and BM25 achieve perfect MRR. Task is too easy (2 passages/query, entity name in query).

## PubMedQA (20 queries) — Biomedical

| Alpha | MRR |
|-------|-----|
| All | 1.000 |

Both SF and BM25 achieve perfect MRR. Task is easy (3-4 passages/query, domain-specific terms).

## Analysis

### Why Hybrid Doesn't Help

1. **NQ-REaR**: BM25 already optimal for factoid QA. SF adds noise.
2. **PopQA/PubMedQA**: Too few candidates. Both methods trivially find gold passage.

### When Would Hybrid Help?

Hybrid scoring helps when:
- **Many candidates** (50+ passages per query)
- **Paraphrased queries** (not exact passage terms)
- **Mixed domain** (some biomedical, some general)
- **Semantic gap** (query uses different vocabulary than passage)

### The Candidate Pool Effect

| Dataset | Candidates | Score Spread | Hybrid Help? |
|---------|-----------|--------------|--------------|
| PopQA | 2 | N/A | No (trivial) |
| PubMedQA | 3-4 | N/A | No (trivial) |
| NQ-REaR | 10 | 11-16% | No (BM25 wins) |
| MuSiQue | 20 | ~20% | Potential |

## Recommendation

| Task Type | Best Approach |
|-----------|---------------|
| Factoid QA (NQ-REaR, PopQA) | BM25 (alpha=0.0) |
| Biomedical QA (PubMedQA) | SF or BM25 (both perfect) |
| Multi-hop QA (MuSiQue) | Test hybrid with 20+ candidates |
| Long-document QA | Test hybrid with many candidates |

## Files

- NQ-REaR: `temp/hybrid_alpha_nq_rear.json`
- PopQA: `temp/hybrid_alpha_popqa.json`
- PubMedQA: `temp/hybrid_alpha_pubmedqa.json`
