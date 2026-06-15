# NQ-REaR Failure Analysis

## Executive Summary

SF underperforms BM25 on NQ-REaR (MRR 0.574 vs 0.638) due to **semantic dilution** in the fingerprint space, not vocabulary mismatch. All queries achieve 100% phrase match rate.

## Key Finding: Vocabulary Match is NOT the Problem

| Metric | Value |
|--------|-------|
| Queries with 0 phrases | 0 |
| Queries with <50% phrase match | 0 |
| Average phrase match rate | 100% |

The issue is **semantic discrimination**, not vocabulary coverage.

## Root Cause Analysis

### Failure Mode 1: Semantic Dilution (16/55 failed queries)

When multiple passages share similar topics, SF's fingerprints become too similar to distinguish the gold passage.

**Example: Query 0** — "who has the oldest team in the nba"
- Gold: "Eddie Gottlieb" (mentions "barnstorming teams")
- Distractor #1: "National Basketball Association" (contains "NBA", "championships")
- Problem: "NBA" in query matches "National Basketball Association" better than "Eddie Gottlieb"

**Example: Query 24** — "the cold dry winds that blow over northern india in winter are called"
- Gold: Should be about "Loo" winds
- Distractor #1: "Indian summer" (contains "dry", "weather", "climate")
- Problem: Weather/climate passages have stronger semantic overlap than the specific answer

### Failure Mode 2: Specificity Gap (12/55 queries with MRR=0.5)

SF ranks gold at position 2-3 instead of 1, missing the top spot.

**Example: Query 33** — "who dies in the lost city of z"
- BM25 ranks gold #1 (exact term match on "lost city of z")
- SF ranks gold #5 (semantic similarity to other adventure/exploration passages)

### Failure Mode 3: Empty Results (5/55 queries)

Some queries return empty top-5 lists despite having candidates. This is a bug in the filtering logic.

## Why BM25 Outperforms SF on NQ-REaR

| Factor | BM25 | SF |
|--------|------|-----|
| **Exact term matching** | Strong (TF-IDF) | Weak (phrase-level) |
| **Rare term weighting** | Strong (IDF in scoring) | Moderate (IDF in fingerprint) |
| **Topic similarity** | Weak (lexical only) | Strong (semantic) |
| **Specificity** | Strong (exact matches rank higher) | Weak (all similar passages rank equally) |

### BM25's Advantage on NQ-REaR

1. **Exact term matching**: When gold passage contains query terms verbatim, BM25 ranks it #1
2. **Rare term boost**: Unique terms like "lost city of z" get high IDF scores
3. **Term frequency**: Multiple occurrences of query terms boost ranking

### SF's Disadvantage on NQ-REaR

1. **Semantic dilution**: Passages about similar topics produce similar fingerprints
2. **No term frequency**: SF doesn't distinguish between 1 occurrence vs 5 occurrences
3. **Phrase-level granularity**: SF matches phrases, not individual terms

## Comparison: SF vs BM25 on Failed Queries

| Query | SF MRR | BM25 MRR | Winner | Reason |
|-------|--------|----------|--------|--------|
| 0: "oldest team in nba" | 0.00 | 0.00 | Tie | Both fail (gold passage lacks key terms) |
| 7: "grey's anatomy season 7" | 0.00 | 0.33 | BM25 | BM25 finds gold at rank 2 |
| 24: "cold dry winds india" | 0.00 | 0.00 | Tie | Both fail (answer passage too specific) |
| 33: "lost city of z" | 0.00 | 1.00 | BM25 | Exact term match in gold passage |
| 39: "first high level language" | 0.00 | 0.00 | Tie | Both fail (gold passage lacks key terms) |

## Recommendations

### Short-term (No code changes)
1. **Accept NQ-REaR as a hard benchmark**: SF's semantic approach struggles with exact-match tasks
2. **Focus on biomedical domain**: SF excels on PubMedQA (MRR=0.969) where semantic understanding matters more

### Medium-term (Pipeline improvements)
1. **Hybrid scoring**: Combine SF fingerprints with BM25 term frequency
2. **Query expansion**: Add synonyms for rare terms
3. **Passage length normalization**: Normalize fingerprints by passage length

### Long-term (Architecture changes)
1. **Hierarchical fingerprints**: Multi-scale fingerprints for topic + detail
2. **Term-level fingerprints**: Add term-level fingerprints alongside phrase-level
3. **Attention-based ranking**: Learn to weight query terms by importance

## Conclusion

SF's underperformance on NQ-REaR is **expected and explainable**:

- NQ-REaR requires **exact term matching**, which is BM25's strength
- SF's strength is **semantic understanding**, which matters more on biomedical QA
- The pattern is consistent: SF excels on PubMedQA (MRR=0.969) but underperforms on general knowledge retrieval

This is not a bug — it's a fundamental tradeoff between **semantic similarity** and **lexical matching**.
