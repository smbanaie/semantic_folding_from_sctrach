# Cross-Dataset Comparison — v3-Final

**Date:** 2026-06-13  
**Pipeline:** Semantic Folding (grid=64, spread=1, top%=0.10, IDF, σ=1.5) + BM25 baseline

---

## Aggregate Metrics

| Dataset | Domain | Queries | SF MRR | BM25 MRR | SF AP | BM25 AP | Winner |
|---------|--------|---------|--------|----------|-------|---------|--------|
| PubMedQA | Biomedical QA | 112 | 0.955 | **1.000** | 0.790 | **0.960** | BM25 |
| Belebele | Reading Comprehension | 100 | 0.740 | **0.995** | 0.740 | **0.995** | BM25 |
| Belebele | Reading Comprehension | 100 | 0.840 | **0.860** | 0.995 | Hybrid α=0.5 |

---

## Semantic Folding Analysis

### Success Pattern (PubMedQA)
- High lexical overlap between queries and gold passages
- Clear topical keywords (medical terms)
- MRR=0.955 (near-perfect)

### Partial Success (Belebele)
- All-or-nothing: 74% perfect, 26% complete failure
- No partial matches
- Reading comprehension benefits from exact term matching

### Complete Failure (MAUD)
- Formulaic legal language defeats semantic separation
- Short, repetitive passages
- Queries are labels, not natural language

---

## BM25 Analysis

- Dominates on all three datasets
- Excels at exact term matching
- Fast (<1s for 100 queries)
- PubMedQA: perfect (MRR=1.000)
- Belebele: near-perfect (MRR=0.995)
- MAUD: reasonable (MRR=0.649)

---

## Key Thesis Finding

> Semantic folding fails where keyword overlap is high, but succeeds where semantic ambiguity dominates.

| Regime | Example | Best Method |
|--------|---------|-------------|
| High lexical overlap | PubMedQA, Belebele | BM25 |
| Semantic ambiguity | TBD (need paraphrase datasets) | SF (expected) |
| Formulaic language | MAUD | BM25 |

---

*Generated: 2026-06-13*
