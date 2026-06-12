# Dataset Benchmark Reports — Semantic Folding Pipeline

**Version:** v3-Final  
**Date:** 2026-06-13  
**Datasets:** PubMedQA, Belebele, MAUD  
**Pipeline:** Semantic Folding Steps 1–6 + BM25 baseline

---

## 1. Cross-Dataset Comparison

| Dataset | Domain | Queries | SF MRR | BM25 MRR | Winner | Gap |
|---------|--------|---------|--------|----------|--------|-----|
| **PubMedQA** | Biomedical QA | 112 | 0.955 | **1.000** | BM25 | −0.045 |
| **Belebele** | Reading Comprehension | 100 | 0.740 | **0.995** | BM25 | −0.255 |
| **MAUD** | Legal Contract QA | 100 | 0.000 | **0.649** | BM25 | −0.649 |

**Headline:** BM25 dominates all three datasets. Semantic folding achieves competitive results on PubMedQA (MRR=0.955) but degrades on reading comprehension (0.740) and fails completely on legal contracts (0.000).

---

## 2. PubMedQA (Previous Baseline)

**Source:** `data/pubmedqa/REPORT.md`  
**Subset:** pqa_labeled, 112 queries with gold passages

| Metric | Semantic Folding | BM25 |
|--------|-----------------|------|
| MRR | 0.955 | **1.000** |
| AP | 0.790 | **0.960** |
| P@1 | 0.955 | **1.000** |
| P@2 | 0.896 | **0.996** |
| Found at rank 1 | 106/111 (95.5%) | 112/112 (100%) |
| Failures | 5 | 0 |

**Key findings:**
- Grid=64, spread=1, top%=0.10, IDF weighting, σ=1.5 is optimal
- Grid=128 hurts (MRR −5.3%) — over-partitions similar sections
- No smoothing catastrophic (MRR −31.2%)
- All 5 SF failures have small candidate pools (2–8 passages) with semantically similar sections
- BM25 succeeds because queries derive from article titles → high keyword overlap with gold passages

**Thesis framing:** PubMedQA is a "high lexical overlap" regime where BM25 excels. Semantic folding's value emerges in "semantic ambiguity" regimes.

---

## 3. Belebele (100 queries)

**Source:** `outputs/belebele_benchmark/benchmarks/benchmark_20260612_223315`  
**Dataset:** facebook/belebele (english split, reading comprehension)  
**Index:** 1,772 unique paragraphs across 100 queries

| Metric | Semantic Folding | BM25 |
|--------|-----------------|------|
| MRR | 0.740 | **0.995** |
| AP | 0.740 | **0.995** |
| P@1 | 0.740 | **0.990** |
| P@2 | 0.370 | **0.500** |
| P@5 | 0.148 | 0.000 |

**Found-at distribution (SF):**
| Rank | Count | % |
|------|-------|---|
| 0 (not found) | 26 | 26% |
| 1 | 74 | 74% |

**Analysis:**
- All-or-nothing pattern: 74 queries perfect (MRR=1.0), 26 complete failures (MRR=0.0)
- No partial matches — pipeline either finds gold at rank 1 or misses entirely
- Reading comprehension questions (e.g., "According to the passage, what...") benefit from exact lexical matching
- Distractor passages from different topics should be distinguishable, but SF struggles with short questions

**Timing:**
- Phase 1 (index): ~3 min
- Phase 2 (100 queries): ~20 min (~12s/query)
- BM25: <1s

---

## 4. MAUD (100 queries)

**Source:** `outputs/maud_benchmark/benchmarks/benchmark_20260612_233820`  
**Dataset:** atticusproject/maud (merger agreement understanding)  
**Index:** 1,972 unique paragraphs across 100 queries

| Metric | Semantic Folding | BM25 |
|--------|-----------------|------|
| MRR | 0.000 | **0.649** |
| AP | 0.000 | **0.649** |
| P@1 | 0.000 | **0.510** |
| P@2 | 0.000 | **0.340** |

**Analysis:**
- Semantic folding completely fails — 0/100 queries find gold passage
- Queries are formulaic legal questions (e.g., "Type of Consideration-Answer")
- Passages are short contract excerpts (~100-300 words)
- All queries get MRR=0.0 — gold passages never appear in top-5
- BM25 achieves reasonable performance (MRR=0.649) via exact term matching

**Why SF fails on MAUD:**
1. Legal language is highly formulaic — many terms repeat across passages
2. Queries are short labels, not natural language questions
3. Gold passages are short excerpts, not full documents
4. The semantic space cannot distinguish between similar legal clauses

**Timing:**
- Phase 1 (index): ~10 min
- Phase 2 (100 queries): ~20 min (~12s/query)
- BM25: <1s

---

## 5. Dataset-Specific Reports

Detailed reports for each dataset are available in:

| Dataset | Location |
|---------|----------|
| PubMedQA | `data/pubmedqa/REPORT.md` |
| PubMedQA (grid=128) | `data/pubmedqa/COMPARISON_GRID128.md` |
| PubMedQA (recommendations) | `data/pubmedqa/RECOMMENDATIONS.md` |

---

## 6. Key Findings for Thesis

### 6.1 Semantic Folding vs BM25 — When Does Each Win?

| Regime | Example | Best Method | Why |
|--------|---------|-------------|-----|
| **High lexical overlap** | PubMedQA (title-derived queries) | BM25 | Exact term matching sufficient |
| **Semantic ambiguity** | TBD (need datasets with paraphrase/synonym queries) | SF (expected) | Grid topology captures latent similarity |
| **Formulaic language** | MAUD (legal contracts) | BM25 | Repeated terminology defeats semantic discrimination |
| **Reading comprehension** | Belebele | BM25 | Questions reference passage content directly |

### 6.2 Semantic Folding Failure Modes

1. **All-or-nothing pattern** — No partial matches; either perfect or complete failure
2. **Small candidate pools** — When few distractors, SF struggles to discriminate similar passages
3. **Formulaic language** — Legal/technical text with repeated terminology defeats semantic separation
4. **Short queries** — Questions with few words don't activate enough grid cells

### 6.3 Recommendations for Improvement

| Priority | Approach | Expected Impact | Effort |
|----------|----------|-----------------|--------|
| High | **Hybrid scoring** (SF + BM25) | Combine strengths of both | Medium |
| High | **Query expansion** (synonyms/acronyms) | Improve short query activation | Medium |
| Medium | **Passage chunking** (split long docs) | Better semantic separation | Low |
| Medium | **TF-IDF re-ranking** (post-SF) | Boost discriminative terms | Low |
| Low | **Multi-scale grids** (64 + 128) | Capture different granularity | High |
| Low | **Negative mining** (hard negatives) | Learn to distinguish similar passages | High |

---

*Generated: 2026-06-13 00:30 UTC*
