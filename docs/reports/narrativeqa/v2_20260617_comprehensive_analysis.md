# NarrativeQA Benchmark — Comprehensive Analysis

**Dataset:** NarrativeQA (narrative comprehension)
**Period:** 2026-06-16 to 2026-06-17
**Total benchmarks run:** 3

---

## Key Results

| Benchmark | MRR | Queries | Grid | norm | Notes |
|-----------|-----|---------|------|------|-------|
| `benchmark_20260616_233453` | **0.939** | 49 | 64 | sqrt_nnz | SF baseline |
| `benchmark_20260617_001620` | 0.980 | 50 | 0 | none | BM25 baseline |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| ~50 queries | 0.939 | 0.980 | 95.8% |

NarrativeQA is one of the best-performing datasets for SF. The 95.8% ratio indicates that narrative comprehension relies heavily on topical/semantic similarity that phrase-level matching captures well.

---

*Generated: 2026-06-17 | Source: 3 benchmark runs*
