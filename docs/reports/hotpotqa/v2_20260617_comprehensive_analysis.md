# HotpotQA Benchmark — Comprehensive Analysis

**Dataset:** HotpotQA (multi-hop question answering)
**Period:** 2026-06-16 to 2026-06-17
**Total benchmarks run:** 2

---

## Key Results

| Benchmark | MRR | Queries | Grid | norm | Notes |
|-----------|-----|---------|------|------|-------|
| `benchmark_20260616_223640` | **0.726** | 48 | 64 | sqrt_nnz | SF baseline |
| `benchmark_20260617_001600` | 0.869 | 50 | 0 | none | BM25 baseline |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| ~50 queries | 0.726 | 0.869 | 83.5% |

HotpotQA requires multi-hop reasoning (connecting facts from two passages). SF's 83.5% ratio is reasonable — the phrase-level matching captures some topical connections but misses explicit multi-hop linking.

---

*Generated: 2026-06-17 | Source: 2 benchmark runs*
