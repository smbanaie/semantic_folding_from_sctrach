# 2WikiMultihopQA Benchmark — Comprehensive Analysis

**Dataset:** 2WikiMultihopQA (multi-hop question answering)
**Period:** 2026-06-16 to 2026-06-17
**Total benchmarks run:** 2

---

## Key Results

| Benchmark | MRR | Queries | Grid | norm | Notes |
|-----------|-----|---------|------|------|-------|
| `benchmark_20260616_230841` | **0.788** | 50 | 64 | sqrt_nnz | SF baseline |
| `benchmark_20260617_001611` | 0.921 | 50 | 0 | none | BM25 baseline |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| 50 queries | 0.788 | 0.921 | 85.6% |

2WikiMultihopQA requires multi-hop reasoning across Wikipedia articles. SF achieves 85.6% of BM25 — similar to HotpotQA (83.5%), suggesting multi-hop tasks are consistently challenging for phrase-level matching.

---

*Generated: 2026-06-17 | Source: 2 benchmark runs*
