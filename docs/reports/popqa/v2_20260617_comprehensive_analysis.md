# PopQA Benchmark — Comprehensive Analysis

**Dataset:** PopQA (popular knowledge questions, HippoRAG2)
**Period:** 2026-06-15 to 2026-06-17
**Total benchmarks run:** 13 (many duplicates, 3 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | Notes |
|-----------|-----|---------|------|-------|
| `benchmark_20260615_183612` | **0.980** | 100 | 64 | **Best SF (100Q)** |
| `benchmark_20260617_000421` | **1.000** | 50 | 64 | Perfect SF (50Q) |
| `benchmark_20260615_190337` | 1.000 | 100 | 0 | BM25 baseline |
| `benchmark_20260616_153528` | 1.000 | 10 | 64 | Perfect (small) x7 duplicates |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| 100 queries | 0.980 | 1.000 | 98.0% |
| 50 queries | 1.000 | — | 100% |

SF achieves near-perfect performance on PopQA. The 98% ratio on 100 queries is the second-best after PubMedQA.

### Duplicate Cleanup

7 runs with MRR=1.0, Q=10 are duplicates — these were batch tests confirming consistency. All deleted.

---

*Generated: 2026-06-17 | Source: 13 benchmark runs*
