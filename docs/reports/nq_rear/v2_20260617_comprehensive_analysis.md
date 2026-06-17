# NQ-REaR Benchmark — Comprehensive Analysis

**Dataset:** NQ-REaR (Natural Questions - Rearranged)
**Period:** 2026-06-15 to 2026-06-17
**Total benchmarks run:** 85 (mostly Q=10 batch experiments, 5 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | Method | Notes |
|-----------|-----|---------|------|--------|-------|
| `benchmark_20260615_193912` | 0.638 | 100 | 0 | BM25 | Baseline |
| `benchmark_20260617_001629` | 0.625 | 50 | 0 | BM25 | 50Q baseline |
| `benchmark_20260616_235034` | 0.521 | 47 | 64 | t-SNE | **Best SF** |
| `benchmark_20260616_012548` | 0.671 | 19 | 64 | t-SNE | Small sample (best MRR but not representative) |
| `benchmark_20260616_023712` | 0.700 | 5 | 64 | t-SNE | Very small sample |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| 100 queries | — | 0.638 | — |
| 47-50 queries | 0.521 | 0.625 | 83.4% |

SF achieves 83.4% of BM25 performance on NQ-REaR. The gap is larger than Belebele (88.4%), suggesting this dataset's queries rely more on lexical matching.

### Batch Q=10 Experiments (80 runs)

The 80+ Q=10 runs were batch experiments testing different query subsets. MRR ranged from 0.49 to 0.70, confirming high variance on small samples. These are not representative of full performance.

### Failure Patterns

NQ-REaR queries are short factual questions. SF's phrase extraction may miss precise entity matches that BM25 captures directly.

---

## Recommendation

- Best SF config: t-SNE, grid=64, sqrt_nnz normalization, Q>=50
- Gap to close: ~17% vs BM25 — focus on entity-level phrase matching

---

*Generated: 2026-06-17 | Source: 85 benchmark runs*
