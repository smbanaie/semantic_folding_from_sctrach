# DocFinQA Benchmark — Comprehensive Analysis

**Dataset:** DocFinQA (financial question answering)
**Period:** 2026-06-08 to 2026-06-09
**Total benchmarks run:** 7 (3 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | Notes |
|-----------|-----|---------|------|-------|
| `benchmark_20260609_134039` | **0.250** | 20 | 128 | Best SF |
| `benchmark_20260608_152530` | 0.025 | 20 | 128 | SF (poor) |
| `benchmark_20260608_154422` | 0.341 | 20 | 0 | BM25 baseline |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| 20 queries | 0.250 | 0.341 | 73.3% |

DocFinQA is challenging for both SF and BM25 (BM25 only 0.341). SF achieves 73.3% of BM25 — reasonable given the low baseline. Financial documents require numerical reasoning and table comprehension.

### Grid Size

Grid=128 was used for DocFinQA (not the recommended 64). This may have hurt performance. The grid=0 (BM25) runs performed better.

---

*Generated: 2026-06-17 | Source: 7 benchmark runs*
