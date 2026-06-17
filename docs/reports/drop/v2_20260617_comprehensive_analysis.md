# DROP Benchmark — Comprehensive Analysis

**Dataset:** DROP (Discrete Reasoning Over Paragraphs)
**Period:** 2026-06-08 to 2026-06-15
**Total benchmarks run:** 7 (3 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | norm | Notes |
|-----------|-----|---------|------|------|-------|
| `benchmark_20260615_134527` | **0.320** | 50 | 64 | l2 | **Best SF** |
| `benchmark_20260615_125648` | 0.280 | 50 | 64 | none | SF baseline |
| `benchmark_20260609_043806` | 0.126 | 198 | 64 | none | SF (large Q) |
| `benchmark_20260608_113254` | 0.762 | 200 | 0 | none | BM25 baseline |
| `benchmark_20260615_141512` | 0.752 | 50 | 0 | none | BM25 (50Q) |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| 198-200 queries | 0.126 | 0.762 | 16.5% |
| 50 queries (best) | 0.320 | 0.752 | 42.6% |

DROP is the hardest dataset for SF. The 42.6% ratio on 50 queries is the worst across all datasets. DROP requires discrete reasoning (counting, sorting, comparison) that phrase-level semantic matching cannot capture.

### Normalization Effect

| norm | MRR | Delta |
|------|-----|-------|
| none | 0.280 | — |
| l2 | 0.320 | +14.3% |

L2 normalization provides a meaningful improvement (+14.3%) on DROP.

### Failure Analysis

DROP queries involve multi-step reasoning: "How many touchdowns did X score in the 4th quarter?" requires counting entities in a passage. SF's phrase fingerprinting captures topical similarity but not the quantitative relationships needed.

---

*Generated: 2026-06-17 | Source: 7 benchmark runs*
