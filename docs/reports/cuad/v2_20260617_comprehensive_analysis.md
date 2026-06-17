# CUAD Benchmark — Comprehensive Analysis

**Dataset:** CUAD (contract understanding)
**Period:** 2026-06-10 to 2026-06-17
**Total benchmarks run:** 6 (3 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | Notes |
|-----------|-----|---------|------|-------|
| `benchmark_20260610_112708` | **0.000** | 200 | 64 | SF baseline — complete failure |
| `benchmark_20260610_134504` | 0.244 | 200 | 0 | BM25 baseline |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| 200 queries | 0.000 | 0.244 | 0% |

CUAD, like MAUD, is a legal contract dataset. SF achieves **MRR=0.000** — complete failure. BM25 also performs poorly (0.244), indicating this is a fundamentally hard dataset. Legal contract clause extraction requires domain-specific reasoning beyond phrase-level matching.

---

*Generated: 2026-06-17 | Source: 6 benchmark runs*
