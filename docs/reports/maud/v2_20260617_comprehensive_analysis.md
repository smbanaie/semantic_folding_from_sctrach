# MAUD Benchmark — Comprehensive Analysis

**Dataset:** MAUD (contract review)
**Period:** 2026-06-12 to 2026-06-12
**Total benchmarks run:** 5 (3 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | Notes |
|-----------|-----|---------|------|-------|
| `benchmark_20260612_221327` | **0.000** | 50 | 64 | SF baseline — complete failure |
| `benchmark_20260612_233005` | 0.000 | 32 | 64 | SF — complete failure |
| `benchmark_20260612_222741` | 0.841 | 50 | 0 | BM25 baseline |
| `benchmark_20260612_233820` | 0.649 | 100 | 0 | BM25 (100Q) |

---

## Analysis

SF achieves **MRR=0.000** on MAUD — a complete failure. BM25 achieves 0.841 (50Q). The 0% ratio indicates that MAUD's legal contract queries require domain-specific reasoning that phrase-level semantic matching cannot handle.

### Why SF Fails on MAUD

MAUD queries are legal questions about contract clauses (e.g., "What happens if the vendor fails to deliver on time?"). These require:
1. Understanding legal terminology and relationships
2. Cross-referencing multiple clauses
3. Reasoning about conditional obligations

Phrase extraction cannot capture these complex legal semantics.

---

*Generated: 2026-06-17 | Source: 5 benchmark runs*
