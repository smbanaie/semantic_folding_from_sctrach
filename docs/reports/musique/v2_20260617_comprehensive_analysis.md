# MuSiQue Benchmark — Comprehensive Analysis

**Dataset:** MuSiQue (multi-hop question answering)
**Period:** 2026-06-17
**Total benchmarks run:** 3 (2 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | Notes |
|-----------|-----|---------|------|-------|
| `benchmark_20260617_025659` | **0.409** | 56 | 64 | SF baseline |
| `benchmark_20260617_121916` | 0.672 | 87 | 0 | BM25 baseline |

---

## Analysis

### SF vs BM25

| Config | MRR (SF) | MRR (BM25) | SF/BM25 |
|--------|----------|------------|---------|
| 56-87 queries | 0.409 | 0.672 | 60.9% |

MuSiQue requires 2-4 hop reasoning across multiple paragraphs. SF achieves only 60.9% of BM25 — the second-worst ratio after DROP (42.6%). The multi-hop nature means queries require connecting facts that phrase-level matching cannot capture.

### Comparison with Other Multi-hop Datasets

| Dataset | SF/BM25 | Hops |
|---------|---------|------|
| HotpotQA | 83.5% | 2-hop |
| 2WikiMultihopQA | 85.6% | 2-hop |
| MuSiQue | 60.9% | 2-4-hop |

Performance degrades with hop count, confirming that multi-hop reasoning is SF's primary limitation.

---

*Generated: 2026-06-17 | Source: 3 benchmark runs*
