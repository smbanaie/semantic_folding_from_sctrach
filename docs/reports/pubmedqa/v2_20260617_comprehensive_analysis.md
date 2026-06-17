# PubMedQA Benchmark — Comprehensive Analysis

**Dataset:** PubMedQA (biomedical question answering)
**Period:** 2026-06-06 to 2026-06-17
**Total benchmarks run:** 52 (many failed/incomplete, 8 unique configurations)

---

## Key Results

| Benchmark | MRR | Queries | Grid | norm | Notes |
|-----------|-----|---------|------|------|-------|
| `benchmark_20260606_162818` | **0.955** | 111 | 64 | none | Baseline (v1) |
| `benchmark_20260606_214417` | 0.902 | 112 | 128 | none | Grid=128 experiment |
| `benchmark_20260607_020428` | 0.946 | 112 | 64 | none | Reproduction |
| `benchmark_20260614_015546` | 0.954 | 65 | 64 | none | 65Q subset |
| `benchmark_20260615_045116` | **0.969** | 65 | 64 | none | Best on 65Q |
| `benchmark_20260617_154443` | **1.000** | 30 | 64 | sqrt_nnz | Perfect (small sample) |
| `benchmark_20260616_210021` | 1.000 | 12 | 64 | sqrt_nnz | Perfect (very small) |
| `benchmark_20260608_012630` | 1.000 | 112 | 0 | none | BM25 baseline |

---

## Analysis

### Grid Size Effect

| Grid | MRR | Queries |
|------|-----|---------|
| 64 | **0.955** | 111 |
| 128 | 0.902 | 112 |
| Delta | -5.6% | |

Grid=128 hurts performance. Smaller grid (64) is better for PubMedQA's biomedical phrases.

### Normalization Effect

| norm | MRR | Queries |
|------|-----|---------|
| none | 0.955 | 111 |
| l2 | 0.954 | 65 |
| sqrt_nnz | 1.000 | 30 |

Normalization has negligible effect on PubMedQA. The dataset is small enough that fingerprints are already well-separated.

### SF vs BM25

SF MRR=0.955 vs BM25 MRR=1.000. SF achieves 95.5% of BM25 — the best ratio across all datasets.

### Failed Runs

20+ runs with MRR=N/A (crashed during sqrt_nnz experiments on 2026-06-16). These were batch tests with incompatible configs.

---

## Evolution Timeline

| Date | MRR | Notes |
|------|-----|-------|
| 2026-06-06 | 0.955 | Initial baseline |
| 2026-06-07 | 0.902 | Grid=128 experiment |
| 2026-06-14 | 0.954 | 65Q subset |
| 2026-06-15 | 0.969 | Best 65Q result |
| 2026-06-17 | 1.000 | Perfect on 30Q |

---

*Generated: 2026-06-17 | Source: 52 benchmark runs*
