# Benchmark Reports — Master Index

## Version History

| Version | Date | Dataset | Grid | MRR (SF) | MRR (BM25) | SF/BM25 | Notes |
|---------|------|---------|------|----------|------------|---------|-------|
| v1 | 2026-06-06 | PubMedQA | 64 | 0.955 | 1.000 | 95.5% | Baseline, 111 queries |
| v1 | 2026-06-07 | PubMedQA | 128 | 0.902 | — | — | Grid=128 experiment |
| v2 | 2026-06-17 | PubMedQA | 64 | **1.000** | 1.000 | 100% | 30 queries, perfect |
| v3 | 2026-06-13 | Belebele | 64 | 0.740 | 0.995 | 74.4% | 100 queries |
| v1 | 2026-06-17 | Belebele | 64 | **0.880** | — | — | 50 queries, t-SNE |
| v2 | 2026-06-17 | Belebele | 64 | **0.880** | 0.995 | 88.4% | Comprehensive (47 runs) |
| v3 | 2026-06-13 | MAUD | 64 | 0.000 | 0.649 | 0% | 100 queries, legal |
| v1 | 2026-06-15 | PopQA | 64 | 0.980 | 1.000 | 98.0% | 100 queries, HippoRAG2 |
| v1 | 2026-06-15 | NQ-REaR | 64 | 0.574 | 0.638 | 89.9% | 100 queries, HippoRAG2 |
| v2 | 2026-06-17 | NQ-REaR | 64 | 0.521 | 0.638 | 83.4% | Comprehensive (85 runs) |
| v1 | 2026-06-17 | MuSiQue | 64 | 0.453 | 0.672 | 67.4% | 100 queries, multi-hop |
| v2 | 2026-06-17 | MuSiQue | 64 | **0.409** | 0.672 | 60.9% | Comprehensive (3 runs) |
| v1 | 2026-06-15 | DROP | 64 | 0.320 | 0.762 | 42.6% | 50 queries, L2 norm |
| v1 | 2026-06-09 | DocFinQA | 128 | 0.250 | 0.341 | 73.3% | 20 queries, financial |
| v1 | 2026-06-16 | HotpotQA | 64 | 0.726 | 0.869 | 83.5% | 48 queries, multi-hop |
| v1 | 2026-06-16 | NarrativeQA | 64 | 0.939 | 0.980 | 95.8% | 49 queries, narrative |
| v1 | 2026-06-16 | 2WikiMultihopQA | 64 | 0.788 | 0.921 | 85.6% | 50 queries, multi-hop |
| v1 | 2026-06-10 | CUAD | 64 | 0.000 | 0.244 | 0% | 200 queries, legal |

---

## Reports by Dataset

### PubMedQA
- `pubmedqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 52 runs, best MRR=1.000 (30Q), grid=64 optimal, normalization negligible**
- `pubmedqa/v1_20260606_162818.md` — Baseline (grid=64, 111 queries, MRR=0.955)
- `pubmedqa/v1_20260607_grid128.md` — Grid=128 experiment (MRR=0.902)
- `pubmedqa/v1_20260617_155600.md` — 50 queries, MRR=1.000 (perfect)

### Belebele
- `belebele/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 47 runs, t-SNE best MRR=0.88, UMAP=0.80, normalization & hybrid experiments**
- `belebele/v3_20260613_223315.md` — 100 queries, MRR=0.740 (baseline)
- `belebele/v1_20260617_150600.md` — 50 queries, MRR=0.880, failure analysis (t-SNE)

### MAUD
- `maud/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 5 runs, SF MRR=0.000, BM25=0.841, legal domain failure**
- `maud/v3_20260613_233820.md` — 100 queries, MRR=0.000

### PopQA
- `popqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 13 runs, SF MRR=0.98, BM25=1.00, near-perfect**
- `popqa/v1_20260615_212400.md` — 100 queries, SF MRR=0.980, BM25 MRR=1.000

### NQ-REaR
- `nq_rear/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 85 runs, SF MRR=0.521, BM25=0.638, entity matching gap**
- `nq_rear/v1_20260615_212400.md` — 100 queries, SF MRR=0.574, BM25 MRR=0.638
- `nq_rear/failure_analysis_v1.md` — Root cause analysis of SF underperformance

### DROP
- `drop/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 7 runs, SF MRR=0.32, BM25=0.762, discrete reasoning gap**
- `drop/v1_20260615_134527.md` — 50 queries, L2 norm, MRR=0.320

### DocFinQA
- `docfinqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 7 runs, SF MRR=0.25, BM25=0.341, financial reasoning gap**
- `docfinqa/v1_20260609_134039.md` — 20 queries, grid=128, MRR=0.250

### HotpotQA
- `hotpotqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 2 runs, SF MRR=0.726, BM25=0.869, multi-hop gap**
- `hotpotqa/v1_20260616_223640.md` — 48 queries, MRR=0.726

### NarrativeQA
- `narrativeqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 3 runs, SF MRR=0.939, BM25=0.980, strong semantic match**
- `narrativeqa/v1_20260616_233453.md` — 49 queries, MRR=0.939

### 2WikiMultihopQA
- `2wikimultihopqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 2 runs, SF MRR=0.788, BM25=0.921, multi-hop gap**
- `2wikimultihopqa/v1_20260616_230841.md` — 50 queries, MRR=0.788

### CUAD
- `cuad/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 6 runs, SF MRR=0.000, BM25=0.244, legal domain failure**

### MuSiQue (Multi-hop QA)
- `musique/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 3 runs, SF MRR=0.409, BM25=0.672, 2-4-hop gap**
- `musique/v1_20260617_122000.md` — 100 queries, SF MRR=0.453, BM25 MRR=0.672

### Cross-Dataset
- `cross-dataset/comparison_20260613.md` — SF vs BM25 across all datasets

---

## Cross-Dataset Summary

| Dataset | SF/BM25 | Category | Notes |
|---------|---------|----------|-------|
| PubMedQA | 95.5% | Biomedical | Best SF performance |
| PopQA | 98.0% | Knowledge | Near-perfect |
| NarrativeQA | 95.8% | Narrative | Strong semantic match |
| Belebele | 88.4% | Reading comprehension | Good |
| 2WikiMultihopQA | 85.6% | Multi-hop | Multi-hop gap |
| NQ-REaR | 83.4% | Knowledge | Entity matching gap |
| HotpotQA | 83.5% | Multi-hop | Multi-hop gap |
| DocFinQA | 73.3% | Financial | Numerical reasoning gap |
| MuSiQue | 60.9% | Multi-hop | Complex reasoning gap |
| DROP | 42.6% | Discrete reasoning | Hardest for SF |
| CUAD | 0% | Legal | Complete failure |
| MAUD | 0% | Legal | Complete failure |

---

## How to Save a New Report

```bash
# 1. Run benchmark
.venv\Scripts\python semantic_folding\dataset_benchmark\run_all_benchmarks.py \
  --datasets <name> --max-queries 100

# 2. Save report with versioned filename
cp outputs/<name>_benchmark/benchmarks/benchmark_<ts>/benchmark_report.md \
   docs/reports/<name>/v<N>_<YYYYMMDD>_<HHMMSS>.md

# 3. Update this index (add row to version history + report list)
```

---

*Last updated: 2026-06-17*
