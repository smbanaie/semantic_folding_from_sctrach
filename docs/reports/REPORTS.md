# Benchmark Reports — Master Index

> **Single source of truth**: [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) — complete consolidated report with all datasets, metrics, and analysis.

## Version History

| Version | Date | Dataset | Grid | MRR (SF) | MRR (BM25) | SF/BM25 | Notes |
|---------|------|---------|------|----------|------------|---------|-------|
| v1 | 2026-06-06 | PubMedQA | 64 | 0.955 | 1.000 | 95.5% | Baseline, 111 queries |
| v2 | 2026-06-17 | PubMedQA | 64 | **1.000** | 1.000 | 100% | 30 queries, perfect |
| **v3** | **2026-06-24** | **Belebele** | 64 | **1.000** (SPLADE) | 0.995 | **100.5%** | **SF+SPLADE surpasses BM25** |
| v2 | 2026-06-17 | Belebele | 64 | 0.880 | 0.995 | 88.4% | Comprehensive (47 runs) |
| v1 | 2026-06-15 | PopQA | 64 | 0.980 | 1.000 | 98.0% | 100 queries, HippoRAG2 |
| v2 | 2026-06-17 | NQ-REaR | 64 | 0.521 | 0.638 | 83.4% | Comprehensive (85 runs) |
| v1 | 2026-06-16 | HotpotQA | 64 | 0.726 | 0.869 | 83.5% | 48 queries, multi-hop |
| v1 | 2026-06-16 | NarrativeQA | 64 | 0.939 | 0.980 | 95.8% | 49 queries, narrative |
| v1 | 2026-06-16 | 2WikiMultihopQA | 64 | 0.788 | 0.921 | 85.6% | 50 queries, multi-hop |
| v1 | 2026-06-17 | MuSiQue | 64 | 0.453 | 0.672 | 67.4% | 100 queries, multi-hop |
| v1 | 2026-06-15 | DROP | 64 | 0.320 | 0.762 | 42.6% | 50 queries, L2 norm |
| v1 | 2026-06-09 | DocFinQA | 128 | 0.250 | 0.341 | 73.3% | 20 queries, financial |
| v2 | 2026-06-10 | CUAD | 64 | 0.000 | 0.244 | 0% | 200 queries, legal |
| v3 | 2026-06-13 | MAUD | 64 | 0.000 | 0.649 | 0% | 100 queries, legal |

---

## Reports by Dataset

### PubMedQA
- `pubmedqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 52 runs, best MRR=1.000, grid=64 optimal**

### Belebele
- `belebele/v3_20260624_splade_perfect.md` — **SF+SPLADE achieves perfect MRR=1.0, surpasses BM25 (0.995)**
- `belebele/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 47 runs, t-SNE best MRR=0.88, UMAP=0.80**
- `belebele/umap_implementation_analysis.md` — UMAP implementation details

### PopQA
- `popqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 13 runs, SF MRR=0.98, near-perfect**

### NQ-REaR
- `nq_rear/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 85 runs, SF MRR=0.521, entity matching gap**
- `nq_rear/failure_analysis_v1.md` — Root cause analysis
- `nq_rear/debug_analysis_v1.md` — Debug analysis

### HotpotQA
- `hotpotqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 2 runs, SF MRR=0.726**

### NarrativeQA
- `narrativeqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 3 runs, SF MRR=0.939**

### 2WikiMultihopQA
- `2wikimultihopqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 2 runs, SF MRR=0.788**

### MuSiQue
- `musique/v1_20260617_122000.md` — 100 queries, SF MRR=0.453, BM25=0.672

### DROP
- `drop/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 7 runs, SF MRR=0.32**

### DocFinQA
- `docfinqa/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 7 runs, SF MRR=0.25**

### CUAD
- `cuad/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 6 runs, SF MRR=0.000**

### MAUD
- `maud/v2_20260617_comprehensive_analysis.md` — **Comprehensive: 5 runs, SF MRR=0.000**

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
# 4. Update docs/reports/BENCHMARK_RESULTS.md (add/update dataset metrics table)
```

---

*Last updated: 2026-06-17*
