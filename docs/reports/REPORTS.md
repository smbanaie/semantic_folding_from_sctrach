# Benchmark Reports — Master Index

> **Single source of truth**: [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) — complete consolidated report with all datasets, metrics, and analysis.

## Version History

| Version | Date | Dataset | Grid | MRR (SF) | MRR (BM25) | SF/BM25 | Notes |
|---------|------|---------|------|----------|------------|---------|-------|
| v1 | 2026-06-06 | PubMedQA | 64 | 0.955 | 1.000 | 95.5% | Baseline, 111 queries |
| v2 | 2026-06-17 | PubMedQA | 64 | **1.000** | 1.000 | 100% | 30 queries, perfect |
| **v3** | **2026-06-24** | **Belebele** | 64 | **1.000** (SPLADE) | 0.995 | **100.5%** | **SF+SPLADE surpasses BM25** |
| v2 | 2026-06-17 | Belebele | 64 | 0.880 | 0.995 | 88.4% | Comprehensive (47 runs) |
| v1 | 2026-06-17 | BioASQ | 64 | 0.195 | — | — | 50 queries, biomedical QA (hard) |
| v1 | 2026-06-15 | PopQA | 64 | 0.980 | 1.000 | 98.0% | 100 queries, HippoRAG2 |
| v2 | 2026-06-17 | NQ-REaR | 64 | 0.521 | 0.638 | 83.4% | Comprehensive (85 runs) |
| v1 | 2026-06-16 | HotpotQA | 64 | 0.726 | 0.869 | 83.5% | 48 queries, multi-hop |
| v1 | 2026-06-16 | NarrativeQA | 64 | 0.939 | 0.980 | 95.8% | 49 queries, narrative |
| v1 | 2026-06-16 | 2WikiMultihopQA | 64 | 0.788 | 0.921 | 85.6% | 50 queries, multi-hop |
| v1 | 2026-06-17 | MuSiQue | 64 | 0.453 | 0.672 | 67.4% | 100 queries, multi-hop |
| **v3** | **2026-06-28** | **MuSiQue** | 64 | **0.554** | — | — | **44 queries, batch processing (20x speedup), snippet-ranking** |
| v1 | 2026-06-15 | DROP | 64 | 0.320 | 0.762 | 42.6% | 50 queries, L2 norm |
| v1 | 2026-06-09 | DocFinQA | 128 | 0.250 | 0.341 | 73.3% | 20 queries, financial |
| v2 | 2026-06-13 | MAUD | 64 | 0.000 | 0.649 | 0% | 100 queries, legal |

---

## Reports by Dataset

### PubMedQA
- PubMedQA results are recorded in BENCHMARK_RESULTS.md (MRR=1.000 at 30Q, 0.955 at 111Q)

### Belebele
- Belebele results are recorded in BENCHMARK_RESULTS.md (SF+SPLADE MRR=1.000, surpassing BM25 0.995)
- `belebele/umap_implementation_analysis.md` — UMAP implementation details

### PopQA
- PopQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.980, 100 queries, HippoRAG2)

### BioASQ
- BioASQ results are recorded in BENCHMARK_RESULTS.md (MRR=0.195, 50 queries, biomedical QA — hard)
- Ablation study details in BENCHMARK_RESULTS.md Section 3.2

### NQ-REaR
- NQ-REaR results are recorded in BENCHMARK_RESULTS.md (MRR=0.521, comprehensive 85 runs)

### HotpotQA
- HotpotQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.726, 48 queries, multi-hop)

### NarrativeQA
- NarrativeQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.939, 49 queries, narrative)

### 2WikiMultihopQA
- 2WikiMultihopQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.788, 50 queries, multi-hop)

### MuSiQue
- MuSiQue results are recorded in BENCHMARK_RESULTS.md (MRR=0.453, 100 queries, multi-hop; v3 MRR=0.554, 44 queries, batch processing)
- `musique/v3_20260628_134311.md` — Batch-processed benchmark report

### DROP
- DROP results are recorded in BENCHMARK_RESULTS.md (MRR=0.320, 50 queries, L2 norm)

### DocFinQA
- DocFinQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.250, 20 queries, financial)

### MAUD
- MAUD results are recorded in BENCHMARK_RESULTS.md (MRR=0.000, 100 queries, legal)

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

*Last updated: 2026-06-28*
