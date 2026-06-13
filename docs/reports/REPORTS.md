# Benchmark Reports — Master Index

## Version History

| Version | Date | Dataset | Grid | MRR (SF) | MRR (BM25) | Notes |
|---------|------|---------|------|----------|------------|-------|
| v1 | 2026-06-06 | PubMedQA | 64 | 0.955 | 1.000 | Baseline, 112 queries |
| v1 | 2026-06-07 | PubMedQA | 128 | 0.902 | — | Grid=128 experiment |
| v3 | 2026-06-13 | Belebele | 64 | 0.740 | 0.995 | 100 queries |
| v3 | 2026-06-13 | MAUD | 64 | 0.000 | 0.649 | 100 queries |

---

## Reports by Dataset

### PubMedQA
- `pubmedqa/v1_20260606_162818.md` — Baseline (grid=64, 112 queries, MRR=0.955)
- `pubmedqa/v1_20260607_grid128.md` — Grid=128 experiment (MRR=0.902)
- Source: `data/pubmedqa/REPORT.md`, `data/pubmedqa/RECOMMENDATIONS.md`

### Belebele
- `belebele/v3_20260613_223315.md` — 100 queries, MRR=0.740

### MAUD
- `maud/v3_20260613_233820.md` — 100 queries, MRR=0.000

### Cross-Dataset
- `cross-dataset/comparison_20260613.md` — SF vs BM25 across all datasets

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

*Last updated: 2026-06-13*
