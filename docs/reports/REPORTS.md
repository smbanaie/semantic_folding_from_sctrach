# Benchmark Reports — Master Index

> **Single source of truth**: [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) — complete consolidated report with all datasets, metrics, and analysis.

## Quick Reference: Metrics & Acronyms

| Term | Meaning |
|------|---------|
| **MRR** | Mean Reciprocal Rank — average of 1/rank of the first relevant result. Primary metric. Range [0,1], higher is better. |
| **AP** | Average Precision — mean precision at each relevant result's rank. Captures ranking quality across all relevant docs. |
| **P@K / R@K** | Precision/Recall at K — fraction of top-K results that are relevant / fraction of relevant docs found in top-K. |
| **NDCG@K** | Normalized Discounted Cumulative Gain — position-aware ranking quality normalized by the ideal ranking. |
| **SF** | Semantic Folding — the unsupervised retrieval method proposed in this thesis. |
| **SPLADE** | Sparse Lexical and Expansion Model — pre-trained learned sparse retriever (Formal et al., 2021). |
| **BM25** | Best Matching 25 — classic lexical retrieval baseline. |
| **DPR** | Dense Passage Retrieval — neural dense vector baseline (Karpukhin et al., 2020). |
| **RRF** | Reciprocal Rank Fusion — rank-level fusion (Cormack et al., 2009). Combines rankings by position, not scores. |
| **α** | Alpha — fusion weight for SF in linear fusion: score = α·SF + (1−α)·SPLADE. Default 0.3. |
| **k (RRF)** | Rank constant — smoothing in RRF formula. Default 60. Higher k = less rank sensitivity. |
| **Δ** | Delta — percentage change between methods. Positive = improvement. |
| **SF/BM25** | SF-to-BM25 ratio — >100% means SF outperforms BM25. |

For full metric definitions and formulae, see [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) §Glossary.

---

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
| **v3** | 2026-06-28 | MuSiQue | 64 | **0.554** | — | — | **44 queries, batch, SPLADE off (SF-only baseline)** |
| v3 SPLADE | 2026-06-28 | MuSiQue | 64 | 0.554 | — | — | *OLD — ran without --corpus, SPLADE didn't actually run* |
| **v4 SPLADE** | **2026-06-28** | **MuSiQue** | 64 | **0.782** | **0.523** | **0.705** | **Fixed SPLADE +41% MRR vs SF-only (44 Q, 954 docs)** |
| v3 OOV | 2026-06-28 | MuSiQue | 64 | **0.541** | — | — | OOV expansion enabled (−2.3% MRR, 7× slower) |
| v1 | 2026-06-15 | DROP | 64 | 0.320 | 0.762 | 42.6% | 50 queries, L2 norm |
| v1 | 2026-06-09 | DocFinQA | 128 | 0.250 | 0.341 | 73.3% | 20 queries, financial |
| **RRF** | **2026-07-07** | **Belebele** | 64 | 0.940 (Lin) | **1.000 (RRF)** | — | **+6.4% MRR with RRF fusion** |
| RRF | 2026-07-07 | NarrativeQA | 64 | 0.940 (Lin) | 0.967 (RRF) | — | +2.8% MRR with RRF |
| RRF | 2026-07-07 | PopQA | 64 | 1.000 (Lin) | 1.000 (RRF) | — | Tie |
| RRF | 2026-07-07 | PubMedQA | 64 | 0.968 (Lin) | 0.968 (RRF) | — | Tie |
| RRF | 2026-07-07 | HotpotQA | 64 | 0.872 (Lin) | 0.857 (RRF) | — | −1.7% (linear wins) |
| RRF | 2026-07-07 | NQ-REaR | 64 | 0.632 (Lin) | 0.631 (RRF) | — | Tie |
| RRF | 2026-07-07 | 2WikiMultihopQA | 64 | 0.901 (Lin) | 0.761 (RRF) | — | −15.5% (linear wins) |

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
