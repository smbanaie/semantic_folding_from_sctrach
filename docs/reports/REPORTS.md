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
| **v5 SPLADE-only** | **2026-07-31** | **MuSiQue** | 64 | **0.876** | **0.644** | — | **SPLADE-only (α=0.0), 44 Q, 954 docs — first measured SPLADE-only value on this pool** |
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

### BEIR zero-shot (tuned, main registry)

Three BEIR datasets tuned via `semantic_folding/dataset_tuner.py` (both profiles, 4-way grid)
and written as **top-level entries** in `config/dataset_registry.yml`. SPLADE wins all three.
**Only SciFact is included in the §3 main matrix (row 9);** NFCorpus and SciDocs are retained
in `docs/reports/` but excluded from the matrix (BM25 outperformed SF+SPLADE on both).

| Version | Date | Dataset | Grid | MRR (SF-only) | MRR (SF+SPLADE) | AP | Best profile | Notes |
|---------|------|---------|------|---------------|-----------------|-----|:------------:|-------|
| v1 | 2026-07-20 | SciFact | 64 | 0.860 | 0.869 | 0.863 | sf_only | 50 Q, untuned SF-only |
| v2 | 2026-07-20 | SciFact | 64 | 0.860 | **0.960** | 0.948 | sf_splade | 50 Q, tuned (tsne, mdf0) |
| v1 | 2026-07-20 | NFCorpus | 64 | 0.650 | 0.670 | 0.404 | sf_only | 50 Q, untuned SF-only |
| v2 | 2026-07-20 | NFCorpus | 64 | 0.650 | **0.760** | 0.414 | sf_splade | 50 Q, tuned (tsne, mdf0) |
| v1 | 2026-07-20 | SciDocs | 64 | 0.800 | 0.830 | 0.474 | sf_only | 50 Q, untuned SF-only |
| v1 | 2026-07-20 | SciDocs | 64 | 0.800 | **0.900** | 0.438 | sf_splade | 50 Q, tuned (tsne, mdf20/pct5) |
| **v3 deep-pool** | **2026-07-21** | **SciFact** | 64 | **0.0109** (SF) | **0.0004** (RRF) | — | deep-pool | **gold+top-100 BM25, ~101 cand/q, n=50 — SF=0.0109, SF+SPLADE RRF=0.0004, BM25=0.0095. 16-doc pool MRR 0.960 is an artifact (NOT leaderboard-comparable)** |

> **Methodology note (BEIR runs):** These are tuned SF benchmarks added as top-level keys in
> `config/dataset_registry.yml` (RRF fusion default, grid 64). Retrieval is over the candidate
> pool built by `BEIRAdapter` (each query's gold passages + 15 distractors from the BEIR corpus),
> NOT the full BEIR corpus — so MRR reflects phrase-fingerprint ranking quality within a small
> pool, comparable to the existing in-project benchmark convention. **This makes the pool MRR
> values an artifact of pool size and NOT comparable to published full-corpus BEIR/SciFact
> leaderboards** — see `scifact/v3_20260721_deeppool.md` and BENCHMARK_RESULTS.md §5.6 for the
> methodologically defensible deep-pool (gold + top-100 BM25) validation. msmarco was dropped
> (1.08 GB passage corpus, too large for this run). See `docs/reports/<dataset>/v2_*_tuned.md`.

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
- n=100 confirmatory core (SF+SPLADE × 7 ops): `hotpotqa/v2_20260824_n100_confirmatory_core.md` (CombSUM 0.746, 4/21 Holm survivors — largely non-separable)

### HotpotQA
- HotpotQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.726, 48 queries, multi-hop)
- **n=100 confirmatory core (2026-08-24)**: `hotpotqa/v2_20260824_n100_confirmatory_core.md` — SF+SPLADE × 7 operators; CombSUM **0.947** vs RRF 0.854 (p_Holm=0.0007), 15/21 pairwise Holm survivors

### NarrativeQA
- NarrativeQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.939, 49 queries, narrative)

### 2WikiMultihopQA
- 2WikiMultihopQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.788, 50 queries, multi-hop)

### MuSiQue
- MuSiQue results are recorded in BENCHMARK_RESULTS.md (MRR=0.453, 100 queries, multi-hop; v3 MRR=0.554, 44 queries, batch processing)
- `musique/v3_20260628_134311.md` — Batch-processed benchmark report
- `musique/v5_20260731_230557_spladeonly.md` — SPLADE-only (α=0.0) benchmark report (MRR=0.876 ± 0.082, 44 Q, 954 docs)
- n=100 confirmatory core (SF+SPLADE × 7 ops): see `hotpotqa/v2_20260824_n100_confirmatory_core.md` (CombSUM/zscore **0.952** vs RRF 0.908; 17/21 Holm survivors)

### DROP
- DROP results are recorded in BENCHMARK_RESULTS.md (MRR=0.320, 50 queries, L2 norm)

### DocFinQA
- DocFinQA results are recorded in BENCHMARK_RESULTS.md (MRR=0.250, 20 queries, financial)

### SciFact (BEIR)
- `scifact/v1_20260720_124345.md` — SF-only benchmark report (MRR=0.869, AP=0.863, 50 Q)
- `scifact/v2_20260720_151348_tuned.md` — Tuned report (MRR=0.960, AP=0.948, sf_splade, tsne/mdf0)
- `scifact/v2_20260720_155945_linear.md` — SF+SPLADE Linear (MRR=0.900)
- `scifact/v2_20260720_162711_bm25.md` — BM25 baseline (MRR=0.900)
- **`scifact/v3_20260721_deeppool.md` — Deep-pool validation (gold + top-100 BM25, n=50): SF MRR=0.0109, BM25 MRR=0.0095. ⚠️ The 16-doc pool MRR 0.960 is a retrieval-recall artifact, NOT comparable to BEIR/SciFact leaderboards. See BENCHMARK_RESULTS.md §5.6.**

### NFCorpus (BEIR)
- `nfcorpus/v1_20260720_124345.md` — SF-only benchmark report (MRR=0.670, AP=0.404, 50 Q)
- `nfcorpus/v2_20260720_151856_tuned.md` — Tuned report (MRR=0.760, AP=0.414, sf_splade, tsne/mdf0)
- `nfcorpus/v2_20260720_160448_linear.md` — SF+SPLADE Linear (MRR=0.680)
- `nfcorpus/v2_20260720_162711_bm25.md` — BM25 baseline (MRR=0.866)

### SciDocs (BEIR)
- `scidocs/v1_20260720_124345.md` — SF-only benchmark report (MRR=0.830, AP=0.474, 50 Q)
- `scidocs/v2_20260720_152707_tuned.md` — Tuned report (MRR=0.900, AP=0.438, sf_splade, tsne/mdf20/pct5)
- `scidocs/v2_20260720_161237_linear.md` — SF+SPLADE Linear (MRR=0.730)
- `scidocs/v2_20260720_162711_bm25.md` — BM25 baseline (MRR=0.952)

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
