# Semantic Folding Pipeline — Final Improvement Roadmap

> **Generated**: 2026-06-26
> **Scope**: Synthesis of pipeline code audit, benchmark results (12 datasets, 47+ runs), and literature-backed recommendations
> **Status**: Consolidation of `docs/recommendations.md`, `docs/reports/BENCHMARK_RESULTS.md`, and pipeline code analysis

---

## 1. Executive Summary

Semantic Folding (SF) achieves strong results on single-hop semantic tasks (PopQA 98%, PubMedQA 95.5%, NarrativeQA 95.8%) but degrades on compositional, numerical, and multi-hop tasks (MuSiQue 67.4%, DROP 42.6%). The pipeline has been extended with 10+ experimental features, most of which showed **zero or negative impact** at benchmark time — the core pipeline is already well-tuned for current datasets.

**The single highest-impact improvement found**: SF+SPLADE hybrid → **MRR 1.0 on Belebele** (+13.6%), **+60.3% on NQ-REaR**, **+35.4% on HotpotQA**.

---

## 2. Current Pipeline Architecture Analysis

### 2.1 Code Quality & Maintainability

| Aspect | Assessment | Evidence |
|--------|-----------|----------|
| Modularity | Good — 15+ discrete files with clear responsibilities | `phrase_extractor.py` → `term_context.py` → `semantic_space.py` → `phrase_fingerprints.py` → `doc_fingerprints.py` → `query_processor.py` |
| Monolith risk | **`lib.py` (2500+ lines)** — core utils, fingerprint loading, similarity functions all in one file | `lib.py` contains 15+ unrelated concerns |
| Cross-cutting concerns | Logging, config loading, CLI argument handling duplicated | `semantic_folder.py` duplicates CLI rename maps, config path maps |
| Test coverage | **Zero automated tests** | AGENTS.md confirms "no test suite" |
| Error handling | Good — graceful degradation (returns `(None, metadata)` tuples) | `construct_query_fingerprint` returns error metadata instead of raising |
| Documentation | Extensive — every function has detailed docstrings with examples | 50% of lines in `query_processor.py` are docstrings |

### 2.2 Feature Implementation Status

| Feature | File(s) | Status | Benchmark Impact | Lines of Code |
|---------|---------|--------|------------------|---------------|
| BM25 Hybrid | `query_processor.py` | ✅ Working | +16.2% Belebele (100Q) | ~120 |
| SPLADE Hybrid | `splade_scorer.py`, `query_processor.py` | ✅ Working | **+13.6% Belebele, +60.3% NQ-REaR** | ~350 |
| Negation Handling | `negation_handler.py` | ✅ Working | 0% on factoid queries | 296 |
| Ontology Expansion | `ontology_expander.py` | ✅ Working | 0% on Belebele, -2.3% PubMedQA | 217 |
| Query Decomposition | `query_decomposer.py` | ⚠️ Basic | +19.6% NQ-REaR, -28.8% HotpotQA | 199 |
| Multi-resolution Spreading | `query_processor.py` | ✅ Working | 0% | ~30 inline |
| Adaptive Spreading | `query_processor.py` | ✅ Working | 0% | ~25 inline |
| LambdaMART Re-ranking | `reranker_train.py`, `reranker_infer.py` | ⚠️ Implemented, no training data | Not measured | ~200 each |
| Feature Extraction | `reranker_features.py` | ✅ Working | Foundation for Phase 5 | 263 |
| Spatial Jaccard | `query_processor.py` | ✅ Working | **-65% PubMedQA, -60% BioASQ** | ~25 inline |
| Score Normalization | `query_processor.py` | ✅ Working | 0% (rank order unchanged) | ~50 |
| Asymmetric Scoring | `query_processor.py` | ✅ Working | 0% | ~20 inline |
| Alternative Metrics (Dice/Overlap/Jaccard/IDF-weighted) | `lib.py`, `query_processor.py` | ✅ Working | 0% on float fingerprints | ~80 |
| OOV Expansion (pseudo-fingerprint) | `query_processor.py` | ✅ Working | Core feature | ~350 |
| LanceDB Storage | `lance_storage.py` | ✅ Implemented | Alternative to NPZ | ~200 |

### 2.3 Performance Profile

| Phase | Time (100Q, 1862 docs) | Bottleneck |
|-------|------------------------|------------|
| Indexing (Steps 1-5) | ~60 min | t-SNE (O(N²)), fingerprint generation |
| Per query (Step 6) | ~30s | OOV expansion via full-vocab cosine sweep |
| BM25 baseline | ~10s | — |
| **SF/BM25 speed ratio** | **360x slower** | 60x indexing, 3000x per query |

### 2.4 Storage

| Artifact | Format | Typical Size (64×64 grid) |
|----------|--------|---------------------------|
| Phrase fingerprints | NPZ (dense float32) | 2-5 MB (n_phrases × 4096) |
| Document fingerprints | NPZ (dense float32) | 7-30 MB (n_docs × 4096) |
| OOV expansion index | In-memory dict | Duplicate of phrase fingerprints |
| LanceDB (optional) | Columnar | Comparable to NPZ |

---

## 3. Prioritized Improvement Roadmap

### Priority Matrix

```
                    High Impact                    Low Impact
                    ──────────────────────────────────────────
Easy                │ SPLADE hybrid (R1)          │ Negation handling (R6)
                    │ L2 normalization (adopted)   │ Ontology expansion (R8)
                    │ Score normalization (P2)     │ Adaptive spreading (R5)
                    │ Asymmetric scoring (P3)      │ Multi-resolution (R4)
                    │ Alternative metrics (P1)     │
                    │                              │
Hard                │ Learned grid mapping (R2)    │ Joint doc-snippet (R10)
                    │ Query-doc cross-attention(R3)│ Multi-scale grids (R4)
                    │ Multi-stage pipeline (R13)   │ Spatial-aware Jaccard (R9)
                    │ LambdaMART cascade (P5)      │
                    │ Query decomposition (R11)    │
                    │ Ontology-guided retrieval(R12)│
```

### P0 — Immediate Wins (Effort: < 4h each)

| # | Improvement | Expected Gain | Current Status | Action |
|---|-------------|--------------|----------------|--------|
| P0.1 | **SPLADE hybrid for all datasets** | **+13.6% Belebele, +60.3% NQ-REaR, +35.4% HotpotQA** | Implemented, needs flag automation | Make SPLADE the default for factoid/multi-hop; SF-only for narrative |
| P0.2 | **Per-dataset parameter auto-select** | +1-4% | Config is manual | Create dataset registry with recommended params |
| P0.3 | **t-SNE perplexity=50 default** | +1.5-4% on Belebele | Configurable, default is 30 | Change `config/semantic_folding.yml` default |
| P0.4 | **L2 doc norm as permanent default** | +4% on Belebele | Already defacto | Lock in config; remove `sqrt_nnz` option |
| P0.5 | **OOV expansion speed optimization** | 30s→0.5s per query | Full-vocab cosine sweep | Pre-build FAISS index or use approximate nearest neighbor |

#### P0.1 Detail: SPLADE as Default Hybrid

**Evidence from benchmarks:**
- Belebele: SF-only 0.880 → SF+SPLADE **1.000** (+13.6%)
- NQ-REaR: SF-only 0.574 → SF+SPLADE **0.920** (+60.3%)
- HotpotQA: SF-only 0.726 → SF+SPLADE **0.983** (+35.4%)
- 2WikiMultihopQA: SF-only 0.788 → SF+SPLADE **0.983** (+24.8%)
- BioASQ: SF-only 0.248 → SF+SPLADE **0.527** (+18.4%)
- Only narrative tasks (NarrativeQA) are hurt by SPLADE (-19%)

**Implementation change:**
- `query_processor.py`: Change default `--splade` to `True`
- Add `--splade-alpha` default 0.3 (SPLADE weight) with auto-detect for narrative datasets
- Cache SPLADE embeddings to disk to avoid re-computation at each query

#### P0.5 Detail: OOV Expansion Vectorization

**Current bottleneck**: `expand_oov_query_terms` does a full-vocab cosine sweep for every query — O(V × D) where V ≈ 2000-10000 phrases.

**Solutions (in order of effort):**
1. **FAISS index** (4h): Build IVFFlat index over `vocab_matrix` at query_processor init. O(log V) per query instead of O(V).
2. **Cache popular OOV expansions** (1h): LRU cache keyed by `(oov_term_frozenset)` to avoid re-expanding common patterns.
3. **SpMM-based batch scoring** (2h): Replace per-OOV-item loop with single sparse matrix multiply.

---

### P1 — High Impact, Moderate Effort (4-8h each)

| # | Improvement | Expected Gain | Current Status | Action |
|---|-------------|--------------|----------------|--------|
| P1.1 | **LambdaMART re-ranking** | +10-15% MRR | Implemented, needs training data | Generate training pairs from existing benchmarks |
| P1.2 | **Query decomposition (rule-based, improve)** | +5-10% on multi-hop | Basic implementation | Expand decomposition patterns; add entity linking |
| P1.3 | **Ontology-guided retrieval for BioASQ** | +5-10% on biomedical | Glossary expansion only | Use MeSH hierarchy; expand to concept IDs |
| P1.4 | **Multi-stage pipeline (SF→SPLADE→Cross-encoder)** | +10-15% MRR | Not implemented | Stage 1: SF (top-100), Stage 2: SPLADE (top-20), Stage 3: cross-encoder (top-5) |
| P1.5 | **LanceDB as default storage backend** | Faster load times | Implemented | Flip default; fall back to NPZ for small corpora |
| P1.6 | **Memory-efficient fingerprint loading** | 2-5x memory reduction | All fingerprints in RAM | Memory-map NPZ files; lazy-load OOV index |

#### P1.1 Detail: LambdaMART Training Protocol

**Training data generation:**
1. For each query in all benchmark datasets, run `reranker_features.py` to get 35-dim feature vectors
2. Generate preference pairs: for each query, all correct (doc_id=gold) vs incorrect (doc_id≠gold) pairs
3. Train LightGBM LambdaMART with `objective='lambdarank'`, `metric='ndcg'`
4. Evaluate with 5-fold cross-validation across datasets

**Cross-dataset evaluation (critical for thesis):**
- Train on: Belebele + PubMedQA + PopQA (diverse domains)
- Evaluate on: MuSiQue, HotpotQA, NQ-REaR (held-out)
- This directly tests whether learned re-ranking generalizes across domains

#### P1.2 Detail: Improved Query Decomposition

**Current limitations:**
- Regex-based patterns only cover 3-4 patterns
- No entity linking — sub-queries don't reference each other's results
- No LLM-assisted decomposition

**Proposed approach:**
1. **Entity extraction**: Use spaCy NER to identify entities in query
2. **Relation extraction**: Use dependency parse to identify relations between entities
3. **Sub-query generation**: For each (entity, relation, entity) triple, generate a sub-query
4. **Result fusion**: RRF as currently implemented, plus optional AND/OR fusion

---

### P2 — High Impact, High Effort (1-3 days each)

| # | Improvement | Expected Gain | Status | Action |
|---|-------------|--------------|--------|--------|
| P2.1 | **Learned grid mapping** | +5-10% MRR | Research opportunity | Replace t-SNE with contrastive-learned 2D mapping |
| P2.2 | **Query-document cross-attention** | +3-8% MRR | Research opportunity | Lightweight attention between query and doc fingerprints |
| P2.3 | **Multi-resolution fingerprints** | +2-5% MRR | Partially tested (128×128 alone hurt) | Combine 32×32 + 64×64 + 128×128 via weighted fusion |
| P2.4 | **Joint document-snippet ranking** | +5-10% MRR | Research opportunity | PDRMM-style ranking using SF fingerprints as features |

#### P2.1 Detail: Learned Grid Mapping

**Problem**: t-SNE optimizes for visualization quality (local neighborhood preservation), not retrieval quality (ranking accuracy). Grid mapping is a fixed preprocessing step.

**Approach**: Train a 2-layer neural network to map high-dim phrase embeddings to 2D grid coordinates using contrastive loss:
- Positive pairs: phrases that co-occur in the same document
- Negative pairs: phrases from different documents
- Loss: `L = -log(sim(pos)) - log(1 - sim(neg))` where `sim` is cosine similarity of resulting fingerprints

**Benefits**:
- Grid optimizes for retrieval rather than visualization
- Can be fine-tuned per domain (biomedical grid vs legal grid)
- Gumbel-Softmax trick enables differentiable discrete grid assignment

---

### P3 — Low Priority / Experimental

| # | Improvement | Expected Gain | Status | Action |
|---|-------------|--------------|--------|--------|
| P3.1 | **Spatial-weighted similarity** | +1-3% | Spatial-Jaccard tested (-65%) | Try different weighting schemes |
| P3.2 | **Score normalization** | 0% (tested) | Already implemented | Skip — no gain found |
| P3.3 | **Binary metrics (Dice/Jaccard/Overlap)** | 0% (tested) | Already implemented | Skip — float fingerprints need float metrics |
| P3.4 | **Adaptive spreading** | 0% (tested) | Already implemented | Skip — fixed radius=1 is optimal |
| P3.5 | **Multi-resolution spreading** | 0% (tested) | Already implemented | Skip — grid_size=64 already optimal |
| P3.6 | **Negation-aware scoring** | 0% on factoid, +5-10% on negation queries | Already implemented | Keep as opt-in; may help on Belebele negation subset |
| P3.7 | **TF-IDF re-ranking** | 0% (tested) | Already implemented | Skip |
| P3.8 | **Query expansion via pseudo-relevance feedback** | <0% (tested) | Already tested | Skip — -2.3% on PubMedQA, 0% on Belebele |

---

## 4. Code-Specific Recommendations

### 4.1 Refactoring Needed

| File | Issue | Recommendation | Effort |
|------|-------|---------------|--------|
| `lib.py` | 2500+ lines, 15+ concerns | Split into: `lib/normalize.py`, `lib/fingerprints.py`, `lib/similarity.py`, `lib/io.py` | 4h |
| `semantic_folder.py` | Duplicated CLI rename maps (CLI_RENAME_MAP, NEGATE_FLAG_MAP) | Move to `lib/cli.py` or shared config | 1h |
| `query_processor.py` | 2700+ lines, 15+ functions | Split into: `query/extract.py`, `query/fingerprint.py`, `query/ranking.py`, `query/hybrid.py`, `query/expansion.py` | 4h |
| `semantic_folder.py` | Hardcoded Python path `E:\\PHD\\...\\.venv\\scripts\\python` | Use `sys.executable` instead | 10min |
| All files | Config paths duplicated across files | Centralize path resolution in `lib.py` or config | 1h |

### 4.2 Bug / Issue Tracking

| Issue | File | Line(s) | Description | Priority |
|-------|------|---------|-------------|----------|
| `load_document_fingerprints` version conflict | `lib.py` | 1053-1116, 1117-1226 | Two functions with same name `load_phrase_fingerprints_sparse` — one returns `Dict[str, np.ndarray]`, the other `Dict[str, csr_matrix]`. Import order determines which is used. | **High** |
| Hybrid corpus alignment | `query_processor.py` | ~2347-2354 | Corpus text loading assumes `doc_id,text` CSV format; silently truncates if sizes don't match | **Medium** |
| SPLADE model cache | `splade_scorer.py` | — | SPLADE model loaded per query (no caching) | **Medium** |
| t-SNE non-determinism | `semantic_space.py` | — | Different embeddings per run → slightly different fingerprints | **Low** |
| Morton encoding in geometric kernel | `query_processor.py` | 1358-1382 | `_unflatten_vector` / `_flatten_grid` are O(grid²) with Python loops | **Low** |
| Duplicate class definitions | `semantic_folder.py` | 132-165, 348-488 | `PhraseVisualizationHandler` has both `handle()` method and `execute()` from base class | **Low** |

### 4.3 Performance Gaps

| Gap | Impact | Root Cause | Fix |
|-----|--------|------------|-----|
| OOV expansion per query | **30s → 0.5s** | Full-vocab cosine sweep O(V×D) | FAISS index or LRU cache |
| Document fetching in hybrid | **+5s per query** | Re-parses CSV corpus for every query | Cache corpus texts in memory |
| No batch query processing | N/A | Each query processed independently | Add `--batch-queries` flag |
| `pos_tag` NLTK per phrase | **~100ms per phrase** | Called inside `normalize_phrase` which is per-candidate | Batch tagging across all candidates |
| JSON serialization of results | **~1s per query** | Writes full results dict to disk | Only serialize top-K results |

---

## 5. Benchmark Data Gaps

| Gap | Issue | Impact | Action |
|-----|-------|--------|--------|
| No BioASQ Tier 1 comparison | SF MRR=0.248 vs SOTA unknown | Cannot position against biomedical SOTA | Run BioASQ Task 12b evaluation |
| No long-doc parameter tuning | Large corpus datasets at default parameters | May improve with tuning | Try long-doc chunking + domain-specific params |
| No cross-encoder comparison | Only compared against BM25 | Misses SOTA comparison | Compare against MiniLM/L-6 cross-encoders |
| Large variance in query count | 20-200 queries per dataset | Some results are statistically weak | Run all datasets at 200+ queries |
| No statistical significance tests | No confidence intervals | Results may be noise | Bootstrap CI for MRR estimates |
| No BEIR benchmark | Only 12 custom datasets | Cannot compare against published BEIR leaderboard | Run SF on BEIR (BioASQ, TREC-COVID, NFCorpus, etc.) |

---

## 6. Thesis & Publication Roadmap

### 6.1 Claims Supported by Evidence

| Claim | Datasets | MRR | Counter-Evidence |
|-------|----------|-----|------------------|
| "SF matches BM25 on single-hop semantic tasks" | PopQA, PubMedQA, NarrativeQA | 94-98% | NQ-REaR (83.4%) — entity matching gap |
| "SF + SPLADE surpasses BM25" | Belebele | **100.5%** | Only one dataset; NarrativeQA -19% |
| "SF degrades on compositional tasks" | MuSiQue, HotpotQA, 2Wiki | 67-84% of BM25 | Consistent across 3 datasets |
| "SF is entirely unsupervised" | All | N/A | True by design |
| "SF is interpretable via fingerprint visualization" | All | N/A | Visualizations exist for phrase/doc fingerprints |

### 6.2 Required Before Publication

1. **Fix the double `load_phrase_fingerprints_sparse` bug** in `lib.py`
2. **Benchmark with statistical significance** (bootstrap CI, n≥100 per dataset)
3. **Compare against a cross-encoder** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
4. **Run on at least one BEIR dataset** for published baseline comparison
5. **Ablation study**: measure each pipeline component's contribution
6. **Parameter sensitivity analysis**: 2D grid of (grid_size × perplexity × smoothing_sigma × spreading_radius)
7. **Visualization examples** for the paper: fingerprint heatmaps, top-K comparison, failure analysis

### 6.3 Narrative Arc

```
                    ┌─ SF + SPLADE (MRR 1.0) ─┐
                    │                            │
     ┌─ BM25 ───────┼──────────────────────────────┼──> SOTA systems
     │              │                              │
─────┴──────────────┴──────────────────────────────┴─────────
     │              │                              │
     │        FF1: Vocabulary mismatch         FF2: Compositional
     │        (80% of BM25 on entity             gap (67% of
     │         tasks with paraphrasing)          BM25 on MuSiQue)
     │
     │  FF3: Semantic dilution
     │       (all docs score 0.034-0.051)
     │
     Key findings:
     • SF matches BM25 on single-hop semantic tasks (95-98%)
     • SF+SPLADE surpasses BM25 on reading comprehension (+13.6%)
     • SF fails on multi-hop, legal, and numerical reasoning
     • Unsupervised advantage: zero training data needed
```

---

## 7. Implementation Timeline

```
Week 1-2: P0 Items
├─ P0.1: SPLADE as default hybrid (2h)
├─ P0.2: Dataset parameter registry (2h)
├─ P0.3: Change t-SNE perplexity default (10min)
├─ P0.4: Lock L2 norm as default (10min)
├─ P0.5: OOV expansion speed optimization (FAISS or LRU) (4h)
├─ Fix: Double load_phrase_fingerprints_sparse bug (30min)
├─ Fix: CPU-only caching for SPLADE model (1h)

Week 3-4: P1 Items
├─ P1.1: LambdaMART training data generation + model (8h)
├─ P1.2: Improved query decomposition (entity + relation) (4h)
├─ P1.3: MeSH ontology integration for BioASQ (4h)
├─ P1.5: LanceDB as default storage (2h)
├─ Refactor: Split lib.py into modules (4h)
├─ Refactor: Split query_processor.py into submodules (4h)

Week 5-6: P2 + Publication Prep
├─ P2.1: Learned grid mapping prototype (2 days)
├─ Benchmark: BEIR datasets + cross-encoder (8h)
├─ Ablation study: Component-by-component (4h)
├─ Statistical significance: bootstrap CI (2h)
├─ Fix: deterministic t-SNE seed (30min)
├─ Visualizations for paper (3h)

Week 7-8: Paper Drafting
├─ Results summary tables
├─ Methodology description
├─ Related work positioning
├─ Thumbnail fingerprint visualizations
```

---

## 8. Summary: What to Keep, Drop, or Fix

| Category | Count | Items |
|----------|-------|-------|
| **✅ Keep / Adopt** | 7 | SPLADE hybrid, L2 norm, t-SNE perplexity=50, dataset registry, FAISS OOV index, single-hop focus, unsupervised advantage |
| **⚠️ Keep but opt-in** | 5 | Negation handling, ontology expansion, query decomposition, LambdaMART, multi-resolution spreading |
| **❌ Drop / Skip** | 7 | Alternative binary metrics (Dice/Jaccard), score normalization, asymmetric scoring, spatial Jaccard, TF-IDF re-rank, adaptive spreading, pseudo-relevance feedback |
| **🔧 Fix** | 5 | Double function definition bug, hardcoded Python path, corpus alignment, SPLADE cache, t-SNE determinism |
| **🔬 Research / Future** | 6 | Learned grid mapping, cross-attention, multi-resolution fingerprints, joint doc-snippet ranking, BEIR benchmark, multi-stage pipeline |

---

*This document is the single source of truth for all pipeline improvements. Update after each completed phase with new benchmark results.*
