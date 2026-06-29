# Strategic Plan: Leveraging Semantic Folding for Specialized QA

**Date:** 2026-06-29
**Analysis based on:** Full 9-dataset benchmark results, SPLADE-only baseline, α-sensitivity curves, pipeline code review
**Branch Strategy:** Each strategy implemented on a dedicated branch per the branch-per-improvement workflow (see AGENTS.md). Branches merged to main only after successful benchmarking and user confirmation.

| Strategy | Branch | Status |
|----------|--------|--------|
| P0: SF as Query Expansion | `feature/sf-query-expansion` | ⬜ Pending |
| P1: SF as Fallback Re-ranker | `feature/sf-fallback-reranker` | ⬜ Pending |
| P2: Alternative Aggregation | `feature/sf-alt-aggregation` | ⬜ Pending |
| P3: Grid-Based Re-Ranker | `feature/sf-grid-reranker` | ⬜ Pending |

---

## 1. The Core Finding

**SF degrades SPLADE on 5/9 datasets when combined linearly.** The α-sensitivity curve is monotonic: as SF weight increases, MRR decreases. The complementarity hypothesis (H2) is falsified — SF and SPLADE signals are correlated, not complementary.

**SF contributes positively on only 2/9 datasets:**
- 2WikiMultihopQA (+8.5% over SPLADE-only)
- PubMedQA (+1.7% over SPLADE-only)

**Why these two?**
- 2Wiki: 2-hop compositional entity chains → SF's phrase-level grid matching captures entity relationships that SPLADE's general-domain training may miss
- PubMedQA: Biomedical domain with specialized MeSH terminology → SF's grid captures domain-specific synonymy better than SPLADE's news-trained embeddings

**SF's value proposition:** Out-of-distribution terminology and compositional entity matching. These are exactly where supervised models (SPLADE) are weakest.

---

## 2. Why Linear Combination Fails

The current hybrid formula (`query_processor.py:2486`):
```
combined = α * sf_score + (1-α) * splade_score
```

**Problem:** SF fingerprints are spatially correlated by design (Gaussian smoothing σ=1.5, Morton encoding, IDF-weighted sum). When aggregated into document fingerprints via IDF-weighted sum + L2 normalization, the resulting vectors produce cosine scores that are highly correlated with SPLADE's scores — but with more noise. Adding noise to a precise signal degrades performance.

**Mathematical intuition:** Let S_sf(d,q) and S_splade(d,q) be the similarity scores. If Cov(S_sf, S_splade) > 0 (they are correlated), then:
```
Var(α·S_sf + (1-α)·S_splade) > Var(S_splade) when α > 0 and Cov dominates
```
More SF adds more correlated variance (noise), not discriminative signal.

---

## 3. Where SF Can Be Leveraged

### 3.1 Strategy A: SF as Query Expansion (Recommended First Try)

Instead of scoring documents with SF, use SF's semantic grid to **expand queries** with grid-adjacent terms, then feed expanded queries to SPLADE/BM25.

**Why this works:**
- SF's grid captures domain-specific synonymy well (that's why PubMedQA +1.7%)
- Query expansion decouples SF's noisy fingerprint correlation from the ranking signal
- SPLADE already knows how to rank well — it just needs better query terms for domain-specific cases

**Implementation:**
1. For each query phrase, find its k-nearest grid neighbors (grid-adjacent phrases)
2. Add these as expansion terms with decay weight γ^(distance)
3. Feed the expanded query to SPLADE (no SF scoring at all)

**Expected gain:** Moderate (+5-10% on domain-specific datasets like PubMedQA, BioASQ)

**Effort:** Low (reuses existing phrase fingerprint index + FAISS)

### 3.2 Strategy B: SF as Low-Confidence Fallback

Use SPLADE as the primary ranker. When SPLADE confidence is low (max score below threshold), use SF to re-rank or expand.

**Why this works:**
- SPLADE is strong on common vocabulary (Belebele 1.000, MuSiQue 0.987)
- SPLADE is weak on domain-specific terminology (PubMedQA 0.952 vs BM25 1.000)
- SF helps exactly where SPLADE is weak — domain synonymy

**Implementation:**
1. Rank with SPLADE
2. If top-1 score < threshold τ, use SF grid to expand query terms
3. Re-rank with expanded query

**Expected gain:** Small (+1-3%) but zero degradation on SPLADE's strong datasets

**Effort:** Low

### 3.3 Strategy C: SF with Alternative Aggregation

**Current aggregation:** IDF-weighted sum → top-10% sparsification → L2 norm
**Problem:** Creates highly correlated document fingerprints

**Alternative aggregations to test:**
1. **Binary OR pooling:** OR all phrase fingerprint bits, threshold at k active bits per doc
2. **Max-pooling:** Take element-wise max across phrase fingerprints (preserves sparse structure better)
3. **No sparsification:** Keep full fingerprint (more information, but less sparse)
4. **Different norm:** L1 norm instead of L2 (produces sparser gradients)

**Expected gain:** Unknown — requires experimentation

**Effort:** Medium (modify doc_fingerprints.py)

### 3.4 Strategy D: SF as Grid-Based Re-Ranker

Instead of linear combination, use SF grid proximity as a re-ranking signal:
1. Rank top-N candidates with SPLADE
2. For each candidate, compute SF grid-proximity score against query phrases
3. Re-rank using a learned or heuristic combination

**Why this differs from current approach:**
- Current: both SF and SPLADE are scored over ALL documents → SF noise propagates
- Proposed: SF only scored on top-N → SF's coarse matching is sufficient on a small set

**Expected gain:** Moderate on multi-hop datasets

**Effort:** Medium

---

## 4. Recommended Implementation Priority

| Priority | Strategy | Datasets to Test | Expected Gain | Effort |
|----------|----------|-----------------|---------------|--------|
| **P0** | SF as Query Expansion (A) | PubMedQA, BioASQ | +5-10% | 1-2 days |
| **P1** | SF as Low-Confidence Fallback (B) | All 9 | +1-3%, zero regression | 2-3 days |
| **P2** | Alternative Aggregation (C) | 2Wiki, HotpotQA | Unknown | 3-5 days |
| **P3** | Grid-Based Re-Ranker (D) | MuSiQue, HotpotQA | Moderate | 5-7 days |

---

## 5. Parameter Space for Optimization

### 5.1 Grid Construction Parameters (Most Impact)

| Parameter | Current | Optimal Range | Notes |
|-----------|---------|---------------|-------|
| grid_size | 64 | 32-128 dataset-dependent | Larger grid = more capacity but sparser fingerprints. BioASQ likely needs 128+. |
| smoothing_sigma | 1.5 | 0.5-2.0 | σ=0 catastrophic (−31.2%). Higher σ = more correlation = more noise in hybrid. |
| method | UMAP | UMAP/t-SNE | UMAP faster; t-SNE better for BioASQ/PubMedQA |

### 5.2 Fingerprint Parameters (Moderate Impact)

| Parameter | Current | Optimal Range | Notes |
|-----------|---------|---------------|-------|
| top_percent | 0.10 | 0.05-0.20 | Less sparsity = more discriminative but higher correlation |
| doc_norm | l2 | l2/l1 | L2 produces smoother scores; L1 may separate better |
| morton | True | True/False | Morton encoding preserves locality but increases correlation |

### 5.3 Query Processing Parameters (Low Impact)

| Parameter | Current | Notes |
|-----------|---------|-------|
| spreading_steps | 1 | radius=2 tested → MRR −7.1% for short queries |
| weighting | idf | uniform tested → MRR −0.86% |
| spreading_decay | 0.5 | Not systematically tested |

### 5.4 Hybrid Parameters (New)

| Parameter | Current | Proposed |
|-----------|---------|----------|
| splade_alpha | 0.3 | **Use 0.0** (SPLADE-only) for most datasets |
| expansion_k | — | k=3-5 grid-neighbors for query expansion (Strategy A) |
| confidence_threshold | — | τ=0.3-0.5 for fallback trigger (Strategy B) |

---

## 6. Implementation Roadmap

### Phase 1: Fix the Foundation (1-2 days)
1. ✅ Fix `generic_benchmark.py` hybrid_alpha bug (DONE)
2. ✅ Run SPLADE-only baselines on all 9 datasets (DONE)
3. ✅ Run α-sensitivity on MuSiQue, Belebele, BioASQ (DONE)
4. Run SPLADE-only as default configuration in registry (TO DO: set splade_alpha=0.0 in registry)

### Phase 2: Query Expansion (2-3 days)
5. Implement SF-based query expansion in query_processor.py
   - For each query phrase, find k=3 nearest grid neighbors
   - Add expansion terms with decay weight
   - Feed to SPLADE
6. Benchmark on PubMedQA, BioASQ, 2Wiki
7. Analyze: does expansion help domain-specific datasets?

### Phase 3: Alternative Aggregation (3-5 days)
8. Implement alternative doc fingerprint aggregation:
   - Binary OR pooling
   - Max-pooling
   - L1 normalization
9. Benchmark on 2Wiki, HotpotQA, MuSiQue
10. Analyze: does less-correlated aggregation improve hybrid?

### Phase 4: Full Re-Benchmark (2-3 days)
11. Run best configuration from Phase 2+3 on all 9 datasets
12. Update all thesis chapters
13. Update BENCHMARK_RESULTS.md

---

## 7. Thesis Impact

After Phase 1-4, the thesis narrative shifts from:
- **Old:** "SF+SPLADE is the best configuration — complementarity wins" (FALSIFIED)
- **New:** "SF provides domain-specific semantic matching that complements SPLADE's general-domain training, but only when deployed as query expansion or fallback re-ranking, not as linear score combination"

This is a stronger thesis because:
1. It acknowledges the failure of linear combination (scientific honesty)
2. It identifies the specific regime where SF adds value (domain terminology, OOD matching)
3. It proposes targeted solutions based on understanding the failure mode
4. It provides a practitioner-ready decision framework for when to use SF

---

## 8. Key Academic Contributions (Revised)

1. **The α-sensitivity framework**: First systematic analysis of hybrid SF-SPLADE weight, revealing monotonic degradation
2. **The spatial correlation problem**: Theoretical + empirical demonstration that Gaussian-smoothed fingerprints are correlated, not orthogonal, invalidating the Orthogonality Constraint application
3. **The feature-invariance principle**: Proof that features duplicating SF signals cannot improve retrieval
4. **SF as query expansion**: Novel application of semantic grid for domain-specific term expansion (if Phase 2 succeeds)
5. **Decision framework**: When to use SF (domain terminology, small candidate pools) vs when not to (general-domain, large corpora)
