# Journal Paper Expansion — PLAN

## Version Control

- **Branch:** `feature/journal-a-expansion` (created 2026-08-22 from `main`)
- All work below is committed to this branch. Merge to `main` only after explicit user confirmation (project convention).

## Overview

This plan breaks down the journal paper expansion into **5 phases** with explicit dependencies. Each phase produces concrete deliverables that feed into the next phase. Results are logged in `EXPANSION-RESULT.md`.

---

## Phase 0: Setup & Baseline Verification (Week 0)

### 0.1 Environment Verification
- [ ] Verify `.venv` works: `.venv\Scripts\python -c "import semantic_folding; print('OK')"`
- [ ] Verify all 9 datasets available in `data/<dataset>/converted/*.jsonl`
- [ ] Verify benchmark infrastructure: `semantic_folding/dataset_benchmark/generic_benchmark.py`
- [ ] Run sanity check on 1 dataset (e.g., Belebele 10 queries)

### 0.2 Current State Audit
- [ ] Extract all existing results from `docs/reports/BENCHMARK_RESULTS.md` into structured format
- [ ] Identify gaps in master table (which cells are empty)
- [ ] Document current operator coverage (Linear, RRF only — need CombSUM, CombMNZ, Borda, z-score, min-max)
- [ ] Document current model pair coverage (SF+SPLADE only — need SF+DPR, BM25+SPLADE, BM25+DPR)

**Deliverable:** `EXPANSION-RESULT.md` initialized with Phase 0 audit results

---

## Phase 1: Complete Operator Matrix on Existing Datasets (Weeks 1-2)

### 1.1 Implement Missing Fusion Operators
**Primary location:** NEW module `semantic_folding/fusion_operators.py` (clean, reusable, unit-testable)
**Wiring:** `semantic_folding/query_processor.py` Stage 4b + `semantic_folding/dataset_benchmark/generic_benchmark.py`

Operators to implement (definitions in SPEC §Fusion Operators):
- [ ] **CombSUM**: `score(d) = s_A(d) + s_B(d)` (Fox & Shaw 1994)
- [ ] **CombMNZ**: `score(d) = (s_A(d) + s_B(d)) × m(d)`, m = #retrievers returning d (Fox & Shaw 1994)
- [ ] **Borda**: `score(d) = Σ_r (N − rank_r(d) + 1)` (rank aggregation)
- [ ] **z-score + Linear**: per-retriever `(x−μ)/σ`, then α-weighted combine
- [ ] **min-max + Linear**: per-retriever `(x−min)/(max−min)`, then α-weighted combine
- [ ] **L2-norm + Linear**: per-retriever unit-L2, then α-weighted combine (secondary)
- [ ] **RRF** (exists, verify k=60 in fusion module)
- [ ] **Linear** (exists, verify α=0.3 in fusion module)

**Code artifacts:**
- `semantic_folding/fusion_operators.py` — `fuse(operator, scores_a, scores_b, **params)` + `rank_from_scores()`
- Patch `query_processor.py` lines ~2576-2610 to delegate to `fusion_operators.fuse`
- Extend `--fusion-method` choices (line 2966) to include all 7
- Patch `generic_benchmark.py` (phase2, ~line 754-770) to loop over `--fusion-operators` list

**Unit test:** `temp/test_fusion_operators.py` — assert RRF invariance under monotonic transform (Proposition 1), CombSUM/CombMNZ/Borda correctness on a tiny fixture.

### 1.2 Add Operator Flag to Benchmark Runner
**Location:** `semantic_folding/dataset_benchmark/generic_benchmark.py`
- [ ] Add `--fusion-operators` flag accepting comma-separated list
- [ ] Add `--fusion-params` for operator-specific params (k for RRF, α for Linear)
- [ ] Ensure all operators work in batch query mode (`--query-file`)

### 1.3 Run Complete Matrix on 8 Core Datasets
**Datasets:** PopQA, PubMedQA, NarrativeQA, Belebele, 2WikiMultihopQA, HotpotQA, MuSiQue, NQ-REaR
**Queries:** 50 per dataset (standard)
**Operators:** Linear, RRF, CombSUM, CombMNZ, Borda, z-score, min-max
**Model Pairs:** SF+SPLADE (baseline), plus SF-only, SPLADE-only, BM25 baselines

```bash
# Example command structure (to be refined)
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset <name> --jsonl data/<name>/converted/<name>.jsonl \
  --max-queries 50 \
  --fusion-operators linear,rrf,combsumb,combmnz,borda,zscore,minmax \
  --model-pairs sf_splade,sf_only,splade_only,bm25 \
  --output outputs/<name>_benchmark/full_matrix_<timestamp>/
```

### 1.4 Statistical Analysis
- [ ] Implement paired bootstrap test in `semantic_folding/dataset_benchmark/benchmark_analyzer.py`
- [ ] Add 95% CI for all MRR values
- [ ] Add Holm correction for multiple comparisons
- [ ] Generate master table with all cells filled + CIs + significance markers

**Deliverable:** Complete master table (Phase 1 section in EXPANSION-RESULT.md)

---

## Phase 2: Second Model Pair (Weeks 2-3)

### 2.1 Add DPR Retriever Integration
**Location:** New `semantic_folding/dpr_scorer.py` or extend `splade_scorer.py` pattern
- [ ] Load pre-trained DPR (e.g., `facebook/dpr-ctx_encoder-single-nq-base`, `facebook/dpr-question_encoder-single-nq-base`)
- [ ] Pre-encode corpus for each dataset (cache embeddings)
- [ ] Implement DPR scoring function compatible with fusion interface
- [ ] Verify DPR scores on 1 dataset

### 2.2 Run 4 Model Pairs on All Datasets
| Signal A | Signal B | Purpose |
|----------|----------|---------|
| BM25 | SPLADE | Baseline learned+lexical |
| BM25 | DPR | Baseline lexical+dense |
| SF | SPLADE | Current main pair |
| SF | DPR | **Critical: SF with different score geometry** |

### 2.3 Full Operator Matrix for Each Pair
Run all 7 operators on all 4 model pairs × 8 datasets = 224 configurations

**Deliverable:** 4×8×7 results table (Phase 2 section in EXPANSION-RESULT.md)

---

## Phase 3: Synthetic Magnitude Experiment (Week 3)

### 3.1 Implement Synthetic Score Generator
**Location:** `semantic_folding/synthetic_magnitude_experiment.py`
- [ ] Generate synthetic retrieval results with controlled rank/magnitude
- [ ] Conditions:
  - Condition 1: Large margin (A=45, B=12, rank_A=1, rank_B=2)
  - Condition 2: Small margin (A=20, B=18, rank_A=1, rank_B=2)
  - Condition 3: Reversed margin (A=12, B=45, rank_A=1, rank_B=2)
  - Additional: Vary margin systematically (45/12, 40/15, 35/20, 30/25, 25/30...)
- [ ] Apply all 7 fusion operators
- [ ] Measure: Which operator correctly ranks A > B?

### 3.2 Rank-Preserving Transformation Test
- [ ] Apply monotonic transformations: log, sqrt, exp, sigmoid, min-max, z-score
- [ ] Verify RRF invariance: RRF(f(s)) == RRF(s) for all monotonic f
- [ ] Measure score-fusion sensitivity: F(f(s)) != F(s)
- [ ] Plot: Performance vs. magnitude separation for each operator

### 3.3 Connect to Real Data
- [ ] Extract real SPLADE score distributions from multi-hop vs single-hop queries
- [ ] Show: multi-hop queries have larger score margins between relevant/irrelevant
- [ ] Demonstrate: synthetic margin manipulation mimics real multi-hop behavior

**Deliverable:** Synthetic experiment results + figures (Phase 3 section in EXPANSION-RESULT.md)

---

## Phase 4: Full-Corpus Evaluation (Weeks 3-4)

### 4.1 Select 2+ Full-Corpus Datasets
**Priority 1 (Standard IR):** SciFact (BEIR, 5,183 docs) — infrastructure exists
**Priority 2 (Multi-hop):** HotpotQA full corpus (5.2M Wikipedia paragraphs) — may need sampling
**Alternative:** MS MARCO dev set (subset)

### 4.2 Implement Full-Corpus Pipeline
- [ ] Modify `generic_benchmark.py` to support full-corpus mode
- [ ] Add candidate generation: BM25 top-1000 → rerank with SF/SPLADE/DPR
- [ ] Or: Direct ANN search for dense retrievers (FAISS)
- [ ] Implement recall@k evaluation (not just MRR on candidate pool)

### 4.3 Run Full-Corpus Experiments
- [ ] SF-only, SPLADE-only, DPR-only baselines
- [ ] All fusion operators on model pairs
- [ ] Compare controlled-reranking vs full-corpus results

**Deliverable:** Full-corpus results showing whether findings generalize (Phase 4 section in EXPANSION-RESULT.md)

---

## Phase 5: Feature Invariance & Score Concentration (Weeks 4-5)

### 5.1 Feature Invariance — Adversarial Features
**Location:** Extend `semantic_folding/reranker_features.py` or new `feature_invariance.py`
- [ ] Implement non-collinear features:
  - Term rarity (IDF-based)
  - Document length normalization
  - Phrase coverage (% of query phrases in doc)
  - Query-term diversity (entropy of query terms)
  - Proximity (min span of query terms in doc)
  - Score margin (top-1 minus top-2 score)
  - Independent BM25 score
- [ ] Measure correlation with SF overlap (qᵀd)
- [ ] Ablation: Add each feature to SF ranking, measure ΔMRR
- [ ] Plot: corr(feature, overlap) vs ΔMRR

### 5.2 Score Concentration — Scaling Experiment
**Location:** `semantic_folding/score_concentration_scaling.py`
- [ ] Candidate sizes: N ∈ {20, 50, 100, 250, 500, 1k, 5k, 10k}
- [ ] For each N: Sample N candidates (1 gold + N-1 BM25 negatives)
- [ ] Measure: mean, std, CV, max score, gold score, gold rank, MRR, Recall@k
- [ ] Compare: SF, BM25, SPLADE, DPR
- [ ] Plot: CV vs N, MRR vs N, gold rank vs N

### 5.3 Theoretical Analysis
- [ ] Derive expected overlap statistics for binary SDRs
- [ ] Show: E[overlap] = Kρ, Var = Kρ(1-ρ)
- [ ] Connect to empirical CV curves

**Deliverable:** Feature invariance extended results + scaling experiment figures (Phase 5 section in EXPANSION-RESULT.md)

---

## Phase 6: Paper Writing & Restructuring (Weeks 5-7)

### 6.1 Restructure Conference Paper to Journal Format
- [ ] Move SF architecture to Appendix A
- [ ] Write new Section 3: Conceptual Framework
- [ ] Expand Section 2: Related Work (add Bruch et al. positioning)
- [ ] Rewrite Section 4: Experimental Methodology (two regimes, statistical protocol)
- [ ] Write Section 5: Zero-Shot Semantic Signal (tone down SF claims)
- [ ] Write Section 6: Fusion Operator Analysis (main empirical contribution)
- [ ] Write Section 7: Magnitude Information Hypothesis (synthetic + real)
- [ ] Write Section 8: Representation and Scaling Boundaries
- [ ] Write Section 9: Discussion (reviewer test answers, what we don't establish)
- [ ] Write Section 10: Limitations and Conclusion
- [ ] Update Abstract, Title, Contributions

### 6.2 Generate All Figures/Tables
- [ ] Master table (8 datasets × 7 operators × 4 model pairs)
- [ ] Synthetic magnitude experiment figure
- [ ] Scaling experiment figure (CV vs N, MRR vs N)
- [ ] Feature invariance figure (corr vs ΔMRR)
- [ ] Operator topology decision matrix
- [ ] Full-corpus vs controlled comparison

### 6.3 Statistical Validation
- [ ] All MRR values with 95% bootstrap CIs
- [ ] All pairwise comparisons with p-values
- [ ] Holm correction applied to confirmatory tests
- [ ] Effect sizes reported

### 6.4 Final Review Against Reviewer Test Questions
- [ ] Answer all 5 reviewer questions with experimental evidence
- [ ] Verify no "must remove" phrases remain
- [ ] Verify title is journal-appropriate
- [ ] Verify contributions match 4 contributions in SPEC

**Deliverable:** Complete journal paper draft (Phase 6 section in EXPANSION-RESULT.md)

---

## Dependency Graph

```
Phase 0 (Setup)
    │
    ├──→ Phase 1 (Operator Matrix) ──────────────────┐
    │                                                │
    ├──→ Phase 2 (Second Model Pair) ←───────────────┤  (can run in parallel after Phase 0)
    │                                                │
    ├──→ Phase 3 (Synthetic Experiment) ←────────────┤  (needs Phase 1 operators)
    │                                                │
    ├──→ Phase 4 (Full Corpus) ←─────────────────────┤  (needs Phase 2 DPR)
    │                                                │
    └──→ Phase 5 (Feature Invariance + Scaling) ─────┘  (independent, needs Phase 0)
                              │
                              ▼
                       Phase 6 (Paper Writing)
                              │
                              ▼
                        SUBMISSION
```

---

## Resource Requirements

| Resource | Needed For | Estimated Time |
|----------|------------|----------------|
| GPU (1x) | DPR encoding, SPLADE encoding | Phase 2, 4 |
| CPU (8+ cores) | Benchmark runs, bootstrap (1000 resamples) | All phases |
| Disk (~50GB) | Cached embeddings, full corpus indices | Phase 2, 4 |
| Time | Full matrix: ~224 configs × 50 queries × 8 datasets | ~40-60 hrs compute |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Full-corpus too slow | Use top-1000 BM25 candidates + rerank; sample 100 queries |
| DPR encoding OOM | Batch encoding, FP16, gradient checkpointing |
| Synthetic experiment not convincing | Add more margin conditions; connect to real score distributions |
| Reviewer rejects "task-operator" claim | Scope claim to tested operators/datasets; emphasize "hypothesis" not "law" |
| Multiple comparison burden | Pre-register primary comparisons; use Holm; report exploratory separately |

---

## Milestone Tracking

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Phase 0 complete | Day 1 | ☐ |
| Phase 1 complete (master table) | Day 7-10 | ☐ |
| Phase 2 complete (4 model pairs) | Day 14-17 | ☐ |
| Phase 3 complete (synthetic) | Day 20 | ☐ |
| Phase 4 complete (full corpus) | Day 25 | ☐ |
| Phase 5 complete (invariance+scaling) | Day 30 | ☐ |
| Phase 6 complete (paper draft) | Day 40-45 | ☐ |
| Internal review | Day 45 | ☐ |
| Submission ready | Day 50 | ☐ |

---

## Commands Reference

### Run Full Operator Matrix
```bash
# For each dataset
for ds in popqa pubmedqa narrativeqa belebele 2wikimultihopqa hotpotqa musique nq_rear; do
  .venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
    --dataset $ds --jsonl data/$ds/converted/$ds.jsonl \
    --max-queries 50 \
    --fusion-operators linear,rrf,combsumb,combmnz,borda,zscore,minmax \
    --model-pairs sf_splade,sf_only,splade_only,bm25 \
    --run-dir outputs/${ds}_benchmark/full_matrix_$(date +%Y%m%d_%H%M%S)
done
```

### Run Synthetic Experiment
```bash
.venv\Scripts\python semantic_folding/synthetic_magnitude_experiment.py \
  --output results/synthetic_magnitude_$(date +%Y%m%d_%H%M%S).json
```

### Run Scaling Experiment
```bash
.venv\Scripts\python semantic_folding/score_concentration_scaling.py \
  --dataset musique --candidate-sizes 20,50,100,250,500,1000,5000,10000 \
  --output results/scaling_$(date +%Y%m%d_%H%M%S).json
```

### Compute Statistics
```bash
.venv\Scripts\python semantic_folding/dataset_benchmark/benchmark_analyzer.py \
  --run-dir outputs/<dataset>_benchmark/full_matrix_<ts> \
  --paired-bootstrap --holm-correction \
  --output results/stats_<dataset>_<ts>.json
```

---

## Next Steps

1. **Immediate:** Run Phase 0 verification commands
2. **Day 1-2:** Implement missing fusion operators (Phase 1.1)
3. **Day 3-7:** Run full operator matrix on all datasets (Phase 1.3)
4. **Parallel:** Start DPR integration (Phase 2.1)

Update `EXPANSION-RESULT.md` after each completed step.