# Two-Stage Neuro-Lexical Pipeline - Implementation Plan

**Branch**: `feature/two-stage-neuro-lexical-pipeline`  
**Date**: 2026-06-30  
**Status**: Planning

## Objective

Implement a two-stage retrieval pipeline that uses BM25 for coarse lexical filtering (top-K) followed by Semantic Folding for fine-grained semantic re-ranking. This addresses the BioASQ scaling limitation where SF-only struggles with large corpora.

**Preliminary Results**: MRR 0.288 (SF-only) → 0.441 (BM25+SF re-rank) on BioASQ

## Architecture

```
Query → [Stage 1: BM25] → Top-K Candidates → [Stage 2: SF Re-ranking] → Final Ranked List
```

**Key Difference from Hybrid Approach**:
- Hybrid: Linearly combines BM25 + SF scores → `α*BM25 + (1-α)*SF`
- Two-Stage: BM25 filters to top-K, SF re-ranks only those K candidates

## Implementation Plan

### Step 1: Modify `generic_benchmark.py`

**File**: `semantic_folding/dataset_benchmark/generic_benchmark.py`

**Changes**:
1. Add `--two-stage` flag to `phase2_benchmark()`
2. Add `--pool-size` flag (default: 100, options: 50/100/200)
3. In batch processing loop:
   - First pass: Run BM25 to get top-K candidate IDs
   - Second pass: Run SF but ONLY score those K candidates (not full corpus)
   - Return re-ranked list

**CLI Flags**:
```bash
--two-stage               # Enable two-stage retrieval
--pool-size 100          # BM25 top-K pool size
--two-stage-alpha 0.5    # Optional: Score fusion weight (BM25 vs SF)
```

### Step 2: Implement Two-Stage Logic

**Function to modify**: `phase2_benchmark()` in `generic_benchmark.py`

**Pseudocode**:
```python
if self.params.get("two_stage", False):
    # Stage 1: BM25 retrieval
    bm25_scorer = BM25Scorer(corpus_texts)
    bm25_results = bm25_scorer.score(query_text)
    top_k_candidates = [doc_id for doc_id, _ in bm25_results[:pool_size]]
    
    # Stage 2: SF re-ranking (constrained to candidates)
    sf_results = run_sf_query(query_text, candidate_ids=top_k_candidates)
    
    # Optional: Score fusion
    if fusion:
        fused_scores = fuse_scores(bm25_results, sf_results, alpha)
        return fused_scores
    else:
        return sf_results  # Pure re-ranking
```

### Step 3: Optimize SF for Candidate Re-ranking

**File**: `semantic_folding/query_processor.py`

**Changes**:
- Add `--candidate-ids` flag to `query_processor.py`
- When set, only load + score those specific document fingerprints
- Massive speedup: Don't need to score all N documents

**CLI**:
```bash
python query_processor.py \
    --query "..." \
    --candidate-ids doc_000001,doc_000002,doc_000003 \
    --doc-fp-dir outputs/run/doc_fingerprints/
```

### Step 4: Benchmark Script

**Test Datasets** (large corpus):
- BioASQ (biomedical, complex queries)
- NQ-REaR (natural questions, large corpus)
- PubMedQA (biomedical, medium corpus)

**Benchmark Matrix**:
| Dataset | SF-Only | BM25-Only | Two-Stage (K=50) | Two-Stage (K=100) | Two-Stage (K=200) |
|---------|----------|------------|-------------------|--------------------|--------------------|
| BioASQ | MRR 0.288 | MRR 0.442 | TBD | TBD | TBD |
| NQ-REaR | TBD | TBD | TBD | TBD | TBD |
| PubMedQA | MRR 0.936 | MRR 0.967 | TBD | TBD | TBD |

### Step 5: Evaluation

**Metrics**:
- MRR, AP, P@K, R@K, NDCG@K
- Found-at distribution
- Per-query analysis (top performers, failures)

**Comparison vs**:
- SF-only baseline
- BM25-only baseline
- SPLADE-only baseline
- Hybrid SF+BM25 (existing)

## Files to Modify

| File | Changes |
|------|---------|
| `semantic_folding/dataset_benchmark/generic_benchmark.py` | Add `--two-stage`, `--pool-size` flags; implement two-stage logic in `phase2_benchmark()` |
| `semantic_folding/query_processor.py` | Add `--candidate-ids` flag for constrained re-ranking |
| `semantic_folding/dataset_benchmark/bm25_benchmark.py` | Extract `BM25Scorer` class for reuse |
| `docs/recommendations.md` | Update with new implementation status |
| `docs/reports/BENCHMARK_RESULTS.md` | Add two-stage results |

## Expected Outcomes

**Hypothesis**:
1. Two-stage will outperform SF-only on large corpora (BioASQ, NQ-REaR)
2. Optimal pool size K will be dataset-dependent (test K=50/100/200)
3. Two-stage may underperform SPLADE-only (SPLADE is stronger lexical model)

**Success Criteria**:
- MRR improvement > 5% over SF-only on BioASQ
- No degradation on small-corpus datasets (Belebele, PopQA)

## Timeline

| Step | Estimated Time | Status |
|------|----------------|--------|
| Step 1: Modify generic_benchmark.py | 2h | Pending |
| Step 2: Implement two-stage logic | 2h | Pending |
| Step 3: Optimize query_processor.py | 1h | Pending |
| Step 4: Run benchmarks | 3h | Pending |
| Step 5: Analyze results | 1h | Pending |
| **Total** | **9h** | |

## Next Steps

1. Implement Step 1-3 (code changes)
2. Test on BioASQ with K=100
3. Run full benchmark matrix
4. Update documentation
5. Merge to main (if successful)

---

**Related Issues**:
- BioASQ scaling limitation (§8.5 in paper)
- Candidate filtering optimization
- Score fusion strategies

**References**:
- Preliminary results: MRR 0.288 → 0.441 on BioASQ
- Hybrid SF+BM25 tested in recommendations.md (0% improvement on Belebele)
- Two-stage differs from hybrid: filtering vs score fusion
