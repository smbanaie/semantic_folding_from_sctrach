# Two-Stage Neuro-Lexical Pipeline - Implementation Plan (UPDATED)

**Branch**: `feature/two-stage-neuro-lexical-pipeline`  
**Date**: 2026-06-30  
**Status**: Partial Implementation (CLI + Structure Done, SF Re-ranking INCOMPLETE)

## Objective

Implement a two-stage retrieval pipeline that uses BM25 for coarse lexical filtering (top-K) followed by Semantic Folding for fine-grained semantic re-ranking. This addresses the BioASQ scaling limitation where SF-only struggles with large corpora.

**Preliminary Results**: MRR 0.288 (SF-only) → 0.441 (BM25+SF re-rank) on BioASQ

## Current Implementation Status

### ✅ COMPLETED

1. **CLI Flags Added** (`generic_benchmark.py`):
   - `--two-stage` - Enable two-stage retrieval
   - `--pool-size 100` - BM25 top-K pool size (default: 100)
   - Flags added to BOTH `benchmark` and `all` subparsers

2. **Parameter Handling** (`generic_benchmark.py`):
   - `params["two_stage"] = args.two_stage`
   - `params["pool_size"] = args.pool_size`

3. **Benchmark Script Created** (`two_stage_benchmark.py`):
   - Loads BM25 index from corpus
   - Loads SF document fingerprints
   - Implements two-stage loop structure
   - Saves results in benchmark format (per_query, CSV, summary.json)

4. **Minimal Working Script** (`two_stage_minimal.py`):
   - Standalone implementation
   - Same structure as `two_stage_benchmark.py`
   - Currently uses PLACEHOLDER for SF re-ranking

### ❌ INCOMPLETE

1. **SF Re-ranking NOT Implemented**:
   - Current code returns BM25 results directly (no SF re-ranking)
   - Placeholder in `two_stage_minimal.py`: "Simulate SF re-ranking by adding small random noise"
   - **Need to integrate actual `query_processor.py` logic**

2. **Candidate Filtering NOT Implemented**:
   - `query_processor.py` does NOT accept `--candidate-ids` flag
   - Currently scores ALL documents (not just top-K candidates)
   - **Need to modify `query_processor.py` to accept candidate list**

## Architecture

```
Query → [Stage 1: BM25] → Top-K Candidates → [Stage 2: SF Re-ranking] → Final Ranked List
```

**Key Difference from Hybrid Approach**:
- Hybrid: Linearly combines BM25 + SF scores → `α*BM25 + (1-α)*SF`
- Two-Stage: BM25 filters to top-K, SF re-ranks ONLY those K candidates

## Next Steps (Priority Order)

### Step 1: Implement SF Re-ranking (CRITICAL)

**File**: `semantic_folding/query_processor.py`

**Changes Needed**:
1. Add `--candidate-ids` flag to `parse_args()`
2. Modify `rank_documents()` to accept optional `candidate_ids` list
3. When `candidate_ids` provided, ONLY score those documents (not full corpus)

**CLI Flags**:
```bash
python query_processor.py \
    --query "..." \
    --candidate-ids "doc_000001,doc_000002,doc_000003" \
    --doc-fp-dir outputs/run/doc_fingerprints/
```

**Estimated Time**: 3-4h

### Step 2: Integrate into Two-Stage Benchmark

**File**: `semantic_folding/dataset_benchmark/two_stage_benchmark.py`

**Changes Needed**:
1. Replace PLACEHOLDER with actual call to `query_processor.py`
2. Pass BM25 top-K candidates as `--candidate-ids`
3. Parse SF re-ranked results
4. Compute metrics on re-ranked list

**Estimated Time**: 2h

### Step 3: Benchmark on BioASQ

**Test Datasets** (large corpus):
- BioASQ (biomedical, complex queries) - **PRIMARY TARGET**
- NQ-REaR (natural questions, large corpus)
- PubMedQA (biomedical, medium corpus)

**Benchmark Matrix**:
| Dataset | SF-Only | BM25-Only | Two-Stage (K=50) | Two-Stage (K=100) | Two-Stage (K=200) |
|---------|----------|------------|-------------------|--------------------|--------------------|
| BioASQ | MRR 0.288 | MRR 0.442 | TBD | TBD | TBD |
| NQ-REaR | TBD | TBD | TBD | TBD | TBD |
| PubMedQA | MRR 0.936 | MRR 0.967 | TBD | TBD | TBD |

**Estimated Time**: 3h (50 queries each)

### Step 4: Evaluate and Document

**Metrics**:
- MRR, AP, P@K, R@K, NDCG@K
- Found-at distribution
- Per-query analysis (top performers, failures)

**Comparison vs**:
- SF-only baseline
- BM25-only baseline
- SPLADE-only baseline
- Hybrid SF+BM25 (existing)

**Estimated Time**: 1h

## Files Modified (So Far)

| File | Changes | Status |
|------|---------|--------|
| `semantic_folding/dataset_benchmark/generic_benchmark.py` | Added `--two-stage`, `--pool-size` flags to `benchmark` and `all` subparsers; added parameter handling | ✅ COMPLETE |
| `semantic_folding/dataset_benchmark/two_stage_benchmark.py` | Created new benchmark script with BM25 + SF structure (SF re-ranking is PLACEHOLDER) | ⚠️ PARTIAL |
| `two_stage_minimal.py` | Standalone minimal implementation (same as above) | ⚠️ PARTIAL |

## Files to Modify (Next)

| File | Changes |
|------|---------|
| `semantic_folding/query_processor.py` | Add `--candidate-ids` flag; modify `rank_documents()` to filter candidates |
| `semantic_folding/dataset_benchmark/two_stage_benchmark.py` | Replace PLACEHOLDER with actual SF re-ranking call |
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

## Timeline (Revised)

| Step | Estimated Time | Status |
|------|----------------|--------|
| Step 1: Modify query_processor.py | 3-4h | ❌ NOT STARTED |
| Step 2: Integrate SF re-ranking | 2h | ❌ NOT STARTED |
| Step 3: Run benchmarks | 3h | ❌ NOT STARTED |
| Step 4: Analyze results | 1h | ❌ NOT STARTED |
| **Total** | **9-10h** | |

## How to Test (After Implementation)

```bash
# Step 1: Index BioASQ dataset
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark index \
    --dataset bioasq \
    --jsonl data/bioasq/converted/bioasq.jsonl \
    --max-queries 50

# Step 2: Run two-stage benchmark (after implementing SF re-ranking)
.venv\Scripts\python -m semantic_folding.dataset_benchmark.two_stage_benchmark \
    --dataset bioasq \
    --jsonl data/bioasq/converted/bioasq.jsonl \
    --run-dir outputs/bioasq_benchmark/runs/run_<timestamp> \
    --pool-size 100 \
    --max-queries 50

# Step 3: Compare with SF-only and BM25-only
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
    --dataset bioasq \
    --jsonl data/bioasq/converted/bioasq.jsonl \
    --max-queries 50

.venv\Scripts\python -m semantic_folding.dataset_benchmark.bm25_benchmark \
    --dataset bioasq \
    --jsonl data/bioasq/converted/bioasq.jsonl \
    --run-dir outputs/bioasq_benchmark/runs/run_<timestamp> \
    --query-end 50
```

## Related Issues

- BioASQ scaling limitation (§8.5 in paper)
- Candidate filtering optimization
- Score fusion strategies

## References

- Preliminary results: MRR 0.288 → 0.441 on BioASQ
- Hybrid SF+BM25 tested in recommendations.md (0% improvement on Belebele)
- Two-stage differs from hybrid: filtering vs score fusion

---

**Next Action**: Implement `--candidate-ids` flag in `query_processor.py` to enable actual SF re-ranking on BM25 candidates.
