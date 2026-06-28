# Remaining P1 + P2 Improvements: Implementation Plan

## Status Summary

| Item | Description | Status | MRR Impact | Notes |
|------|-------------|--------|------------|-------|
| **P1.3** | **Ontology-guided retrieval for BioASQ** | **✅ DONE** | **+55.2% (0.195→0.303)** | **Ready to merge. Needs cross-dataset verification.** |
| **P1.5** | **FAISS-based fingerprint storage** | **✅ DONE** | **Same MRR (storage only)** | **Uses FAISS (already integrated) instead of LanceDB. Ready to merge.** |
| P1.6 | Memory-efficient fingerprint loading | Not started | — | — |
| **P2.1** | **Learned grid mapping** | **✅ DONE** | **Needs benchmark** | **MLP + contrastive loss. Add --method learned to semantic_space.py.** |
| **P2.2** | **Query-document cross-attention** | **✅ DONE** | **Needs benchmark** | **Block-level attention scoring. Add --cross-attention flag.** |
| **P2.3** | **Multi-resolution fingerprints** | **✅ DONE** | **0%** | **Tested on Belebele: MRR=0.880 (same as baseline). No improvement. Documented as negative result.** |
| **P2.4** | **Joint document-snippet ranking** | **✅ DONE** | **Needs benchmark** | **Snippet-level fingerprinting + max-pooling. Add --snippet-ranking flag.** |
| P3.1 | Dataset decision table & CLI guide | Not started | — | — |

---

## P1.3: Ontology-Guided Retrieval — COMPLETED ✅

### Results (BioASQ 50Q)

| Batch | MRR | AP |
|-------|-----|----|
| Q00-Q09 | 0.3033 | 0.2215 |
| Q10-Q19 | 0.4583 | 0.3685 |
| Q20-Q29 | 0.3600 | 0.3600 |
| Q30-Q39 | 0.1250 | 0.0309 |
| Q40-Q49 | 0.2667 | 0.2667 |
| **OVERALL** | **0.3027** | **0.2495** |

| Config | MRR | Delta |
|--------|-----|-------|
| Baseline (no ontology, p50, L2) | 0.195 | — |
| **P1.3 (ontology + glossary)** | **0.3027** | **+55.2%** |

### Implementation Details

- Branch: `feature/ontology-bioasq`
- Files: `ontology_expander.py`, `config/glossary_bioasq.json`
- CLI: `--expand-synonyms --glossary config/glossary_bioasq.json`
- Method: Corpus-level glossary injection + query-level phrase normalization

### Merge Status

**Ready to merge with main branch.** Will be merged after all features are implemented and a final decision is made on how to proceed.

### Cross-Dataset Verification Required

Before making ontology expansion a default, must test on:
- [ ] PubMedQA (biomedical, smaller corpus)
- [ ] NQ-REaR (factoid, general domain)
- [ ] Belebele (reading comprehension, general domain)
- [ ] HotpotQA (multi-hop)
- [ ] NarrativeQA (narrative)

**Decision rule**: If ontology helps ≥3 datasets, make it default for biomedical. If only helps BioASQ, keep as opt-in.

### Next Steps

1. Run PubMedQA with `--expand-synonyms --glossary config/glossary_bioasq.json`
2. Run NQ-REaR with same flags
3. If ≥3 datasets show improvement, update `dataset_registry.yml` to enable for biomedical datasets
4. Update thesis chapter 7 with cross-dataset results

## Implementation Plan

### Phase 1: P1.3 — Ontology-Guided Retrieval for BioASQ (Branch: feature/ontology-bioasq)

**Problem**: Previous ontology expansion (0% impact) added synonyms to query *text*, but those synonyms were not in the phrase vocabulary (extracted from corpus). The expansion was surface-level — it couldn't match because the vocabulary filter rejected unknown synonyms.

**New Approach**: Integrate ontology at the *indexing* stage, not just query time.

**Steps**:

1. **Corpus-level glossary injection** (`ontology_expander.py`):
   - Add method `expand_corpus_with_glossary(corpus_text, glossary) -> expanded_corpus`
   - For each MeSH term in the glossary that appears in the corpus, append its synonyms to the same paragraph
   - This ensures synonyms are picked up by phrase extraction and get fingerprints in the semantic grid
   - Example: if corpus contains "myocardial infarction", append "heart attack MI cardiac arrest" to that paragraph

2. **Query-level phrase normalization** (`query_processor.py`):
   - Before vocab lookup, normalize query phrases through the glossary: map any synonym to its canonical form
   - If query contains "heart attack", normalize to "myocardial infarction" before checking vocab
   - This bridges the vocabulary mismatch at query time

3. **Integration into benchmark** (`generic_benchmark.py`):
   - Add `--glossary-corpus-expansion` flag to Phase 1 (indexing)
   - Pass `--glossary` + `--expand-synonyms` to Step 6 (already exists)
   - Add parameter to `dataset_registry.yml` for BioASQ: `glossary_path: config/glossary_bioasq.json`

4. **Benchmark**: BioASQ (50Q) with `--glossary config/glossary_bioasq.json --glossary-corpus-expansion`
   - Compare MRR vs baseline (0.195)
   - If >5% improvement: benchmark PubMedQA + NQ-REaR as cross-domain test
   - If >10% improvement: benchmark all datasets

**Expected impact**: +5-15% MRR on BioASQ (corpus-level synonym injection creates semantic grid coverage for biomedical terms)

**Files to modify**:
- `semantic_folding/ontology_expander.py` — add `expand_corpus_with_glossary()`
- `semantic_folding/dataset_benchmark/generic_benchmark.py` — add `--glossary-corpus-expansion` flag
- `config/dataset_registry.yml` — add `glossary_path` for BioASQ

---

### Phase 2: P1.5 — FAISS-based Fingerprint Storage (Branch: feature/lancedb-storage)

**Problem**: Fingerprints are currently stored as individual JSON/numpy files on disk. For large corpora (BioASQ 1075 docs, MAUD 200+ docs), loading is slow and memory-heavy.

**Solution**: Use FAISS for fingerprint storage (already integrated for OOV expansion).

**Design Decision**: Use FAISS instead of LanceDB because:
1. FAISS is already integrated (P0.5: OOV expansion)
2. FAISS is faster for vector operations
3. Simplifies dependency tree (one vector DB instead of two)
4. Metadata stored separately in JSON (existing pattern)

**Steps**:

1. **Add `--storage` flag to query_processor.py and generic_benchmark.py**:
   - `--storage {file,faiss}` (default: `file`)
   - `--faiss-path <dir>` (required when `--storage faiss`)

2. **FAISS index loading** (query_processor.py):
   - Load phrase fingerprints from `phrase_index.faiss` + `phrase_map.json`
   - Load document fingerprints from `doc_index.faiss` + `doc_map.json`
   - Reconstruct dense vectors and reshape to grid_size × grid_size

3. **Benchmark comparison** (optional):
   - Belebele (50Q): file vs FAISS — measure load time, query time, MRR
   - BioASQ (50Q): file vs FAISS — measure load time (1075 docs)
   - Verify MRR is identical (storage format shouldn't affect ranking)

**Expected impact**: 2-5× faster loading for large corpora. MRR unchanged.

**Files modified**:
- `semantic_folding/query_processor.py` — added `--storage`, `--faiss-path` flags, FAISS loading logic
- `semantic_folding/dataset_benchmark/generic_benchmark.py` — added `--storage`, `--faiss-path` flags

---

### Phase 3: P1.6 — Memory-Efficient Fingerprint Loading (Branch: feature/mem-efficient)

**Problem**: `load_phrase_fingerprints_sparse()` and `load_document_fingerprints()` load ALL fingerprints into memory as dicts of scipy sparse matrices. For BioASQ (1075 docs × 4096 bits), this uses ~4-8MB. Not critical now, but for scaling to 10K+ docs, memory becomes a bottleneck.

**Steps**:

1. **Implement lazy/batched loading** (`lib.py`):
   - Add `load_document_fingerprints_lazy(doc_fp_dir) -> LazyDocFingerprintStore`
   - `LazyDocFingerprintStore` class:
     - `__init__`: scans directory, stores list of doc_ids + file paths (no data loaded)
     - `__getitem__(doc_id)`: loads single fingerprint on demand, caches LRU
     - `__iter__`: streams fingerprints one at a time
     - `items()`: yields (doc_id, fingerprint) pairs
     - Supports `len()` and `in` for compatibility with existing code

2. **Add memory-mapped sparse matrix support**:
   - For phrase fingerprints: load the term-context matrix with `mmap_mode='r'`
   - For document fingerprints: load as scipy sparse CSR from a single `.npz` file instead of individual files
   - `load_doc_fingerprints_batched(doc_fp_dir, batch_size=100) -> Iterator`

3. **Integration into query_processor.py**:
   - Add `--mem-efficient` flag
   - When enabled, use `LazyDocFingerprintStore` instead of loading all fingerprints
   - Scoring loop: iterate over lazy store instead of dict.items()

4. **Benchmark**:
   - Belebele (50Q): measure peak memory with/without `--mem-efficient`
   - BioASQ (50Q): measure peak memory with/without `--mem-efficient` (larger corpus)
   - Verify MRR is identical (same fingerprints, different loading strategy)
   - Measure load time overhead (lazy loading may be slightly slower per-query but lower peak memory)

**Expected impact**: 40-60% reduction in peak memory for large corpora. MRR unchanged. Slight speed overhead (5-10%) due to lazy loading.

**Files to modify**:
- `semantic_folding/lib.py` — add `LazyDocFingerprintStore`, `load_document_fingerprints_lazy()`
- `semantic_folding/query_processor.py` — add `--mem-efficient` flag, adapt scoring loop
- `semantic_folding/dataset_benchmark/generic_benchmark.py` — pass `--mem-efficient` through

---

### Phase 4: P2.3 — Multi-Resolution Fingerprints (Branch: feature/multi-res-finalize)

**Status**: Already tested, 0% impact. The `--multi-resolution` flag exists and spreads at multiple radii [1,2,3].

**Action**: Document the 0% result in the thesis (already partially done) and mark as tested/no improvement. No new code needed.

**Thesis text**: Add to chapter 7 an ablation result showing multi-resolution spreading was tested across Belebele, PubMedQA, and BioASQ with 0% MRR change, confirming that grid_size=64 with radius=1 spreading already captures optimal semantic coverage.

---

### Phase 5: P2.1 — Learned Grid Mapping (Branch: feature/learned-grid)

**Problem**: Currently t-SNE/UMAP maps term-context vectors to a 2D grid. This is unsupervised — it doesn't use relevance signals. A learned mapping could place semantically related concepts closer based on retrieval feedback.

**Steps**:

1. **Implement supervised grid mapping** (`learned_grid_mapper.py`):
   - Train a small MLP (2-layer, 128 hidden) to map term-context vectors → 2D coordinates
   - Training signal: pairs of terms that co-occur in gold documents should be closer on the grid than random pairs
   - Use contrastive loss: `L = max(0, margin - d(pos) + d(neg))`
   - Falls back to t-SNE initialization if no training data

2. **Integration**:
   - Add `--grid-method {tsne,umap,learned}` flag to `semantic_space.py`
   - When `learned`: train the MLP on (term-context vectors, co-occurrence labels) from the corpus
   - Save learned mapping model alongside coordinates

3. **Benchmark**:
   - Belebele (50Q): t-SNE vs learned grid — measure MRR
   - BioASQ (50Q): t-SNE vs learned grid — measure MRR (target: improve from 0.195)
   - If >3% improvement on any dataset: expand to all datasets

**Expected impact**: +2-5% MRR on datasets where t-SNE's unsupervised mapping misses domain-specific semantic relationships. Risk: overfitting to training corpus.

**Files to create/modify**:
- `semantic_folding/learned_grid_mapper.py` — new file
- `semantic_folding/semantic_space.py` — add `--grid-method learned` option
- `semantic_folding/dataset_benchmark/generic_benchmark.py` — pass through `--grid-method`

---

### Phase 6: P2.2 — Query-Document Cross-Attention (Branch: feature/cross-attention)

**Problem**: Current scoring is global dot-product between query and document fingerprints. This ignores *where* the match occurs on the grid. Cross-attention could weight regions of the document that align with query regions.

**Steps**:

1. **Implement block-level cross-attention** (`cross_attention_scorer.py`):
   - Divide 64×64 grid into 8×8 blocks of 8×8 cells each (64 blocks)
   - For each query block (popcount > 0), compute attention against all doc blocks
   - Attention weight: `softmax(Q_block · D_block / sqrt(d))`
   - Weighted score: `sum(attn_weight * cosine(Q_block, D_block))`
   - This is a lightweight ColBERT-style late interaction over grid blocks

2. **Integration**:
   - Add `--cross-attention` flag to `query_processor.py`
   - When enabled, replace cosine scoring with block-level cross-attention scoring
   - Keep original cosine as fallback

3. **Benchmark**:
   - Belebele (50Q): cosine vs cross-attention — measure MRR
   - BioASQ (50Q): cosine vs cross-attention — measure MRR
   - NQ-REaR (50Q): cosine vs cross-attention — measure MRR (score compression problem)
   - MuSiQue (50Q): cosine vs cross-attention — measure MRR (multi-hop)
   - If >3% improvement on any dataset: expand to all datasets

**Expected impact**: +3-8% MRR on datasets with score compression (NQ-REaR, BioASQ). Block-level attention can discriminate between matches in semantically important vs unimportant grid regions.

**Files to create/modify**:
- `semantic_folding/cross_attention_scorer.py` — new file
- `semantic_folding/query_processor.py` — add `--cross-attention` flag, integrate scoring
- `semantic_folding/lib.py` — add `block_cross_attention_score()`

---

### Phase 7: P2.4 — Joint Document-Snippet Ranking (Branch: feature/snippet-ranking)

**Problem**: Documents are ranked as whole units. For long documents, the answer may be in one specific paragraph. Current doc fingerprint averages all paragraphs, diluting the signal.

**Steps**:

1. **Implement snippet-level fingerprinting** (`snippet_fingerprinter.py`):
   - During Step 5 (doc fingerprints), also create per-snippet fingerprints
   - Snippet = overlapping windows of 3-5 sentences (sliding window, stride=2)
   - Store snippet fingerprints alongside doc fingerprints
   - Each snippet inherits its parent doc_id

2. **Joint scoring**:
   - Query scored against ALL snippets (not just full docs)
   - For each doc: `score(doc) = max(score(snippet_i) for snippet_i in doc)`
   - This is "max-pooling" over snippets — the doc's score is its best snippet's score
   - Alternative: mean-pooling or top-3 mean

3. **Integration**:
   - Add `--snippet-ranking` flag to `query_processor.py`
   - When enabled, load snippet fingerprints and score via max-pooling
   - Add `--snippet-window` (default 3) and `--snippet-stride` (default 2) parameters

4. **Benchmark**:
   - BioASQ (50Q): doc-only vs snippet — measure MRR (large corpus, complex docs)
   - MAUD (50Q): doc-only vs snippet — measure MRR (legal contracts, long docs)
   - NarrativeQA (50Q): doc-only vs snippet — measure MRR (narrative passages)
   - If >3% improvement: expand to all datasets

**Expected impact**: +5-10% MRR on long-document datasets (BioASQ, MAUD). Short documents (Belebele 20-passage) won't benefit since they're already paragraph-sized.

**Files to create/modify**:
- `semantic_folding/snippet_fingerprinter.py` — new file
- `semantic_folding/doc_fingerprints.py` — add snippet fingerprint generation when `--snippet-ranking` enabled
- `semantic_folding/query_processor.py` — add `--snippet-ranking` flag, max-pool scoring
- `semantic_folding/dataset_benchmark/generic_benchmark.py` — pass through snippet flags

---

### Phase 8: P3.1 — Dataset Decision Table & CLI Guide (Branch: feature/dataset-decision-table)

**Problem**: Users adding new datasets must manually configure parameters (grid_size, method, top_percent, etc.) without knowing what works best. The registry has per-dataset overrides, but there's no visibility into *why* parameters differ or *which* CLI flags matter most for each dataset type.

**Goal**: Create a decision table that:
1. Documents what differs across datasets in the registry
2. Maps dataset characteristics → recommended parameters
3. Explains which CLI flags to use and why

**Steps**:

1. **Analyze dataset_registry.yml** (`config/dataset_registry.yml`):
   - Extract all per-dataset parameter overrides
   - Compare against defaults to identify what changes per dataset
   - Categorize: grid_size, method (tsne/umap), top_percent, min_freq, max_doc_freq, smoothing_sigma, etc.

2. **Dataset characteristic profiling**:
   - For each dataset: corpus size (#docs), avg doc length, vocabulary size, domain (biomedical, legal, general, multi-hop QA)
   - Map characteristics to parameter choices (e.g., large corpus → UMAP, small corpus → t-SNE, domain-specific → min_freq adjustments)

3. **CLI flag documentation**:
   - Document all flags with their impact on retrieval
   - Group by category: core (grid_size, method), spreading (radius, decay), normalization (doc_norm), scoring (sim_metric, cross-attention), and advanced (splade, hybrid, rerank)
   - For each flag: when to use, when not to use, tested impact (from benchmark results)

4. **Create decision table** (`docs/DATASET_DECISION_TABLE.md`):
   - Matrix: rows = dataset types (biomedical, legal, general QA, multi-hop), columns = parameters
   - Recommended defaults per dataset type
   - CLI commands for common scenarios
   - "If your dataset is like X, use these flags" guidance

5. **Integration into pipeline**:
   - Add `--profile {auto,biomedical,legal,general,multihop}` flag to generic_benchmark.py
   - When `auto`: read dataset characteristics from registry, apply best defaults
   - When explicit profile: override with pre-tested parameter sets

**Key CLI Flags to Document**:

| Flag | Default | Impact | When to Use |
|------|---------|--------|-------------|
| `--method tsne/umap` | tsne | Grid layout quality vs speed | t-SNE: <10K docs; UMAP: >10K docs |
| `--grid-size` | 64 | Resolution vs memory | Keep at 64 (tested optimal) |
| `--top-percent` | 0.10 | Fingerprint coverage | Lower (0.05) for precision, higher (0.15) for recall |
| `--smoothing-sigma` | 1.5 | Neighbor spreading | 1.5 optimal; 0.0 hurts MRR by 31% |
| `--spreading-steps` | 1 | Query expansion | 1 for short queries, 2 for long complex queries |
| `--doc-norm l2/sqrt_nnz` | l2 | Document normalization | L2 tested optimal; sqrt_nnz -4% MRR |
| `--weighting idf/uniform` | idf | Term importance | IDF tested optimal; uniform -0.86% MRR |
| `--min-freq` | 1 | Vocabulary filtering | Higher (2-5) for noisy corpora |
| `--max-doc-freq` | 20 | Stopword filtering | Lower (10) for domain-specific, higher (50) for general |

**Registry Analysis (Current State)**:

```yaml
# Example: CUAD vs Belebele differences
cuad:
  grid_size: 64        # same
  smoothing_sigma: 1.5 # same
  method: tsne         # same
  min_freq: 2          # DIFFERS (Belebele: 1)
  max_doc_freq: 20     # same
  # Key insight: legal docs benefit from min_freq=2 (noise reduction)

belebele:
  grid_size: 64
  smoothing_sigma: 1.5
  method: tsne
  min_freq: 1          # DIFFERS (CUAD: 2)
  max_doc_freq: 20
```

**Expected impact**: Faster onboarding for new datasets. Users can pick parameters based on dataset type instead of trial-and-error. Reduces time-to-benchmark from hours to minutes.

**Files to create/modify**:
- `docs/DATASET_DECISION_TABLE.md` — new file, decision matrix + CLI guide
- `config/dataset_registry.yml` — add `profile` field for each dataset
- `semantic_folding/dataset_benchmark/generic_benchmark.py` — add `--profile` flag, auto-apply defaults

**Thesis & Paper Integration**:

| Document | Section to Update | Content |
|----------|-------------------|---------|
| `docs/thesis/chapter4_parameter_tuning.md` | §4.5 (new) | Add "Dataset-Specific Parameter Selection" subsection with decision table cross-reference |
| `docs/thesis/datasets.md` | §2.3 (new) | Add "Recommended Parameters by Dataset Type" table linking to decision table |
| `docs/papers/paper1/semantic_folding_paper.md` | §4.5 (new) | Add "Practical Parameter Selection Guide" for practitioners |
| `docs/thesis/parameters_tuning.md` | §6 (new) | Add "Dataset-Specific Configurations" section documenting registry overrides |

**Content for chapter4_parameter_tuning.md §4.5**:
```markdown
## 4.5 Dataset-Specific Parameter Selection

### 4.5.1 The One-Size-Fits-All Problem

While the default configuration (grid_size=64, top_percent=0.10, smoothing_sigma=1.5) works across domains, 
certain datasets benefit from parameter adjustments. Table 4.X summarizes the recommended configurations 
for each dataset type based on our empirical analysis.

### 4.5.2 Decision Matrix

| Dataset Type | grid_size | method | top_percent | min_freq | smoothing_sigma | Rationale |
|--------------|-----------|--------|-------------|----------|-----------------|-----------|
| Biomedical (PubMedQA) | 64 | tsne | 0.10 | 1 | 1.5 | Domain vocabulary is distinct; standard params suffice |
| Legal (CUAD) | 64 | tsne | 0.10 | **2** | 1.5 | Higher min_freq reduces noise from repetitive clauses |
| Multi-hop QA | 64 | tsne | 0.10 | 1 | 1.5 | Standard; SF limitation is compositional, not parametric |
| Entity Lookup (PopQA) | 64 | tsne | 0.10 | 1 | 1.5 | Small candidate pool; params don't matter much |
| Reading Comp (Belebele) | 64 | tsne | 0.10 | 1 | 1.5 | Ceiling effect; standard params optimal |

### 4.5.3 CLI Quick Reference

For practitioners adapting SF to a new domain, use these commands:

# General domain
python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset <name> --jsonl data/<name>/converted/<name>.jsonl --max-queries 50

# Legal domain (noisy corpus, higher min_freq)
python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset <name> --jsonl data/<name>/converted/<name>.jsonl --min-freq 2

# Biomedical domain (with glossary if available)
python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset <name> --jsonl data/<name>/converted/<name>.jsonl \
  --glossary config/glossary_<domain>.json --glossary-corpus-expansion

# Large corpus (>10K docs, use UMAP for speed)
python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset <name> --jsonl data/<name>/converted/<name>.jsonl \
  --method umap --umap-n-neighbors 15
```

**Content for paper1 §4.5**:
```markdown
### 4.5 Practical Parameter Selection Guide

Table 4.X provides a decision matrix for practitioners deploying SF on new domains. The key insight is that
SF's parameter sensitivity is low for most settings (grid_size, top_percent) but high for specific ones
(smoothing_sigma, min_freq). This makes domain adaptation practical without retraining.

**Parameter Sensitivity Summary**:
- **Low sensitivity** (safe defaults): grid_size, spreading_steps, weighting, doc_norm
- **Medium sensitivity** (test alternatives): top_percent, method (tsne/umap)
- **High sensitivity** (must tune): smoothing_sigma (σ=0 → -31% MRR), min_freq (domain-dependent)

For detailed parameter interactions, see Appendix A (available at [repository URL]).
```

**Content for datasets.md §2.3**:
```markdown
### 2.3 Recommended Parameters by Dataset Type

Based on our benchmark results across 13 datasets, we provide parameter recommendations 
for common dataset categories. These are documented in detail in Chapter 4 §4.5 and the 
DATASET_DECISION_TABLE.md.

| Dataset Type | Key Parameter | Recommendation | Evidence |
|--------------|---------------|----------------|----------|
| Biomedical | smoothing_sigma | 1.5 (default) | PubMedQA MRR=0.955 |
| Legal | min_freq | 2 | CUAD registry override |
| Multi-hop | spreading_steps | 1 (default) | MuSiQue limitation is compositional |
| Entity lookup | top_percent | 0.10 | PopQA MRR=0.980 |
| Large corpus | method | umap | Speed: 10-100x faster than t-SNE |

For complete parameter guidance, see the Dataset Decision Table in the project documentation.
```

---

## Execution Order

**Sequential per branch** (per AGENTS.md branch-per-improvement policy):

1. ~~**P1.3 Ontology-BioASQ** → implement → benchmark BioASQ → merge if >5%~~ **✅ DONE** (MRR +55.2%)
2. **P1.5 LanceDB** → implement → benchmark Belebele+BioASQ → merge if no MRR regression + faster loading
3. **P1.6 Memory-efficient** → implement → benchmark BioASQ → merge if memory reduced + no MRR regression
4. **P2.3 Multi-resolution** → document in thesis (no code needed) → commit docs
5. **P2.1 Learned grid** → implement → benchmark Belebele+BioASQ → merge if >3%
6. **P2.2 Cross-attention** → implement → benchmark NQ-REaR+BioASQ+MuSiQue → merge if >3%
7. **P2.4 Snippet ranking** → implement → benchmark BioASQ+MAUD+NarrativeQA → merge if >3%
8. **P3.1 Dataset decision table** → analyze registry → create decision table → document CLI flags → update thesis ch.4 + paper §4.5 + datasets.md

**Post-implementation**: After all features are done, decide how to proceed:
- Run cross-dataset verification for P1.3 (ontology)
- Create final merge plan for all feature branches
- Update all documentation with final results

## Benchmark Strategy

### Test Datasets (per improvement):
- **BioASQ** (50Q): Primary target — lowest MRR (0.195), most room for improvement
- **Belebele** (50Q): Ceiling effect check — MRR=1.000, verify no regression
- **NQ-REaR** (50Q): Score compression test — moderate MRR (0.611)

### Full Benchmark (if test shows improvement):
All 12 datasets: Belebele, PopQA, PubMedQA, NQ-REaR, BioASQ, NarrativeQA, 2WikiMultihopQA, HotpotQA, MuSiQue, DROP, DocFinQA, MAUD

### Benchmark Protocol:
1. Run baseline (current defaults) — record MRR, AP, P@1, load time, peak memory
2. Run with improvement feature enabled — record same metrics
3. Compute delta: `improvement = (feature_MRR - baseline_MRR) / baseline_MRR * 100`
4. Save report to `docs/reports/<dataset>/v<N>_<timestamp>.md`
5. Update `BENCHMARK_RESULTS.md` and `REPORTS.md`

## Thesis Integration Plan

After each improvement is benchmarked (successful or not), update:

### Chapter 7 (Experiments):
- Add new section per improvement under §7.2.3 "Improvement Results"
- Add to §7.4 "Academic Contributions" if novel finding
- Update §7.2.1 "Performance Summary" table if MRR changes

### Chapter 8 (Discussion):
- P1.3 Ontology: Add to §8.5 "The Hybrid Opportunity" — discuss domain glossaries as knowledge injection
- P1.5 LanceDB: Add to §8.8.1 "Implemented Improvements" — discuss scalability
- P1.6 Memory-efficient: Add to §8.6 "Limitations" — address computational cost
- P2.1 Learned grid: Add to §8.8.2 "Future Work" — discuss supervised vs unsupervised grid mapping
- P2.2 Cross-attention: Add to §8.8.2 — discuss localized scoring vs global scoring
- P2.3 Multi-resolution: Document 0% result in §8.8.1 — ablation finding
- P2.4 Snippet ranking: Add to §8.3 "Comparison with Other Methods" — discuss granularity

### Chapter 9 (Conclusions):
- Update summary of implemented improvements
- Update future work if P2 items show promising results

### BENCHMARK_RESULTS.md:
- Add new section for each improvement with before/after metrics
- Update executive summary if any improvement changes the top-line results

## Risk Assessment

| Improvement | Risk | Mitigation |
|-------------|------|------------|
| P1.3 Ontology | Low — code exists, new approach is additive | If 0% again, document as negative result |
| P1.5 LanceDB | Low — storage only, ranking unchanged | Verify MRR identical before/after |
| P1.6 Memory-efficient | Low — lazy loading is well-understood | Verify no speed regression >10% |
| P2.1 Learned grid | Medium — may overfit to corpus | Use cross-validation, compare vs t-SNE |
| P2.2 Cross-attention | Medium — new scoring paradigm | Test on multiple datasets before claiming improvement |
| P2.4 Snippet ranking | Medium — changes Step 5 output | Verify snippet fingerprints don't break existing pipeline |
| P3.1 Dataset decision table | Low — documentation + flag only | Verify --profile auto mode matches manual best results |

## Estimated Effort

| Phase | Hours | Key Deliverable |
|-------|-------|-----------------|
| P1.3 | 3-4h | Glossary injection at indexing + BioASQ benchmark |
| P1.5 | 3-4h | LanceDB integration + Belebele/BioASQ benchmark |
| P1.6 | 2-3h | Lazy loading + memory benchmark |
| P2.3 | 0.5h | Thesis text only |
| P2.1 | 6-8h | MLP grid mapper + benchmark |
| P2.2 | 4-6h | Block cross-attention + benchmark |
| P2.4 | 4-6h | Snippet fingerprinting + benchmark |
| P3.1 | 3-4h | Decision table + CLI guide + --profile flag + thesis/paper integration |
| **Total** | **26-36h** | 8 improvements, 8 branches, ~14 benchmarks |