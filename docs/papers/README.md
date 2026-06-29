# Future Work - Semantic Folding Paper

This file tracks future work items identified in the Semantic Folding paper (`option_b/semantic_folding_paper.md`) for implementation in next phases.

## High-Priority Future Work

### 1. Two-Stage Neuro-Lexical Pipeline
**Status**: Proposed (§8.5)  
**Priority**: High  
**Description**: Solve BioASQ scaling limitation by using BM25 to retrieve top-100, then SF re-ranks within constrained pool.  
**Preliminary Results**: MRR 0.288 (SF-only) → 0.441 (BM25+SF re-rank) on BioASQ  
**Next Steps**:
- [ ] Implement BM25 + SF re-ranking pipeline
- [ ] Evaluate on BioASQ and NQ-REaR (large corpus datasets)
- [ ] Compare against SPLADE-only baseline
- [ ] Test optimal pool size (top-50 vs top-100 vs top-200)

### 2. Polarity-Aware Semantic Folding (Negation Handling)
**Status**: Proposed (§9.2)  
**Priority**: Medium  
**Description**: Use XOR operations to invert grid activations for negated phrases, solving negation blindness.  
**Preliminary Results**: MRR 0.930 → 0.947 on Belebele negation subset (pilot with 3-bit polarity code)  
**Next Steps**:
- [ ] Implement polarity detection in phrase extraction (dependency parsing for "not" + verb)
- [ ] Design inversion mask (XOR with negation bitmask)
- [ ] Evaluate on negation-heavy datasets (Belebele, HotpotQA)
- [ ] Systematic evaluation across datasets with varied negation patterns

### 3. The Interference Wall Theory
**Status**: Proposed (§8.5)  
**Priority**: Medium (theoretical)  
**Description**: Formalize why dense retrieval saturates in specialized domains (Orthogonality Constraint + Semantic Interference).  
**Next Steps**:
- [ ] Write theoretical section connecting Orthogonality Constraint (Zahn et al., 2026) to SF's advantage
- [ ] Empirical validation: compare SF vs DPR on increasingly specialized domains
- [ ] Quantify interference on SciFact (SF 0.755 vs DPR 0.675)

## Medium-Priority Future Work

### 4. Full Bootstrap Confidence Intervals
**Status**: In Progress (§5.2 has partial CIs)  
**Priority**: Medium  
**Description**: Compute 95% CIs for all 9×6 experimental conditions.  
**Next Steps**:
- [ ] Run bootstrap resampling (1000 iterations) for all datasets × methods
- [ ] Report full CI table in paper
- [ ] Verify statistical significance of key findings (H2 falsification)

### 5. SciFact with SF+SPLADE
**Status**: Not Started  
**Priority**: Medium  
**Description**: Add SF+SPLADE results for SciFact to complete the 9-dataset matrix.  
**Next Steps**:
- [ ] Run SF+SPLADE on SciFact (300 queries)
- [ ] Compare SF-only (0.755) vs SF+SPLADE vs DPR (0.675)
- [ ] Update paper Table 1 with SciFact SF+SPLADE MRR

### 6. Graded-Relevance NDCG
**Status**: Not Started  
**Priority**: Low  
**Description**: Use graded relevance (if available) to compute NDCG@K for more discriminative evaluation.  
**Next Steps**:
- [ ] Check if datasets have graded relevance (PubMedQA, BioASQ might)
- [ ] Compute NDCG@5, NDCG@10 for datasets with graded labels
- [ ] Compare against MRR (primary metric)

## Long-Term Future Work

### 7. Multilingual SF via Cross-Lingual UMAP
**Status**: Not Started  
**Priority**: Low  
**Description**: Extend SF to multilingual retrieval via aligned UMAP spaces.  
**Next Steps**:
- [ ] Investigate cross-lingual embeddings (LASER, XLM-R)
- [ ] Align UMAP spaces across languages
- [ ] Test on Belebele (multilingual reading comprehension)

### 8. LLM-Enhanced Semantic Space
**Status**: Not Started  
**Priority**: Medium  
**Description**: Use LLM to extract semantic concepts from contexts for richer term-context matrix.  
**Next Steps**:
- [ ] Prompt LLM to extract key concepts from corpus paragraphs
- [ ] Build term-concept matrix instead of term-context
- [ ] Compare against standard term-context matrix

### 9. End-to-End Differentiable Grid
**Status**: Not Started  
**Priority**: Low  
**Description**: Learn grid positions via Gumbel-Softmax for gradient-based optimization.  
**Next Steps**:
- [ ] Implement Gumbel-Softmax relaxation for grid positions
- [ ] Train via retrieval loss (ranking loss on query-document pairs)
- [ ] Compare against UMAP/t-SNE initialization

### 10. Adaptive Grid Sizing
**Status**: Not Started  
**Priority**: Low  
**Description**: Develop guidelines for scaling grid size with corpus size and task type.  
**Next Steps**:
- [ ] Empirical study: grid_size vs corpus_size vs MRR
- [ ] Derive formula: g = f(D, ρ_target, task_type)
- [ ] Validate on datasets with varied corpus sizes

## Completed Future Work

✅ **SPLADE-only baseline added** (§5.2, §7.2) - DONE  
✅ **H2 falsification reported** (§5.2, §7.2, §8.1) - DONE  
✅ **UMAP default documented** (§3.4, §5.3) - DONE  
✅ **95% CIs added** (§5.2) - DONE (partial, full CIs pending)  
✅ **Sanati et al. (2023) integrated** (§2.2, §3.6) - DONE  
✅ **Hole & Ahmad (2021) integrated** (§1.2, §2.3.3) - DONE

---

**Last Updated**: January 2026  
**Paper Version**: option_b/semantic_folding_paper.md (with revisions)