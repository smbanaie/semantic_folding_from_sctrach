# Chapter 9: Conclusions and Future Work

## 9.1 Summary of Contributions

This thesis has presented Semantic Folding (SF), an unsupervised retrieval architecture that represents text as sparse binary fingerprints over a 2D semantic grid. Through a comprehensive 9-dataset benchmark and systematic evaluation of 7 feature variants, we established the conditions under which SF succeeds, fails, and can be improved.

### 9.1.1 Theoretical Contributions

1. **The Complementarity Principle**: Features that duplicate existing SF signals cannot improve performance. Only genuinely non-overlapping signals (SPLADE's learned sparse expansion) provide consistent gains. Validated across 7 feature variants.

2. **The α-Sensitivity Framework**: The SF+SPLADE hybrid weight α ∈ [0,1] produces monotonic degradation on most datasets — as SF weight increases, MRR decreases. This falsifies the complementarity hypothesis (H2) and reveals that SF and SPLADE signals are correlated, not complementary.

3. **Orthogonality Constraint Caveat**: The Orthogonality Constraint (Zahn et al., 2026) applies to independent random SDRs, but SF fingerprints are spatially correlated by design (Gaussian smoothing σ=1.5, Morton encoding, IDF aggregation). Empirical pairwise cosine distributions have higher mean and variance than the random-SDR prediction.

### 9.1.2 Methodological Contributions

1. **Complete Unsupervised Pipeline**: A six-stage architecture converting raw text to ranked retrieval results without training data. The SF component is fully unsupervised; the hybrid uses off-the-shelf pre-trained SPLADE.

2. **Systematic Parameter Tuning**: Comprehensive analysis of grid size, spreading steps, top percent, IDF weighting, Gaussian smoothing, Morton encoding, and document normalization with theoretical and empirical justification (Chapter 4).

3. **Nine-Dataset Benchmark**: Evaluation across 9 datasets spanning 6 task types, establishing SF's task-type dependency with statistical rigor (Chapter 7).

4. **Systematic Negative Results**: Documentation of 7 failed improvement attempts (cross-attention, learned grid, ontology expansion, etc.) to prevent future dead ends.

### 9.1.3 Empirical Contributions

The key empirical findings are:

1. **SPLADE-only outperforms SF-only on 5/9 datasets** (Chapter 7, Table 7.2). The SF+SPLADE hybrid is beneficial on only 2/9 datasets (2WikiMultihopQA +8.5%, PubMedQA +1.7%).

2. **SF+SPLADE achieves MRR=0.782 on MuSiQue**, outperforming BM25 (0.482) by +62.2% — the largest relative gain in the matrix — though it trails HippoRAG2's dense baseline (0.865, −9.6%) and SPLADE-only alone (0.876) is the strongest single system. This remains the best unsupervised-sparse result on this multi-hop QA dataset.

3. **SF matches or exceeds DPR on three datasets** (MuSiQue, HotpotQA, PopQA) without any training data.

4. **UMAP matches or beats t-SNE on 7/9 datasets** (average +1.3% MRR) with 10× faster indexing (Chapter 7, §7.3.4).

---

## 9.2 Key Findings

*For the complete cross-dataset performance tables, see Chapter 7, Table 7.1 and Table 7.2.*

### 9.2.1 When SF Excels

SF's success follows a clear pattern:

| Condition | SF Performance | Example |
|-----------|:--------------:|---------|
| High vocabulary mismatch + small candidate pool | **Excellent** (MRR > 0.90) | MuSiQue |
| Low vocabulary mismatch + small pool | **Competitive** (MRR 0.85–0.90) | HotpotQA, Belebele |
| Large candidate pool + complex queries | **Poor** (MRR < 0.60) | BioASQ, NQ-REaR |

**Predictive rule**: SF excels when (a) query and document vocabularies differ substantially, and (b) the candidate pool is small enough to avoid score compression (< 100 docs).

### 9.2.2 When SF Struggles

1. **Compositional gap**: SF cannot compose facts across passages. Performance degrades on multi-hop tasks requiring reasoning across documents.

2. **Score compression**: On large corpora (BioASQ: 1075 docs), SF produces near-uniform scores. Sparse dot-product lacks dynamic range for large candidate pools.

3. **Negation blindness**: SF treats "not considered" identically to "considered." Predicate-level scope analysis is needed.

### 9.2.3 The Sparse-Dense Trade-off

**Table 9.1: Comparison of Retrieval Paradigms**

| Aspect | SF (Sparse) | BM25 (Lexical) | DPR (Dense) |
|--------|-------------|----------------|-------------|
| Training data | None | None | 50K+ pairs |
| Memory per document | 512 bytes | ~1KB | 3KB |
| Interpretability | Grid visualization | Term frequency | Black box |
| Best dataset MRR | 0.782 (MuSiQue) | 0.995 (Belebele) | 0.863 (NQ) |
| Worst dataset MRR | 0.288 (BioASQ) | 0.482 (MuSiQue) | — |

SF occupies a unique quadrant: unsupervised semantic matching + interpretability + memory efficiency. No other method provides all three simultaneously.

---

## 9.3 What Works and What Doesn't

### 9.3.1 Verified Improvements

**Table 9.2: Verified Improvements (Part of Default Pipeline)**

| Improvement | Impact | Status |
|-------------|--------|--------|
| SF+SPLADE hybrid (α=0.3) | Best config for 2/9 datasets | ✓ Verified |
| UMAP dimensionality reduction | Matches or beats t-SNE on 7/9 datasets | ✓ Verified |
| L2 doc normalization | +4.0% MRR | ✓ Verified |
| FAISS OOV expansion | 400× speedup | ✓ Verified |
| Batch query processing | ~25× speedup | ✓ Verified |

### 9.3.2 Tested and Failed

| Attempt | Impact | Status |
|---------|:------:|--------|
| Cross-attention | −87% (SF-Only) | ✗ Failed |
| Learned grid | −79% (SF-Only) | ✗ Failed |
| Snippet ranking | 0% (identical) | ✗ No effect |
| Adaptive spreading | 0% (identical) | ✗ No effect |
| MeSH ontology | 0% to −3.8% | ✗ No benefit |

**The only verified improvement to SF is SPLADE.** All other tested features either degrade performance or have zero effect.

---

## 9.4 Implications for Retrieval Research

### 9.4.1 The Value of Unsupervised Methods

Our results demonstrate that unsupervised semantic matching can achieve competitive performance on specific task types. SF provides:

1. **Zero-shot domain adaptation**: No labeled data required
2. **Interpretability**: Grid visualizations explain retrieval decisions
3. **Memory efficiency**: 512 bytes per document (6× smaller than DPR)

These properties make SF valuable for scenarios where training data is unavailable, interpretability is required, or resource constraints prevent dense retrieval.

### 9.4.2 The Vocabulary Mismatch Problem

SF's strong performance on MuSiQue (+62% vs BM25) provides evidence that vocabulary mismatch remains a significant challenge for lexical retrieval. However, the broader 9-dataset pattern shows that vocabulary mismatch is only one component of retrieval quality. Lexical precision, entity matching, and score discrimination are equally important.

### 9.4.3 The Complementarity Principle

The Phase 2c/3/4 results establish a general principle: **improvements must add genuinely non-overlapping signal**. This explains why SPLADE works (learned expansion, distinct from grid proximity) while cross-attention, snippet ranking, and adaptive spreading fail (they duplicate existing SF signals).

---

## 9.5 Limitations

### 9.5.1 Current Limitations

1. **Compositional gap**: SF cannot compose facts across passages
2. **Score compression**: Sparse dot-product lacks dynamic range for large corpora
3. **Negation blindness**: No predicate-level scope analysis
4. **Computational cost**: Indexing takes ~5 minutes for 100 passages (with UMAP)
5. **Grid size sensitivity**: 64×64 grid is optimal for 20-doc corpora; scaling to larger corpora requires re-tuning

### 9.5.2 Methodological Limitations

1. **Binary relevance**: Ground truth uses binary relevance (supporting passage or not)
2. **Dimensionality reduction stochasticity**: t-SNE and UMAP results vary with random seed (±0.015 MRR)
3. **Fixed candidate pools**: Benchmark evaluates retrieval within curated pools (20 passages/query), not open-domain retrieval

---

## 9.6 Future Work

### 9.6.1 High-Priority Directions

**Table 9.3: Future Work Priorities**

| Priority | Direction | Impact | Feasibility |
|:--------:|-----------|:------:|:-----------:|
| 1 | Compositional retrieval (graph fusion, LLM-guided decomposition) | High | Medium |
| 2 | Learned grid with UMAP pretraining | Medium | High |
| 3 | Large-corpus scaling guidelines | Medium | High |
| 4 | Negation-aware processing | Low-Medium | Medium |

### 9.6.2 Open Questions

1. **Why does SF succeed on MuSiQue but not on 2Wiki/HotpotQA, despite all three being multi-hop?** The candidate-pool structure and entity distinctiveness likely interact differently across datasets.

2. **What is the upper bound of SF+SPLADE performance?** Adding a cross-encoder re-ranker could push MRR higher but would require training data.

3. **Is the Complementarity Principle universal or architecture-specific?** — *Partially answered (journal extension)*: the four-pair matrix (SF+SPLADE, SF+DPR, BM25+SPLADE, BM25+DPR) shows the operator-selection effect generalizes across retriever pairs, with signal-B score geometry (SPLADE vs DPR) setting the winning family. Testing on non-QA retrieval tasks and additional learned sparse checkpoints beyond SPLADE-v3 remains open.

4. **How do SF scores behave at corpus sizes between 1K and 1M documents?** The mathematical derivation (Chapter 7, §7.3.3) predicts O(√N) score range scaling; the two-pairing padded-pool sweep now empirically covers N up to 494 with flat operator MRR throughout, while full-corpus SciFact (5,183) shows collapse. The 10K–1M range remains unverified.

5. **Which magnitude-preserving operator should practitioners default to? (new)** CombSUM leads at n=50 everywhere it matters, CombMNZ wins on large pools (NQ-REaR), yet almost no pairwise difference survives Holm correction. Whether larger samples reveal reliable separation — or whether operators within the magnitude family are practically interchangeable — is open and directly relevant to deployment guidance.

---

## 9.7 Conclusion

Semantic Folding provides unsupervised semantic matching that is competitive with supervised methods on specific task types. The key findings are:

1. **SPLADE-only outperforms SF-only on 5/9 datasets**. SF's contribution is positive on only 2/9 datasets.

2. **The complementarity hypothesis (H2) is falsified**. SF and SPLADE signals are correlated, not complementary.

3. **The only verified improvement to SF is SPLADE**. All other feature variants either degrade or have zero effect.

4. **SF excels when vocabulary mismatch is high and candidate pools are small**. These conditions predict SF's performance across datasets.

SF occupies a unique position in the retrieval landscape: the only method providing unsupervised semantic matching, interpretable grid visualizations, and memory-efficient storage simultaneously. For scenarios where training data is unavailable or interpretability is required, SF+SPLADE is the strongest available approach.

The negative results documented in this thesis — 7 failed improvement attempts — are as valuable as the positive findings. They establish that **SF's architecture is well-optimized**, and that future improvements must add genuinely non-overlapping signal rather than duplicating existing capabilities.

**Journal-extension findings** strengthen and refine these conclusions:

1. **Fusion-operator selection is real but geometry-conditioned, not topology-determined.** The complete seven-operator matrix shows magnitude-preserving operators (CombSUM/CombMNZ) leading on every multi-hop/factoid dataset at n=50 — an ordering that replicates under a second SPLADE checkpoint (v3) — while single-hop rows are operator-invariant. After Holm correction, however, individual operator pairs are almost never separable at n=50; the claim rests on replicated orderings, not single tests.
2. **Rank vs magnitude information is causally separable on real retrieval outputs**: rank-preserving magnitude transforms leave RRF bit-identical (τ=1.000); rank destruction collapses it maximally; score-space operators respond to magnitude alone.
3. **Pool size does not separate operators**: padded-pool sweeps in two pairings show flat MRR from N=20 to N=494, with gap structure tracking signal-A score geometry — refining the deep-pool collapse account into a two-regime picture (small/medium pools: operator choice matters; full corpus: operator choice vanishes).

---

## 9.8 Practitioner's Decision Guide

Based on the 11-dataset benchmark results, we provide the following decision rule for retrieval system selection:

**Table 9.4: Decision Guide for Retrieval Method Selection**

| Condition | Recommended Method | MRR (Expected) | Rationale |
|-----------|-------------------|:----------------:|-----------|
| High vocabulary mismatch + small pool (< 100 docs) | **SF+SPLADE** | 0.85–0.93 | Semantic matching catches synonyms |
| High vocabulary mismatch + large pool (> 1000 docs) | **BM25 + SPLADE re-ranking** | 0.44–0.68 | Avoid SF score compression |
| Low vocabulary mismatch + entity lookup | **BM25** | 0.95–1.00 | Exact match suffices |
| Multi-hop reasoning (2+ hops) | **SPLADE-only** or **DPR** | 0.80–0.99 | SF cannot compose facts |
| No training data available | **SF-only** | 0.20–0.93 | Zero-shot deployment |
| GPU available + training data exists | **SPLADE or DPR** | 0.68–0.99 | Peak performance |
| Interpretability required | **SF+SPLADE** | 0.85–0.93 | Grid visualization explains matches |
| Hybrid reranking over heterogeneous-score signals (new) | **SF+SPLADE with CombSUM** | 0.89–0.99 | Magnitude-preserving fusion leads at n=50 on compositional/factoid tasks; ordering checkpoint-robust |
| Hybrid reranking over uniform-scale signals, e.g. DPR pair (new) | **α-blend linear** | — | Scale-invariant operators waste magnitude-free signals; explicit α controls the trade-off |

**Decision tree**:
1. Is training data available? → No: Use SF-only or SF+SPLADE (off-the-shelf SPLADE)
2. Is the candidate pool large (> 1000 docs)? → Yes: Use BM25 baseline + SPLADE re-ranking
3. Is the task multi-hop reasoning? → Yes: Use SPLADE-only or dense method
4. Is vocabulary mismatch high? → Yes: Use SF+SPLADE; No: Use BM25
5. (new) Fusing two signals for reranking? → Check signal-B score geometry: heterogeneous/log-pooled magnitudes → CombSUM-family; L2-normalized/uniform → α-blend linear; and note that within the winning family, individual operator choice is second-order (orderings replicate, pairwise gaps rarely family-wise significant)

---

## 9.9 Scalability Warnings

### 9.9.1 Score Compression Mechanism

SF's sparse dot-product scoring suffers from **score compression** on large corpora. The mathematical derivation is as follows:

For a corpus of N documents, the expected dot-product score between query q and document d is:

E[s] = ‖f_q‖₁ × ρ

where ρ ≈ 0.10 is the fingerprint density. The standard deviation is:

σ[s] ≈ √ (‖f_q‖₁ × ρ × (1-ρ))

For a 64×64 grid with 10% density, this gives E[s] ≈ 41 and σ[s] ≈ 6.07.

**The dynamic range problem**: The maximum expected score for N documents approaches E[s] + z × σ[s], where z scales with N. For N = 1,075 (BioASQ), z ≈ 3.5 (extreme value theory), giving a maximum of ~62. The dynamic range (62 - 41 = 21 units) is comparable to small-pool settings, but the ratio of relevant to irrelevant documents degrades from 1:19 (20-doc pool) to 1:1074 (1075-doc pool).

**Practical consequence**: When N > 1000, pre-filter with BM25 or use SF as a re-ranker on a smaller candidate set (top-100 BM25 results).

### 9.9.2 Grid Size Scaling

The 64×64 grid is optimal for 20–200 document corpora. For larger corpora:

| Corpus Size | Recommended Grid | Expected MRR (Belebele-scale) |
|-------------|-----------------|:------------------------------:|
| 20–200 | 64×64 | 0.88–0.93 |
| 200–1000 | 128×128 | 0.85–0.90 |
| 1000–5000 | 128×128 or 256×256 | 0.70–0.85 |
| >5000 | 256×256 + BM25 prefiltering | 0.60–0.75 |

**Warning**: These are extrapolations. The largest corpus evaluated in this thesis is BioASQ (1075 docs, MRR=0.288). Scaling beyond 5000 documents requires further empirical validation.

---

## References

- Formal, T., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *Proceedings of SIGIR 2021*.
- Furnas, G. W., et al. (1987). The vocabulary problem in human-system communication. *Communications of the ACM*, 30(11), 964–971.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP 2020*.
- Trivedi, H., et al. (2022). MuSiQue: Multi-hop Synthetic Question Answering. *Proceedings of NAACL 2022*.
- Zahn, O., et al. (2026). Attention Is Not Retention: The Orthogonality Constraint in Infinite-Context Architectures. arXiv:2601.15313.
