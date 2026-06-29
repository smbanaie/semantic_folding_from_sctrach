# Chapter 2: Literature Review

## 2.1 Introduction

This chapter reviews the theoretical foundations and related work informing Semantic Folding. We cover: (1) Sparse Distributed Representations and the Orthogonality Constraint, (2) Semantic Folding and Sparse Distributed Representations, (3) Dimensionality Reduction for Semantic Folding (t-SNE, UMAP), (4) Morton Encoding for Spatial Locality, (5) Similarity Metrics for Sparse Representations, and (6) The Current State of Sparse Retrieval for Closed-Domain Question Answering (2023–2025).

## 2.2 Sparse Distributed Representations and the Orthogonality Constraint

### 2.2.1 Sparse Distributed Memory

Sparse Distributed Memory (SDM) was proposed by Kanerva (1988) as a theoretical model of biological memory. In SDM, each memory is a sparse binary vector (Sparse Distributed Representation, SDR) where only a small fraction of bits are active (typically 1–10%). The key property is **high-dimensional orthogonality**: for random sparse binary vectors with density ρ, the expected cosine similarity is ρ with variance ρ(1-ρ)/d. For d=4096 and ρ=0.10, 99.9% of random pairs have cosine similarity < 0.15.

However, this formula assumes **independent random bits**. In Semantic Folding, fingerprints are spatially correlated by design (Gaussian smoothing, Morton encoding, IDF aggregation), which increases both the mean and variance of pairwise cosine similarity (see Chapter 5, §5.2.3 for details).

### 2.2.2 The Orthogonality Constraint (Zahn et al., 2026)

Recent theoretical work (Zahn et al., 2026) identifies a fundamental limitation: reliable memory requires orthogonal keys, but semantic embeddings cannot be orthogonal because training clusters similar concepts together. This creates **Semantic Interference** — memory collapse when storing many related facts.

**Formal Statement**: Let k_i, k_j ∈ ℝ^d be key vectors for facts i and j. For reliable retrieval:

cos(k_i, k_j) ≈ 0  ∀i≠j

However, training on semantically related facts forces:

cos(k_i, k_j) > 0  when sem(i, j) > θ

This creates interference that limits the number of facts that can be reliably stored.

**Caveat**: The Orthogonality Constraint applies strictly to independent random SDRs. SF fingerprints are spatially correlated, which reduces but does not eliminate the constraint's relevance. Future work should report empirical pairwise cosine distributions before invoking this framework.

## 2.3 Semantic Folding and Sparse Distributed Representations

### 2.3.1 The Semantic Folding Pipeline

Semantic Folding (Dengel, 2015) applies SDM principles to text retrieval. Text is converted to sparse binary fingerprints over a 2D semantic grid. The grid is constructed by dimensionality reduction (t-SNE or UMAP) on the term-context co-occurrence matrix, mapping semantically similar phrases to nearby grid cells.

### 2.3.2 Related Work on Sparse Representations for Retrieval

Several approaches use sparse representations for retrieval:

| Method | Sparsity Mechanism | Training | Performance |
|--------|-------------------|----------|-------------|
| BM25 | Term frequency | None | Strong baseline |
| SPLADE | Learned sparse expansion | Supervised | SOTA sparse method |
| SF (this work) | Distributional + grid | Unsupervised | Competitive, zero-shot |

## 2.4 Dimensionality Reduction for Semantic Folding

### 2.4.1 t-SNE (van der Maaten & Hinton, 2008)

t-SNE minimizes KL divergence between pairwise probability distributions in high and low dimensions. It is excellent at preserving local structure but computationally expensive (O(N²) per iteration).

### 2.4.2 UMAP (McInnes et al., 2018)

UMAP uses a cross-entropy objective that preserves both local and global structure. It is 10× faster than t-SNE and matches or beats it on 7/9 datasets in our benchmark (Chapter 7, §7.3.4).

**Mathematical advantage**: UMAP's cross-entropy has both attractive and repulsive terms, while t-SNE's KL divergence only has an attractive term. This gives UMAP better global structure preservation.

## 2.5 Morton Encoding for Spatial Locality

Morton encoding (Z-order curve, Morton, 1966) interleaves the bits of x and y coordinates to preserve 2D spatial locality in a 1D bitstring. This ensures that nearby grid cells have similar binary representations, improving compression and cache efficiency.

## 2.6 The Current State of Sparse Retrieval for Closed-Domain Question Answering (2023–2025)

### 2.6.1 SPLADE Advances

SPLADE (Formal et al., 2021) remains the dominant learned sparse retrieval method. Recent improvements include:

- **Mistral-SPLADE** (arXiv:2408.11119): Decoder-only LLMs outperform encoder-only variants
- **Two-Step SPLADE** (arXiv:2404.13357): 30× speedup for in-domain with minimal quality loss
- **SPLATE** (arXiv:2404.13950): ColBERTv2 + SPLADE adapter for CPU-efficient late interaction

### 2.6.2 Hybrid Retrieval Systems

Recent work confirms hybrid sparse+dense pipelines outperform single-method baselines:

- **RRF Fusion** (arXiv:2604.13728): Sparse+dense reciprocal rank fusion outperforms sparse-only by 14.9%
- **HiRAG** (arXiv:2408.11875): Hierarchical sparse+dense for multi-hop QA
- **GeAR** (arXiv:2412.18431): Graph expansion + sparse retriever, >10% improvement on MuSiQue

### 2.6.3 Semantic Folding in Context

SF's unique contribution is **unsupervised semantic matching**. While SPLADE requires training data, SF adapts instantly to new domains. This is valuable for closed-domain QA where labeled data may not exist (e.g., emerging biomedical subfields).

## 2.7 Summary

The literature establishes that:

1. **Sparse methods** (BM25, SPLADE, SF) provide interpretability and memory efficiency
2. **Dense methods** (DPR, ColBERT) provide compositional reasoning but require training
3. **Hybrid approaches** combine the best of both paradigms
4. **SF's unique position**: Unsupervised semantic matching with grid-based interpretability

The Orthogonality Constraint (Zahn et al., 2026) provides a theoretical framework for understanding sparse vs dense trade-offs, but SF's spatially correlated fingerprints require empirical validation before applying this framework.

---

## References

- Dengel, A. (2015). Semantic Folding. *Technical Report*.
- Formal, T., et al. (2021). SPLADE. *SIGIR 2021*.
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- McInnes, L., et al. (2018). UMAP. *arXiv:1802.03426*.
- Morton, G. M. (1966). *A Computer Oriented Geodetic Data Base*. IBM.
- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.
- Zahn, O., et al. (2026). Attention Is Not Retention. *arXiv:2601.15313*.
