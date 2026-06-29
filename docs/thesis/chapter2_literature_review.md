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

### 2.2.3 Information-Theoretic Analysis of SDR Sparsification

The theoretical justification for sparse representations extends beyond memory capacity to **information preservation**. Recent work by Sanati et al. (2023) provides an information-theoretic analysis of the sparsification process in the HTM Spatial Pooler (SP) algorithm, proving that sparser representations improve estimation performance.

#### Information Bottleneck Framework

The **Information Bottleneck (IB)** method (Tishby et al., 2000) formulates representation learning as an optimization problem: find a compressed representation Y of input X that preserves relevant information about target Z. The objective is:

```
L_IB = I(X; Y) - β·I(Y; Z)
```

where I(·;·) is mutual information, and β controls the trade-off between compression (first term) and information preservation (second term).

Sanati et al. (2023) introduce a **modified IB upper bound** that directly compares X and Z (skipping Y), providing a tighter bound for analyzing sparsification:

```
L_MOD-IB = I(X; Y) - β₂·I(X; Z)
```

This modified bound is proven to be an upper bound of the standard IB: `L_MOD-IB ≥ L_IB` (Sanati et al., 2023, §3.1).

#### Cramér-Rao Bound and Sparsity

The key result from Sanati et al. (2023) is a mathematical proof that **increased sparsity improves estimation performance**. Using the **Cramér-Rao Lower Bound (CRLB)** and **Fisher Information Matrix (FIM)**, they show:

> *Under the Cauchy distribution assumption, the CRLB of the estimation error decreases as the output sparsity increases* (Sanati et al., 2023, §3.2).

The FIM measures the amount of information that data provides about an unknown parameter. A higher FIM (lower CRLB) means more accurate parameter estimation. Sanati et al. prove that sparsification increases the FIM diagonal entries, leading to better estimation.

#### Noise Robustness

The information-theoretic analysis also explains the noise robustness of sparse representations. Sanati et al. (2023) show that the HTM-SP algorithm with learning is resistant to up to 40% input noise without discernible output change. This robustness arises from the IB optimization: sparsification acts as a natural denoising process that discards irrelevant information while preserving task-relevant signals.

#### Relevance to Semantic Folding

The information-theoretic results support Semantic Folding's design choices:

1. **Sparsity level**: SF uses 10% density (top_percent=0.10), which is higher than the ~2% typical in HTM. Sanati et al.'s proof suggests that even 10% provides benefits over dense representations, though the optimal sparsity likely depends on the task.

2. **Noise robustness**: SF's Gaussian smoothing (σ=1.5) provides noise robustness analogous to the SP algorithm's learned connectivity. Both methods use spatial smoothing to handle noisy or incomplete input.

3. **Information preservation**: SF's use of distributional semantics (Term-Context matrix) preserves semantic relationships while compressing the representation. The IB framework suggests this is optimal when β is chosen to balance compression and preservation.

However, SF does not explicitly optimize the IB objective. Future work could incorporate IB-guided sparsification to determine the optimal top_percent value for different datasets.

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

### 2.3.3 Biological Foundation: Thousand Brains Theory and HTM

The theoretical foundation for Sparse Distributed Representations (SDRs) originates in computational neuroscience, specifically the **Thousand Brains Theory** (Hawkins, 2021) and its realization in **Hierarchical Temporal Memory (HTM)** (Hole & Ahmad, 2021). This section reviews the biological constraints that motivate SDR-based approaches and positions Semantic Folding within this framework.

#### Biological Constraints for General AI

Hole & Ahmad (2021) identify six properties that distinguish biologically plausible AI from narrow AI systems:

1. **Sparse data representations**: The neocortex uses SDRs where only 2-5% of neurons are active at any time. This sparsity enables robustness to noise, massive pattern storage capacity, and multiple simultaneous predictions (Hole & Ahmad, 2021, §4.2, §5.2).

2. **Realistic neuron model**: Biological neurons (pyramidal cells) have separate dendritic integration zones (apical, basal, proximal) that process different information streams. In contrast, artificial neural networks use simplified "point neurons" with a single weighted sum (Hole & Ahmad, 2021, §4.2, §5.3).

3. **Reference frames**: The neocortex represents knowledge in allocentric reference frames (object-centered coordinates) that enable invariant predictions despite sensor movements. Grid cells in the entorhinal cortex provide location signals that modulate sensory predictions (Hole & Ahmad, 2021, §4.2, §5.5).

4. **Continuous online learning**: The neocortex learns continuously from streaming sensory data without requiring labeled training sets or retraining from scratch. HTM implements this via Hebbian-like permanence updates that reinforce correct predictions (Hole & Ahmad, 2021, §4.2, §5.4).

5. **Sensorimotor integration**: Every cortical column integrates sensory input with motor commands, enabling active perception through environmental interaction. This embodied reasoning shapes even abstract cognitive processes (Hole & Ahmad, 2021, §4.2, §5.6).

6. **Single general-purpose algorithm**: All neocortical regions run a "common cortical algorithm" despite processing different sensory modalities. Understanding this algorithm is potentially the only path to scalable general-purpose AI (Hole & Ahmad, 2021, §4.2, §5.8).

#### HTM and the Spatial Pooler

Hierarchical Temporal Memory (HTM) is a specific computational model implementing the Thousand Brains Theory. The **Spatial Pooler (SP)** is the component responsible for encoding input streams into SDRs. Sanati et al. (2023) provide an information-theoretic analysis of the SP algorithm, proving that:

1. **Sparsification improves reconstruction**: Using the Cramér-Rao lower bound and Fisher information matrix, they show that sparser representations (e.g., 2% sparsity) yield better estimation performance under the Cauchy distribution assumption (Sanati et al., 2023, §3.2).

2. **Noise robustness**: The SP algorithm with learning is resistant to input noise up to 40% without discernible output change (Sanati et al., 2023, Abstract).

3. **Information bottleneck optimization**: The SP algorithm implicitly optimizes the information bottleneck trade-off between compressing input and preserving relevant information (Sanati et al., 2023, §3.1).

#### Semantic Folding's Relationship to Biological SDRs

Semantic Folding adopts the principle of **sparse distributed representations** from HTM but implements it in a simplified computational framework suitable for text retrieval. The key correspondences and differences are:

| Biological SDR (HTM) | Semantic Folding |
|---------------------|------------------|
| Sparse binary vectors (2-5% density) | Sparse binary fingerprints (10% density via top_percent) |
| Grid cells provide location signals | 2D semantic grid provides spatial layout |
| Online learning via permanence updates | Offline dimensionality reduction (t-SNE/UMAP) |
| Sensorimotor integration | Purely textual (no sensorimotor component) |
| Reference frames for invariance | Distributional similarity for semantic invariance |
| Continuous adaptation | Fixed after index phase |

SF does not implement all six biological constraints — notably lacking continuous online learning, sensorimotor integration, and reference frames. However, it retains the foundational principle of **sparse distributed representations** that provides robustness to vocabulary mismatch and zero-shot adaptation to new domains.

The information-theoretic analysis of Sanati et al. (2023) supports SF's design choice of sparse representations: their proof that "more sparsity leads to better performance" (under the Cauchy distribution assumption) justifies SF's use of sparse fingerprints. However, SF's 10% density (top_percent=0.10) is higher than the ~2% typical in HTM, reflecting the different requirements of text retrieval versus sensory encoding.

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

## 2.8 Semantic Folding Evaluation Baselines

Recent work by Cai et al. (2024) introduces SSDB-100, the first evaluation dataset specifically designed for Semantic Folding Theory (SFT). The dataset contains 3,215 sentences manually labeled into 100 semantic grids by 10 expert annotators (average consistency index 0.856).

**Key contributions relevant to this thesis:**

1. **Expert-validated semantic grids**: SSDB-100 provides ground-truth semantic divisions that can be used to evaluate the quality of automatically constructed semantic spaces (Chapter 3, Step 3).

2. **Evaluation methodology**: Cai et al. use clustering metrics (homogeneity, completeness, NMI) to evaluate semantic division quality. These metrics complement the retrieval metrics (MRR, AP) used in this thesis.

3. **Dataset availability**: SSDB-100 is open-sourced at https://github.com/cks1999/SSDB-100, enabling direct comparison of our semantic space construction against their expert-validated grids.

**Connection to our work**: While Cai et al. focus on sentence-level semantic division for SFT validation, our work applies Semantic Folding to retrieval tasks. However, the quality of the underlying semantic space (Step 3) is critical for both applications. We include SSDB-100 as an additional benchmark in Chapter 7 to validate our dimensionality reduction approach (t-SNE/UMAP) against expert-validated semantic grids.

## References

- Cai, K., Chen, Z., Guo, H., Wang, S., Li, G., Li, J., Chen, F., & Feng, H. (2024). An Evaluative Baseline for Sentence-Level Semantic Division. *Machine Learning and Knowledge Extraction*, 6(1), 41–52. https://doi.org/10.3390/make6010003
- Dengel, A. (2015). Semantic Folding. *Technical Report*.
- Formal, T., et al. (2021). SPLADE. *SIGIR 2021*.
- Hole, K. J., & Ahmad, S. (2021). A thousand brains: toward biologically constrained AI. *SN Applied Sciences*, 3(8), 743. https://doi.org/10.1007/s42452-021-04715-0
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- McInnes, L., et al. (2018). UMAP. *arXiv:1802.03426*.
- Morton, G. M. (1966). *A Computer Oriented Geodetic Data Base*. IBM.
- Sanati, S., Rouhani, M., & Hodtani, G. A. (2023). Information-theoretic analysis of Hierarchical Temporal Memory-Spatial Pooler algorithm with a new upper bound for the standard information bottleneck method. *Frontiers in Computational Neuroscience*, 17, 1140782. https://doi.org/10.3389/fncom.2023.1140782
- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.
- Zahn, O., et al. (2026). Attention Is Not Retention. *arXiv:2601.15313*.
