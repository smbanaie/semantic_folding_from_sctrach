# Chapter 6: Similarity Metrics for Sparse Distributed Representations

## 6.1 Introduction

Semantic Folding represents text as sparse binary fingerprints over a 2D grid. Retrieval requires comparing query and document fingerprints via similarity metrics. This chapter evaluates metrics for sparse binary vectors, justifies the choice of Cosine Similarity as the default for Semantic Folding, and analyzes the impact of fingerprint correlation on metric performance.

## 6.2 Sparse Binary Fingerprint Representation

A document fingerprint is a sparse binary vector **d** ∈ {0,1}^d where d = grid_size². For a 64×64 grid, d = 4,096. Sparsity is controlled by top_percent (default 10%), producing fingerprints with ~410 active bits.

**Key properties**:
- **Sparsity**: Most bits are zero (90% for top_percent=0.10)
- **Spatial correlation**: Active bits cluster in semantically meaningful regions (not independent random)
- **IDF weighting**: Active bits are weighted by phrase IDF before aggregation

## 6.3 Similarity Metrics for Sparse Binary Vectors

### 6.3.1 Cosine Similarity (Default)

Cosine similarity measures the angle between two vectors:

$$\text{cosine}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

**Advantages**:
- Normalizes for fingerprint magnitude (document length)
- Works well with weighted fingerprints (IDF)
- Standard in retrieval evaluation

**Disadvantages**:
- Sensitive to fingerprint correlation (spatial clustering reduces discriminability)

### 6.3.2 Dice Coefficient

Dice measures overlap between two sets:

$$D(\mathbf{q}, \mathbf{d}) = \frac{2|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{A}| + |\mathcal{B}|}$$

where A and B are the sets of active bit indices for query and document.

**Advantages**:
- Emphasizes overlap
- Robust to differing fingerprint densities

**Disadvantages**:
- Does not account for IDF weighting
- Less discriminative than Cosine for SF fingerprints

### 6.3.3 Jaccard Coefficient

Jaccard measures set intersection over union:

$$J(\mathbf{q}, \mathbf{d}) = \frac{|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{A} \cup \mathcal{B}|}$$

**Advantages**:
- Standard for binary vectors
- Robust to density differences

**Disadvantages**:
- Ignores IDF weighting
- Lower discriminability than Dice or Cosine for SF

### 6.3.4 Hamming Distance

Hamming distance counts bit positions where two vectors differ:

$$H(\mathbf{q}, \mathbf{d}) = \sum_{i=1}^{d} \mathbb{1}[q_i \neq d_i]$$

**Advantages**:
- Computationally efficient (XOR + popcount)
- Natural for binary vectors

**Disadvantages**:
- Does not account for spatial correlation
- Ignores IDF weighting
- Poor performance for SF (Chapter 7, §7.3.5)

## 6.4 Empirical Comparison

### 6.4.1 Experimental Setup

We compare similarity metrics on Belebele (50 queries, 20 passages/query) using SF-only configuration. Metrics are applied to the same fingerprints (grid_size=64, top_percent=0.10, IDF weighting, L2 normalization).

### 6.4.2 Results

**Table 6.1: Similarity Metric Comparison (Belebele 50Q, SF-Only)**

| Metric | MRR | AP | P@1 | NDCG@5 | Interpretation |
|--------|:---:|:--:|:---:|:------:|-----------------|
| **Cosine** | **0.880** | **0.670** | **0.720** | **0.850** | **Optimal** |
| Dice | 0.840 | 0.630 | 0.680 | 0.810 | Good but lower than Cosine |
| Jaccard | 0.820 | 0.610 | 0.660 | 0.790 | Lower discriminability |
| Hamming | 0.720 | 0.510 | 0.600 | 0.710 | Poor — ignores weighting |

**Finding**: Cosine Similarity is optimal for SF fingerprints. Dice is a viable alternative but provides no advantage over Cosine.

## 6.5 Impact of Fingerprint Correlation

### 6.5.1 Theoretical Analysis

Independent random sparse binary vectors have expected cosine similarity ρ with variance ρ(1-ρ)/d. However, SF fingerprints are **spatially correlated** by design:

1. **Gaussian smoothing** (σ=1.5) spreads activation to neighbors
2. **Morton encoding** clusters nearby cells
3. **IDF aggregation** weights frequent phrases higher

This correlation increases both the mean and variance of pairwise cosine similarity, reducing discriminability.

### 6.5.2 Empirical Evidence

**Table 6.2: Pairwise Cosine Similarity Statistics (Belebele 50Q)**

| Statistic | Random SDR Prediction | SF Fingerprint (Observed) |
|-----------|----------------------:|--------------------------:|
| Mean | 0.10 | 0.15 |
| Std Dev | 0.007 | 0.025 |
| Max | 0.15 (99.9% CI) | 0.42 |

**Finding**: SF fingerprints have higher mean and higher variance than random SDRs. The maximum observed cosine (0.42) is 2.8× higher than the random-SDR 99.9% CI upper bound (0.15). This confirms significant spatial correlation.

### 6.5.3 Implications for Metric Selection

Higher fingerprint correlation reduces discriminability for all metrics. Cosine Similarity is most robust because it normalizes for magnitude. For highly correlated fingerprints, consider:

1. **Increasing grid size** (reduces density, reduces correlation)
2. **Reducing top_percent** (retains fewer cells, increases sparsity)
3. **Using UMAP instead of t-SNE** (UMAP's cross-entropy objective produces less correlated embeddings)

## 6.6 Normalization Strategies

### 6.6.1 L2 Normalization (Recommended)

L2 normalization treats each fingerprint as a unit vector:

$$\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2}$$

**Advantages**:
- Canonical retrieval metric
- Works well with IDF-weighted fingerprints
- Optimal per Chapter 4 tuning

### 6.6.2 sqrt(nnz) Normalization

sqrt(nnz) normalization divides by √(number of non-zero bits):

$$\hat{\mathbf{d}} = \frac{\mathbf{d}}{\sqrt{\text{nnz}(\mathbf{d})}}$$

**Disadvantages**:
- Over-penalizes long documents (Chapter 4, §4.7)
- −4.5% MRR vs L2 on Belebele

### 6.6.3 No Normalization

**Disadvantages**:
- Long documents (more phrases) dominate scores
- −10.2% MRR vs L2 on Belebele

## 6.7 LambdaMART Re-ranking (Proof-of-Concept)

LambdaMART is a learned re-ranking method that uses gradient-boosted decision trees to optimize retrieval metrics. We evaluated LambdaMART as a proof-of-concept extension to SF+SPLADE.

### 6.7.1 Architecture

LambdaMART re-ranks the top-K documents from SF+SPLADE using 35 features per (query, document) pair:

| Feature Type | Examples |
|--------------|----------|
| SF scores | Cosine, Dice, Jaccard, Hamming |
| SPLADE scores | Expansion overlap, term weights |
| Document features | Length, IDF sum, fingerprint density |
| Query features | Length, IDF sum, entity count |

### 6.7.2 Results

**Table 6.3: LambdaMART Re-ranking Results (Belebele 50Q)**

| Method | MRR | vs SF+SPLADE |
|--------|:---:|:------------:|
| SF+SPLADE (baseline) | 0.930 | — |
| LambdaMART (same-dataset) | 0.945 | +1.6% |
| LambdaMART (cross-dataset) | 0.649 | −30.2% |

**Finding**: LambdaMART provides marginal improvement on same-dataset evaluation but degrades on cross-dataset. The improvement is within expected noise (±0.015 MRR from t-SNE seed variation). LambdaMART is **not recommended** for production use due to:
1. Ceiling effect (SF+SPLADE already near-perfect on Belebele)
2. Insufficient training data (50 queries)
3. Cross-dataset generalization failure

## 6.8 Recommended Configuration

**Table 6.4: Recommended Similarity Configuration for SF+SPLADE**

| Component | Value | Justification |
|-----------|-------|----------------|
| **Similarity metric** | **Cosine Similarity** | Optimal across all datasets |
| **Query normalization** | **L2** | Standard retrieval practice |
| **Document normalization** | **L2** | Optimal per Chapter 4 tuning |
| **Fingerprint density** | **10% (top_percent=0.10)** | Optimal balance of signal and noise |
| **Re-ranking** | **None** | LambdaMART provides marginal benefit |

## 6.9 Integration with Pipeline

The similarity metric is applied in **Step 6 (Query Processing)** of the SF pipeline (Chapter 3, §3.7.4). The query fingerprint is scored against all document fingerprints using the selected metric. For SF+SPLADE, the hybrid score combines SF similarity and SPLADE similarity via the α-weighted equation (Chapter 3, §3.8.1).

**Pipeline completeness**: The similarity metric is the final scoring step. All previous steps (phrase extraction, term-context matrix, dimensionality reduction, fingerprint construction) produce the inputs to this step. The pipeline is complete with Cosine Similarity as the default metric.

---

## References

- Beyer, K., et al. (1999). When is "nearest neighbor" meaningful? *ICDT*, 217–235.
- Chen, J., et al. (2024). GeAR. *ACL 2025 Findings*. arXiv:2412.18431.
- Liu, T.-Y. (2009). Learning to Rank for Information Retrieval. *Foundations and Trends in Information Retrieval*, 3(3), 225–331.
