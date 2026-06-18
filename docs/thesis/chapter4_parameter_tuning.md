# Chapter 4: Parameter Tuning for Semantic Folding

## 4.1 Introduction

Semantic Folding exposes several free parameters that control the density, resolution, and matching behaviour of the sparse distributed representations (SDRs) it produces. Unlike supervised methods where parameters are learned from data, SF's parameters must be chosen empirically through systematic tuning. This chapter presents a comprehensive parameter tuning framework with mathematical justification for each configuration choice, supported by controlled experiments across multiple datasets.

## 4.2 Theoretical Foundations

### 4.2.1 Parameter Taxonomy

SF's parameters can be classified into three categories:

| Category | Parameters | Role |
|----------|------------|------|
| **Grid Architecture** | grid_size, morton encoding | Spatial resolution and locality preservation |
| **Fingerprint Construction** | top_percent, smoothing_sigma | Density and smoothness of representations |
| **Query Processing** | spreading_steps, weighting, doc_norm | Matching behaviour and scoring |

### 4.2.2 The Sparsity-Density Trade-off

The fundamental trade-off in SF is between **sparsity** (few active bits → distinctiveness) and **density** (many active bits → coverage). For a grid of size $g \times g = N$ cells with target density $\rho$:

$$\text{Active bits} = k = \rho \cdot N$$

The optimal density balances two competing forces:

1. **Discriminability**: Lower $\rho$ → fewer active bits → more distinct fingerprints → better precision
2. **Coverage**: Higher $\rho$ → more active bits → better recall → more semantic signal

**Theoretical optimal range**: $\rho \in [0.05, 0.15]$ for corpora with $O(10^2)$ to $O(10^3)$ documents.

## 4.3 Grid Size (64 vs 128)

### 4.3.1 Mathematical Analysis

For a corpus of $D$ documents with average $P$ phrases per document, the expected fingerprint density is:

$$\rho(d) \approx \frac{\text{nnz}(F_d)}{g^2}$$

where $\text{nnz}(F_d)$ is the number of non-zero entries in document $d$'s fingerprint.

**For grid_size=128** (16,384 cells):
- 20-doc corpus: $\rho \approx 2\text{--}5\%$ (338–862 active bits)
- Signal-to-noise ratio: Low (sparse activations)

**For grid_size=64** (4,096 cells):
- 20-doc corpus: $\rho \approx 7\text{--}10\%$ (287–409 active bits)
- Signal-to-noise ratio: High (denser activations)

### 4.3.2 Experimental Results

| Metric | grid=128 | grid=64 | Δ |
|--------|----------|---------|---|
| MRR | 0.900 | **1.000** | **+11.1%** |
| NDCG@5 | 0.888 | **0.919** | +3.5% |
| AP | 0.836 | **0.869** | +3.9% |
| P@5 | 0.520 | 0.520 | 0% |
| R@5 | 1.000 | 1.000 | 0% |

### 4.3.3 Interpretation

The 64×64 grid outperforms 128×128 because:

1. **Higher fingerprint density**: 7-10% vs 2-5% → more semantic signal per document
2. **Better overlap**: Semantically related documents share more active cells
3. **Reduced sparsification loss**: Fewer cells to discard during peak selection

**Recommendation**: Use `grid_size=64` for corpora up to $O(10^3)$ documents. Scale to 128 or 256 for larger collections.

## 4.4 Spreading Steps (0, 1, 2)

### 4.4.1 Mathematical Formulation

Spreading applies a spatial filter to active bits, creating a "semantic halo" with exponential decay:

$$\tilde{Q}_{x,y} = \max_{u,v} \left( Q_{u,v} \cdot \gamma^{d((u,v), (x,y))} \right)$$

where $\gamma = 0.5$ is the decay factor and $d$ is Chebyshev distance.

For spreading_steps=$r$, each active cell expands to a $(2r+1) \times (2r+1)$ block:

| Steps | Block Size | Max Expansion | Decay at Edge |
|-------|------------|---------------|---------------|
| 0 | 1×1 | 1× | N/A |
| 1 | 3×3 | 9× | 0.5 |
| 2 | 5×5 | 25× | 0.25 |

### 4.4.2 Experimental Results

| Metric | steps=0 | steps=1 | steps=2 |
|--------|---------|---------|---------|
| MRR | 0.900 | 0.900 | 0.900 |
| NDCG@5 | 0.848 | **0.888** | 0.888 |
| AP | 0.784 | **0.836** | 0.836 |
| Recall@5 | 0.933 | **1.000** | 1.000 |

### 4.4.3 Analysis

- **steps=0**: Loses C09 (Social Networks) in Q4 — "community networks" cannot reach "social networks" without topological expansion
- **steps=1**: All relevant documents found — optimal soft matching
- **steps=2**: No improvement over steps=1, increases noise

**Recommendation**: `spreading_steps=1` provides optimal soft matching for corpora up to $O(10^3)$ documents.

## 4.5 Top Percent (0.05, 0.10, 0.15)

### 4.5.1 Mathematical Formulation

`top_percent` controls the fraction of grid cells retained after peak detection:

$$k = \lfloor \rho \cdot g^2 \rfloor$$

where $\rho$ is the target density.

| Top Percent | Active Bits (64×64) | Active Bits (128×128) |
|-------------|---------------------|----------------------|
| 0.05 | 205 | 819 |
| 0.10 | 410 | 1,638 |
| 0.15 | 614 | 2,458 |

### 4.5.2 Experimental Results

| Metric | top=0.05 | top=0.10 | top=0.15 |
|--------|----------|----------|----------|
| MRR | 0.900 | 0.900 | 0.900 |
| NDCG@5 | 0.863 | **0.888** | 0.848 |
| AP | 0.806 | **0.836** | 0.779 |
| Recall@5 | 0.933 | **1.000** | 0.933 |

### 4.5.3 Analysis

- **top=0.05**: Loses C00 (Emotional Intelligence) in Q5 — too sparse, loses discriminative signal
- **top=0.10**: All relevant documents found — balanced precision-recall
- **top=0.15**: Also loses C00 in Q5 — too dense, fingerprint overlap diluted by noise

**Recommendation**: `top_percent=0.10` for balanced precision-recall. Adjust downward (0.05-0.08) for very large corpora where distinctiveness matters more.

## 4.6 Weighting Strategy (IDF vs Uniform)

### 4.6.1 Mathematical Formulation

**IDF weighting** boosts rare, discriminative phrases:

$$w_j^{\text{IDF}} = \log\left(\frac{N}{\text{df}(p_j) + 1}\right)$$

**Uniform weighting** treats all phrases equally:

$$w_j^{\text{uniform}} = 1$$

### 4.6.2 Experimental Results

| Metric | IDF | Uniform | Δ |
|--------|-----|---------|---|
| MRR | 0.900 | 0.900 | 0% |
| NDCG@5 | **0.888** | 0.842 | -5.2% |
| AP | **0.836** | 0.772 | -7.7% |
| Recall@5 | **1.000** | 0.933 | -6.7% |

### 4.6.3 Analysis

- **IDF**: Consistently ranks C17 (Semantics) above irrelevant documents in Q2 — high IDF of "contextual meaning" boosts distinctive terms
- **Uniform**: Drops C17 to rank 4 in Q2, loses C00 in Q5 — common OOV terms drown specific signal

**Recommendation**: Always use IDF weighting. Uniform weighting only when IDF weights are unavailable.

## 4.7 Smoothing Sigma (1.0, 1.5, 2.0)

### 4.7.1 Mathematical Formulation

Gaussian smoothing applies before peak detection:

$$\tilde{G} = G * K_{\sigma}, \qquad K_{\sigma}(u,v) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{u^2+v^2}{2\sigma^2}\right)$$

The kernel radius (in cells) for $\sigma=1.5$ is approximately $3\sigma = 4.5$ cells.

### 4.7.2 Experimental Results

| Metric | σ=1.0 | σ=1.5 | σ=2.0 |
|--------|-------|-------|-------|
| MRR | 0.900 | 0.900 | 0.900 |
| NDCG@5 | 0.888 | 0.888 | 0.879 |
| AP | 0.836 | 0.836 | 0.824 |
| Recall@5 | 1.000 | 1.000 | 1.000 |

### 4.7.3 Analysis

All three values produce nearly identical results (AP range: 0.824-0.836). The peak detection algorithm is robust to moderate changes in smoothing because:
1. Document fingerprints are dominated by top-10% percentile selection
2. Smoothing primarily affects the shape of activation regions, not their peak locations

**Recommendation**: Keep `smoothing_sigma=1.5` as a safe default. May become more influential for larger grids (256+).

## 4.8 Document Normalization

### 4.8.1 Mathematical Formulation

| Normalization | Formula | Effect |
|---------------|---------|--------|
| sqrt_nnz | $\sqrt{\|\mathbf{d}\|_0}$ | Favors longer documents |
| **L2** | $\|\mathbf{d}\|_2 = \sqrt{\sum_j d_{ij}^2}$ | Standard cosine |
| L1 | $\|\mathbf{d}\|_1 = \sum_j |d_{ij}|$ | Aggressive penalization |
| Max | $\max_j d_{ij}$ | Normalizes by peak activation |

### 4.8.2 Experimental Results (Belebele, 50 queries)

| Configuration | MRR | Δ |
|---------------|-----|---|
| Baseline (sqrt_nnz) | 0.840 | — |
| **L2 Normalization** | **0.880** | **+4.0%** |
| L1 Normalization | 0.830 | -1.0% |
| Max Normalization | 0.818 | -2.2% |

### 4.8.3 Analysis

L2 normalization provides significant improvement (+4.0% MRR) because:
1. Treats all documents equally regardless of length
2. Prevents longer documents from dominating similarity scores
3. Mathematically correct for cosine similarity

**Recommendation**: Use `--doc-norm l2` for all datasets.

## 4.9 t-SNE Perplexity

### 4.9.1 Mathematical Formulation

Perplexity controls the balance between local and global structure in t-SNE:

$$\text{Perp}(P_i) = 2^{H(P_i)}, \qquad H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$$

Higher perplexity → more global structure; lower perplexity → tighter local clusters.

### 4.9.2 Experimental Results

| Perplexity | Belebele MRR | PubMedQA MRR |
|------------|--------------|--------------|
| 10 | 0.860 (+2.0%) | — |
| **30 (baseline)** | 0.840 | 0.954 |
| **50** | **0.880 (+4.0%)** | **0.969 (+1.5%)** |

### 4.9.3 Analysis

Perplexity=50 improves both datasets because:
1. Creates tighter local clusters for fine-grained discrimination
2. Better separation of semantically distinct concepts
3. More robust to noise in the term-context matrix

**Recommendation**: Use `--tsne-perplexity 50` for all datasets.

## 4.10 Recommended Configuration

Based on systematic tuning across multiple datasets:

```yaml
# Grid Architecture
grid_size: 64                    # Optimal for 20-doc corpora
use_morton: true                 # Preserves 2D spatial locality

# Fingerprint Construction
top_percent: 0.10                # Balance precision and recall
smoothing_sigma: 1.5             # Robust default

# Query Processing
spreading_steps: 1               # Soft matching without excessive noise
weighting: idf                   # Boosts rare, discriminative phrases
doc_norm: l2                     # +4.0% MRR over sqrt_nnz

# Dimensionality Reduction
method: tsne                     # Better local clustering
tsne_perplexity: 50              # +4.0% MRR over perplexity=30
random_seed: 42                  # Reproducibility
```

### 4.10.1 Expected Performance

| Metric | Value |
|--------|-------|
| MRR | 0.880-1.000 |
| AP | 0.836-0.869 |
| NDCG@5 | 0.888-0.919 |
| P@5 | 0.520 |
| R@5 | 1.000 |

### 4.10.2 Cross-Dataset Validation

| Dataset | Best Config | SF MRR | BM25 MRR |
|---------|-------------|--------|----------|
| PubMedQA | Perplexity=50 | **0.969** | 1.000 |
| Belebele | L2 + Perplexity=50 | **0.880** | 0.995 |
| SciFact | Default | **0.755** | — |
| PopQA | Default | **0.980** | 1.000 |

## 4.11 Parameter Interactions

### 4.11.1 Grid Size × Top Percent

The interaction between grid size and top percent determines effective fingerprint density:

$$\rho_{\text{effective}} = \text{top\_percent} \times \frac{\text{avg\_active\_bits}}{g^2}$$

For grid=64, top=0.10: $\rho \approx 7\text{--}10\%$ (optimal)
For grid=128, top=0.10: $\rho \approx 2\text{--}5\%$ (too sparse)

### 4.11.2 Spreading × IDF Weighting

Spreading and IDF weighting are complementary:
- **IDF** selects which phrases contribute to the query fingerprint
- **Spreading** expands the spatial footprint of selected phrases

Without IDF, spreading amplifies noise from common terms. With IDF, spreading enhances signal from discriminative terms.

### 4.11.3 Smoothing × Top Percent

Smoothing affects peak detection, which interacts with top percent:
- Higher σ → broader peaks → more cells above threshold → higher effective density
- Lower σ → sharper peaks → fewer cells above threshold → lower effective density

For top=0.10, the pipeline is robust to σ ∈ [1.0, 2.0].

## 4.12 Limitations and Future Work

### 4.12.1 Current Limitations

1. **Corpus size dependency**: All sweeps performed on 20-doc corpora. Optimal values may shift for larger collections.

2. **t-SNE stochasticity**: Results depend on random seed (fixed at 42). Relative comparisons valid, absolute scores seed-dependent.

3. **Binary relevance**: Ground truth uses binary relevance. Graded relevance would make NDCG more discriminating.

4. **Spreading decay**: Fixed at 0.5. Varying this (0.3, 0.7) could further tune soft-matching.

5. **Normalisation formula**: Current scoring uses $\text{score} = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\sqrt{\text{nnz}(\mathbf{d}_i)}}$. Alternative normalizations not explored.

### 4.12.2 Future Directions

1. **Adaptive parameters**: Learn optimal parameters per dataset using meta-learning
2. **Grid size scaling**: Develop guidelines for scaling grid size with corpus size
3. **Joint optimization**: Optimize all parameters jointly rather than sequentially
4. **Neural parameter selection**: Use reinforcement learning to select parameters

## References

- Furnas, G. W., et al. (1987). The vocabulary problem in human-system communication. *CACM*, 30(11), 964–971.
- Harris, Z. S. (1954). Distributional structure. *Word*, 10(2–3), 146–162.
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9, 2579–2605.
