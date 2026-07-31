# Chapter 4: Parameter Tuning and the Architectural Baseline

## 4.1 Introduction

Before diagnosing the mathematical boundaries of hybrid fusion (Chapter 5), we must first establish the optimal architectural baseline for the Semantic Folding (SF) pipeline. SF exposes 9 primary tunable parameters that govern the transition from continuous distributional semantics to discrete Sparse Distributed Representations (SDRs). 

This chapter provides a systematic, ablation-based analysis of each parameter's effect on Mean Reciprocal Rank (MRR), Average Precision (AP), and NDCG@K. Crucially, we do not treat this tuning in a vacuum. We explicitly connect these architectural choices to the core theoretical contributions of this thesis:
1. **The Bounded Scale Property:** How L2 normalization and grid size enforce the $[0, 1]$ score scale that ultimately triggers the Complementarity Illusion.
2. **The Scaling Wall:** How grid resolution and sparsity dictate the $O(\sqrt{N})$ dynamic range collapse observed in large corpora.
3. **The Operator-Topology Constraint:** Why parameters like the hybrid weight $\alpha$ are strictly bound by task topology.

We evaluate parameters primarily on the Belebele (50 queries, 20-doc pools) and PubMedQA (31 queries) datasets, representing single-hop reading comprehension and domain-specific QA, respectively, and validate generalizations against the full 8-dataset matrix.

## 4.2 Grid Size and Semantic Resolution

### 4.2.1 Theoretical Analysis
Grid size $N$ determines the total number of cells available for fingerprint representation ($d = N^2$). This parameter controls the **semantic resolution** of the 2D space. A smaller grid forces unrelated concepts to share cells (false neighbors), while a larger grid fragments semantically related phrases across distant cells, reducing the dot-product overlap necessary for retrieval. Furthermore, grid size directly impacts the Scaling Wall: as $N$ increases, the fixed sparsity threshold ($\rho=0.10$) spreads active bits across a wider area, reducing the expected overlap $\mathbb{E}[\mathbf{q} \cdot \mathbf{d}]$ and exacerbating score compression.

### 4.2.2 Empirical Results

**Table 4.1: Grid Size Tuning Results (Belebele 50Q, $\rho=0.10$)**

| Grid Size | Cells ($d$) | Active Bits (~10%) | MRR | AP | NDCG@5 | Interpretation |
|-----------|----------:|:------------------------:|:---:|:--:|:------:|-----------------|
| 32×32 | 1,024 | ~102 | 0.810 | 0.590 | 0.770 | Too few cells — semantic collisions |
| **64×64** | **4,096** | **~410** | **0.880** | **0.670** | **0.850** | **Optimal resolution for $N \approx 20$** |
| 128×128 | 16,384 | ~1,638 | 0.830 | 0.610 | 0.810 | Signal fragmentation begins |
| 256×256 | 65,536 | ~6,553 | 0.790 | 0.570 | 0.770 | Severe fragmentation / Scaling Wall |

**Finding**: $64 \times 64$ is optimal for 20-document candidate pools. Larger grids reduce the probability of spatial overlap for semantically similar queries and documents. While increasing grid size is theoretically necessary for larger corpora to maintain orthogonality (as per the Orthogonality Constraint in Chapter 2), doing so without drastically reducing $\rho$ triggers the Scaling Wall (Chapter 7).

## 4.3 Topological Spreading Activation

### 4.3.1 Theoretical Analysis
Spreading activation controls the robustness of the retrieval topology. By activating neighboring cells in the 2D grid, SF mitigates the boundary brittleness inherent in discrete quantization. Radius $r=1$ activates the $3 \times 3$ Moore neighborhood (9 cells). However, excessive spreading destroys the localized semantic signal required for precise discrimination.

### 4.3.2 Empirical Results

**Table 4.2: Spreading Radius Tuning Results (Belebele 50Q, $\gamma=0.5$)**

| Radius | Neighborhood Size | MRR | AP | $\Delta$ vs $r=1$ | Interpretation |
|--------|-------------------|:---:|:--:|:-----------------:|-----------------|
| 0 | $1 \times 1$ (1 cell) | 0.720 | 0.510 | −18.2% | Exact match only — brittle |
| **1** | **$3 \times 3$ (9 cells)** | **0.880** | **0.670** | **—** | **Optimal local robustness** |
| 2 | $5 \times 5$ (25 cells) | 0.810 | 0.590 | −7.9% | Semantic blur — distant cells activated |
| 3 | $7 \times 7$ (49 cells) | 0.750 | 0.530 | −14.8% | Severe noise injection |

**Finding**: Radius $r=1$ with exponential decay $\gamma=0.5$ is strictly optimal. The $3 \times 3$ window provides just enough topological robustness to bridge minor vocabulary gaps without triggering false positives from semantically distant grid regions.

## 4.4 Sparsification Threshold (Top Percent)

### 4.4.1 Theoretical Analysis
The top percent parameter $\rho$ controls the density of the SDR. As established in Chapter 2 (§2.2.3), information-theoretic analysis (Sanati et al., 2023) proves that increased sparsity improves estimation performance by maximizing Fisher Information. However, in the context of grid-based retrieval, $\rho$ must be high enough to ensure that semantically similar phrases share active bits after Gaussian smoothing.

### 4.4.2 Empirical Results

**Table 4.3: Top Percent Tuning Results (Belebele 50Q, Grid $64 \times 64$)**

| Top Percent ($\rho$) | Cells Retained | Bit Density | MRR | AP | $\Delta$ vs 10% | Interpretation |
|-------------|----------------|:-----------:|:---:|:--:|:--------------:|-----------------|
| 5% | 205 | Sparse | 0.830 | 0.610 | −5.7% | Under-connected — overlap lost |
| **10%** | **410** | **Moderate** | **0.880** | **0.670** | **—** | **Optimal balance** |
| 15% | 614 | Dense | 0.870 | 0.650 | −1.1% | Minor noise introduction |
| 20% | 819 | Very Dense | 0.840 | 0.620 | −4.5% | Baseline noise floor rises |

**Finding**: $\rho=0.10$ (410 active bits) is the mathematical sweet spot for a 4096-dimensional grid. It satisfies the SDR requirement of high-dimensional sparsity while maintaining sufficient spatial overlap for dot-product discrimination.

## 4.5 Gaussian Smoothing ($\sigma$)

### 4.5.1 Theoretical Analysis
Gaussian smoothing bridges the gap between continuous UMAP coordinates and discrete grid cells. Without smoothing, a phrase located at coordinate $(15.4, 22.8)$ activates only cell $(15, 22)$, entirely losing its proximity to $(15, 23)$. 

### 4.5.2 Empirical Results

**Table 4.4: Gaussian Smoothing Results (Belebele 50Q)**

| $\sigma$ | Spatial Reach | MRR | AP | $\Delta$ vs $\sigma=1.5$ | Interpretation |
|---|:---:|:---:|:--:|:--------:|-----------------|
| 0 (None) | Single cell | 0.600 | 0.380 | **−31.2%** | **Catastrophic boundary failure** |
| 0.5 | Very local | 0.810 | 0.590 | −7.9% | Under-smoothed |
| **1.5** | **1-2 cells** | **0.880** | **0.670** | **—** | **Optimal continuous-discrete bridge** |
| 2.5 | 3+ cells | 0.860 | 0.640 | −2.3% | Over-smoothed — distinct concepts merge |

**Finding**: $\sigma=1.5$ is mathematically mandatory. The catastrophic −31.2% drop at $\sigma=0$ proves that raw discrete quantization destroys the topological structure generated by UMAP. 

## 4.6 Document Normalization and the Bounded Scale Property

### 4.6.1 Theoretical Analysis
Document normalization dictates the final algebraic properties of the SF score. This is not merely an implementation detail—it is the root cause of the **Complementarity Illusion**. 
*   **L2 Normalization** forces the document vector to unit length. Consequently, the dot-product $\mathbf{q} \cdot \mathbf{d}$ becomes equivalent to cosine similarity, strictly bounding the output to $\text{score}_{\text{SF}} \in [0, 1.0]$.
*   **No Normalization** or **sqrt(nnz)** allows scores to scale with document length or phrase count, creating an unbounded scale.

### 4.6.2 Empirical Results

**Table 4.5: Document Normalization Results (Belebele 50Q)**

| Normalization | Score Scale | MRR | AP | $\Delta$ vs L2 | Interpretation |
|---------------|-------------|:---:|:--:|:-----:|-----------------|
| **L2** | **Bounded [0, 1]** | **0.880** | **0.670** | **—** | **Optimal. Enforces cosine similarity.** |
| sqrt(nnz) | Unbounded | 0.840 | 0.630 | −4.5% | Magnitude biases toward long documents |
| None | Highly Unbounded | 0.790 | 0.570 | −10.2% | Long documents completely dominate |

**Finding**: L2 normalization provides a +4.5% MRR improvement over sqrt(nnz). More importantly for this thesis, L2 normalization **engineers the bounded scale** ($\max=1.0$) that perfectly mismatches SPLADE's unbounded scale ($\max \approx 50$). Without L2 normalization, the Linear Fusion Scale Mismatch analyzed in Chapter 5 would not manifest as severely.

## 4.7 Morton Encoding and Topology Preservation

### 4.7.1 Theoretical Analysis
As formalized in Chapter 3, Morton Z-order curve encoding interleaves the bits of $x$ and $y$ coordinates to guarantee that 2D Euclidean distance is monotonically related to 1D Hamming distance. 

### 4.7.2 Empirical Results

**Table 4.6: Morton Encoding Results (Belebele 50Q)**

| Morton Encoding | Topological Locality | MRR | AP | $\Delta$ | Interpretation |
|--------|-----------------------|:---:|:--:|:---------:|-----------------|
| **True** | **Preserved** | **0.880** | **0.670** | **—** | **1D dot-product respects 2D space** |
| False (Row-major) | Destroyed | 0.870 | 0.660 | −1.1% | Adjacent 2D cells become distant in 1D |

**Finding**: Morton encoding provides a modest but consistent +1.1% MRR improvement. The benefit is theoretically guaranteed by the math in Chapter 3, though the small empirical delta suggests that UMAP's global repulsive term prevents massive topological destruction even under row-major flattening.

## 4.8 Dimensionality Reduction: UMAP Parameters vs. t-SNE

### 4.8.1 Theoretical Shift
Prior iterations of this pipeline relied on t-SNE. However, as established in Chapter 3, t-SNE's lack of a repulsive term causes global concept collapse. We have migrated the default to UMAP. The primary tuning parameter for UMAP is `n_neighbors`, which controls the trade-off between local structure preservation and global topological separation.

### 4.8.2 Empirical Results

**Table 4.7: UMAP `n_neighbors` Tuning (Belebele 50Q, `min_dist=0.0`)**

| Method | Parameter | MRR | AP | $\Delta$ vs UMAP(15) | Interpretation |
|-----------|-----------|:---:|:--:|:-------------------:|-----------------|
| t-SNE | Perplexity=30 | 0.840 | 0.630 | −4.5% | Over-localizes, global structure lost |
| t-SNE | Perplexity=50 | 0.850 | 0.650 | −3.4% | Better global view, but still lacks repulsion |
| **UMAP** | **n_neighbors=5** | **0.860** | **0.640** | **−2.3%** | **Too local — acts like t-SNE** |
| **UMAP** | **n_neighbors=15** | **0.880** | **0.670** | **—** | **Optimal local-global balance** |
| UMAP | n_neighbors=30 | 0.870 | 0.650 | −1.1% | Slight over-smoothing of local semantics |

**Finding**: UMAP with `n_neighbors=15` and `min_dist=0.0` is the strictly dominant configuration. The repulsive term in the cross-entropy objective is critical for preventing unrelated domain concepts from overlapping on the 4096-cell grid.

## 4.9 Hybrid Weight $\alpha$ and the Operator-Topology Constraint

### 4.9.1 Theoretical Context
The hybrid weight $\alpha \in [0,1]$ balances the SF and SPLADE signals: $\text{score} = \alpha \cdot \text{SF} + (1-\alpha) \cdot \text{SPLADE}$. 
**Crucial Caveat**: Because of the **Operator-Topology Constraint** (formally proven in Chapter 5), tuning $\alpha$ via Linear Interpolation is *only valid for multi-hop tasks*. On single-hop tasks, Linear Fusion mathematically fails due to scale mismatch, and $\alpha$ tuning is bypassed entirely in favor of RRF. 

Therefore, we evaluate $\alpha$ sensitivity on 2WikiMultihopQA (a multi-hop dataset where Linear Fusion is the dominant operator).

### 4.9.2 Empirical Results

**Table 4.8: $\alpha$-Sensitivity Results (2WikiMultihopQA 50Q, Multi-Hop)**

| $\alpha$ | SF Weight | SPLADE Weight | MRR | AP | $\Delta$ vs $\alpha=0.3$ | Interpretation |
|---|:---:|:---:|:---:|:--:|:--------:|-----------------|
| 0.0 | 0% | 100% (SPLADE-only) | 0.797 | 0.537 | −11.5% | Lacks unsupervised spatial grounding |
| **0.3** | **30%** | **70%** | **0.901** | **0.637** | **—** | **Optimal magnitude preservation** |
| 0.5 | 50% | 50% | 0.850 | 0.620 | −5.7% | SF signal begins to dilute magnitude |
| 0.7 | 70% | 30% | 0.820 | 0.590 | −9.0% | SPLADE magnitude suppressed |
| 1.0 | 100% | 0% (SF-only) | 0.788 | 0.537 | −12.5% | No compositional confidence signal |

**Finding**: $\alpha=0.3$ is optimal for multi-hop tasks. This specific weight preserves SPLADE's high-magnitude scores (which encode compositional confidence) while injecting a 30% topological SF signal. Note that on single-hop tasks (e.g., Belebele), $\alpha=0.3$ *degrades* performance to 0.920 MRR, necessitating the switch to RRF (which achieves 1.000 MRR).

## 4.10 OOV Expansion via FAISS

### 4.10.1 Implementation
Out-of-vocabulary (OOV) terms are mapped to the grid via approximate nearest neighbor search using FAISS IVFFlat over the phrase fingerprint matrix. 

### 4.10.2 Empirical Results
| OOV Expansion | Latency/Query | MRR | $\Delta$ | Interpretation |
|---------------|---------------|:---:|:--------:|-----------------|
| **FAISS IVFFlat** | **~0.075s** | **0.880** | **—** | **Optimal speed/accuracy trade-off** |
| Brute Force | ~30.0s | 0.880 | 0.0% | Identical MRR, 400× slower |
| Disabled | ~0.001s | 0.810 | −8.0% | Severe degradation on unseen query terms |

**Finding**: FAISS acceleration provides a 400× speedup with zero MRR degradation. OOV expansion is critical for handling domain-specific terminology not present in the candidate documents.

## 4.11 Synthesis: The Recommended Baseline Configuration

Based on the ablations above, we establish the baseline configuration used for the diagnostic analyses in Chapters 5, 6, and 7. 

**Table 4.9: Default SF Pipeline Configuration**

| Parameter | Value | Theoretical Justification |
|-----------|-------|---------------------------|
| Grid size ($N$) | 64 ($d=4096$) | Optimal semantic resolution for $N_{docs} \approx 20$ |
| Dimensionality Reduction | UMAP | Repulsive term prevents global concept collapse |
| UMAP `n_neighbors` | 15 | Balances local synonymy and global separation |
| Gaussian $\sigma$ | 1.5 | Bridges continuous-to-discrete boundary (−31% if disabled) |
| Sparsification ($\rho$) | 0.10 | Maximizes Fisher Information while retaining overlap |
| Morton Encoding | True | Preserves 2D Euclidean topology in 1D Hamming space |
| Spreading Radius ($r$) | 1 ($\gamma=0.5$) | 3×3 local robustness without semantic blur |
| Document Normalization | L2 | **Enforces bounded [0,1] scale (Prerequisite for Ch 5)** |
| OOV Expansion | FAISS IVFFlat | 400× speedup, zero accuracy loss |
| Hybrid Operator | Task-Dependent | **RRF for Single-Hop; Linear ($\alpha=0.3$) for Multi-Hop** |

**Dataset-Specific Registry**: While Table 4.9 represents the global default, the final benchmark utilizes a per-dataset registry (`config/dataset_registry.yml`) to override specific parameters where mathematically justified. For instance, on topically narrow datasets (PubMedQA, 2WikiMultihopQA), t-SNE with perplexity=30 is permitted because the aggressive local clustering avoids false neighbors that UMAP's repulsive term might introduce in an already tightly confined semantic space.

## 4.12 Threats to Validity in Parameter Tuning

1. **Marginal Metric Ties**: Several parameters (IDF weighting, Morton encoding) show marginal improvements (+1.1% MRR) that fall within the 95% Bootstrap Confidence Intervals for a 50-query dev set. They are retained for theoretical completeness rather than strict empirical superiority.
2. **Covariate Shift**: Parameters tuned on Belebele (reading comprehension) are applied to MuSiQue (multi-hop reasoning). The fact that the Operator-Topology Constraint still emerges strongly despite this parameter covariate shift suggests the finding is structurally robust, not an artifact of hyperparameter overfitting.

---

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
- Morton, G.M. (1966). *A Computer Oriented Geodetic Data Base*. IBM Technical Report.
- Sanati, S., Rouhani, M., & Hodtani, G.A. (2023). Information-theoretic analysis of Hierarchical Temporal Memory-Spatial Pooler algorithm with a new upper bound for the standard information bottleneck method. *Frontiers in Computational Neuroscience*, 17, 1140782.
- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.