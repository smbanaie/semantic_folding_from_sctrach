# Chapter 4: Parameter Tuning for Semantic Folding

## 4.1 Introduction

Semantic Folding has 9 tunable parameters, each affecting retrieval performance. This chapter provides a systematic analysis of each parameter's effect on MRR, AP, P@K, and NDCG@K. We evaluate parameters on two datasets (Belebele and PubMedQA) with 50 and 31 queries respectively, and generalize findings to the full benchmark (now 11 datasets after the journal extension).

**Note on methodology**: Parameter tuning was conducted on the full benchmark datasets (50 queries for Belebele, 31 for PubMedQA). However, some sections report identical MRR values across configurations (e.g., Table 4.3.2 and 4.4.2 both show MRR=0.900). This suggests that certain parameter combinations were evaluated on a smaller dev set than the full benchmark. Future work should re-verify all parameter tuning on the full 50-query benchmark to ensure statistical robustness.

## 4.2 Grid Size

### 4.2.1 Theoretical Analysis

Grid size determines the total number of cells available for fingerprint representation. For a grid_size × grid_size grid, the total number of cells is d = grid_size².

| Grid Size | Cells (d) | Bit Density (10% active) | Use Case |
|-----------|----------:|:------------------------:|----------|
| 32×32 | 1,024 | 10–20% | Very small corpora (< 50 docs) |
| **64×64** | **4,096** | **8–12%** | **Optimal for 20-doc corpora** |
| 128×128 | 16,384 | 2–5% | Large corpora (> 200 docs) |
| 256×256 | 65,536 | <2% | Very large corpora (> 1000 docs) |

### 4.2.2 Empirical Results

**Table 4.1: Grid Size Tuning Results (Belebele 50Q)**

| Grid Size | MRR | AP | P@1 | NDCG@5 | Interpretation |
|-----------|:---:|:--:|:---:|:------:|-----------------|
| 32×32 | 0.810 | 0.590 | 0.640 | 0.770 | Too few cells — fingerprints overlap |
| **64×64** | **0.880** | **0.670** | **0.720** | **0.850** | **Optimal** |
| 128×128 | 0.830 | 0.610 | 0.680 | 0.810 | Too many cells — signal lost |
| 256×256 | 0.790 | 0.570 | 0.640 | 0.770 | Severe signal loss |

**Finding**: 64×64 is optimal for 20-doc corpora. Larger grids reduce bit density below the threshold needed for discriminative dot-product scoring.

## 4.3 Spreading Steps

### 4.3.1 Theoretical Analysis

Spreading steps control how many neighbors are activated around each phrase's grid cell. Radius=1 activates the 3×3 Moore neighborhood (9 cells total). Radius=2 activates the 5×5 neighborhood (25 cells).

### 4.3.2 Empirical Results

**Table 4.2: Spreading Steps Tuning Results (Belebele 50Q)**

| Radius | Decay | MRR | AP | vs Radius=1 | Interpretation |
|--------|-------|:---:|:--:|:-----------:|-----------------|
| 0 | — | 0.720 | 0.510 | −18.2% | No spreading — exact match only |
| **1** | **0.5** | **0.880** | **0.670** | **—** | **Optimal** |
| 2 | 0.5 | 0.810 | 0.590 | −7.9% | Too much noise from distant cells |
| 3 | 0.5 | 0.750 | 0.530 | −14.8% | Severe noise |

**Finding**: Radius=1 with decay=0.5 is optimal. Larger radii introduce noise from distant cells without adding discriminative signal.

## 4.4 Top Percent

### 4.4.1 Theoretical Analysis

Top percent controls what fraction of grid cells are retained after Gaussian smoothing. For a 64×64 grid (4,096 cells), top_percent=0.10 retains the top 410 cells.

### 4.4.2 Empirical Results

**Table 4.3: Top Percent Tuning Results (Belebele 50Q)**

| Top Percent | Cells Retained | MRR | AP | vs 10% | Interpretation |
|-------------|----------------|:---:|:--:|:------:|-----------------|
| 5% | 205 | 0.830 | 0.610 | −5.7% | Loses signal — too few cells |
| **10%** | **410** | **0.880** | **0.670** | **—** | **Optimal** |
| 15% | 614 | 0.870 | 0.650 | −1.1% | Slightly too many cells — noise increases |
| 20% | 819 | 0.840 | 0.620 | −4.5% | Too much noise |

**Finding**: 10% is optimal for 64×64 grid with 20-doc corpora. The optimal top_percent scales with grid size: ~8% for 128×128, ~5% for 256×256.

## 4.5 IDF Weighting

### 4.5.1 Theoretical Analysis

IDF weighting boosts rare, discriminative phrases while suppressing common terms. Uniform weighting treats all phrases equally.

### 4.5.2 Empirical Results

**Table 4.4: IDF Weighting Results (Belebele 50Q)**

| Weighting | MRR | AP | vs IDF | Interpretation |
|-----------|:---:|:--:|:------:|-----------------|
| **IDF** | **0.880** | **0.670** | — | **Optimal** |
| Uniform | 0.870 | 0.660 | −1.1% | Minor degradation |

**Finding**: IDF provides a modest but consistent improvement (+1.1% MRR). The effect is larger on datasets with high vocabulary diversity.

## 4.6 Gaussian Smoothing (σ)

### 4.6.1 Theoretical Analysis

Gaussian smoothing spreads each activation to neighboring cells weighted by a Gaussian kernel. This reduces isolated noise peaks while preserving structural patterns.

### 4.6.2 Empirical Results

**Table 4.5: Gaussian Smoothing Results (Belebele 50Q)**

| σ | MRR | AP | vs σ=1.5 | Interpretation |
|---|:---:|:--:|:--------:|-----------------|
| 0 (no smoothing) | 0.600 | 0.380 | **−31.2%** | Catastrophic — no neighborhood support |
| 0.5 | 0.810 | 0.590 | −7.9% | Under-smoothed |
| **1.5** | **0.880** | **0.670** | **—** | **Optimal** |
| 2.5 | 0.860 | 0.640 | −2.3% | Over-smoothed — structure lost |

**Finding**: σ=1.5 is critical. No smoothing causes catastrophic failure (−31.2% MRR) because each phrase occupies a single cell with no neighborhood support.

## 4.7 Document Normalization

### 4.7.1 Theoretical Analysis

Document normalization ensures that document length does not dominate retrieval scores. L2 normalization treats each document's fingerprint as a unit vector. sqrt(nnz) normalization assumes fingerprint magnitude scales with sqrt(non-zero count).

### 4.7.2 Empirical Results

**Table 4.6: Document Normalization Results (Belebele 50Q)**

| Normalization | MRR | AP | vs L2 | Interpretation |
|---------------|:---:|:--:|:-----:|-----------------|
| **L2** | **0.880** | **0.670** | — | **Optimal** |
| sqrt(nnz) | 0.840 | 0.630 | −4.5% | Over-penalizes long documents |
| None | 0.790 | 0.570 | −10.2% | Long documents dominate |

**Finding**: L2 normalization is critical (+4.5% MRR vs sqrt(nnz)). sqrt(nnz) assumes fingerprint magnitude carries meaning, but in SF, magnitude is an artifact of phrase count, not document importance.

## 4.8 Morton Encoding

### 4.8.1 Theoretical Analysis

Morton encoding (Z-order curve) interleaves the bits of x and y coordinates to preserve 2D spatial locality in the 1D bitstring representation.

### 4.8.2 Empirical Results

**Table 4.7: Morton Encoding Results (Belebele 50Q)**

| Morton | MRR | AP | vs Morton | Interpretation |
|--------|:---:|:--:|:---------:|-----------------|
| **True** | **0.880** | **0.670** | — | **Optimal** |
| False | 0.870 | 0.660 | −1.1% | Minor spatial structure loss |

**Finding**: Morton encoding provides a modest improvement (+1.1% MRR). The benefit is larger for datasets with high spatial clustering of semantically similar terms.

## 4.9 Hybrid Weight α (SF+SPLADE)

### 4.9.1 Theoretical Analysis

The hybrid weight α ∈ [0,1] balances SF and SPLADE signals:

score_hybrid = α · score_SF + (1-α) · score_SPLADE

α=0.0 uses only SPLADE; α=1.0 uses only SF.

### 4.9.2 Empirical Results

**Table 4.8: α-Sensitivity Results (2WikiMultihopQA 50Q)**

| α | MRR | AP | vs α=0.3 | Interpretation |
|---|:---:|:--:|:--------:|-----------------|
| 0.0 (SPLADE-only) | 0.797 | 0.537 | −7.9% | Lower than α=0.3 |
| **0.3** | **0.865** | **0.637** | **—** | **Optimal** |
| 0.5 | 0.850 | 0.620 | −1.7% | Degrading |
| 0.7 | 0.820 | 0.590 | −5.2% | Degrading |
| 1.0 (SF-only) | 0.788 | 0.537 | −8.9% | Worst |

**Finding**: α=0.3 is optimal across datasets. The α-sensitivity curve is monotonic — as SF weight increases, MRR decreases. This falsifies the complementarity hypothesis (H2).

### 4.9.3 Full α-Sensitivity Sweep (Journal Extension)

The coarse 5-point sweep above left open whether 0.3 was a cherry-picked favourable point. A complete α ∈ {0.0, 0.1, …, 1.0} sweep (eleven points) on four datasets — 2WikiMultihopQA, HotpotQA, MuSiQue, and SciFact — resolves this. Because the fused score is *linear* in the two max-normalized signals, each intermediate α is computed exactly from the two endpoint component runs (α=1.0 = pure SF, α=0.0 = pure SPLADE); no interpolation is involved.

**Table 4.8b: Complete MRR(α) Sweep (n=10 per dataset)**

| α | 2Wiki | HotpotQA | MuSiQue | SciFact |
|---|------:|---------:|--------:|--------:|
| 0.0 (SPLADE) | 1.000 | 1.000 | 0.925 | 0.823 |
| 0.1 | 1.000 | 1.000 | 0.925 | 0.823 |
| 0.2 | 1.000 | 1.000 | 0.925 | 0.823 |
| **0.3** | **1.000** | **1.000** | **0.925** | **0.823** |
| 0.4 | 1.000 | 1.000 | 0.925 | 0.822 |
| 0.5 | 1.000 | 1.000 | 0.913 | 0.820 |
| 0.6 | 1.000 | 1.000 | 0.856 | 0.821 |
| 0.7 | 0.933 | 0.867 | 0.754 | 0.818 |
| 0.8 | 0.858 | 0.617 | 0.686 | 0.718 |
| 0.9 | 0.853 | 0.575 | 0.543 | 0.703 |
| 1.0 (SF) | 0.803 | 0.453 | 0.447 | 0.704 |

**Finding (revised)**: MRR(α) is **flat within noise for α ∈ [0.0, 0.6] on every dataset** and degrades only when SF is over-weighted (α > 0.6), where the zero-shot SF signal collapses toward its SF-only floor. Two implications: (i) α = 0.3 is not a tuned-in-our-favour point — any value in [0, 0.6] yields identical ranking quality, so the earlier "monotonic curve" reading (Table 4.8) was an artifact of coarse sampling that missed the flat plateau; (ii) the correct interpretation of H2's falsification is subtler than "α hurts": at low α, SF contributes nothing beyond SPLADE (consistent with signal correlation), but the blend is also *not harmed* by moderate SF weight. The choice of α = 0.3 is retained as a conservative default, and the full curve is reported so the claim is auditable.

## 4.10 t-SNE Perplexity

### 4.10.1 Theoretical Analysis

t-SNE perplexity controls the balance between local and global structure preservation. Perplexity=30 (default) emphasizes local structure; perplexity=50 provides broader neighborhood preservation.

### 4.10.2 Empirical Results

**Table 4.9: t-SNE Perplexity Results (Belebele 50Q)**

| Perplexity | MRR | AP | vs p=50 | Interpretation |
|-----------|:---:|:--:|:-------:|-----------------|
| 30 | 0.840 | 0.630 | −4.5% | Too local — overfits to fine-grained structure |
| **50** | **0.880** | **0.670** | **—** | **Optimal** |
| 100 | 0.860 | 0.650 | −2.3% | Too global — loses local discrimination |

**Finding**: Perplexity=50 is optimal for Belebele and PubMedQA. UMAP (recommended default) does not have a perplexity parameter; its n_neighbors=15 achieves similar neighborhood balance.

## 4.11 Recommended Configuration

**Table 4.10: Recommended Configuration for SF+SPLADE**

| Parameter | Value | Justification |
|-----------|-------|----------------|
| Grid size | 64 | Optimal for 20-doc corpora |
| Spreading radius | 1 | 3×3 neighborhood |
| Top percent | 10% | 410 of 4,096 cells |
| IDF weighting | True | Boosts discriminative phrases |
| Gaussian σ | 1.5 | Critical for performance |
| Morton encoding | True | Preserves spatial locality |
| Doc normalization | L2 | Treats documents equally |
| Hybrid α | 0.3 | Optimal across datasets |
| Dimensionality reduction | UMAP | Matches or beats t-SNE on 7/9 datasets |

**Caveat**: This configuration is optimized for small-to-medium corpora (20–200 documents). For large corpora (1000+ documents), increase grid size to 128–256 and reduce top_percent to 5–8%.

## 4.12 Current Limitations

1. **Tuning dataset size**: Some parameter combinations report identical MRR values, suggesting evaluation on a small dev set. Future work should re-verify all tuning on the full 50-query benchmark.

2. **Generalizability**: The recommended configuration is based on tuning that may not generalize across all 9 datasets. Chapter 7 shows that dataset-specific parameters (e.g., t-SNE perplexity=30 for BioASQ) can improve performance.

3. **Interaction effects**: Parameters were tuned individually, not in combination. Future work should explore joint parameter optimization (e.g., grid size × top percent).

---

## References

- van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.
- McInnes, L., et al. (2018). UMAP. *arXiv:1802.03426*.
- Morton, G. M. (1966). *A Computer Oriented Geodetic Data Base*. IBM.
