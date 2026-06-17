# Dimensionality Reduction for Semantic Space Construction: UMAP vs t-SNE vs PCA and Alternatives

**Date:** 2026-06-17
**Context:** Comparing dimensionality reduction techniques for constructing a 2D semantic space from high-dimensional text embeddings, preserving semantic similarity between contexts.

---

## Executive Summary

For projecting high-dimensional text embeddings into 2D while preserving semantic similarity, **UMAP is the strongest general-purpose choice**, followed by t-SNE for local structure, with PCA as a fast baseline that sacrifices non-linear relationships. The optimal method depends on whether you prioritize local neighborhoods (semantically similar phrases stay close), global structure (distant semantic clusters are correctly positioned), or computational scalability.

---

## 1. Method Comparison

### PCA (Principal Component Analysis)
- **Type:** Linear
- **Preserves:** Global variance (maximizes variance along orthogonal axes)
- **Local fidelity:** Poor — collapses non-linear manifold structure
- **Global fidelity:** Good — distances between distant clusters are roughly meaningful
- **Speed:** O(n·d²) — fastest by far, trivially scalable to millions
- **Out-of-sample:** Yes (learned projection applies to new points)
- **Key limitation:** Semantic similarity in embedding spaces is inherently non-linear. PCA treats Euclidean distances as meaningful, but in high-dimensional NLP embeddings, cosine similarity on a hypersphere is the true metric. PCA flattens this, often merging semantically distinct regions.
- **When to use:** As a preprocessing step before t-SNE/UMAP (reducing 768-dim BERT embeddings to 50-dim before nonlinear reduction), or when speed is paramount and approximate structure suffices.

### t-SNE (t-distributed Stochastic Neighbor Embedding)
- **Type:** Non-linear, neighbor-based
- **Preserves:** Local neighborhood structure (pairwise similarities between nearby points)
- **Local fidelity:** Excellent — nearby phrases in embedding space stay nearby in 2D
- **Global fidelity:** Poor — inter-cluster distances are unreliable (Wattenberg et al., Distill 2016: "distances between well-separated clusters in a t-SNE plot may mean nothing")
- **Speed:** O(n²) naive, O(n·log n) with Barnes-Hut; still slow at >100k points
- **Out-of-sample:** No — must recompute from scratch for new data
- **Key parameters:**
  - **Perplexity** (5–50 typical): Controls the balance between local and global attention. Low perplexity → fragmented local clusters; high → merged blobs. Distill (2016) showed that no single perplexity value captures both local and global geometry for mixed-density data.
  - **Iterations** (1000 default, often needs 3000+): Belkina et al. (Nature Comms, 2019) showed standard 1000 iterations fail on large datasets; opt-SNE achieves superior results with dataset-scaled learning rates.
  - **Early exaggeration** (factor α=4–12): Critical for forming initial cluster structure. Too short EE → fragmented islands.
- **Key limitation:** Non-parametric — no mapping function, so you can't project new queries without re-running. Cluster sizes are meaningless (t-SNE equalizes densities by design). Non-deterministic across runs.
- **Citation:** van der Maaten & Hinton, JMLR 2008; Belkina et al., Nature Comms 2019; Wattenberg et al., Distill 2016.

### UMAP (Uniform Manifold Approximation and Projection)
- **Type:** Non-linear, manifold-based (Riemannian geometry + algebraic topology)
- **Preserves:** Both local neighborhood structure AND more global structure than t-SNE
- **Local fidelity:** Excellent — comparable to t-SNE
- **Global fidelity:** Significantly better than t-SNE — inter-cluster distances are more meaningful
- **Speed:** O(n^1.14) empirically — 10–100x faster than t-SNE at scale
- **Out-of-sample:** Yes (parametric variant available; transform() method in the Python library)
- **Key parameters:**
  - **n_neighbors** (5–50): Analogous to perplexity. Low → very local focus; high → more global. Default 15 is a solid starting point for semantic data.
  - **min_dist** (0.0–1.0): Controls how tightly points pack. Lower → tighter clusters (better for discrete grid mapping). 0.1 default; for grid-based semantic folding, 0.0–0.05 may be better.
  - **metric**: Can use cosine distance directly, which is the correct metric for NLP embeddings (t-SNE uses Euclidean by default).
- **Key advantage for semantic spaces:** UMAP's theoretical grounding in topological data analysis means it preserves the "shape" of the manifold — both local neighborhoods and the relative positioning of clusters. McInnes et al. (2018) state: "arguably preserves more of the global structure with superior run time performance."
- **Citation:** McInnes, Healy & Melville, arXiv:1802.03426, 2018.

---

## 2. Comparative Analysis for Semantic Embedding Projection

| Criterion | PCA | t-SNE | UMAP | Kernel PCA | Isomap | LLE |
|---|---|---|---|---|---|---|
| Local semantic similarity | Poor | Excellent | Excellent | Moderate | Good | Good |
| Global cluster positioning | Good | Poor | Good | Moderate | Good | Poor |
| Speed (10k–100k phrases) | Trivial | Slow | Fast | Moderate | Slow | Moderate |
| Out-of-sample projection | Yes | No | Yes* | Yes | No | No |
| Cosine metric support | Implicit | No** | Yes | Kernel-dependent | No | No |
| Deterministic | Yes | No | Yes*** | Yes | Yes | Yes |
| Scalability to 100k+ | Excellent | Poor | Good | Moderate | Poor | Poor |

*UMAP has `transform()` for new data; quality degrades slightly.
**t-SNE uses Euclidean by default; pre-computing a distance matrix with cosine is possible but slow.
***UMAP with `random_state` set is deterministic.

---

## 3. Hybrid and Alternative Approaches

### PCA → t-SNE/UMAP Pipeline (Recommended for NLP)
The standard practice in the literature (Kobak & Berens, Nature Comms 2019; Becht et al., Nature Biotech 2019) is:
1. Reduce from high-dim (e.g., 768) to ~50 dims with PCA (removes noise, speeds up)
2. Apply t-SNE or UMAP on the 50-dim PCA output

This two-step approach is both faster and produces better embeddings than applying nonlinear reduction directly to raw high-dimensional data.

### UMAP + Discrete Grid Mapping
A UMAP-based semantic folding pipeline would:
1. UMAP with `n_neighbors=15, min_dist=0.0, metric='cosine'` → 2D continuous coordinates
2. Map to 64×64 grid with Gaussian smoothing (σ=1.5) and Morton encoding

This should preserve MORE global semantic relationships than t-SNE while maintaining local structure.

### Other Alternatives
- **TriMap** (Amid & Warmuth, NeurIPS 2019): Preserves global structure better than t-SNE, competitive with UMAP. Worth testing.
- **PaCMAP** (Wang et al., 2020): Specifically designed to preserve both local and global structure. Claims to outperform UMAP on global structure.
- **FIt-SNE** (Linderman et al., Nature Methods 2019): FFT-accelerated t-SNE, 10x faster than Barnes-Hut, enabling 1M+ points. Same quality as standard t-SNE.
- **openTSNE** (Poličar et al., 2019): High-performance t-SNE with better defaults and support for precomputed metrics.
- **Autoencoders**: Can learn a parametric mapping from high-dim to 2D, supporting out-of-sample projection. More complex to train but potentially better quality for domain-specific data.

---

## 4. Empirical Evidence from Literature

### UMAP Outperforms t-SNE on Global Structure
- **Kobak & Berens (Nature Comms, 2019)**: Showed t-SNE with proper parameter tuning (PCA init, early exaggeration) dramatically improves quality, but global distances remain unreliable.
- **Becht et al. (Nature Biotech, 2019)**: Comprehensive benchmark of UMAP, t-SNE, and others on single-cell RNA-seq data. Found UMAP best balances local and global structure preservation.
- **McInnes et al. (2018)**: UMAP paper demonstrates superior runtime and comparable or better visualization quality vs t-SNE on multiple benchmarks.

### t-SNE Limitations on Semantic Data
- **Wattenberg et al. (Distill, 2016)**: "Distances between well-separated clusters in a t-SNE plot may mean nothing." Cluster sizes are meaningless. Low perplexity creates artificial clusters in noise.
- **Belkina et al. (Nature Comms, 2019)**: Standard t-SNE fails on datasets >500k points; requires careful EE and learning rate tuning (opt-SNE).

### PCA Inadequacy for Semantic Embeddings
- Multiple studies show PCA projections of NLP embeddings lose semantic neighborhood structure. Cosine similarity manifolds in 768-dim space are highly non-linear.

---

## 5. Recommendation for Semantic Folding Pipeline

**Switch from t-SNE to UMAP** as the primary dimensionality reduction method, based on:

1. **Better global structure preservation**: The semantic folding pipeline uses a 64×64 discrete grid. Global cluster positions directly affect which phrases land in nearby cells, which determines retrieval quality. UMAP's superior global fidelity should improve MRR.

2. **Cosine metric support**: NLP embeddings live on a hypersphere; cosine distance is the correct metric. UMAP supports this natively; t-SNE does not.

3. **10–100x faster**: At 10k–100k phrases, UMAP completes in seconds vs minutes for t-SNE.

4. **Deterministic**: Reproducible results across runs (set `random_state`).

5. **Out-of-sample projection**: Can project new query embeddings without re-running the full pipeline.

**Suggested parameters for initial test:**
```python
import umap
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,      # local neighborhood size
    min_dist=0.0,        # tight packing for grid mapping
    metric='cosine',     # correct for NLP embeddings
    random_state=42
)
```

**Risk:** If current MRR is specifically tuned to t-SNE's local-focus behavior, switching methods may require re-tuning spreading parameters. Start with the same grid_size=64, sigma=1.5, top_percent=0.10 and measure.

---

## 6. Key References

1. McInnes, L., Healy, J. & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv:1802.03426.
2. van der Maaten, L. & Hinton, G. (2008). "Visualizing data using t-SNE." JMLR 9, 2579–2605.
3. Wattenberg, M., Viégas, F. & Johnson, I. (2016). "How to Use t-SNE Effectively." Distill. doi:10.23915/distill.00002.
4. Belkina, A.C. et al. (2019). "Automated optimized parameters for T-distributed stochastic neighbor embedding improve visualization and analysis of large datasets." Nature Communications 10, 5415.
5. Kobak, D. & Berens, P. (2019). "The art of using t-SNE for single-cell transcriptomics." Nature Communications 10, 5416.
6. Becht, E. et al. (2019). "Dimensionality reduction for visualizing single-cell data using UMAP." Nature Biotechnology 37, 38–44.
7. Linderman, G.C. et al. (2019). "Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data." Nature Methods 16, 243–245.
8. Amid, E. & Warmuth, M.K. (2019). "TriMap: Example-based Dimension Embedding." NeurIPS 2019.
9. Wang, Y. et al. (2020). "Parametric Contrastive Learning." arXiv:2005.09221 (PaCMAP).
10. Kobak, D. & Linderman, G.C. (2021). "Initialization is critical for preserving global data structure in both t-SNE and UMAP." Nature Biotechnology 39, 156–157.

---

## 7. Empirical Validation: UMAP vs t-SNE on Belebele (2026-06-17)

*Full details: `docs/reports/belebele/umap_implementation_analysis.md`*

### Benchmark Results (50 queries, 64×64 grid, sigma=1.5, top%=0.10)

| Metric | t-SNE | UMAP | Delta |
|--------|-------|------|-------|
| **MRR** | **0.880** | 0.800 | -9.1% |
| **AP** | **0.880** | 0.800 | -9.1% |
| **P@1** | **0.880** | 0.780 | -11.4% |
| **R@5** | 0.880 | **0.820** | -6.8% |
| Found at rank 1 | **44/50** (88%) | 39/50 (78%) | -5 queries |
| Not found | 6/50 (12%) | **9/50** (18%) | +3 failures |

### Key Findings

1. **t-SNE outperforms UMAP** on Belebele by 8% MRR — confirming the theoretical Section 5 recommendation was wrong for this task type.
2. **UMAP's global structure preservation** does not help when the task is local phrase matching. t-SNE's local focus is better for retrieval.
3. **10-query UMAP results were misleading** (MRR=1.000) due to small sample bias — the 50-query benchmark revealed the true gap.
4. **UMAP's speed advantage** (10-100x faster) remains valuable for large datasets (>10k contexts).
5. **Grid collision rate** with UMAP was excellent (0.3%), so the issue is not grid mapping quality but semantic neighborhood structure.

### Updated Recommendation

**Keep t-SNE as default** for datasets ≤10k contexts. Switch to UMAP only when:
- Dataset size > 10k contexts (speed becomes critical)
- Out-of-sample projection is needed (UMAP supports `transform()`)
- Global structure preservation is prioritized over local matching
