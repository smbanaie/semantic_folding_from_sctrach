# UMAP Implementation Analysis & Benchmark Results

**Date:** 2026-06-17
**Context:** Analysis of semantic_space.py UMAP implementation and benchmark comparison

---

## Implementation Issues Found & Fixed

### 1. No PCA Pre-reduction Before UMAP
**Issue:** t-SNE had TruncatedSVD pre-reduction for dim>100 (line 409-412), but UMAP didn't. For high-dimensional phrase vectors, this slowed down UMAP significantly.

**Fix:** Added TruncatedSVD pre-reduction to 100 dimensions when input dim > 100, matching t-SNE behavior.

### 2. CLI Default min_dist=0.25 Too Loose
**Issue:** The CLI default `--min-dist` was 0.25, but research recommends 0.0 for tight grid mapping. The function default was 0.1.

**Fix:** Changed CLI default to 0.0, matching the function default and research recommendation.

### 3. n_neighbors Clamping Too Aggressive
**Issue:** Clamping used `n_samples // 2`, which was too generous for small datasets.

**Fix:** Changed to `n_samples // 3`, more conservative and matching t-SNE perplexity clamping behavior.

### 4. Benchmark Runner Missing UMAP Support
**Issue:** Step 3 invocation in generic_benchmark.py only passed t-SNE parameters (`--perplexity`, `--tsne-iter`) and never passed `--method umap`.

**Fix:** 
- Added `method`, `umap_n_neighbors`, `umap_min_dist`, `umap_metric` to PIPELINE_DEFAULTS
- Updated step 3 invocation to pass method and method-specific params
- Added CLI args for `--method`, `--umap-n-neighbors`, `--umap-min-dist`, `--umap-metric` to both `index` and `all` subparsers

---

## Benchmark Results

### t-SNE Baseline (50 queries, benchmark_20260615_033122)
| Metric | Value |
|--------|-------|
| MRR | 0.88 |
| AP | 0.88 |
| P@1 | 0.88 |
| R@5 | 0.88 |
| Found at rank 1 | 44/50 (88%) |
| Not found | 6/50 (12%) |

### UMAP (50 queries, benchmark_20260617_144934)
| Metric | Value |
|--------|-------|
| MRR | 0.80 |
| AP | 0.80 |
| P@1 | 0.78 |
| R@5 | 0.82 |
| Found at rank 1 | 39/50 (78%) |
| Not found | 9/50 (18%) |

### UMAP (10 queries, benchmark_20260617_142719) — Small Sample Bias
| Metric | Value |
|--------|-------|
| MRR | 1.000 |
| AP | 1.000 |

### Comparison
- **t-SNE outperforms UMAP** on Belebele 50-query benchmark: MRR 0.88 vs 0.80
- The 10-query UMAP result (MRR=1.000) was misleading due to small sample size
- UMAP has 3 more failures (9 vs 6) on the full benchmark

---

## Key Findings

1. **t-SNE outperforms UMAP** on Belebele dataset with identical grid parameters (64x64, sigma=1.5, top%=0.10)

2. **UMAP's global structure preservation** may not help when the task is local phrase matching — t-SNE's local focus is actually better for this use case

3. **min_dist=0.0** creates tight clusters but may over-compress some semantic neighborhoods

4. **Cosine metric** is correct for NLP embeddings, but t-SNE with Euclidean on L2-normalized vectors approximates cosine similarity well enough

5. **Grid collision rate** with UMAP: 0.3% (3 collisions out of 926 contexts) — excellent, but not the bottleneck

---

## Recommendation

Keep **t-SNE as the default** for Belebele. UMAP can be used as an alternative when:
- Dataset size > 10k contexts (UMAP is 10-100x faster)
- Global structure preservation is more important than local matching
- Out-of-sample projection is needed

The config should remain `method: "tsne"` for production use.
