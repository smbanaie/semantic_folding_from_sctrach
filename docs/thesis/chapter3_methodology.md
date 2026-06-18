# Chapter 3: Methodology — Semantic Folding Pipeline

## 3.1 Overview

Semantic Folding is an unsupervised retrieval architecture that represents words, phrases, and documents as sparse binary vectors (Sparse Distributed Representations, SDRs) over a fixed 2D semantic grid. The pipeline proceeds through six stages:

1. **Phrase Extraction** — Domain vocabulary built from raw text via noun-chunk parsing and n-gram discovery
2. **Term-Context Matrix** — Sparse co-occurrence matrix of phrases × contexts, weighted by TF-IDF
3. **Semantic Space** — Contexts embedded onto a g×g integer grid via t-SNE dimensionality reduction
4. **Phrase Fingerprints** — Each phrase assigned a centroid on the grid, convolved with Gaussian kernel, linearized via Morton encoding
5. **Document Fingerprints** — Phrase fingerprints accumulated onto reconstructed 2D grid, sparsified via topology-preserving peak selection
6. **Query Processing** — Query decomposed into phrases, weighted, converted to SDR, optionally spread, scored against document SDRs

## 3.2 Key Design Decisions

### 3.2.1 Grid Resolution (64×64)

The semantic grid uses a 64×64 resolution (4,096 cells). This provides:
- Sufficient capacity for ~1,000 unique phrases
- 5-15% fingerprint density (optimal for discrimination)
- Fast computation (~2-5 min for Steps 1-5)

Grid=128 was tested but reduced MRR by 5.3% due to over-partitioning.

### 3.2.2 Morton Z-order Encoding

Phrases are mapped to 1D fingerprints using Morton (Z-order) encoding, which preserves 2D spatial locality. This ensures that semantically similar phrases (adjacent on the grid) have similar fingerprint indices.

### 3.2.3 Gaussian Smoothing (σ=1.5)

Phrase fingerprints are convolved with a 2D Gaussian kernel before peak detection. This:
- Creates soft activation regions around phrase centroids
- Makes fingerprints robust to small coordinate shifts
- Enables partial matching between similar phrases

### 3.2.4 Document Normalization (L2)

Document fingerprints are normalized using L2 norm (Euclidean distance). This:
- Treats all documents equally regardless of fingerprint length
- Prevents longer documents from dominating similarity scores
- Improved MRR by +4.0% on Belebele vs. sqrt(nnz) normalization

### 3.2.5 t-SNE Perplexity (50)

The perplexity parameter controls the balance between local and global structure in the t-SNE embedding. Perplexity=50:
- Creates tighter local clusters for fine-grained discrimination
- Improved MRR by +4.0% on Belebele and +1.5% on PubMedQA vs. perplexity=30

## 3.3 Scoring Formula

Given a query fingerprint **q** and document fingerprint **d**, the score is:

\[
\text{score}(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}^T}{\|\mathbf{q}\|_2 \cdot \|\mathbf{d}\|_2}
\]

Where:
- **q** is L2-normalized query fingerprint
- **d** is L2-normalized document fingerprint
- The denominator uses L2 norms for both query and document

## 3.4 Optional Enhancements

### 3.4.1 Hybrid SF+BM25 Scoring

Combines semantic folding with BM25 lexical matching:

\[
\text{score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{score}_{\text{SF}}(q, d) + (1 - \alpha) \cdot \text{score}_{\text{BM25}}(q, d)
\]

**Effect:** +2.0% MRR on Belebele, -3.1% on PubMedQA. Use as optional flag.

### 3.4.2 Query Expansion with Glossary

Expands query terms using domain-specific synonym mappings. **Effect:** No improvement on either dataset. Skip.

### 3.4.3 TF-IDF Re-ranking

Post-SF re-ranking with TF-IDF scores. **Effect:** No improvement. Skip.

## 3.5 Comparison to HippoRAG

| Aspect | HippoRAG | Semantic Folding |
|--------|---------|------------------|
| Index representation | Dense embeddings + KG | Sparse SDRs on 2D grid |
| Retrieval mechanism | PageRank + dense retrieval | Dot-product over SDRs |
| Requires LLM | Yes (OpenIE) | No |
| Interpretability | KG paths | Spatial grid positions |
| Computational cost | High | Low |

## 3.6 Limitations

1. **Computational cost:** ~2-5 min for indexing, ~15s per query
2. **Grid size sensitivity:** Optimal for 20-passage corpora; larger pools need scaling
3. **t-SNE stochasticity:** Results depend on random seed (fixed at 42)
4. **Binary relevance:** No graded relevance judgments

---

*References: Furnas et al. (1987), Harris (1954), Hawkins & George (2006), Purdy (2016), Rocchio (1971), Xu & Croft (1996)*
