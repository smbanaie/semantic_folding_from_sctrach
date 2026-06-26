# Chapter 6: Similarity Metrics for Sparse Distributed Representations

## 6.1 Introduction

The choice of similarity metric fundamentally affects retrieval performance. For Sparse Distributed Representations (SDRs), the metric must account for the high-dimensional, sparse, and often binary nature of the vectors. This chapter provides a comprehensive analysis of similarity metrics applicable to SF, with mathematical formulations, theoretical properties, and empirical validation.

## 6.2 Theoretical Framework

### 6.2.1 Metric Properties

A valid similarity metric $\text{sim}: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ should satisfy:

| Property | Definition | Importance |
|----------|------------|------------|
| **Symmetry** | $\text{sim}(x, y) = \text{sim}(y, x)$ | Bidirectional comparison |
| **Boundedness** | $\text{sim}(x, y) \in [-1, 1]$ or $[0, 1]$ | Normalized comparison |
| **Self-similarity** | $\text{sim}(x, x) = \max$ | Consistency |
| **Triangle inequality** | $\text{sim}(x, z) \geq \text{sim}(x, y) + \text{sim}(y, z) - 1$ | Metric space property |

**Note**: Retrieval is inherently **asymmetric** — we want to score how well a document $D$ satisfies a query $Q$, not vice versa. Standard symmetric metrics may not capture this directionality.

### 6.2.2 SDR-Specific Considerations

For SDRs with dimensions $d = g^2$ (e.g., 4,096 for 64×64 grid) and density $\rho \approx 0.10$:

1. **Sparsity**: Only 10-25% of bits are active → efficient set-based operations
2. **Binary/Continuous**: Fingerprints can be binary $\{0,1\}$ or continuous $\mathbb{R}^+$
3. **Locality preservation**: Morton encoding ensures spatial proximity → Hamming-like distance

## 6.3 Cosine Similarity (Default)

### 6.3.1 Formulation

Given query fingerprint $\mathbf{q} \in \mathbb{R}^{g^2}$ and document fingerprint $\mathbf{d} \in \mathbb{R}^{g^2}$:

$$\text{sim}_{\cos}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \cdot \|\mathbf{d}\|_2}$$

With L2-normalized query ($\|\mathbf{q}\|_2 = 1$):

$$\text{sim}_{\cos}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{d}\|_2}$$

### 6.3.2 Properties

| Property | Value | Interpretation |
|----------|-------|----------------|
| Range | $[-1, 1]$ | 1 = identical, 0 = orthogonal, -1 = opposite |
| Symmetry | Yes | $\text{sim}(q, d) = \text{sim}(d, q)$ |
| Computational cost | $O(d)$ | Linear in dimensionality |
| Sensitivity to magnitude | Yes | Higher activation → higher score |

### 6.3.3 Advantages for SF

1. **Leverages activation magnitudes**: Continuous-valued SF fingerprints encode semantic strength
2. **Well-understood**: Standard metric in information retrieval
3. **Efficient**: Fast sparse matrix operations

### 6.3.4 Limitations

1. **Ignores bit position**: Two documents with same number of active bits in different positions score identically
2. **Sparsity bias**: Very sparse vectors tend to have higher cosine similarity
3. **Loses asymmetry**: $\text{sim}(q, d) = \text{sim}(d, q)$ even though retrieval is asymmetric

### 6.3.5 Empirical Performance

| Dataset | Cosine MRR | Best Alternative | Δ |
|---------|------------|------------------|---|
| Belebele | 0.880 | Jaccard: 0.840 | +4.8% |
| PubMedQA | 0.955 | Dice: 0.950 | +0.5% |

**Conclusion**: Cosine is preferred for float-valued SF fingerprints at ~7.8% density.

## 6.4 Dice Coefficient (Sørensen–Dice)

### 6.4.1 Formulation

$$D(\mathbf{q}, \mathbf{d}) = \frac{2|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{A}| + |\mathcal{B}|}$$

where $\mathcal{A} = \{i : q_i > 0\}$ and $\mathcal{B} = \{i : d_i > 0\}$ are active bit sets.

### 6.4.2 Properties

| Property | Value | Interpretation |
|----------|-------|----------------|
| Range | $[0, 1]$ | 1 = identical active sets |
| Symmetry | Yes | Symmetric |
| Bias | Toward smaller set | Favors queries over documents |
| Binary equivalence | $D = \cos$ for binary vectors | Mathematically equivalent |

### 6.4.3 Advantages

1. **Set-based**: Ignores activation magnitudes, focuses on bit positions
2. **Biased toward query**: Appropriate for asymmetric retrieval
3. **Efficient**: Set intersection operations on sparse vectors

### 6.4.4 Limitations

1. **Ignores magnitudes**: Cannot distinguish strong from weak activations
2. **Equivalent to cosine for binary**: No advantage over cosine for binarized fingerprints

### 6.4.5 When to Use

- Binary fingerprints ($\{0,1\}^d$)
- Short queries where query-document asymmetry matters
- When activation magnitudes are not informative

## 6.5 Jaccard Index

### 6.5.1 Formulation

$$J(\mathbf{q}, \mathbf{d}) = \frac{|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{A} \cup \mathcal{B}|}$$

### 6.5.2 Properties

| Property | Value | Interpretation |
|----------|-------|----------------|
| Range | $[0, 1]$ | 1 = identical, 0 = disjoint |
| Symmetry | Yes | Symmetric |
| Shared absence | Excludes $M_{00}$ | Appropriate for sparse data |

### 6.5.3 Theoretical Justification

Jaccard is the standard metric for binary sets (Broder, 1997) because:
1. Excludes shared absences ($M_{00}$) which are uninformative for sparse data
2. Normalizes by union size, accounting for different set sizes
3. Has strong theoretical properties for MinHash-based approximate similarity

### 6.5.4 Relationship to Other Metrics

For binary vectors with active bit counts $|\mathcal{A}| = a$, $|\mathcal{B}| = b$, intersection $|\mathcal{A} \cap \mathcal{B}| = c$:

$$J = \frac{c}{a + b - c}$$

$$D = \frac{2c}{a + b}$$

$$O = \frac{c}{\min(a, b)}$$

### 6.5.5 Empirical Performance

| Dataset | Jaccard MRR | Cosine MRR | Δ |
|---------|-------------|------------|---|
| Belebele | 0.840 | 0.880 | -4.5% |
| SciFact | 0.720 | 0.755 | -4.6% |

**Conclusion**: Jaccard underperforms cosine for float-valued SF fingerprints.

## 6.6 Overlap Coefficient

### 6.6.1 Formulation

$$O(\mathbf{q}, \mathbf{d}) = \frac{|\mathcal{A} \cap \mathcal{B}|}{\min(|\mathcal{A}|, |\mathcal{B}|)}$$

### 6.6.2 Properties

| Property | Value | Interpretation |
|----------|-------|----------------|
| Range | $[0, 1]$ | 1 = query is subset of document |
| Symmetry | Yes | Symmetric |
| Robustness | Maximum | Immune to set size differences |

### 6.6.3 Theoretical Justification

Overlap coefficient is optimal when:
1. Queries are shorter than documents (typical in IR)
2. Gold document contains all query concepts plus additional context
3. Set size differences are large

Returns 1.0 when $\mathcal{A} \subseteq \mathcal{B}$ — ideal for passage retrieval where the gold passage contains all query concepts.

### 6.6.4 Empirical Performance

| Dataset | Overlap MRR | Cosine MRR | Δ |
|---------|-------------|------------|---|
| Belebele | 0.860 | 0.880 | -2.3% |
| MuSiQue | 0.470 | 0.453 | +3.8% |

**Interesting**: Overlap outperforms cosine on MuSiQue (multi-hop) where query-document size asymmetry is extreme.

## 6.7 IDF-Weighted Intersection

### 6.7.1 Formulation

$$S_{\text{idf}}(\mathbf{q}, \mathbf{d}) = \frac{\sum_{i \in \mathcal{A} \cap \mathcal{B}} w_i^{\text{idf}}}{\sum_{i \in \mathcal{A}} w_i^{\text{idf}}}$$

### 6.7.2 Properties

| Property | Value | Interpretation |
|----------|-------|----------------|
| Range | $[0, 1]$ | 1 = all query bits match with weight |
| Asymmetry | Inherently asymmetric | Query-weighted |
| Rare term emphasis | Yes | High IDF → high weight |

### 6.7.3 Theoretical Justification

Analogous to BM25's term frequency saturation (Sparck Jones, 1972):
- Rare concepts (high IDF) contribute more to the score
- Common concepts (low IDF) are down-weighted
- Matches the intuition that rare terms are more discriminative

### 6.7.4 Implementation Requirement

Requires mapping from bit positions back to the concepts that activated them, maintained in the term-context matrix.

### 6.7.5 Empirical Performance

| Dataset | IDF MRR | Cosine MRR | Δ |
|---------|---------|------------|---|
| Belebele | 0.870 | 0.880 | -1.1% |
| NQ-REaR | 0.590 | 0.574 | +2.8% |

**Interesting**: IDF weighting outperforms cosine on NQ-REaR (factoid retrieval) where rare entity terms are critical.

## 6.8 Asymmetric Scoring

### 6.8.1 Motivation

Standard set similarity is symmetric: $J(A,B) = J(B,A)$. But retrieval is inherently asymmetric — we want to score how well a document satisfies a query.

### 6.8.2 Query Containment (Recall-like)

$$S_{\text{contain}}(\mathbf{q}, \mathbf{d}) = \frac{|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{A}|}$$

Measures what fraction of query concepts are present in the document.

**Example**: If $\mathcal{A} = \{\text{NBA, basketball, oldest}\}$ and $\mathcal{B} = \{\text{NBA, basketball, history, Eddie Gottlieb}\}$, then $S_{\text{contain}} = 2/3$.

### 6.8.3 Document Coverage (Precision-like)

$$S_{\text{cover}}(\mathbf{q}, \mathbf{d}) = \frac{|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{B}|}$$

Measures what fraction of the document's concepts are relevant to the query.

### 6.8.4 Combined Asymmetric Score

$$S_{\text{asym}}(\mathbf{q}, \mathbf{d}) = \alpha \cdot S_{\text{contain}}(\mathbf{q}, \mathbf{d}) + (1 - \alpha) \cdot S_{\text{cover}}(\mathbf{q}, \mathbf{d})$$

where $\alpha \in [0, 1]$ is the containment weight (default $\alpha = 0.7$).

**Analogy**: This is analogous to the $F_\beta$ score in classification, where $\alpha$ controls the precision-recall trade-off.

### 6.8.5 Empirical Performance

| α | Belebele MRR | PubMedQA MRR |
|---|--------------|--------------|
| 0.5 | 0.860 | 0.950 |
| 0.7 | 0.870 | 0.955 |
| 0.9 | 0.850 | 0.945 |

**Conclusion**: $\alpha = 0.7$ (favoring recall) provides best overall performance.

## 6.9 Score Normalization

### 6.9.1 The Score Compression Problem

When all documents score within a narrow range (e.g., 0.034–0.051 on NQ-REaR), fine-grained ranking becomes impossible.

### 6.9.2 Z-Score Normalization

$$S_z = \frac{S - \mu_S}{\sigma_S}$$

**Advantages:**
- Adapts to each query's score distribution
- Amplifies signal without amplifying noise
- Differentiable (enabling end-to-end training)

### 6.9.3 Percentile Rank

$$S_{\text{pct}} = \frac{\text{rank}(S)}{N}$$

**Advantages:**
- Distribution-free
- Handles any score distribution
- Robust to outliers

### 6.9.4 Min-Max Normalization

$$S_{\text{norm}} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$

**Advantages:**
- Maps to $[0, 1]$ range
- Intuitive interpretation

**Disadvantage:** More sensitive to outliers than z-score.

### 6.9.5 Ranking Equivalence

For ranking-only tasks (without learning), z-score, percentile rank, and min-max produce **identical rankings** since they are monotonic transformations. The choice only matters when:
1. Absolute score values are needed
2. Combining scores from different queries
3. Training a downstream model

## 6.10 LambdaMART Re-ranking Features

### 6.10.1 Feature Design

The 35-feature vector captures complementary aspects of query-document similarity:

| Category | Features | Count | Purpose |
|----------|----------|-------|---------|
| Binary similarity | Jaccard, Dice, overlap, Hamming, cosine | 5 | Set overlap from different angles |
| Asymmetric | Containment, coverage, IDF-weighted intersection | 3 | Directionality of retrieval |
| Bit-density | popcount(Q), popcount(D), intersection, union, mismatch, density(Q), density(D) | 8 | Document length and specificity |
| Block histogram | Per-block Jaccard (16 blocks of 256 bits) | 16 | Spatial distribution across grid |
| Auxiliary | BM25 score, query length, document length | 3 | Lexical matching signal |

### 6.10.2 Feature Importance (50-query Belebele training)

| Rank | Feature | Gain | Category |
|------|---------|------|----------|
| 1 | cosine | 730.3 | Binary similarity |
| 2 | bm25_score | 714.7 | Auxiliary |
| 3 | block_13_jaccard | 42.3 | Block histogram |
| 4 | overlap | 37.9 | Asymmetric |
| 5 | block_12_jaccard | 35.4 | Block histogram |

**Key finding**: Cosine and BM25 dominate (730 and 714 gain respectively), but block histogram features contribute significantly when trained on 50+ queries.

### 6.10.3 SiDR Validation

Mallia et al. (2022) demonstrate that binary sparse first-stage + learned re-ranking achieves 49.5% top-1 on NQ with $m=20$ candidates, matching full neural retrieval (49.1%).

## 6.11 Metric Selection Guide

### 6.11.1 Decision Tree

```
Is your fingerprint binary {0,1}?
├─ Yes → Use Jaccard or Dice
│        ├─ Short queries? → Dice (query-biased)
│        └─ Balanced comparison? → Jaccard
└─ No (float-valued) → Use Cosine
         ├─ Need asymmetry? → Asymmetric scoring (α=0.7)
         ├─ Rare terms critical? → IDF-weighted intersection
         └─ Multi-hop tasks? → Overlap coefficient
```

### 6.11.2 Summary Table

| Metric | Best For | Limitation | SF Performance |
|--------|----------|------------|----------------|
| **Cosine** | Float-valued fingerprints | Ignores bit position | Best overall |
| **Dice** | Binary fingerprints, short queries | Biased toward smaller set | Good for binary |
| **Jaccard** | Binary fingerprints, balanced comparison | Sensitive to union size | Moderate |
| **Overlap** | Subset matching, asymmetric retrieval | Ignores document size | Best for multi-hop |
| **IDF-weighted** | Rare term matching | Requires concept mapping | Best for factoid |

## 6.12 Conclusion

The choice of similarity metric depends on fingerprint characteristics and task requirements:

1. **Default**: Cosine similarity for float-valued SF fingerprints at ~7.8% density
2. **Binary fingerprints**: Jaccard or Dice for truly binarized representations
3. **Asymmetric tasks**: Overlap coefficient or asymmetric scoring (α=0.7)
4. **Rare term matching**: IDF-weighted intersection for factoid retrieval
5. **Maximum quality**: LambdaMART re-ranking with 35-feature vector

The optimal metric is task-dependent, and no single metric dominates across all datasets. This underscores the importance of empirical validation when deploying SF in new domains.

## References

- Broder, A. Z. (1997). On the resemblance and containment of documents. *Compression and Complexity of Sequences*.
- Mallia, A., et al. (2022). Learning sparse indexes for text retrieval. *arXiv:2405.01924*.
- Sparck Jones, K. (1972). A statistical interpretation of term specificity. *Journal of Documentation*, 28(1), 11–21.
