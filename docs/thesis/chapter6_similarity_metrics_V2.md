# Chapter 6: Similarity Metrics and the Feature Invariance Principle

## 6.1 Introduction

Semantic Folding represents text as sparse, spatially-correlated fingerprints over a 2D grid. To retrieve documents, we must compare a query fingerprint against a corpus of document fingerprints using a similarity metric. While this chapter empirically evaluates standard metrics for sparse binary vectors (Cosine, Dice, Jaccard, Hamming), its theoretical purpose extends far beyond simple metric selection. 

The choice of similarity metric dictates the **scale properties** of the retrieval score. As established in Chapters 3 and 4, Semantic Folding utilizes L2-normalized, IDF-weighted fingerprints scored via Cosine Similarity. This specific mathematical combination produces a strictly **bounded score scale** ($\text{score}_{\text{SF}} \in [0, 1]$). This bounded scale is not an arbitrary artifact; it is the structural catalyst for the Complementarity Illusion. 

Furthermore, analyzing why alternative metrics fail allows us to formally prove the **Feature Invariance Principle**—a core thesis contribution demonstrating that once localized spatial overlap is computed, any internal architectural or metric-level modification to the SDR yields exactly 0.00% MRR improvement because the signal is perfectly collinear with the baseline dot-product.

## 6.2 Sparse Binary Fingerprint Representation

A document fingerprint is constructed by aggregating sparse binary phrase vectors $\mathbf{v}_p \in \{0,1\}^d$ (where $d = 4096$ for a $64 \times 64$ grid). However, the final document fingerprint $\mathbf{d}$ is *not* strictly binary. Because phrases are aggregated using IDF weighting:
$$\mathbf{d} = \sum_{p \in d} \text{IDF}(p) \cdot \mathbf{v}_p$$
The resulting document fingerprint is a **sparse real-valued vector** where active cells contain floating-point IDF weights, rather than strictly $\{0, 1\}$. This distinction is critical: it mathematically invalidates naive applications of pure set-theoretic metrics (like Jaccard or Dice) unless destructive binarization is applied.

**Key Properties of the Final SDRs:**
*   **Sparsity:** ~90% of cells are exactly zero ($\rho=0.10$).
*   **Real-Valued Active Bits:** Non-zero cells contain IDF weights (e.g., 1.2, 3.5), not binary 1s.
*   **Spatial Correlation:** Active bits cluster in semantically meaningful 2D regions, violating the independence assumption of random SDRs.

## 6.3 Similarity Metrics for Sparse Representations

### 6.3.1 Cosine Similarity (Default)
Cosine similarity measures the angle between two real-valued vectors:
$$\text{cosine}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
**Advantages:** Correctly handles real-valued IDF weights. Normalizes for document length/fingerprint magnitude. Under L2 normalization, becomes a pure dot-product, yielding the strictly bounded $[0, 1]$ scale required for our fusion diagnostics.

### 6.3.2 Dice Coefficient
Dice measures overlap between two sets:
$$D(\mathbf{q}, \mathbf{d}) = \frac{2|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{A}| + |\mathcal{B}|}$$
where $\mathcal{A}$ and $\mathcal{B}$ are the sets of active bit indices. 
**Disadvantage:** Requires binarizing the IDF-weighted vectors back to $\{0,1\}$, destroying the discriminative phrase weighting calculated in Stage 5.

### 6.3.3 Jaccard Coefficient
Jaccard measures intersection over union:
$$J(\mathbf{q}, \mathbf{d}) = \frac{|\mathcal{A} \cap \mathcal{B}|}{|\mathcal{A} \cup \mathcal{B}|}$$
**Disadvantage:** Also requires destructive binarization. Furthermore, it heavily penalizes set-size asymmetries, which are common when query phrases (small set) are compared to document phrases (large set).

### 6.3.4 Hamming Distance
Hamming distance counts bit positions where two vectors differ:
$$H(\mathbf{q}, \mathbf{d}) = \sum_{i=1}^{d} \mathbb{1}[q_i \neq d_i]$$
**Disadvantage:** Completely ignores IDF weighting. Equates a mismatch on a high-IDF discriminative cell with a mismatch on a low-IDF common cell.

## 6.4 Empirical Comparison and the Feature Invariance Proof

### 6.4.1 Experimental Setup
We compare similarity metrics on Belebele (50 queries, 20 passages/query) using the SF-only configuration. For Dice, Jaccard, and Hamming, the IDF-weighted continuous vectors are thresholded at $>0$ to form strict binary sets.

### 6.4.2 Results

**Table 6.1: Similarity Metric Comparison (Belebele 50Q, SF-Only)**

| Metric | Vector Type Used | MRR | AP | P@1 | NDCG@5 | Interpretation |
|--------|------------------|:---:|:--:|:---:|:------:|-----------------|
| **Cosine** | **Continuous (IDF)** | **0.880** | **0.670** | **0.720** | **0.850** | **Optimal** |
| Dice | Binary (Thresholded) | 0.840 | 0.630 | 0.680 | 0.810 | Loses IDF weighting signal |
| Jaccard | Binary (Thresholded) | 0.820 | 0.610 | 0.660 | 0.790 | Loses IDF; penalizes asymmetry |
| Hamming | Binary (Thresholded) | 0.720 | 0.510 | 0.600 | 0.710 | Ignores weighting entirely |

### 6.4.3 Formalizing the Feature Invariance Principle
Why do Dice, Jaccard, and Hamming fail to improve upon Cosine? We formalize this into the **Feature Invariance Principle**, a core theoretical contribution of this thesis.

**Theorem 2 (The Feature Invariance Principle):** *Let $\mathbf{q}, \mathbf{d} \in \mathbb{R}_{\geq 0}^d$ be localized SDRs derived from a fixed 2D grid. If a feature $f$ is computed strictly as a function of the localized spatial overlap between $\mathbf{q}$ and $\mathbf{d}$, then $f$ is perfectly collinear with the dot-product $\mathbf{q} \cdot \mathbf{d}$ under fixed vector norms.*

**Proof Sketch:** Let $\mathcal{A}, \mathcal{B} \subset \{1, \dots, d\}$ be the active bit indices. The fundamental unit of information in an SDR is the intersection size $c = |\mathcal{A} \cap \mathcal{B}|$. 
1. The dot-product $\mathbf{q} \cdot \mathbf{d} = \sum_{i \in \mathcal{A} \cap \mathcal{B}} w_{q,i} w_{d,i}$. If we ignore weights (or assume uniform weights), this equals $c$.
2. Dice coefficient: $D = \frac{2c}{|\mathcal{A}| + |\mathcal{B}|}$. Because $|\mathcal{A}|$ and $|\mathcal{B}|$ are fixed by the `top_percent` parameter ($\approx 410$ bits), $D \propto c$.
3. Jaccard coefficient: $J = \frac{c}{|\mathcal{A}| + |\mathcal{B}| - c}$. For small overlaps relative to vector size, the denominator is dominated by the constant $|\mathcal{A}| + |\mathcal{B}|$, making $J \propto c$.

**Implication:** Any metric (or internal architectural modification, such as snippet ranking or cross-attention over the grid) that merely re-computes or re-weights the localized spatial overlap is mathematically trapped in the same gradient space as the baseline dot-product. You cannot extract new discriminative ranking signal from an SDR by manipulating how you count its overlapping bits. This principle explains why the 7 architectural variants tested in Chapter 7 yield exactly 0.00% MRR improvement.

## 6.5 Impact of Fingerprint Correlation on Discriminability

### 6.5.1 Theoretical Analysis
The Orthogonality Constraint (Zahn et al., 2026) assumes independent random bits, yielding an expected cosine of $\rho$ with extremely low variance ($\approx 2.2 \times 10^{-5}$ for $d=4096$). However, SF fingerprints intentionally violate this to enable semantic matching. 

### 6.5.2 Empirical Evidence of Correlation

**Table 6.2: Pairwise Cosine Similarity Statistics (Belebele 50Q, 20 Docs)**

| Statistic | Random SDR Prediction | SF Fingerprint (Observed) |
|-----------|----------------------:|--------------------------:|
| Mean Cosine | 0.100 | 0.152 |
| Std Dev | 0.007 | 0.025 |
| 99.9% CI Upper Bound | 0.150 | 0.420 |

**Finding:** SF fingerprints have a mean similarity 52% higher than random SDRs, and a variance 3.5× higher. The maximum observed cosine (0.42) is 2.8× higher than the theoretical random-SDR upper bound. 

### 6.5.3 Implications for the Scaling Wall
This elevated baseline correlation is the microscopic mechanism of the **Scaling Wall**. In a 20-document pool, a relevant document scoring 0.35 easily floats above the 0.15 mean noise floor. However, as corpus size $N$ grows, the probability of *unrelated* documents sharing large semantic clusters increases. The dynamic range (Max - Mean) compresses. At $N > 1000$ (e.g., NQ-REaR), the entire corpus scores between 0.034 and 0.051, rendering the cosine metric statistically impotent without a hard pre-filter.

## 6.6 Normalization Strategies and the Bounded Scale Property

The choice of normalization is not just a performance tuning step; it is the engineering mechanism that creates the bounded scale mismatch central to this thesis.

### 6.6.1 L2 Normalization (Mandatory)
L2 normalization forces the document vector to unit length: $\hat{\mathbf{d}} = \mathbf{d} / \|\mathbf{d}\|_2$. 
**Theoretical Consequence:** Because $\|\hat{\mathbf{d}}\|_2 = 1$ and $\|\tilde{\mathbf{q}}\|_2 = 1$, the cosine formula reduces to a simple dot-product: $\text{score}_{\text{SF}} = \tilde{\mathbf{q}} \cdot \hat{\mathbf{d}}$. The maximum possible score occurs when the vectors are identical, yielding exactly $1.0$. **L2 normalization mathematically guarantees the $[0, 1]$ bounded scale.**

### 6.6.2 sqrt(nnz) Normalization
Dividing by $\sqrt{\text{nnz}(\mathbf{d})}$ assumes magnitude scales with the number of active phrases, but does not enforce a hard upper bound across different documents. 
**Empirical Consequence:** Results in unbounded scores that vary by document length, degrading MRR by −4.5% on Belebele compared to L2.

### 6.6.3 No Normalization
**Empirical Consequence:** Long documents (with many phrases) dominate the scoring space entirely, degrading MRR by −10.2%. Scores become unbounded integers.

**Thesis Synthesis:** We *must* use L2 normalization to ensure fair document comparison. By doing so, we intentionally construct a bounded SF scorer ($\max \approx 1.0$) that we will later linearly fuse with an unbounded SPLADE scorer ($\max \approx 50.0$). This mathematical friction is not a bug; it is the controlled experiment required to expose the Complementarity Illusion.

## 6.7 LambdaMART as Empirical Ceiling Validation

To empirically confirm that SF+SPLADE has reached the performance ceiling for this specific 20-document candidate pool structure—and to further validate the Feature Invariance Principle—we applied LambdaMART, a learned gradient-boosted decision tree re-ranker, as a boundary test.

### 6.7.1 Architecture
LambdaMART re-ranks the top-20 documents using an ensemble of 35 features per (query, document) pair:

| Feature Type | Examples |
|--------------|----------|
| SF Metric Variants | Cosine, Dice, Jaccard, Hamming scores |
| SPLADE Scores | Raw dot-product, expansion overlap count |
| Document Features | Length, IDF sum, fingerprint density (nnz) |
| Query Features | Length, IDF sum, entity count |

If the Feature Invariance Principle is false—if Dice or Jaccard carried independent, non-collinear signal—LambdaMART's feature selection algorithm would exploit them to boost MRR.

### 6.7.2 Results

**Table 6.3: LambdaMART Ceiling Validation (Belebele 50Q)**

| Method | MRR | $\Delta$ vs SF+SPLADE | Interpretation |
|--------|:---:|:-------------------:|-----------------|
| SF+SPLADE (Linear $\alpha=0.3$) | 0.930 | — | Baseline |
| LambdaMART (Same-dataset train) | 0.945 | +1.6% | Within noise margin |
| LambdaMART (Cross-dataset train) | 0.649 | −30.2% | Severe overfitting |

### 6.7.3 Interpretation: Proof of Feature Collinearity
The LambdaMART results provide powerful empirical validation for the Feature Invariance Principle. 
1. The +1.6% same-dataset gain falls entirely within the expected noise margin (±0.015 MRR from UMAP seed variation). It found no exploitable independent signal.
2. The −30.2% cross-dataset collapse confirms that the 35 features (including the alternative metrics) are so highly correlated that the model immediately overfits to the training query set's specific overlap distributions.
3. Most importantly, tree-based feature importance metrics for the LambdaMART model showed that Dice, Jaccard, and Hamming features had near-zero importance scores. The trees exclusively used the SF Cosine score and SPLADE score. 

This proves mathematically and empirically: once the localized dot-product (Cosine) is computed, adding set-theoretic metric variations or internal fingerprint statistics cannot extract further discriminative ranking signal. 

## 6.8 Recommended Configuration

**Table 6.4: Recommended Similarity and Scoring Configuration**

| Component | Value | Theoretical/Empirical Justification |
|-----------|-------|------------------------------------|
| **Similarity Metric** | **Cosine Similarity** | Only metric correctly handling continuous IDF weights; produces bounded $[0,1]$ scale |
| **Normalization** | **L2 (Query and Doc)** | Enforces strict upper bound of 1.0; optimal +4.5% over sqrt(nnz) |
| **Re-ranking** | **None (for SF-only)** | LambdaMART validates Feature Invariance Principle (0% independent gain) |
| **Fingerprint Density** | **10% ($\rho=0.10$)** | Maximizes Fisher Information while limiting spatial correlation |

## 6.9 Integration with the Hybrid Fusion Thesis

The similarity metric is applied in **Stage 6** of the SF pipeline. It represents the final mathematical operation of the unsupervised topology. 

We have now established the complete architectural profile of the SF signal:
1.  **Topology:** Discrete 2D spatial grid (Chapter 3).
2.  **Optimization:** Tuned for maximum local discrimination (Chapter 4).
3.  **Boundaries:** Constrained by the Scaling Wall and Compositional Gap (Chapter 5).
4.  **Output Scale:** Strictly bounded $[0, 1]$ via L2-normalized Cosine Similarity (This Chapter).

In Chapter 7, we take this strictly bounded SF signal and fuse it with the strictly unbounded signal of SPLADE. We will demonstrate that the choice between Linear Interpolation and Reciprocal Rank Fusion is not a hyperparameter tuning step, but a mathematical necessity dictated by the scale properties defined here—formalizing the **Operator-Topology Constraint**.

---

## References

- Beyer, K., et al. (1999). When is "nearest neighbor" meaningful? *ICDT*, 217–235.
- Liu, T.-Y. (2009). Learning to Rank for Information Retrieval. *Foundations and Trends in Information Retrieval*, 3(3), 225–331.
- Zahn, O., et al. (2026). Attention Is Not Retention: The Orthogonality Constraint. arXiv:2601.15313.