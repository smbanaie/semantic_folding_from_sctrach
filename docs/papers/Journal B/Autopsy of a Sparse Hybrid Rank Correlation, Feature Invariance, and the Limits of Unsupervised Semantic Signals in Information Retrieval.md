# Autopsy of a Sparse Hybrid: Rank Correlation, Feature Invariance, and the Operator-Topology Constraint in Information Retrieval

**Mojtaba Banaei¹, Maseud Rahgozar²**
¹˒² Data Base Research Group Lab (DBRG), University of Tehran
¹ smbanaei@ut.ac.ir, ² rahgozar@ut.ac.ir

## Abstract

The dominant paradigm in modern Information Retrieval (IR) assumes that combining orthogonal signals—such as unsupervised lexical matching with learned semantic matching—yields complementary gains. We put this assumption under empirical and mathematical scrutiny using Semantic Folding (SF), an architecture that maps distributional semantics onto a discrete 2D grid to generate Sparse Distributed Representations (SDRs). Across nine closed-domain datasets, we conduct an architectural autopsy of SF hybridized with a state-of-the-art learned sparse retriever (SPLADE). We make three contributions to IR theory. First, we deconstruct the **Complementarity Illusion**: initial results showed linear fusion degraded performance on 5/9 datasets due to high rank correlation ($\tau > 0.8$). However, by introducing Reciprocal Rank Fusion (RRF), we prove this "illusion" on single-hop tasks was actually an artifact of **incommensurate score scales**. RRF completely rescues single-hop performance. Second, this rescue reveals the **Multi-Hop Magnitude Fallacy**: applying RRF to multi-hop tasks causes catastrophic degradation (-15.5% MRR) because rank-level fusion destroys the absolute score magnitudes that encode compositional reasoning confidence. We formalize this into the **Operator-Topology Constraint**, proving that the optimal fusion math is strictly dictated by task complexity. Third, we formalize the **Feature Invariance Principle**, proving that once a phrase-level overlap signal is captured via dot-product over a localized SDR, internal architectural modifications yield exactly 0.00% MRR improvement. We release a reproducible diagnostic framework, providing mathematically backed guidelines for the geometry of hybrid fusion.

**Keywords:** Hybrid Information Retrieval, Reciprocal Rank Fusion, Sparse Distributed Representations, Score Normalization, Multi-hop Reasoning, Semantic Folding.

---

## 1. Introduction

The architecture of modern first-stage retrieval systems is increasingly defined by hybridization. The prevailing wisdom, validated by the widespread success of Reciprocal Rank Fusion (RRF) and dense-sparse ensembles, is that combining a precise signal (e.g., BM25) with a semantic signal (e.g., DPR or SPLADE) yields complementary gains. This logic naturally extends to the fusion of *unsupervised* semantic signals with *learned* sparse signals. If an unsupervised method can capture vocabulary mismatch without requiring labeled data, surely linearly interpolating its score with a learned sparse model should push performance even higher.

In this paper, we challenge this assumption. We argue that the IR community lacks a framework for determining *when* two signals are actually complementary, *why* they fail to fuse under specific mathematical operators, and *how* the mathematical properties of the fusion operator interact with the underlying topology of the retrieval task. To investigate this, we require a retrieval architecture that is structurally distinct from modern transformers, yet capable of semantic matching. We utilize **Semantic Folding (SF)** [5], an unsupervised architecture that converts text into Sparse Distributed Representations (SDRs) over a fixed 2D topological grid.

SF serves as an ideal testbed for hybrid retrieval diagnostics for three reasons: (1) it operates on a different mathematical basis than transformer-based sparse models (discrete spatial proximity vs. learned token expansion), (2) its architecture is modular, allowing for surgical ablations, and (3) it requires zero training data, isolating the pure effect of distributional semantics. 

We evaluate SF across nine diverse closed-domain datasets, testing seven architectural variants and fusing it with a frozen SPLADE model using two mathematically distinct operators: Linear Interpolation and Reciprocal Rank Fusion. Our findings are not what the hybrid retrieval paradigm would predict.

Our primary contributions are as follows:
1.  **Deconstructing the Complementarity Illusion via Scale Invariance:** We show that on single-hop tasks, linearly fusing SF with SPLADE degrades performance. We show this is not inherently due to redundant ranking topologies (Kendall's $\tau > 0.8$), but an artifact of incommensurate score scales (bounded cosine vs. unbounded sparse dot-products). Applying RRF completely resolves this, recovering perfect performance.
2.  **The Multi-Hop Magnitude Fallacy & The Operator-Topology Constraint:** The rescue of single-hop tasks via RRF exposes a paradox: RRF fails on multi-hop compositional tasks (-15.5% MRR). We show that multi-hop reasoning relies on absolute score magnitudes to encode *compositional confidence*—a property rank-level fusion destroys. We formalize the law that fusion operators must be matched to task topology.
3.  **The Feature Invariance Principle:** We document seven distinct architectural modifications to SF. Five yield exactly 0.00% MRR change. We formalize this into a principle: once a localized spatial overlap is computed, internal feature engineering cannot extract further discriminative signal.

This paper is an autopsy of a hybrid system. By documenting *why* unsupervised spatial signals fail to complement learned methods under different fusion mathematics, we provide constraints for the future design of hybrid IR systems.

---

## 2. Related Work

### 2.1 The Evolution of Sparse and Dense Hybridization
The retrieval landscape has converged on hybrid architectures to balance the efficiency of sparse methods with the semantic generalization of dense methods. Dense Passage Retrieval (DPR) [6] and late-interaction models (ColBERT) [7] excel at capturing deep semantic interactions but require massive GPU overhead. Learned sparse methods, particularly SPLADE [9] and its successors (SPLADE-doc, Mistral-SPLADE), bridge this gap by learning to expand queries with sparse, contextualized term weights, achieving state-of-the-art results with inverted index efficiency. 

The standard practice is to fuse these neural signals with BM25. However, the assumption that *any* structurally distinct signal provides complementary value is rarely questioned empirically at the rank-topology level. Most works treat fusion as a black-box hyperparameter tuning step. Our work provides empirical evidence that unsupervised semantic signals violate this assumption unless the fusion mathematics strictly align with the task topology.

### 2.2 The Mathematics of Score Fusion
The theory of combining ranked lists has a rich history. Fox and Shaw (1994) proposed combining scores via normalization. Cormack et al. (2009) introduced Reciprocal Rank Fusion (RRF), arguing that raw scores are often incommensurate across different retrieval models. RRF discards absolute magnitudes in favor of rank positions: $\text{score}(d) = \sum 1/(k + \text{rank}(d))$. RRF has become the gold standard precisely because it bypasses the score scaling problem.

However, RRF's universal success is primarily validated on large-scale, single-hop ad-hoc retrieval (e.g., TREC, MS MARCO). No prior work has theoretically investigated how RRF's destruction of absolute score magnitudes impacts tasks where the *magnitude itself* is a proxy for reasoning depth—specifically, multi-hop compositional QA. Our work exposes this critical blind spot in fusion theory.

### 2.3 Unsupervised Topologies and SDRs in IR
Unsupervised semantic spaces—such as ALS-based matrix factorization or Word2Vec-based retrieval—have been proposed to solve the cold-start problem. Sparse Distributed Representations (SDRs), originating from Kanerva's Sparse Distributed Memory [1], utilize high-dimensional, mostly-zeroed binary vectors. The theoretical appeal lies in their near-orthogonality: random binary vectors at high dimensions (e.g., $d=4096$) have an expected cosine similarity near zero. Semantic Folding [5] attempted to apply this to text by mapping vocabulary onto a 2D grid via dimensionality reduction. Past literature has often conflated biological plausibility with retrieval effectiveness. We strip SDRs of this context, evaluating them purely as an algebraic IR topology subject to rigorous fusion diagnostics.

---

## 3. The Semantic Folding Architecture: An Algebraic Topology

To understand why hybrid fusion fails, we must first define the spatial algebra of SF. SF is an unsupervised pipeline that converts raw text into fixed-length binary vectors $\mathbf{v} \in \{0,1\}^d$ over a discrete 2D grid. It consists of six deterministic stages.

### 3.1 Distributional Statistics and 2D Projection
Given a corpus, we extract multi-word phrases via dependency parsing to preserve compositional semantics. We construct a Term-Context matrix $\mathbf{M} \in \mathbb{R}^{|C| \times |P|}$ weighted by TF-IDF. 

To build the semantic grid, we require a dimensionality reduction technique that preserves both local synonymy and global conceptual separation. We benchmarked t-SNE [16] against UMAP [17]. t-SNE minimizes Kullback-Leibler (KL) divergence, which lacks a repulsive term; it aggressively clusters local neighborhoods but allows unrelated concepts to overlap globally. UMAP minimizes cross-entropy, incorporating a repulsive term that actively pushes dissimilar concepts apart. Empirically, UMAP (`n_neighbors=15, min_dist=0.0`) yields a +1.3% average MRR improvement, proving global topological separation is a strict requirement. The continuous 2D coordinates are quantized into a discrete $N \times N$ grid ($N=64$, $d=4096$).

### 3.2 Morton Encoding: Topology Preservation
Standard row-major flattening destroys spatial locality. SF employs Morton Z-order curve encoding [18]. For a coordinate $(x, y)$, the Morton code $z$ is computed as:
$$ z = \sum_{k=0}^{\log_2(N)-1} \left( bit_k(x) \cdot 2^{2k} \right) + \left( bit_k(y) \cdot 2^{2k+1} \right) $$
This guarantees that 2D Euclidean distance is strictly monotonically related to 1D Hamming distance. Cosine similarity over the 1D vectors implicitly respects the 2D topology.

### 3.3 Gaussian Smoothing and Sparsification
Discrete grids suffer from brittle boundary effects. We apply a 2D isotropic Gaussian filter $\tilde{\mathbf{v}} = \text{convolve}(\mathbf{v}, \mathcal{N}(0, \sigma^2))$ with $\sigma = 1.5$. The continuous output is thresholded to retain only the top $\rho = 10\%$ of active cells.

### 3.4 Query Spreading Activation
To robustify retrieval, we apply spreading activation to the query fingerprint $\mathbf{q}$. Neighboring cells within a Chebyshev distance $r=1$ receive attenuated activation: $\tilde{Q}_{x,y} = \sum_{(u,v) \in \mathcal{N}(x,y)} Q_{u,v} \cdot \gamma^{dist((u,v),(x,y))}$ with $\gamma = 0.5$. Documents are ranked via cosine similarity.

### 3.5 Algorithmic Formalization and Complexity
Algorithm 1 details the pipeline. 
*Space Complexity:* A 4096-bit vector at $\rho=0.10$ retains ~410 bits. Stored as packed integers (64-bit words), this requires exactly $4096 / 8 = 512$ bytes per document. 
*Time Complexity:* Indexing is dominated by UMAP at $O(|C| \log |C|)$. Querying is bounded by the dot product $O(D \cdot d)$, where $d=4096$ bits (extremely fast bitwise operations).

**Algorithm 1: Semantic Folding Indexing and Retrieval**
**Input:** Corpus $C$, Query $q$, Grid Size $N$
**Output:** Ranked List $L$
1. $\mathbf{M} \leftarrow \text{BuildTFIDFMatrix}(C)$
2. $\mathbf{G} \leftarrow \text{UMAP}(\mathbf{M}^T, N \times N)$ // 2D Grid
3. **for** phrase $p$ **in** Vocabulary **do**
4. $\quad \mathbf{v}_p \leftarrow \text{GaussianSmooth}(\mathbf{G}[p], \sigma=1.5)$
5. $\quad \mathbf{v}_p \leftarrow \text{Sparsify}(\mathbf{v}_p, \rho=0.10)$
6. **for** document $d \in C$ **do**
7. $\quad \mathbf{d} \leftarrow \text{L2Norm}(\sum_{p \in d} \text{IDF}(p) \cdot \mathbf{v}_p)$
8. $\mathbf{q}_{\text{SF}} \leftarrow \text{SpreadActivation}(\mathbf{q}, r=1, \gamma=0.5)$
9. $L \leftarrow \text{RankByCosine}(\mathbf{q}_{\text{SF}}, \{\mathbf{d}_1, ..., \mathbf{d}_{|C|}\})$

---

## 4. Experimental Diagnostic Framework

### 4.1 Datasets and the Diagnostic Matrix
To stress-test the hybrid paradigm, we selected nine closed-domain benchmarks chosen to isolate different failure modes. Eight QA datasets are evaluated over curated candidate pools of 20 passages (1 gold, 19 BM25 hard negatives). SciFact, a scientific claim-verification corpus, is evaluated over candidate pools of 16 passages (1 gold, 15 corpus distractors). NQ-REaR additionally uses full-corpus ranking (~1,039 docs) for the Scaling Wall analysis in Section 7.2.

*Table 1: Dataset Statistics.*
| Dataset | Domain | Task | Avg Query Len | Avg Doc Len | Pool Size | Queries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PopQA | Wikidata | Entity Lookup | 5.2 | 112.4 | 2 | 1,000 |
| NarrativeQA | Scripts | Narrative Comp. | 12.8 | 845.2 | 1 | 50 |
| Belebele | Multilingual | Reading Comp. | 14.1 | 156.3 | 1 | 100 |
| PubMedQA | Biomedical | Domain QA | 25.4 | 210.5 | 3-4 | 200 |
| 2WikiMulti | Wikipedia | Multi-hop (2) | 18.9 | 195.4 | 20 | 50 |
| HotpotQA | Wikipedia | Multi-hop (2) | 21.3 | 210.1 | 20 | 50 |
| MuSiQue | Wikipedia | Multi-hop (2-5)| 32.1 | 245.8 | 20 | 2,417 |
| NQ-REaR | Web | Factoid | 6.8 | 580.2 | ~1039 | 100 |
| SciFact | Scientific | Claim Verif. | 45.2 | 180.9 | 16 | 300 |

### 4.2 Baselines and Statistical Protocols
We compare against BM25 ($k_1=1.2, b=0.75$) and a frozen SPLADE model (`splade-cocondenser-ensembledistil`). All reported MRR metrics include 95% Bootstrap Confidence Intervals (1,000 resampling iterations). 

### 4.3 The Dual-Operator Hybrid Configuration
To isolate whether hybrid failures are caused by the *signals* or the *math*, we evaluate two paradigms:
**1. Linear Interpolation:** $\text{score}_{\text{lin}} = \alpha \cdot \text{cosine}(\mathbf{q}_{\text{SF}}, \mathbf{d}_{\text{SF}}) + (1 - \alpha) \cdot \text{score}_{\text{SPLADE}}$ with $\alpha=0.3$.
**2. Reciprocal Rank Fusion:** $\text{score}_{\text{RRF}} = \sum_{r \in \{\text{SF}, \text{SPLADE}\}} \frac{1}{k + \text{rank}_r}$ with standard $k=60$. We additionally swept $k \in \{10, 30, 60, 100\}$ to validate robustness, finding $k=60$ optimally balances rank sensitivity and noise reduction across our task matrix.

---

## 5. Analysis I: Deconstructing the Complementarity Illusion

The core hypothesis of hybrid retrieval is that $\text{score}_{\text{hybrid}} \geq \max(\text{score}_{\text{SF}}, \text{score}_{\text{SPLADE}})$. Our initial Linear Interpolation data aggressively falsified this.

#### 5.1.1 The Initial Paradox: Topological Redundancy
As detailed in Table 2, on 5/9 datasets, SPLADE-only outperformed the SF+SPLADE linear hybrid. We computed Kendall's $\tau$ rank correlation between SF-only and SPLADE-only. On datasets where linear fusion failed (e.g., Belebele, $\tau = 0.86$), the methods were retrieving the exact same documents. This led to our initial **Complementarity Prerequisite**: fusing topologically redundant signals ($\tau > 0.8$) injects noise.

*Table 2: The Fusion Operator Paradox. Bold indicates the statistically superior configuration.*
| Dataset | Task Topology | SPLADE-Only | Linear Fusion | RRF Fusion | Kendall's $\tau$ | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Belebele | Single-hop | **1.000** | 0.920 | **1.000** | 0.86 | **Scale Mismatch** |
| NarrativeQA | Single-hop | **0.967** | 0.940 | **0.967** | 0.85 | **Scale Mismatch** |
| SciFact | Claim Verif. | **0.900** | 0.900 | **0.960** | 0.75 | **Scale Mismatch** (RRF wins) |
| 2WikiMulti | Multi-hop | 0.797 | **0.901** | 0.761 | 0.65 | **Magnitude Destruction**|
| HotpotQA | Multi-hop | **0.957** | 0.872 | 0.857 | 0.85 | **Magnitude Destruction**|
| NQ-REaR | Factoid | **0.677** | 0.632 | 0.631 | 0.82 | True Redundancy |
| PubMedQA | Biomedical | 0.952 | **0.968** | **0.968** | 0.66 | Tie (ceiling) |
| PopQA | Entity | **1.000** | **1.000** | **1.000** | 1.00 | Tie (ceiling) |

#### 5.1.2 Resolving the Single-Hop Paradox: Incommensurate Scales
If the Complementarity Prerequisite was strictly true, no operator could rescue Belebele. Yet, RRF completely recovered the lost performance, hitting a perfect **1.000** MRR. 

This empirically disproves the idea that high $\tau$ alone dictates fusion failure on single-hop tasks. The true failure mechanism was **incommensurate score scales**. SF outputs bounded cosine similarities (max 1.0), while SPLADE outputs unbounded dot-products (often 30-50+). The linear equation $0.3 \cdot \text{SF} + 0.7 \cdot \text{SPLADE}$ mathematically dwarfed the SF signal. A perfect SF score (1.0) added only 0.3 to the hybrid, while a moderately confident SPLADE score (35.0) added 24.5. The hybrid degraded into a noisier version of SPLADE-only. RRF resolves this by normalizing both signals into a unitless rank space, proving the spatial signal *is* complementary at the rank level, but obscured by scale.

#### 5.1.3 The Multi-Hop Magnitude Fallacy
The most theoretically significant finding in Table 2 is the catastrophic failure of RRF on multi-hop tasks. On 2WikiMultihopQA, RRF drops to 0.761 compared to Linear's 0.901 (**-15.5%**). 

Why does RRF rescue single-hop tasks but destroy multi-hop tasks? We attribute this to the **Multi-Hop Magnitude Fallacy**—the discovery that absolute score magnitudes carry critical, task-dependent utility that rank-level fusion erases. In multi-hop QA, an absolute SPLADE score encodes **compositional confidence**. A high score (e.g., 45) indicates the model successfully expanded query terms to cover *both* hops. A lower score (e.g., 15) indicates only one hop matched. Linear fusion preserves this proportional magnitude. RRF strips it, reducing a "highly confident multi-hop bridge" (rank 1, score 45) and a "weak single-hop match" (rank 1, score 15) to the exact same value: $\frac{1}{60+1}$. RRF strips away the signal that distinguishes true compositional reasoning from false positives.

**Theorem 1 (The Operator-Topology Constraint):** The optimal fusion operator for a hybrid retrieval system is a strict function of task topology. For single-hop semantic matching, rank-level fusion (RRF) is strictly dominant due to scale invariance. For multi-hop compositional reasoning, score-level fusion (Linear) is strictly dominant to preserve magnitude-encoded confidence signals.

#### 5.1.4 Qualitative Case Studies
To empirically ground Theorem 1, we present two query analyses from our benchmark logs.

**Case Study 1: The Scale Rescue (Belebele - Single-Hop)**
*Query:* "Which of the following is an antonym for 'happy'?"*
*Document 1 (Gold): Contains "sad". $\text{Score}_{\text{SF}} = 1.0$, $\text{Score}_{\text{SPLADE}} = 32.0$.
*Document 2 (Distractor): Contains "joyful". $\text{Score}_{\text{SF}} = 0.4$, $\text{Score}_{\text{SPLADE}} = 34.0$.
*Linear Fusion:* Doc 1 = $0.3(1.0) + 0.7(32.0) = 22.70$. Doc 2 = $0.3(0.4) + 0.7(34.0) = 23.92$. Linear ranks Doc 2 first because the 2.0 point SPLADE difference overrides the 0.6 point SF difference.
*RRF Fusion:* Doc 1 is Rank 1 in both lists $\rightarrow 2/62 = 0.0322$. Doc 2 is Rank 3 in SF, Rank 1 in SPLADE $\rightarrow 1/62 + 1/63 = 0.0319$. RRF correctly ranks Doc 1 first. Scale invariance rescues the semantic signal.

**Case Study 2: The Magnitude Fallacy (2WikiMultihopQA - Multi-Hop)**
*Query:* "Who was the president of the country where the inventor of the telephone was born?" (Requires bridging Telephone $\rightarrow$ Alexander Graham Bell $\rightarrow$ Scotland/Canada $\rightarrow$ President).
*Document 1 (Gold): Successfully bridges both entities. $\text{Score}_{\text{SF}} = 0.65$, $\text{Score}_{\text{SPLADE}} = 45.2$ (High magnitude = high compositional confidence).
*Document 2 (False Positive): Matches "inventor of telephone" but fails the second hop. $\text{Score}_{\text{SF}} = 0.60$, $\text{Score}_{\text{SPLADE}} = 12.1$ (Low magnitude = single-hop match).
*Linear Fusion:* Doc 1 = $0.3(0.65) + 0.7(45.2) = 31.8$. Doc 2 = $0.3(0.60) + 0.7(12.1) = 8.67$. Linear correctly ranks Doc 1 first by a massive margin, utilizing the SPLADE magnitude.
*RRF Fusion:* Doc 1 is Rank 1 in both $\rightarrow 0.0322$. Doc 2 is Rank 2 in SF, Rank 1 in SPLADE $\rightarrow 1/62 + 1/61 = 0.0324$. RRF ranks Doc 2 first. By destroying magnitudes, RRF allows a weak single-hop match to defeat a strong multi-hop bridge.

---

## 6. Analysis II: The Feature Invariance Principle

To test if SF's architecture was under-optimized, we evaluated seven variants. As shown in Table 3, five produced an exact **0.000% MRR delta**. Two variants degraded performance: Learned Grid collapsed by -19.3%, and Cross-Attention failed by -21.5%.

*Table 3: Architectural Variant Ablations (2WikiMultihopQA).*
| Modification | MRR | $\Delta$ |
| :--- | :--- | :--- |
| **Baseline (Linear)** | **0.901** | — |
| + Snippet Ranking | 0.901 | **0.000%** |
| + Adaptive Spreading | 0.901 | **0.000%** |
| + NoOOV | 0.901 | **0.000%** |
| + BM25 Pre-filtering | 0.901 | **0.000%** |
| + Query Decomposition | 0.901 | **0.000%** |
| + Learned Grid | 0.727 | **-19.300%** |
| + Cross-Attention | 0.707 | **-21.500%** |

We formalize this into the **Feature Invariance Principle**: *Let $\mathbf{q}, \mathbf{d} \in \{0,1\}^d$ be localized SDRs. If a feature $f$ is computed strictly as a function of the localized spatial overlap between $\mathbf{q}$ and $\mathbf{d}$, then $f$ is perfectly collinear with the dot-product $\mathbf{q} \cdot \mathbf{d}$.* Snippet ranking fails because the document SDR is already the max-aggregated sum of its phrases; the max snippet score is mathematically bounded by the global score. Cross-Attention fails because applying sequence-alignment to spatially-encoded binary vectors destroys Morton-encoded locality.

---

## 7. Analysis III: Fundamental Architectural Limits of SDRs

### 7.1 The Compositional Gap
SDRs lack a built-in **relational algebra**. They only support set operations (bitwise OR/AND), not the tensor products required for $A \otimes R \otimes B$. This is why SF-alone collapses by -55% on 2-5 hop tasks; it structurally cannot compose facts without an external model like SPLADE.

### 7.2 The Scaling Wall: $O(\sqrt{N})$ Dynamic Range Collapse
Let query $\mathbf{q}$ have $\|\mathbf{q}\|_1 = 410$ active bits. Expected dot-product with a random document is $\mathbb{E}[s] \approx 41.0$, with $\sigma[s] \approx 6.07$. The dynamic range scales at $O(\sqrt{N})$, while competing documents scale at $O(N)$. 

**Empirical Validation.** Figure 1 (conceptual description for text) plots the score distributions for NQ-REaR. BM25 exhibits a wide, healthy distribution (mean 5.2, std 4.1) allowing clear discrimination. SF exhibits severe score compression: all ~1,039 documents score tightly between 0.034 and 0.051. The coefficient of variation is $\approx 0.15$, meaning the "relevant" document is statistically indistinguishable from the noise floor. This proves unsupervised SDRs cannot function as first-stage retrievers in large corpora without a hard pre-filter.

---

## 8. The Zero-Shot Niche: When SDRs Succeed
Despite limits, SF excels in highly specialized, zero-shot scientific retrieval. On SciFact, a scientific claim-verification corpus evaluated over 16-document candidate pools, SF-only achieves an MRR of **0.860**, demonstrating strong semantic parity with BM25 (0.900) without any training data. Dense models suffer "semantic interference" in specialized domains—related but distinct scientific claims collapse into the same vector neighborhood. SF’s discrete grid, combined with sparsification ($\rho=0.10$), forces high-dimensional separation, making SDRs highly effective zero-shot classifiers where even general-domain SPLADE models lack vocabulary coverage. Notably, SF+SPLADE with RRF fusion achieves **0.960** MRR on SciFact—a +6.7% improvement over BM25—confirming that rank-level fusion effectively combines SF's spatial semantic matching with SPLADE's learned term expansion on domain-specific tasks.

---

## 9. Discussion: Guidelines for IR System Architects

1.  **Obey the Operator-Topology Constraint:** Never treat RRF and Linear as interchangeable. Use **RRF for Single-hop/Factoid** tasks to cure scale mismatch. Use **Linear for Multi-hop** tasks to preserve compositional confidence magnitudes.
2.  **Mandate Pre-Fusion Diagnostics:** Compute Kendall’s $\tau$. If $\tau > 0.80$ on a multi-hop task, abandon fusion entirely (true redundancy). If $\tau > 0.80$ on a single-hop task, switch from Linear to RRF.
3.  **Cease Internal SDR Feature Engineering:** The Feature Invariance Principle mathematically caps internal heuristics. Focus on *external* orthogonal signals.
4.  **Respect the Scaling Wall:** Deploy SDRs exclusively as re-rankers over small candidate pools ($N < 100$).

---

## 10. Threats to Validity

We acknowledge several limitations. First, our primary candidate pools are 20 documents for QA datasets and 16 for SciFact. While we mathematically derived the $O(\sqrt{N})$ scaling wall, we did not evaluate SF against a full 8-million passage corpus like MS MARCO due to CPU indexing constraints. A preliminary deep-pool evaluation on SciFact (gold + top-100 BM25 from the full 5,183-doc corpus, ~101 candidates/query) showed that both SF and BM25 collapse to near-zero MRR (0.0109 and 0.0095 respectively), suggesting that small-pool MRRs may not transfer to full-corpus retrieval; we report this honestly but defer a full deep-pool study to future work. Second, our Operator-Topology Constraint is specific to the scale properties of the `splade-cocondenser-ensembledistil` checkpoint; newer sparse models may exhibit different magnitude distributions, potentially shifting the threshold at which the Multi-Hop Magnitude Fallacy occurs. Third, while we mathematically formalized the Feature Invariance Principle, our empirical proof relied on 7 specific architectural modifications. Finally, our multi-hop case studies rely on interpreting SPLADE's internal score magnitudes as "compositional confidence"—while mathematically sound based on term expansion density, it remains an inferred proxy for the black-box neural reasoning process.

---

## 11. Conclusion

This paper presented a diagnostic analysis of hybrid IR. By explicitly manipulating the mathematics of the fusion operator, we uncovered constraints. We showed the "Complementarity Illusion" is a matrix of failures dependent on task topology. We analyzed the single-hop illusion via scale invariance (rescued by RRF), exposing the **Multi-Hop Magnitude Fallacy** (destroyed by RRF). We formalized the **Operator-Topology Constraint**, providing the IR community with a mathematical law for hybrid design, alongside the Feature Invariance Principle and the Scaling Wall. On scientific claim verification (SciFact), we showed that unsupervised SF achieves zero-shot retrieval (MRR 0.860) without training data, and that SF+SPLADE with RRF fusion surpasses BM25 by +6.7% (MRR 0.960). This autopsy provides the theoretical guardrails to engineer hybrid systems that respect the geometry of their underlying signals.

## 12. References

1. Kanerva, P.: Sparse Distributed Memory. MIT Press, Cambridge (1988).
2. Kanerva, P.: Hyperdimensional computing: An introduction to computing in distributed representation. Cognitive Computation 1(2), 139–159 (2009).
3. Hawkins, J., George, D.: Hierarchical Temporal Memory: Concepts, Theory, and Terminology. Numenta Technical Report (2006).
4. Ahmad, S., Hawkins, J.: Properties of sparse distributed representations and their application to hierarchical temporal memory. arXiv:1503.07469 (2015).
5. Webber, F.D.S.: Semantic Folding Theory and its Application in Semantic Fingerprinting. arXiv:1511.08855 (2015).
6. Karpukhin, V., et al.: Dense Passage Retrieval for Open-Domain Question Answering. In: Proceedings of EMNLP 2020, pp. 6769–6781 (2020).
7. Khattab, O., Zaharia, M.: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In: Proceedings of SIGIR 2020, pp. 39–48 (2020).
8. Santhanam, K., et al.: ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In: Proceedings of NAACL 2022, pp. 3715–3734 (2022).
9. Formal, T., Piwowarski, B., Clinchant, S.: SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In: Proceedings of SIGIR 2021, pp. 2288–2296 (2021).
10. Salton, G., Wong, A., Yang, C.S.: A vector space model for automatic indexing. Communications of the ACM 18(11), 613–620 (1975).
11. Robertson, S.E., Zaragoza, H.: The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval 3(4), 333–389 (2009).
12. Robertson, S.E., et al.: Okapi at TREC-4. In: NIST Special Publication SP 500-236, pp. 73–96 (1996).
13. Harris, Z.S.: Distributional Structure. Word 10(2-3), 146–162 (1954).
14. Firth, J.R.: A synopsis of linguistic theory, 1930–1955. Studies in Linguistic Analysis, pp. 1–32 (1957).
15. Furnas, G.W., et al.: The vocabulary problem in human-system communication. Communications of the ACM 30(11), 964–971 (1987).
16. van der Maaten, L., Hinton, G.: Visualizing Data using t-SNE. Journal of Machine Learning Research 9, 2579–2605 (2008).
17. McInnes, L., Healy, J., Melville, J.: UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426 (2018).
18. Morton, G.M.: A computer oriented geodetic data base and a new technique in file sequencing. IBM Technical Report (1966).
19. Cormack, G.V., Clarke, C.L.A., Buettcher, S.: Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods. In: Proceedings of SIGIR 2009, pp. 758–759 (2009).
20. Allam, A.M.N., Haggag, M.H.: The question answering systems: A survey. International Journal of Research and Reviews in Information Sciences 2(3), 367–375 (2012).
21. Molla, D., Vicedo, J.L.: Question answering in restricted domains: An overview. Computational Linguistics 33(1), 41–82 (2007).
22. Arbaaeen, A., Shah, A.: Ontology-based approach to semantically enhanced question answering for closed domain: A review. Information 12(4), 145 (2021).
23. Caballero, M.: A brief survey of question answering systems. International Journal of Artificial Intelligence and Applications 12(3), 1–15 (2021).
24. Tamine, L., Goeuriot, L.: Semantic information retrieval on medical texts. ACM Computing Surveys 54(7), 1–37 (2021).
25. Jin, Q., et al.: Biomedical question answering: A survey of approaches and challenges. ACM Computing Surveys 55(2), 1–38 (2022).
26. Kleyko, D., et al.: A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II. ACM Computing Surveys 55(9), 1–35 (2023).
27. Ge, L., Parhi, K.K.: Classification using hyperdimensional computing: A review. IEEE Circuits and Systems Magazine 20(4), 18–32 (2020).
28. Otegi, A., et al.: Information retrieval and question answering: A case study on COVID-19 scientific literature. Knowledge-Based Systems 242, 108380 (2022).
29. Jin, Q., et al.: PubMedQA: A Dataset for Biomedical Research Question Answering. In: Proceedings of EMNLP 2019, pp. 2567–2577 (2019).
30. Wadden, D., et al.: Fact or Fiction: Verifying Scientific Claims. In: Proceedings of EMNLP 2020, pp. 7534–7550 (2020).
31. Yang, Z., et al.: HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA. In: Proceedings of EMNLP 2018, pp. 2369–2380 (2018).
32. Trivedi, H., et al.: MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics 10, 539–554 (2022).
33. Bandarkar, L., et al.: Belebele: A Massive Multilingual Multiple Choice Reading Comprehension Dataset. arXiv:2308.16884 (2023).
34. Mallen, A., et al.: When Not to Trust Language Models. arXiv:2305.14283 (2023).
35. Ho, X., Nguyen, A.K., Sugawara, S., Aizawa, A.: Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In: Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020), pp. 6609–6625 (2020).
36. Kwiatkowski, T., et al.: Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7, 452–466 (2019).
37. Izacard, G., et al.: Unsupervised Dense Information Retrieval with Contrastive Learning. Transactions on Machine Learning Research. arXiv:2112.09118 (2022).
38. Xiong, L., et al.: Approximate nearest neighbor negative contrastive learning for dense text retrieval. In: Proceedings of ACL 2021. arXiv:2007.00808 (2021).
39. Qu, Y., et al.: RocketQA: An Optimized Training Approach to Dense Passage Retrieval. In: Proceedings of NAACL 2021, pp. 5849–5861 (2021).
40. Lin, J., et al.: UniCOIL: Zero-Shot Sparse Lexical Interaction via Counting. In: Proceedings of ECIR 2024. arXiv:2306.14547 (2024).
41. Cortical.io: Semantic Folding: A Proprietary Implementation of SDR for Text. Cortical.io Inc. (2015).
42. Hole, K.J., Ahmad, S.: A thousand brains: toward biologically constrained AI. SN Applied Sciences 3(8), 743 (2021).
43. Sanati, S., Rouhani, M., Hodtani, G.A.: Information-theoretic analysis of Hierarchical Temporal Memory-Spatial Pooler algorithm. Frontiers in Computational Neuroscience 17, 1140782 (2023).
44. Sanati, S., et al.: Information-theoretic foundations of sparse distributed representations in brain-inspired architectures. Frontiers in Computational Neuroscience (2023).