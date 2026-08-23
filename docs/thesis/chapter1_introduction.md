# Chapter 1: Introduction

## 1.1 Motivation: The Closed-Domain Cold-Start Barrier
Closed-domain question answering (QA) systems serving specialized communities—such as medical professionals querying clinical guidelines, lawyers searching legal precedents, or scientists navigating research literature—operate under a highly specific set of constraints. Unlike open-domain retrieval, these closed-domain systems operate in bounded corpora where domain-specific terminology, complex entity relationships, and conceptual hierarchies define the retrieval landscape. 

The dominant paradigm in modern first-stage retrieval is hybridization, typically fusing a precise lexical signal (e.g., BM25) with a semantic signal (e.g., dense embeddings like DPR or learned sparse methods like SPLADE). The foundational assumption in modern IR is that combining these structurally distinct signals inherently yields complementary gains. However, deploying these neural methods in emerging domains faces a severe **cold-start barrier**: they require tens of thousands of annotated query-document pairs to adapt to evolving domain terminology. Standard lexical methods like BM25 require no training, but they catastrophically fail under *vocabulary mismatch*—when a query uses "myocardial infarction" but the document uses "heart attack" [15]. 

To bypass this cold-start barrier, our thesis proposes utilizing **Semantic Folding (SF)** [1], an unsupervised architecture that maps text into Sparse Distributed Representations (SF) topology. By converting text into a discrete 2D grid, SF provides a mathematically distinct, zero-shot alternative to learned methods. Furthermore, the discrete, highly compressed nature of SF (512 bytes per document) offers extreme memory efficiency and interpretability, which are highly valued in specialized fields. 

However, our thesis is not merely an evaluation of Semantic Folding as a standalone retriever. We utilize SF as a rigorous empirical testbed to expose a fundamental blind spot in hybrid retrieval theory. Because SF outputs bounded cosine similarities (max 1.0) and modern sparse methods output highly unbounded sparse dot-products (often 30-50+), standard linear fusion mathematically dwarfs the SF signal. We use this architectural disconnect to expose the mechanics behind what we call the **Complementarity Illusion**, leading to our primary theoretical contribution: the **Operator-Topology Constraint**.

## 1.2 The Complementarity Illusion and the Operator-Topology Constraint

The core premise of hybrid retrieval is that $\text{score}_{\text{hybrid}} \geq \max(\text{score}_{\text{SF}}, \text{score}_{\text{SPLADE}})$. Initial tests using standard linear interpolation ($\alpha \cdot \text{SF} + (1-\alpha) \cdot \text{SPLADE}$) aggressively falsify this assumption. On single-hop tasks, the hybrid strictly degrades performance compared to SPLADE-only. 

However, we prove this "illusion" is not a singular phenomenon; it is a matrix of failures dictated by task topology. By introducing Reciprocal Rank Fusion (RRF) [40], we prove that the failure on single-hop tasks is not inherently due to redundant ranking topologies (Kendall’s $\tau > 0.80$), but an artifact of **incommensurate score scales**. RRF resolves this by normalizing both signals into a unitless rank space, recovering perfect single-hop performance.

Conversely, applying RRF to multi-hop compositional tasks (e.g., 2-hop QA) causes catastrophic degradation (-15.5% MRR). We attribute this to the **Multi-Hop Magnitude Fallacy**. In multi-hop QA, an absolute SPLADE score encodes *compositional confidence*—indicating whether multiple hops were successfully bridged. RRF destroys these absolute magnitudes, reducing a "highly confident multi-hop bridge" and a "weak single-hop match" to the exact same rank-based value. 

This leads to our primary theoretical contribution: **The Operator-Topology Constraint**, refined in this thesis into a **score-geometry-conditioned** account. The optimal fusion operator is not a free hyperparameter, nor purely a function of task complexity alone: rank-level fusion (RRF) preserves only ordering and is therefore invariant to any monotone magnitude change, while score-level fusion (linear, CombSUM) additionally preserves magnitude — and which of these information classes matters is set by the task topology *interacting with* the score geometry of the fused signals. Our evidence base for this refined claim spans eleven datasets, seven operators, four retriever pairs, two SPLADE checkpoints, confirmatory paired statistics at n=50 with Holm–Bonferroni correction, causal perturbation of real retrieval scores, and candidate-pool sweeps from 20 to 494 documents. Treating fusion operators as interchangeable hyperparameters is a mathematical error; treating task topology as their sole determinant is an empirical one.

## 1.3 Research Questions

This thesis addresses three core research questions in the context of utilizing Semantic Folding for closed-domain QA and hybrid fusion diagnostics:

**RQ1 (The Zero-Shot Boundary):** Can an unsupervised, highly compressed SDR topology achieve competitive retrieval performance against supervised dense methods on closed-domain QA benchmarks (e.g., matching a trained DPR model on SciFact with zero training data)?

**RQ2 (The Operator-Topology Constraint):** How does the complexity of the reasoning task (single-hop vs. multi-hop) dictate the optimal fusion math? We mathematically deconstruct why linear fusion fails on single-hop tasks (and how RRF rescues it), and why RRF catastrophically fails on multi-hop tasks (the Multi-Hop Magnitude Fallacy).

**RQ3 (The Feature Invariance Principle):** What are the theoretical limits of grid-based SDRs? We mathematically prove that once a localized spatial overlap is computed, internal architectural modifications (cross-attention, snippet ranking, adaptive spreading) yield exactly 0.00% MRR improvement because they are perfectly collinear with the baseline dot-product.

## 1.4 Thesis Contributions

This work makes the following contributions to Information Retrieval theory:

1.  **The Operator-Topology Constraint (refined as score-geometry-conditioned):** We mathematically formalize why rank-level fusion (RRF) is invariant to magnitude while score-level fusion (Linear, CombSUM) preserves it, and show empirically that the winning operator family is determined by task topology *interacting with* the score geometry of the fused signals — not by topology alone, and not freely choosable.
2.  **Resolution of the Linear Fusion Scale Mismatch:** We empirically prove that the "Complementarity Illusion" on single-hop tasks is an artifact of incommensurate score scales, not an inherent property of the spatial signals.
3.  **Formalization of the Feature Invariance Principle:** We mathematically prove that internal SDR modifications cannot extract further discriminative signal once spatial overlap is computed, establishing a hard empirical ceiling for the architecture.
4.  **The Scaling Wall:** We derive the mathematical proof that SDR dot-product dynamic ranges scale at $O(\sqrt{N})$, defining a hard upper bound for their use as first-stage retrievers in large corpora.
5.  **Causal Separation of Rank and Magnitude Information (new):** Beyond the definitional proof, we establish the rank/magnitude separation *empirically and causally* on real retrieval outputs: perturbing score magnitudes with rank-preserving transforms (doubling, log-compression, power transforms, random remaps) leaves RRF's fused ranking bit-identical (Kendall τ = 1.000) while measurably reordering score-space fusion; conversely, shuffling scores across documents destroys rank-only fusion maximally while magnitude-carrying operators retain partial signal. This is corroborated by confirmatory paired statistics (bootstrap CIs + Wilcoxon + Holm at n=50), replication under a second SPLADE checkpoint, and pool-growth sweeps showing operator separation tracks signal geometry rather than pool size.

## 1.5 Thesis Outline

- **Chapter 2** reviews related work in unsupervised SDRs, modern sparse retrieval (SPLADE), and the mathematical theory of score fusion (RRF vs. Linear).
- **Chapter 3** presents the Semantic Folding architecture, stripping away the biological context to evaluate it purely as an algebraic IR topology explicitly tailored for closed-domain QA.
- **Chapter 4** presents a comprehensive multi-dataset benchmark across closed-domain QA, serving as the empirical testbed for the Operator-Topology Constraint.
- **Chapter 5** details the qualitative case studies proving the Multi-Hop Magnitude Fallacy on multi-hop closed-domain datasets.
- **Chapter 6** formalizes the Feature Invariance Principle and the mathematical formulation of the Scaling Wall.
- **Chapter 7** synthesizes the findings into actionable constraints for the future design of hybrid IR systems.
- **Chapter 8** concludes with a summary of contributions and future work regarding the validation of the Operator-Topology Constraint across diverse retrieval modalities.

## 1.6 References

1. Webber, F.D.S.: Semantic Folding Theory And its Application in Semantic Fingerprinting. Technical Report (2015).
...
40. Thakur, N., et al.: BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. arXiv:2104.08663 (2021).
*Cormack, G.V., Clarke, C.L.A., & Buettcher, S.: Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods. In: Proceedings of SIGIR 2009, pp. 758-759 (2009).*