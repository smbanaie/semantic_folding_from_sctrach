# Chapter 1: Introduction

## 1.1 Motivation

Closed-domain question answering (QA) systems serve specialized user communities — medical professionals querying clinical guidelines, lawyers searching legal precedents, scientists navigating research literature. These systems operate within bounded corpora where domain-specific terminology, entity relationships, and conceptual hierarchies define the retrieval landscape. Unlike open-domain QA, closed-domain systems require retrieval methods that are not only accurate but also interpretable, parameter-tunable, and rapidly adaptable to domain-specific terminology.

Information retrieval (IR) systems face a fundamental tension between two competing paradigms: **lexical matching** (exact term overlap) and **semantic matching** (meaning-based similarity). Traditional methods like BM25 excel at lexical precision but fail when queries and documents use different vocabulary for the same concepts — the *vocabulary mismatch problem* (Furnas et al., 1987; Fernández et al., 2011). Neural dense retrieval methods (DPR, ColBERT) address this through learned embeddings but require massive labeled training datasets and GPU infrastructure — resources often unavailable in emerging domains.

This thesis presents **Semantic Folding (SF)**, an unsupervised retrieval architecture that represents text as **Sparse Distributed Representations (SDRs)** over a fixed 2D semantic grid. SF bridges the lexical-semantic gap without training data by encoding distributional similarity as spatial proximity on a discrete grid, drawing on neuroscientific parallels with cortical sparse coding (Kanerva, 1988; Hawkins & George, 2006). SF is uniquely suited for closed-domain QA because:

1. **Domain glossaries can be integrated directly** into the semantic grid without retraining (Dramé et al., 2014; Chen et al., 2013)
2. **Parameters can be tuned quickly** for new domains in minutes, not days (Sarrouti & El Alaoui, 2020; Abacha & Zweigenbaum, 2015)
3. **Interpretable grid visualizations** explain retrieval decisions to domain experts (Liu et al., 2025; Vazrala & Mohammed, 2025)

## 1.2 From Narrow AI to Biologically Constrained Systems

Current AI systems are predominantly "narrow AI" — systems that perform a single well-defined task in a single domain (Hole & Ahmad, 2021). These systems, including deep learning approaches like DPR and ColBERT, exhibit four fundamental limitations: they are **greedy** (requiring massive training sets), **brittle** (failing when test data differs mildly from training data), **rigid** (unable to adapt after initial training), and **opaque** (functioning as black boxes) (Hole & Ahmad, 2021, §3.2). Most critically, narrow AI systems lack abstract reasoning abilities and common sense about the world — they "do not know what they do" and cannot transfer performance to other domains without redesign and retraining (Hole & Ahmad, 2021, §3.2).

The limitations of narrow AI stem from its mathematical and logical foundations, which differ fundamentally from biological intelligence. As Hole & Ahmad (2021) argue, "the non-biological path of narrow AI does not lead to intelligent machines that understand and act similarly to humans" (§1, p. 6). The authors contend that "continued work on today's narrow AI techniques cannot lead to general AI because the techniques are missing necessary biological properties" (§4.2, p. 90).

Biologically constrained AI offers an alternative path. The neocortex — the center of intelligence in the human brain — employs **Sparse Distributed Representations (SDRs)** where only 2-5% of neurons are active at any time (Hole & Ahmad, 2021, §5.2). This sparse coding provides three advantages: (1) **robustness to noise**, (2) **capacity for massive pattern storage**, and (3) **ability to perform multiple simultaneous predictions** (Hole & Ahmad, 2021, §5.2). The Thousand Brains Theory (Hawkins, 2021) proposes that the neocortex runs a "common cortical algorithm" across all regions, suggesting that a single biologically plausible algorithm could achieve general intelligence.

This thesis positions Semantic Folding within the biologically constrained AI paradigm. By representing text as SDRs on a 2D semantic grid, SF employs sparse representations analogous to neocortical coding. The Semantic Folding pipeline's use of distributional semantics (Term-Context matrices) mirrors the brain's learning of statistical regularities from sensory input. While SF does not model all six biological constraints identified by Hole & Ahmad (2021) — sparse representations, realistic neuron models, reference frames, continuous online learning, sensorimotor integration, and single general-purpose algorithm — it incorporates the foundational principle of sparse distributed representations.

### 1.2 Research Questions

This thesis addresses three core research questions in the context of closed-domain and open-domain QA:

**RQ1**: Can unsupervised sparse binary representations achieve competitive retrieval performance against supervised dense methods on closed-domain QA benchmarks?

**RQ2**: What is the *performance boundary* — on which task types does Semantic Folding match or surpass BM25, and where does it fail?

**RQ3**: Does a hybrid (SF+SPLADE) combine unsupervised semantic matching with learned term expansion to outperform both approaches individually?

**Important clarification on supervision.** The SF pipeline is fully unsupervised—no labelled pairs, no gradient updates, no GPU training. SPLADE is used as an **off-the-shelf, frozen pre-trained model** (naver/splade-cocondenser-ensembledistil) with **no domain-specific fine-tuning**. The hybrid system requires **zero labelled data for new target domains**, distinguishing it from DPR or ColBERT which need 10K–500K domain-specific training pairs. This makes SF+SPLADE a **training-free neuro-retrieval architecture** that can be deployed instantly in emerging domains.

## 1.3 Contributions

This work makes the following contributions to closed-domain and general-purpose QA:

1. **A complete unsupervised retrieval pipeline** (Semantic Folding) that converts raw text into sparse binary fingerprints through six stages. The pipeline requires no training data for its core SF component. A hybrid SF+SPLADE variant (using off-the-shelf pre-trained SPLADE) is also evaluated.

2. **A domain adaptation framework** demonstrating that SF parameters can be tuned for new QA tasks in under 10 minutes. We provide a systematic parameter tuning methodology with mathematical justification.

3. **A glossary integration mechanism** that allows domain-specific terminologies to be directly incorporated into the semantic grid.

4. **A comprehensive multi-dataset benchmark** across 9 datasets with three configurations: SF-only (unsupervised), SPLADE-only (supervised, off-the-shelf), and SF+SPLADE (hybrid). Key finding: SPLADE-only is the best configuration on 5/9 datasets. SF contributes positively on only 2/9 datasets; the complementarity hypothesis (H2) is falsified.

5. **The feature-invariance principle**: Features duplicating existing SF signals (cross-attention, snippet ranking, adaptive spreading) contribute ≤0% MRR. Only genuinely non-overlapping signals (SPLADE learned expansion) provide gains. This negative result establishes the empirical ceiling for SF-based architectures.

6. **The α-sensitivity framework**: We demonstrate that the SF+SPLADE hybrid weight α ∈ [0,1] produces a monotonic degradation curve on most datasets — as SF weight increases, MRR decreases. This falsifies the complementarity hypothesis and establishes that SF and SPLADE signals are correlated, not complementary.

## 1.4 Thesis Outline

- **Chapter 2** reviews related work in information retrieval, sparse distributed representations, and semantic matching.
- **Chapter 3** presents the Semantic Folding methodology with full mathematical formulation of each pipeline stage.
- **Chapter 4** provides a systematic parameter tuning analysis with academic justification for each configuration choice.
- **Chapter 5** presents a comprehensive analysis of sparse vs dense retrieval paradigms, including the theoretical Orthogonality Constraint framework.
- **Chapter 6** provides a detailed analysis of similarity metrics for sparse distributed representations.
- **Chapter 7** describes the experimental setup and multi-dataset benchmark results.
- **Chapter 8** discusses findings, implications, and the unique position of SF in the retrieval landscape.
- **Chapter 9** concludes with summary of contributions and future research directions.

## 1.5 Publications

Parts of this work have been published or are under review:

*[Note to candidate: Populate this section with actual publication records before final submission. If no papers have been published yet, remove this section or state "No publications to date; manuscripts in preparation."]*

## References

- Abacha, A., & Zweigenbaum, P. (2015). MEANS: A medical question-answering system combining NLP techniques and semantic Web technologies. *Information Processing & Management*, 51(5), 570-584.
- Chen, Y. J., Chu, H. C., Chen, Y. M., & Chao, C. Y. (2013). Adapting domain ontology for personalized knowledge search and recommendation. *Information & Management*, 50(6), 278-288.
- Dramé, K., Diallo, G., Delva, F., Dartigues, J. F., & Mouillet, V. (2014). Reuse of termino-ontological resources and text corpora for building a multilingual domain ontology. *Journal of Biomedical Informatics*, 48, 1-10.
- Fernández, M., Cantador, I., López, V., Vallet, D., Castells, P., & Motta, E. (2011). Semantically enhanced information retrieval: An ontology-based approach. *Journal of Web Semantics*, 9(4), 413-434.
- Furnas, G. W., et al. (1987). The vocabulary problem in human-system communication. *Communications of the ACM*, 30(11), 964–971.
- Hawkins, J., & George, D. (2006). *Hierarchical Temporal Memory: Concepts, Theory, and Terminology*. Numenta Technical Report.
- Hole, K. J., & Ahmad, S. (2021). A thousand brains: toward biologically constrained AI. *SN Applied Sciences*, 3(8), 743. https://doi.org/10.1007/s42452-021-04715-0
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Liu, Y., Li, X., Luo, Y., Du, J., Zhang, Y., Lv, T., ... & Tang, X. (2025). Toward a large language model-driven medical knowledge retrieval and QA system. *Engineering*, in press.
- Sarrouti, M., & El Alaoui, S. O. (2020). SemBioNLQA: A semantic biomedical question answering system. *Artificial Intelligence in Medicine*, 102, 101776.
- Vazrala, S., & Mohammed, T. K. (2025). RBTM: A hybrid gradient Regression-Based transformer model for biomedical question answering. *Biomedical Signal Processing and Control*, 104, 107489.
- Zahn, O., Beton, M., & Chana, S. (2026). Attention Is Not Retention: The Orthogonality Constraint in Infinite-Context Architectures. arXiv:2601.15313.
