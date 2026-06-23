# Chapter 1: Introduction

## 1.1 Motivation

Closed-domain question answering (QA) systems serve specialized user communities—medical professionals querying clinical guidelines, lawyers searching legal precedents, scientists navigating research literature. These systems operate within bounded corpora where domain-specific terminology, entity relationships, and conceptual hierarchies define the retrieval landscape. Unlike open-domain QA, closed-domain systems require retrieval methods that are not only accurate but also interpretable, parameter-tunable, and rapidly adaptable to domain-specific terminology.

Information retrieval (IR) systems face a fundamental tension between two competing paradigms: **lexical matching** (exact term overlap) and **semantic matching** (meaning-based similarity). Traditional methods like BM25 excel at lexical precision but fail when queries and documents use different vocabulary for the same concepts — the *vocabulary mismatch problem* (Furnas et al., 1987; Fernández et al., 2011). Neural dense retrieval methods (DPR, ColBERT) address this through learned embeddings but require massive labeled training datasets and GPU infrastructure — resources often unavailable in emerging domains.

This thesis presents **Semantic Folding (SF)**, an unsupervised retrieval architecture that represents text as **Sparse Distributed Representations (SDRs)** over a fixed 2D semantic grid. SF bridges the lexical-semantic gap without training data by encoding distributional similarity as spatial proximity on a discrete grid, drawing on neuroscientific parallels with cortical sparse coding (Kanerva, 1988; Hawkins & George, 2006). SF is uniquely suited for closed-domain QA because:

1. **Domain glossaries can be integrated directly** into the semantic grid without retraining (Dramé et al., 2014; Chen et al., 2013)
2. **Parameters can be tuned quickly** for new domains in minutes, not days (Sarrouti & El Alaoui, 2020; Abacha & Zweigenbaum, 2015)
3. **Interpretable grid visualizations** explain retrieval decisions to domain experts (Liu et al., 2025; Vazrala & Mohammed, 2025)

## 1.2 Research Questions

This thesis addresses three core research questions in the context of closed-domain QA:

**RQ1:** Can unsupervised sparse binary representations achieve competitive retrieval performance against supervised dense methods on domain-specific QA benchmarks?

**RQ2:** How can domain-specific glossaries be integrated into the semantic grid to improve retrieval for specialized terminology?

**RQ3:** What is the minimal parameter tuning effort required to adapt SF to a new closed-domain QA task, and how does this compare to retraining dense methods?

## 1.3 Contributions

This work makes the following contributions to closed-domain QA:

1. **A complete unsupervised retrieval pipeline** (Semantic Folding) that converts raw text into sparse binary fingerprints through six stages: phrase extraction, term-context matrix construction, semantic space mapping, phrase fingerprinting, document fingerprinting, and query processing. The pipeline is specifically designed for domain-specific deployment with minimal setup.

2. **A domain adaptation framework** demonstrating that SF parameters can be tuned for new closed-domain QA tasks in under 10 minutes, compared to days or weeks required for retraining dense methods. We provide a systematic parameter tuning methodology with mathematical justification for each configuration choice.

3. **A glossary integration mechanism** that allows domain-specific terminologies (MeSH terms, legal citations, chemical formulas) to be directly incorporated into the semantic grid, improving retrieval for specialized vocabulary without retraining.

4. **A comprehensive multi-dataset benchmark** across 10 datasets (PubMedQA, Belebele, NarrativeQA, PopQA, SciFact, HotpotQA, 2WikiMultihopQA, NQ-REaR, MuSiQue, BioASQ) demonstrating that SF achieves 88-98% of BM25 performance on single-hop tasks and matches/exceeds DPR on SciFact (0.755 vs 0.675).

5. **A hybrid SF+BM25 architecture** that improves reading comprehension by +13.6% MRR on Belebele (0.8800→1.0000), providing a practical deployment strategy combining semantic coverage with lexical precision for closed-domain systems.

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

- [To be completed with actual publication records before final submission]

## References

- Abacha, A., & Zweigenbaum, P. (2015). MEANS: A medical question-answering system combining NLP techniques and semantic Web technologies. *Information Processing & Management*, 51(5), 570-584.
- Chen, Y. J., Chu, H. C., Chen, Y. M., & Chao, C. Y. (2013). Adapting domain ontology for personalized knowledge search and recommendation. *Information & Management*, 50(6), 278-288.
- Dramé, K., Diallo, G., Delva, F., Dartigues, J. F., & Mouillet, V. (2014). Reuse of termino-ontological resources and text corpora for building a multilingual domain ontology. *Journal of Biomedical Informatics*, 48, 1-10.
- Fernández, M., Cantador, I., López, V., Vallet, D., Castells, P., & Motta, E. (2011). Semantically enhanced information retrieval: An ontology-based approach. *Journal of Web Semantics*, 9(4), 413-434.
- Furnas, G. W., et al. (1987). The vocabulary problem in human-system communication. *Communications of the ACM*, 30(11), 964–971.
- Hawkins, J., & George, D. (2006). *Hierarchical Temporal Memory: Concepts, Theory, and Terminology*. Numenta Technical Report.
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Liu, Y., Li, X., Luo, Y., Du, J., Zhang, Y., Lv, T., ... & Tang, X. (2025). Toward a large language model-driven medical knowledge retrieval and QA system. *Engineering*, in press.
- Sarrouti, M., & El Alaoui, S. O. (2020). SemBioNLQA: A semantic biomedical question answering system. *Artificial Intelligence in Medicine*, 102, 101776.
- Vazrala, S., & Mohammed, T. K. (2025). RBTM: A hybrid gradient Regression-Based transformer model for biomedical question answering. *Biomedical Signal Processing and Control*, 104, 107489.
- Zahn, O., Beton, M., & Chana, S. (2026). Attention Is Not Retention: The Orthogonality Constraint in Infinite-Context Architectures. arXiv:2601.15313.
