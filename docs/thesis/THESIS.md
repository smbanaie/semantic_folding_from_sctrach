# Semantic Folding: Evaluating Brain-Inspired Sparse Representations for Closed-Domain Question Answering

## A PhD Thesis

**Author**: [Author Name]
**Supervisor**: [Supervisor Name]
**Institution**: [Institution Name]
**Date**: June 2026

---

## Abstract

This thesis presents Semantic Folding (SF), an unsupervised retrieval architecture that represents text as sparse binary fingerprints over a 2D semantic grid. Unlike dense retrieval methods (DPR, ColBERT) that require labeled training data, SF is entirely unsupervised — it encodes semantic similarity through spatial proximity on the grid without any gradient-based optimization. SF is uniquely suited for closed-domain question answering (QA) because: (1) domain-specific glossaries can be integrated directly into the semantic grid, (2) parameters can be tuned quickly for new domains without retraining, and (3) interpretable grid visualizations explain retrieval decisions to domain experts.

We make the following contributions:

1. **A complete unsupervised retrieval pipeline** that converts raw text into sparse binary fingerprints through six stages: phrase extraction, term-context matrix construction, semantic space mapping, phrase fingerprinting, document fingerprinting, and query processing. The pipeline is specifically designed for domain-specific deployment with minimal setup.

2. **A domain adaptation framework** demonstrating that SF parameters can be tuned for new closed-domain QA tasks in under 10 minutes, compared to days or weeks required for retraining dense methods. We provide a systematic parameter tuning methodology with mathematical justification for each configuration choice.

3. **A glossary integration mechanism** that allows domain-specific terminologies (MeSH terms, legal citations, chemical formulas) to be directly incorporated into the semantic grid, improving retrieval for specialized vocabulary without retraining.

4. **A comprehensive multi-dataset benchmark** across 9 datasets spanning biomedical, narrative, reading comprehension, multi-hop QA, and discrete reasoning domains, demonstrating that SF achieves 88-98% of BM25 performance on single-hop tasks and matches/exceeds DPR on SciFact (0.755 vs 0.675).

5. **A hybrid SF+SPLADE architecture** that achieves **perfect MRR=1.0** on Belebele (+13.6% over baseline), **surpassing BM25 (0.995)** — the first configuration where SF outperforms a strong lexical baseline on a standard benchmark.

6. **A diagnostic theory of hybrid retrieval** that formalizes score geometry, proves the Operator Information Preservation claim (RRF preserves only order, linear preserves order+magnitude), defines the Complementarity Illusion and Hybrid Compatibility Profile, and provides a taxonomy of hybrid failures (Signal, Operator, Representation) with a pre-fusion diagnostic pipeline.

Our results demonstrate that sparse methods trade peak performance for zero-shot capability — a fundamental architectural choice with clear implications for deployment in emerging domains where training data is unavailable and interpretability is required.

---

## Table of Contents

| Chapter | Title | File |
|---------|-------|------|
| 1 | Introduction | [chapter1_introduction.md](chapter1_introduction.md) |
| 2 | Literature Review | [chapter2_literature_review.md](chapter2_literature_review.md) |
| 3 | The Semantic Folding Pipeline | [chapter3_sf_pipeline.md](chapter3_sf_pipeline.md) |
| 4 | Parameter Tuning | [chapter4_parameter_tuning.md](chapter4_parameter_tuning.md) |
| 5 | Sparse vs Dense Retrieval | [chapter5_sparse_vs_dense.md](chapter5_sparse_vs_dense.md) |
| 6 | Similarity Metrics | [chapter6_similarity_metrics.md](chapter6_similarity_metrics.md) |
| 7 | Experiments and Benchmark Results | [chapter7_experiments.md](chapter7_experiments.md) |
| 8 | Discussion | [chapter8_discussion.md](chapter8_discussion.md) |
| 9 | Conclusions and Future Work | [chapter9_conclusions.md](chapter9_conclusions.md) |

---

## List of Tables

| Table | Title | Chapter |
|-------|-------|---------|
| 3.1 | Pipeline Architecture | 3 |
| 3.2 | Phrase Extraction Passes | 3 |
| 3.3 | Dimensionality Reduction Methods | 3 |
| 4.1 | Parameter Taxonomy | 4 |
| 4.2 | Grid Size Comparison | 4 |
| 4.3 | Spreading Steps Comparison | 4 |
| 4.4 | Top Percent Comparison | 4 |
| 4.5 | Weighting Strategy Comparison | 4 |
| 4.6 | Smoothing Sigma Comparison | 4 |
| 4.7 | Document Normalization Comparison | 4 |
| 4.8 | t-SNE Perplexity Comparison | 4 |
| 4.9 | Recommended Configuration | 4 |
| 5.1 | Training Data Requirements | 5 |
| 5.2 | Memory and Speed Trade-off | 5 |
| 5.3 | Interpretability Comparison | 5 |
| 6.1 | Similarity Metrics Comparison | 6 |
| 6.2 | LambdaMART Feature Importance | 6 |
| 7.1 | Dataset Overview | 7 |
| 7.2 | Cross-Dataset Results | 7 |
| 7.3 | Performance by Task Type | 7 |

---

## List of Figures

| Figure | Title | Chapter |
|--------|-------|---------|
| 3.1 | Pipeline Architecture Diagram | 3 |
| 3.2 | Semantic Grid Visualization | 3 |
| 3.3 | Morton Z-order Curve | 3 |
| 4.1 | Parameter Interaction Heatmap | 4 |
| 5.1 | Sparse-Dense Spectrum | 5 |
| 5.2 | Zero-Shot Capability Comparison | 5 |
| 6.1 | Metric Selection Decision Tree | 6 |
| 7.1 | MRR by Dataset | 7 |
| 7.2 | Performance vs Hop Count | 7 |

---

## Appendix A: Reproduction Instructions

### A.1 Environment Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy scipy spacy plotly scikit-learn pyyaml
python -m spacy download en_core_web_sm
```

### A.2 Running Benchmarks

```bash
# Single dataset
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl --max-queries 100

# All datasets
.venv\Scripts\python -m semantic_folding.dataset_benchmark.run_all_benchmarks.py \
  --datasets belebele --max-queries 100

# BM25 baseline
.venv\Scripts\python -m semantic_folding.dataset_benchmark.bm25_benchmark \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl
```

### A.3 Best Configuration

```bash
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset belebele \
  --grid-size 64 \
  --spreading-steps 1 \
  --top-percent 0.10 \
  --weighting idf \
  --smoothing-sigma 1.5 \
  --doc-norm l2 \
  --tsne-perplexity 50 \
  --morton
```

### A.4 Performance Optimizations

**FAISS-Accelerated OOV Expansion:**
- OOV lookup: ~30s/query → ~0.075s/query (400× speedup)
- Uses FAISS IVFFlat index, built once during phrase fingerprint generation
- Memory overhead: ~15KB (negligible)

**Per-Dataset Parameter Registry:**
- Config file: `config/dataset_registry.yml`
- Stores dataset-specific optimal parameters (perplexity, normalization, hybrid weight)
- Impact: +1–4% MRR across datasets

**Query Decomposition:**
- Multi-hop queries decomposed into sub-queries via spaCy NER entity extraction
- Independent retrieval + result fusion
- +19.6% NQ-REaR, −28.8% HotpotQA (depends on NER entity extraction quality)

---

## Appendix B: Mathematical Notation

| Symbol | Definition |
|--------|------------|
| $g$ | Grid size (side length) |
| $N = g^2$ | Total grid cells |
| $\rho$ | Fingerprint density (active bits / total bits) |
| $k$ | Number of active bits |
| $\mathbf{q}$ | Query fingerprint |
| $\mathbf{d}$ | Document fingerprint |
| $\sigma$ | Gaussian smoothing parameter |
| $\gamma$ | Spreading decay factor |
| $r$ | Spreading radius |
| $\alpha$ | Asymmetric scoring weight |
| $P$ | Number of phrases |
| $C$ | Number of contexts |
| $D$ | Number of documents |
| $\mathbf{M}$ | Term-context matrix |
| $\text{IDF}(p)$ | Inverse document frequency of phrase $p$ |

---

## Appendix C: Dataset Details

### C.1 PubMedQA

- **Domain**: Biomedical
- **Task**: Question answering with context
- **Queries**: 111
- **Passages/query**: ~20
- **SF MRR**: 0.955
- **BM25 MRR**: 1.000

### C.2 Belebele

- **Domain**: Multilingual
- **Task**: Multiple choice reading comprehension
- **Queries**: 50 (3-way comparison)
- **Passages/query**: ~20
- **SF-only MRR**: 0.880
- **SF+BM25 MRR**: 0.880 (no improvement)
- **SF+SPLADE MRR**: **1.000** (perfect, surpasses BM25)
- **BM25 MRR**: 0.995

### C.3 NarrativeQA

- **Domain**: Narrative
- **Task**: Script comprehension
- **Queries**: 49
- **Passages/query**: ~20
- **SF MRR**: 0.939
- **BM25 MRR**: 0.980

### C.4 PopQA

- **Domain**: Wikipedia
- **Task**: Entity lookup
- **Queries**: 50 (new defaults)
- **Passages/query**: ~20
- **SF MRR**: **1.000** (with SPLADE, perplexity=50, L2)
- **BM25 MRR**: 1.000

### C.5 SciFact

- **Domain**: Scientific
- **Task**: Claim verification
- **Queries**: 300
- **Passages/query**: ~5K docs
- **SF MRR**: 0.755
- **DPR MRR**: 0.675

### C.6 2WikiMultihopQA

- **Domain**: Wikipedia
- **Task**: 2-hop QA
- **Queries**: 50
- **Passages/query**: 20
- **SF MRR**: 0.788
- **BM25 MRR**: 0.921

### C.7 HotpotQA

- **Domain**: Wikipedia
- **Task**: 2-hop QA
- **Queries**: 48
- **Passages/query**: 20
- **SF MRR**: 0.726
- **BM25 MRR**: 0.869

### C.8 NQ-REaR

- **Domain**: Web
- **Task**: Factoid retrieval
- **Queries**: 100
- **Passages/query**: 20
- **SF MRR**: 0.574
- **BM25 MRR**: 0.638

### C.9 MuSiQue

- **Domain**: Wikipedia
- **Task**: 2-5 hop QA
- **Queries**: 100
- **Passages/query**: 20
- **SF MRR**: 0.453
- **BM25 MRR**: 0.672

### C.10 BioASQ

- **Domain**: Biomedical
- **Task**: Biomedical QA (factoid, yes/no, list, summary)
- **Queries**: 50
- **Passages/query**: ~1075 docs
- **SF MRR**: 0.195 (p50, L2) / 0.210 (p30, L2)
- **Note**: Old 0.248 baseline was inflated by batched 10Q evaluation. SPLADE has 0% effect on BioASQ.

### C.11 DROP

- **Domain**: Discrete reasoning
- **Task**: Counting, sorting, comparison over text passages
- **Queries**: 50
- **Passages/query**: ~20
- **SF MRR**: 0.320 (L2 norm)
- **BM25 MRR**: 0.762
- **Note**: L2 norm provides +14.3% improvement. Requires numerical reasoning beyond phrase matching.

### C.12 DocFinQA

- **Domain**: Financial
- **Task**: Financial question answering
- **Queries**: 20
- **Passages/query**: ~20
- **SF MRR**: 0.250
- **BM25 MRR**: 0.341
- **Note**: Grid=128 used (not optimal). Both methods struggle — financial documents require numerical reasoning.

---

## Appendix D: Support Files (Archive)

The following files in `docs/thesis/archive/` contain detailed technical documentation for individual pipeline stages. These were written as standalone references before the chapter reorganization and are retained for completeness.

| File | Corresponding Chapter | Content |
|------|----------------------|---------|
| `phrase_extractor.md` | Ch3 (Pipeline) | Full v3.1 technical documentation of phrase extraction |
| `term_context.md` | Ch3 (Pipeline) | Term-context matrix construction details |
| `semantic_space.md` | Ch3 (Pipeline) | Semantic space construction with t-SNE/UMAP math |
| `fingerprints.md` | Ch3 (Pipeline) | Deep technical description of Steps 4-5 with full math |
| `query_processing.md` | Ch3 (Pipeline) / Ch6 | Comprehensive query processing with LambdaMART, negation, hybrid scoring |
| `parameters_tuning.md` | Ch4 (Parameter Tuning) | Parameter sweep results on 20-doc corpus |
| `datasets.md` | Ch7 (Experiments) / Ch8 (Discussion) | Dataset characteristics matrix, performance tiers |
| `metrics.md` | Ch6 (Similarity Metrics) | Retrieval metrics framework |
| `benchmarks.md` | Ch7 / Ch8 | Benchmarking methodology and multi-dataset results |
| `lib.md` | — | API reference for lib.py utility module (developer docs) |
| `review-comments.md` | — | Review comments and feedback |

---

## References

### Information Retrieval

- Broder, A. Z. (1997). On the resemblance and containment of documents. *Compression and Complexity of Sequences*.
- Furnas, G. W., et al. (1987). The vocabulary problem in human-system communication. *Communications of the ACM*, 30(11), 964–971.
- Sparck Jones, K. (1972). A statistical interpretation of term specificity. *Journal of Documentation*, 28(1), 11–21.

### Dense Retrieval

- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.
- Santhanam, K., et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. *NAACL 2022*.

### Sparse Retrieval

- Formal, T., et al. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR 2021*.
- Formal, T., et al. (2024). Mistral-SPLADE: LLMs for Sparse Retrieval. arXiv:2408.11119.
- Lin, J., et al. (2024). Two-Step SPLADE for Efficient In-Domain Retrieval. *ECIR 2024 Findings*. arXiv:2404.13357.
- Mallia, A., et al. (2022). Learning sparse indexes for text retrieval. arXiv:2405.01924.
- Paria, B., et al. (2024). SPLATE: ColBERTv2 + SPLADE Adapter. *SIGIR 2024*. arXiv:2404.13950.

### Semantic Folding & Sparse Distributed Memory

- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Hawkins, J., & George, D. (2006). Hierarchical temporal memory. *Numenta Technical Report*.

### Hyperdimensional Computing

- Plate, T. A. (1995). Holographic reduced representations. *IEEE Transactions on Neural Networks*, 6(3), 629–642.
- Gayler, R. W. (2003). Vector symbolic architectures as a computing framework for emerging hardware. *Proceedings of the IEEE*.

### Dimensionality Reduction

- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9, 2579–2605.
- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426.

### Closed-Domain QA

- Abacha, A. B., & Zweigenbaum, P. (2015). Medical question answering: Hybrid approach. *Journal of Biomedical Informatics*.
- Chen, Q., et al. (2013). Dataset and evaluation for biomedical QA. *ACL 2013*.
- Dramé, B., et al. (2014). Building a biomedical QA system. *JNLP*.
- Jin, Q., et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. *EMNLP 2019*.
- Malayi, M., et al. (2023). Belebele: A Massive Multilingual Multiple Choice Reading Comprehension Dataset. arXiv:2308.16884.
- Nentidis, A., et al. (2025). Overview of BioASQ 2024. arXiv:2508.20532.
- Sarrouti, M., & El Alaoui, S. O. (2020). Evidence-based biomedical question answering. *JNLP*.
- Trivedi, H., et al. (2022). MuSiQue: Multihop Questions via Single Question Decomposition. *TACL*.
- Wang, L., et al. (2025). BioRAGent: A RAG System for Biomedical Q&A. arXiv:2412.12358.
- Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA. *EMNLP 2018*.

### Orthogonality Constraint & Memory Interference

- Zahn, O., et al. (2026). Attention Is Not Retention: The Orthogonality Constraint. arXiv:2601.15313.

### Hybrid Retrieval & Score Geometry

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*.
- Fox, E. A., & Shaw, J. A. (1994). Combination of Multiple Searches. *TREC-2*.
- Hermosillo-Valadez, J., et al. (2022). Exploiting Hierarchical Dependence Structures for Unsupervised Rank Fusion in Information Retrieval. arXiv:2208.05574.
- Montague, M., & Aslam, J. A. (2001). Relevance score normalization for metasearch. *CIKM 2001*.

### Medical & Biomedical Retrieval

- Lee, J., et al. (2020). BioBERT: A Pre-trained Biomedical Language Representation Model. *Bioinformatics*, 36(4), 1234–1240.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
