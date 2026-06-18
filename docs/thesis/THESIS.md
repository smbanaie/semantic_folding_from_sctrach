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

4. **A comprehensive multi-dataset benchmark** across 9 datasets demonstrating that SF achieves 88-98% of BM25 performance on single-hop tasks and matches/exceeds DPR on SciFact (0.755 vs 0.675).

5. **A hybrid SF+BM25 architecture** that improves reading comprehension by +16.2% MRR on Belebele, providing a practical deployment strategy combining semantic coverage with lexical precision for closed-domain systems.

Our results demonstrate that sparse methods trade peak performance for zero-shot capability — a fundamental architectural choice with clear implications for deployment in emerging domains where training data is unavailable and interpretability is required.

---

## Table of Contents

| Chapter | Title | File |
|---------|-------|------|
| 1 | Introduction | [chapter1_introduction.md](chapter1_introduction.md) |
| 2 | Literature Review | [chapter2_literature_review.md](chapter2_literature_review.md) |
| 3 | The Semantic Folding Pipeline | [chapter3_sf_pipeline.md](chapter3_sf_pipeline.md) |
| 4 | Parameter Tuning | [chapter4_parameter_tuning.md](chapter4_parameter_tuning.md) |
| 5 | Experiments and Benchmark Results | [chapter5_experiments.md](chapter5_experiments.md) |
| 6 | Sparse vs Dense Retrieval | [chapter6_sparse_vs_dense.md](chapter6_sparse_vs_dense.md) |
| 7 | Similarity Metrics | [chapter7_similarity_metrics.md](chapter7_similarity_metrics.md) |
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
| 5.1 | Dataset Overview | 5 |
| 5.2 | Cross-Dataset Results | 5 |
| 5.3 | Performance by Task Type | 5 |
| 6.1 | Training Data Requirements | 6 |
| 6.2 | Memory and Speed Trade-off | 6 |
| 6.3 | Interpretability Comparison | 6 |
| 7.1 | Similarity Metrics Comparison | 7 |
| 7.2 | LambdaMART Feature Importance | 7 |

---

## List of Figures

| Figure | Title | Chapter |
|--------|-------|---------|
| 3.1 | Pipeline Architecture Diagram | 3 |
| 3.2 | Semantic Grid Visualization | 3 |
| 3.3 | Morton Z-order Curve | 3 |
| 4.1 | Parameter Interaction Heatmap | 4 |
| 5.1 | MRR by Dataset | 5 |
| 5.2 | Performance vs Hop Count | 5 |
| 6.1 | Sparse-Dense Spectrum | 6 |
| 6.2 | Zero-Shot Capability Comparison | 6 |

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
- **Queries**: 100
- **Passages/query**: ~20
- **SF MRR**: 0.880
- **BM25 MRR**: 0.995

### C.3 SciFact

- **Domain**: Scientific
- **Task**: Claim verification
- **Queries**: 300
- **Passages/query**: ~5K docs
- **SF MRR**: 0.755
- **DPR MRR**: 0.675

### C.4 MuSiQue

- **Domain**: Wikipedia
- **Task**: 2-5 hop QA
- **Queries**: 100
- **Passages/query**: 20
- **SF MRR**: 0.453
- **BM25 MRR**: 0.672

---

## References

See individual chapter reference lists for complete citations.
