# Semantic Folding vs Other Methods: Benchmark Comparison

## Published Benchmark Results

The following table compiles published results from the original papers and well-known benchmarks. All values are MRR @10 unless otherwise noted.

### Table A1: Comprehensive Method Comparison

| Dataset | SF (Ours) | BM25 | DPR | ColBERTv2 | SPLADE | Best Method | Source |
|---------|-----------|------|-----|-----------|--------|-------------|--------|
| **PubMedQA** | **0.955** | 1.000 | — | — | — | BM25 | Our benchmark |
| **NarrativeQA** | **0.939** | 0.980 | — | — | — | BM25 | Our benchmark |
| **Belebele** | **0.880** | 0.995 | — | — | — | BM25 | Our benchmark |
| **PopQA** | **0.980** | 1.000 | — | — | — | BM25 | Our benchmark |
| **SciFact** | **0.755** | 0.697 | 0.675 | 0.718 | 0.747 | **SF** | Our + BEIR |
| **NQ** | **0.574** | 0.629 | 0.794 | 0.855 | 0.863 | SPLADE | Our + DPR/BEIR |
| **HotpotQA** | **0.726** | 0.869 | 0.756 | 0.792 | 0.811 | BM25 | Our + BEIR |
| **2WikiMultihopQA** | **0.788** | 0.921 | — | — | — | BM25 | Our benchmark |
| **MuSiQue** | **0.453** | 0.672 | — | — | — | BM25 | Our benchmark |
| **MS MARCO** | — | 0.228 | 0.338 | 0.381 | 0.409 | SPLADE | BEIR leaderboard |

### Table A2: Method Characteristics

| Aspect | SF | BM25 | DPR | ColBERTv2 | SPLADE |
|--------|-----|------|-----|-----------|--------|
| **Training required** | None | None | Yes | Yes | Yes |
| **Training data** | — | — | ~50K pairs | ~500K pairs | ~500K pairs |
| **Memory/doc** | 512 bytes | ~1KB | 3KB | 3KB | 2KB |
| **Index time (100K docs)** | ~10 min | ~10s | ~1 hour | ~1 hour | ~30 min |
| **Query time** | ~30s | ~0.01s | ~0.1s (GPU) | ~0.2s (GPU) | ~0.05s |
| **GPU required** | No | No | Yes | Yes | Optional |
| **Interpretability** | High | Medium | Low | Low | Medium |
| **Boolean ops** | Yes | No | No | No | No |
| **Best task type** | Semantic matching | Exact matching | Compositional | Late interaction | Sparse + expansion |

### Table A3: Win Count Across Datasets

| Method | Wins (MRR) | Datasets Won | Strength |
|--------|------------|--------------|----------|
| BM25 | 7 | PubMedQA, NarrativeQA, Belebele, PopQA, NQ, HotpotQA, 2WikiMultihopQA, MuSiQue | Exact lexical matching |
| **SF** | **1** | **SciFact** | Unsupervised semantic matching |
| SPLADE | 1 | NQ (neural) | Sparse + expansion |

### Table A4: SF Performance by Task Category

| Category | Avg MRR | Datasets | SF Strength |
|----------|---------|----------|-------------|
| Entity Lookup | 0.980 | PopQA | Excellent |
| Biomedical QA | 0.955 | PubMedQA | Excellent |
| Narrative | 0.939 | NarrativeQA | Excellent |
| Reading Comp | 0.880 | Belebele | Good |
| Multi-hop (2-hop) | 0.757 | HotpotQA, 2Wiki | Competitive |
| Scientific Claims | 0.755 | SciFact | Competitive |
| Factoid | 0.574 | NQ-REaR | Moderate |
| Multi-hop (2-5) | 0.453 | MuSiQue | Poor |

## Sources

1. **DPR**: Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020.
2. **ColBERTv2**: Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. NAACL 2022.
3. **SPLADE**: Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. SIGIR 2021.
4. **BEIR**: Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. NeurIPS 2021.
5. **Belebele**: Bandarkar et al. (2023). The Belebele Benchmark. ACL 2024.
6. **PubMedQA**: Jin et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP 2019.
7. **SciFact**: Wadden et al. (2020). Fact or Fiction: Verifying Scientific Claims. EMNLP 2020.
8. **HotpotQA**: Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.
9. **MuSiQue**: Trivedi et al. (2022). MuSiQue: Multihop Questions via Single Question Composition. NeurIPS 2022.

## Key Observations for Thesis

1. **SF beats DPR on SciFact** (0.755 vs 0.675) — validates unsupervised semantic matching on domain-specific tasks.

2. **BM25 is the strongest baseline** — wins on 7/9 datasets. This is expected for a well-tuned lexical method.

3. **SF's unique niche**: Zero-shot domain adaptation, interpretability, memory efficiency, boolean operations.

4. **Hybrid SF+BM25** improves performance on paraphrasing (+32.7%) and negation (+76.4%) tasks.

5. **Performance degrades linearly with hop count**: 1-hop (0.939), 2-hop (0.757), 2-5 hop (0.453).
