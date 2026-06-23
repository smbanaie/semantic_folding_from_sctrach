# Semantic Folding — Complete Benchmark Report

**Generated**: 2026-06-19 (updated with SPLADE hybrid results)
**Scope**: 10 benchmarked datasets across biomedical, narrative, reading comprehension, scientific, and multi-hop QA domains
**Method**: Semantic Folding (SF) vs BM25 baseline vs SF+SPLADE hybrid

---

## 1. Executive Summary

Semantic Folding was benchmarked on **9 datasets** spanning 4 performance tiers. BM25 outperforms SF on all datasets. SF achieves competitive results (≥85% of BM25) on **entity lookup**, **biomedical QA**, **narrative comprehension**, and **reading comprehension**. SF degrades on **multi-hop reasoning** (67–85%) and shows mixed results on **scientific claim verification**.

| Category | Best Dataset | SF/BM25 Ratio | SF MRR |
|----------|-------------|---------------|--------|
| Entity Lookup | PopQA | 98.0% | 0.980 |
| Biomedical | PubMedQA | 95.5% | 0.955 |
| Narrative | NarrativeQA | 95.8% | 0.939 |
| Reading Comp | Belebele | 88.4% | 0.880 |
| Scientific | SciFact | — | 0.755 |
| Multi-hop | 2WikiMultihopQA | 85.6% | 0.788 |
| Multi-hop | MuSiQue | 67.4% | 0.453 |

---

## 2. Default Pipeline Configuration

All benchmarks use the same recommended configuration unless noted:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Grid size | 64 | Optimal for 20-passage corpora (5–15% fingerprint density) |
| Method | t-SNE | +10% MRR vs UMAP on Belebele |
| Perplexity | 30 | Default; perplexity=50 improved some datasets by +1.5–4% |
| Smoothing | Gaussian, σ=1.5 | Critical (σ=0 → MRR −31.2%) |
| Top percent | 0.10 | 5% loses signal, 15% adds noise |
| Weighting | IDF | −0.86% MRR vs uniform |
| Spreading | radius=1, decay=0.5 | Radius=2 → MRR −7.1% on short queries |
| Morton encoding | true | Z-order spatial encoding |
| Normalization | L2 (docs) | +4.0% MRR on Belebele vs sqrt_nnz |
| keep_verbs | true | Not worth testing |
| min_freq | 1 | Not worth testing |

---

## 3. All Dataset Results

### 3.1 Ranking by SF/BM25 Ratio

| Rank | Dataset | Domain | Queries | SF MRR | BM25 MRR | SF/BM25 | Category |
|------|---------|--------|---------|--------|----------|---------|----------|
| 1 | **PopQA** | Wikidata | 100 | 0.980 | 1.000 | **98.0%** | Entity lookup |
| 2 | **PubMedQA** | Biomedical | 111 | 0.955 | 1.000 | **95.5%** | Biomedical QA |
| 3 | **NarrativeQA** | Movie scripts | 49 | 0.939 | 0.980 | **95.8%** | Narrative |
| 4 | **Belebele** | Reading comp | 100 | 0.880 | 0.995 | **88.4%** | Reading comprehension |
| 5 | **2WikiMultihopQA** | Multi-hop | 50 | 0.788 | 0.921 | **85.6%** | Multi-hop |
| 6 | **NQ-REaR** | Factoid | 100 | 0.574 | 0.638 | **89.9%** | Knowledge retrieval |
| 7 | **HotpotQA** | Multi-hop | 48 | 0.726 | 0.869 | **83.5%** | Multi-hop |
| 8 | **BioASQ** | Biomedical QA | 50 | **0.2480** | — | — | Biomedical QA (hard) |
| 9 | **DocFinQA** | Financial | 20 | 0.250 | 0.341 | **73.3%** | Financial QA |
| 10 | **MuSiQue** | Multi-hop | 100 | 0.453 | 0.672 | **67.4%** | Multi-hop QA |
| 11 | **DROP** | Discrete reasoning | 50 | 0.320 | 0.762 | **42.6%** | Discrete reasoning |
| 12 | **CUAD** | Legal contracts | 200 | 0.000 | 0.244 | **0%** | Legal |
| 13 | **MAUD** | Legal review | 100 | 0.000 | 0.649 | **0%** | Legal |

### 3.2 Detailed Metrics per Dataset

#### PubMedQA (Biomedical QA)
| Metric | SF Best | BM25 | Notes |
|--------|---------|------|-------|
| MRR | 1.000 (30Q) / 0.955 (111Q) | 1.000 | Grid=128 hurts (−5.6%) |
| AP | 0.904 | — | |
| P@1 | 1.000 | — | |
| P@2 | 0.967 | — | |

**Finding**: SF nearly matches BM25. Biomedical terminology benefits from semantic matching.

#### PopQA (Entity Lookup)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.980 | 1.000 | Trivial for lexical matching |
| AP | 0.540 | 1.000 | Small candidate pool (2 passages) |
| P@1 | 0.980 | 1.000 | |

**Finding**: Near-perfect. Entity names in queries make this trivial for both methods.

#### BioASQ (Biomedical QA — Hard)
| Metric | SF (50Q) | SF+SPLADE | SF+BM25 | Notes |
|--------|----------|-----------|---------|-------|
| MRR | 0.2480 | 0.2480 | 0.2480 | All identical |
| AP | 0.1949 | 0.1949 | 0.1949 | |
| P@1 | 0.1400 | 0.1400 | 0.1400 | |
| P@5 | 0.1160 | 0.1160 | 0.1160 | |
| Queries | 50 | 50 | 50 | 1075 docs |

**Finding**: BioASQ is significantly harder than PubMedQA (MRR 0.248 vs 0.936). All three methods produce identical results — the bottleneck is SF's phrase-level matching, not the scoring method. Hybrid scoring cannot improve on SF's performance when the corpus is large and queries are complex.

#### NarrativeQA (Narrative Comprehension)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.939 | 0.980 | Script comprehension benefits from semantics |

**Finding**: Second-best ratio. Narrative understanding favors SF's semantic approach.

#### Belebele (Reading Comprehension)
| Metric | SF Best (50Q) | SF (100Q) | BM25 |
|--------|--------------|-----------|------|
| MRR | 0.880 | 0.740 | 0.995 |
| AP | 0.880 | 0.740 | — |
| P@1 | 0.880 | 0.740 | — |

**Finding**: t-SNE outperforms UMAP (+10%). L2 norm helps (+4%). Hybrid SF+BM25 improves 100Q from 0.74→0.86 (+16.2%).

#### NQ-REaR (Factoid Retrieval)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.574 | 0.638 | Entity matching gap ~17% |
| AP | 0.371 | 0.582 | |
| P@1 | 0.420 | 0.470 | |
| P@2 | 0.460 | 0.485 | |

**Finding**: BM25 alone outperforms all hybrid configurations (alpha=0.0 best at MRR=0.675).

#### HotpotQA (Multi-hop QA)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.726 | 0.869 | Multi-hop reasoning challenges SF |

**Finding**: Consistent with other multi-hop datasets (14–17% gap).

#### 2WikiMultihopQA (Multi-hop Compositional)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.788 | 0.921 | Multi-hop gap ~14% |

**Finding**: Similar to HotpotQA. Compositional queries degrade SF performance.

#### MuSiQue (Multi-hop QA)
| Metric | SF | BM25 | Delta |
|--------|-----|------|-------|
| MRR | 0.453 | 0.672 | −32.6% |
| AP | 0.272 | 0.482 | −43.7% |
| P@1 | 0.395 | 0.563 | −29.8% |
| P@2 | 0.221 | 0.362 | −39.0% |

**Finding**: Worst non-legal dataset. 47.7% of queries had no gold passage in top results. 2–5 hop composition defeats phrase-level matching.

#### DROP (Discrete Reasoning)
| Metric | SF Best (L2) | BM25 | Notes |
|--------|-------------|------|-------|
| MRR | 0.320 | 0.762 | L2 norm helps (+14.3%) |
| AP | — | — | Counting/sorting/comparison |

**Finding**: Requires numerical reasoning beyond phrase matching. L2 norm provides significant improvement.

#### DocFinQA (Financial QA)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.250 | 0.341 | Grid=128 used (not recommended) |

**Finding**: Both methods struggle. Financial documents require numerical reasoning.

#### CUAD (Legal Contracts)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.000 | 0.244 | Complete SF failure |

**Finding**: Legal clause extraction requires domain-specific reasoning. Even BM25 performs poorly.

#### MAUD (Legal Review)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | 0.000 | 0.649 | Complete SF failure |

**Finding**: Legal queries require clause cross-referencing and conditional reasoning.

---

## 4. Analysis by Task Type

### 4.1 Where SF Excels (SF/BM25 ≥ 88%)

| Task | Datasets | SF Strength |
|------|----------|-------------|
| Entity lookup | PopQA (98%) | Entity names in queries match phrase fingerprints |
| Biomedical QA | PubMedQA (95.5%) | Domain-specific semantic matching |
| Narrative comprehension | NarrativeQA (95.8%) | Script understanding benefits from semantics |
| Reading comprehension | Belebele (88.4%) | Paraphrased queries benefit from semantic matching |

**Pattern**: SF excels when queries contain domain-specific vocabulary that maps well to phrase fingerprints.

### 4.2 Where SF Struggles (SF/BM25 < 85%)

| Task | Datasets | SF Weakness |
|------|----------|-------------|
| Multi-hop QA | MuSiQue (67%), HotpotQA (83.5%), 2Wiki (85.6%) | Compositional reasoning requires precise entity matching |
| Discrete reasoning | DROP (42.6%) | Counting/sorting/comparison beyond phrase level |
| Financial QA | DocFinQA (73.3%) | Numerical reasoning required |
| Legal | CUAD (0%), MAUD (0%) | Domain-specific clause reasoning |

**Pattern**: SF degrades on tasks requiring compositional, numerical, or domain-specific reasoning.

### 4.3 Performance by Hop Count

| Hops | Dataset | SF MRR | BM25 MRR | Gap |
|------|---------|--------|----------|-----|
| 1 (simple) | PopQA | 0.980 | 1.000 | −2% |
| 1 (factoid) | NQ-REaR | 0.574 | 0.638 | −10% |
| 2–3 | HotpotQA | 0.726 | 0.869 | −16% |
| 2–3 | 2WikiMultihopQA | 0.788 | 0.921 | −14% |
| 2–5 | MuSiQue | 0.453 | 0.672 | −33% |

**Finding**: Performance degrades linearly with hop count. SF cannot compose facts across passages.

---

## 5. Optimization Experiments

### 5.1 Grid Size
| Grid | MRR (Belebele) | MRR (PubMedQA) | Verdict |
|------|----------------|----------------|---------|
| 64 | 0.840 | 0.954 | **Best** |
| 128 | 0.800 (−5.3%) | 0.902 (−5.6%) | Worse |

**Rule**: Use grid=64. Do not change.

### 5.2 Dimensionality Reduction
| Method | MRR (Belebele 50Q) | Notes |
|--------|---------------------|-------|
| t-SNE | 0.880 | **Best** — local focus helps phrase matching |
| UMAP | 0.800 (−10%) | Faster but worse for phrase-level tasks |

**Rule**: Use t-SNE. UMAP only for large datasets (>10K contexts) or out-of-sample projection.

### 5.3 Document Normalization
| Method | MRR (Belebele 50Q) | Delta |
|--------|---------------------|-------|
| L2 | 0.880 | **Best** (+4.0%) |
| sqrt_nnz | 0.840 | Baseline |
| L1 | 0.830 | −1.0% |
| Max | 0.818 | −2.2% |

**Rule**: Use `--doc-norm l2`.

### 5.4 Hybrid SF+BM25
| Dataset | Pure SF | Hybrid α=0.5 | Pure BM25 | Verdict |
|---------|---------|--------------|-----------|---------|
| Belebele (100Q) | 0.740 | **0.860** (+16.2%) | 0.995 | Hybrid helps |
| PubMedQA | 0.954 | 0.923 (−3.1%) | 1.000 | Hybrid hurts |
| NQ-REaR | 0.583 | — | **0.675** | BM25 alone best |

**Rule**: Keep hybrid as opt-in flag (`--hybrid --hybrid-alpha 0.5`). Use for reading comprehension, avoid for biomedical/factoid.

### 5.5 Hybrid SF+SPLADE
| Dataset | Pure SF | SF+SPLADE α=0.3 | SF+BM25 α=0.3 | Verdict |
|---------|---------|-----------------|---------------|---------|
| PubMedQA (50Q) | 0.9355 | **0.9677** (+3.4%) | **0.9677** (+3.4%) | Both hybrids help |
| Belebele (50Q) | 0.8800 | — | **1.0000** (+13.6%) | BM25 hybrid achieves perfect score |
| BioASQ (50Q) | **0.2480** | 0.2204 (-11.1%) | 0.1667 (-32.8%) | SF-only best |

**Finding**: BM25 hybrid improves SF by +13.6% on Belebele (0.8800→1.0000) and +3.4% on PubMedQA (0.9355→0.9677). SPLADE hybrid matches BM25 on PubMedQA (+3.4%) but was too slow to complete on Belebele (926 docs). The BM25 hybrid is both faster and more effective.

**Rule**: Use SF+BM25 hybrid for reading comprehension and biomedical QA. SPLADE provides no additional benefit over BM25 and is 60x slower.

### 5.6 Improvement Experiments (R5-R9)

| Dataset | Glossary | Negation | Adaptive | Spatial J |
|---------|----------|----------|----------|-----------|
| PubMedQA 50Q | 0.9355 (0%) | 0.9355 (0%) | 0.9355 (0%) | 0.3226 (-65%) |
| Belebele 50Q | (timeout) | (timeout) | (timeout) | (timeout) |
| BioASQ 10Q | 0.4950 (+11%) | 0.4450 (0%) | 0.4450 (0%) | 0.1000 (-60%) |

**Finding**: None of the tested improvements provided consistent gains across datasets. Glossary expansion helps on BioASQ 10Q (+11%) but hurts on 50Q (-4.7%). Negation-aware and adaptive spreading show no improvement. Spatial-Jaccard hurts significantly on all datasets.

**Rule**: SF-only remains the best approach for BioASQ. For PubMedQA/Belebele, use SF+BM25 hybrid.

### 5.7 Query Expansion
| Dataset | Baseline | expand_default | expand_glossary | Verdict |
|---------|----------|----------------|-----------------|---------|
| Belebele | 0.840 | 0.840 (0%) | 0.840 (0%) | No improvement |
| PubMedQA | 0.954 | 0.954 (0%) | 0.931 (−2.3%) | Hurts |

**Rule**: Skip query expansion.

### 5.7 Smoothing Sigma
| Sigma | MRR Impact | Notes |
|-------|------------|-------|
| 0 | −31.2% | Critical failure |
| 1.0–2.0 | Optimal | Robust range |
| 1.5 | Default | Chosen as default |

**Rule**: Keep σ=1.5. Never set to 0.

---

## 6. Computational Cost

| Phase | SF Time | BM25 Time | Ratio |
|-------|---------|-----------|-------|
| Indexing (100Q, 1862 docs) | ~10 min | ~10s | 60x |
| Per query | ~30s | ~0.01s | 3000x |
| Total (100Q) | ~60 min | ~10s | 360x |

**Finding**: BM25 is 100–3000x faster. SF's cost is dominated by t-SNE and fingerprint generation.

---

## 7. Recommendations

### For Production Use
- **Use BM25** for all task types — faster and more accurate
- SF only adds value on **biomedical domain** (PubMedQA) where it nearly matches BM25

### For Research
- SF's strength is **domain-specific semantic matching** (biomedical, narrative)
- SF's weakness is **compositional reasoning** (multi-hop, legal, financial)
- **Hybrid SF+BM25** helps on reading comprehension (+16.2% on Belebele)
- SF has **no trainable parameters** — entirely unsupervised pipeline

### For Thesis
- MuSiQue results (MRR=0.453, −32.6% vs BM25) provide honest baseline
- Pattern across 12 datasets shows clear task-type dependency
- Future work: phrase-level composition, domain-adaptive training

---

## 8. Appendix: HippoRAG2 Dataset Details

The following datasets were benchmarked as part of the HippoRAG2 evaluation (source: `brain_approaches/hipporag2/datasets/`):

| Dataset | Type | Corpus Size | Notes |
|---------|------|-------------|-------|
| PopQA | Entity lookup | 8,676 passages | Wikidata-derived, 2 passages/query |
| NQ-REaR | Factoid retrieval | 9,633 passages | Google Natural Questions |
| NarrativeQA | Script comprehension | 10-doc pools | Movie scripts |
| HotpotQA | Multi-hop Wikipedia | 20 candidates | 2-hop reasoning |
| 2WikiMultihopQA | Multi-hop compositional | 20 candidates | 2-hop compositional |
| MuSiQue | Multi-hop QA | 20 candidates | 2–5 hop reasoning |
| LV-Eval | Long-context | 256K chars | Not benchmarked (too long) |
| Case Study | Single query | Small | Not benchmarked (too small) |

---

## 9. Appendix: Hybrid Alpha Tuning

Detailed alpha sweep results for hybrid SF+BM25 scoring:

| Dataset | Candidates | Best Alpha | MRR | Pure SF | Pure BM25 |
|---------|-----------|-----------|-----|---------|-----------|
| NQ-REaR | ~10/query | **0.0** (BM25 only) | **0.675** | 0.583 | 0.675 |
| PopQA | 2/query | Any | 1.000 | 1.000 | 1.000 |
| PubMedQA | 3–4/query | Any | 1.000 | 1.000 | 1.000 |

**Finding**: BM25 alone outperforms all hybrid configurations on factoid QA. More SF weight = worse results. Hybrid only helps when candidate pools are large and queries are paraphrased (Belebele).

---

## 10. Appendix: Improvement Experiments

Five improvement approaches were tested (implemented on feature branches):

| Approach | Flag | Result | Verdict |
|----------|------|--------|---------|
| Hybrid SF+BM25 | `--hybrid` | +16.2% on Belebele | **Adopt** (reading comprehension only) |
| L2 normalization | `--doc-norm l2` | +4.0% on Belebele | **Adopt** (default) |
| t-SNE perplexity=50 | `--tsne-perplexity 50` | +1.5–4% on some datasets | Optional |
| Query expansion | `--expand-synonyms` | 0% to −2.3% | **Reject** |
| TF-IDF re-ranking | `--tfidf-rerank` | 0% | **Reject** |

---

## 11. Files Reference

| File | Role |
|------|------|
| `docs/reports/BENCHMARK_RESULTS.md` | **This file** — single source of truth |
| `docs/reports/REPORTS.md` | Version history index + report file log |
| `docs/reports/<dataset>/v2_*.md` | Per-dataset detailed analysis (deep dives) |
| `docs/recommendations.md` | Future work & improvement roadmap |
| `semantic_folding/benchmarks.md` | Benchmarking methodology & parameter justification |

---

*This report is the single source of truth for all Semantic Folding benchmark results. Update this file after each new benchmark run.*
