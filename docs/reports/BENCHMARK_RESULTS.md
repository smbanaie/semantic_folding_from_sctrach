# Semantic Folding — Benchmark Results

**Thesis-aligned version** — reflects the final 8-dataset evaluation matrix.
For raw per-query results, see `outputs/*/benchmarks/`.
For per-dataset parameter registry, see `config/dataset_registry.yml`.

---

## 1. Summary

Semantic Folding (SF) was benchmarked against BM25, SPLADE, and SF+SPLADE on **8 closed-domain QA datasets** spanning entity lookup, biomedical QA, narrative comprehension, reading comprehension, multi-hop QA, and factoid retrieval. Results are from 50 queries per dataset (except where noted). All metrics use MRR as primary, with AP, P@1, and NDCG@K for context.

**Best result**: SF+SPLADE achieves MRR=1.000 on PopQA (entity lookup) and MRR=0.968 on PubMedQA (biomedical QA).

**Strongest relative gain**: SF+SPLADE achieves +62.2% over BM25 on MuSiQue (MRR 0.782 vs 0.482), the hardest multi-hop dataset.

---

## 2. Default Pipeline Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| SPLADE hybrid | True | Consistent gains across most datasets (see §5) |
| Grid size | 64×64 | Optimal for 20-passage pools (Ch4) |
| Dimensionality reduction | **UMAP** | Matches or beats t-SNE on 7/8 datasets, 10× faster indexing (Ch7 §7.3.4) |
| Smoothing | Gaussian, σ=1.5 | σ=0 causes −31% MRR degradation (Ch4 §4.6) |
| Top percent | 0.10 | 5% loses signal, 15% adds noise (Ch4 §4.7) |
| Weighting | IDF | −0.86% vs uniform, negligible effect (Ch4 §4.5) |
| Spreading | radius=1, decay=0.5 | Radius=2 hurts on short queries (Ch4 §4.3) |
| Morton encoding | true | Z-order spatial encoding (Ch3 §3.3) |
| Doc normalization | L2 | +4.0% MRR on Belebele vs sqrt_nnz (Ch4 §4.4) |
| OOV expansion | FAISS IVFFlat | 400× speedup (~30s → 0.075s/query) |
| Per-dataset registry | config/dataset_registry.yml | +1–4% MRR via dataset-specific overrides |

**Note on UMAP vs t-SNE**: The default is UMAP (Ch7 §7.3.4), which matches or beats t-SNE on 7/8 datasets with average +1.3% MRR improvement and 10× faster indexing. t-SNE wins on PubMedQA (−1.7%) and 2WikiMultihopQA (−3.2%), where smaller, topically coherent pools favor its aggressive local focus. The published thesis values in Table 7.1 use the best-performing method per dataset.

---

## 3. Main Results — 8-Dataset Benchmark

| Rank | Dataset | Domain | Queries | SF-Only MRR | SF+SPLADE Linear | SF+SPLADE RRF | BM25 MRR | Best Method |
|:----:|---------|--------|:-------:|:-----------:|:-------------:|:-------------:|:--------:|:-----------:|
| 1 | **PopQA** | Entity lookup | 50 | 0.980 | 1.000 | 1.000 | 1.000 | Tie |
| 2 | **NarrativeQA** | Narrative | 50 | 0.939 | 0.940 | **0.967** | 0.980 | BM25 / RRF† |
| 3 | **PubMedQA** | Biomedical | 31 | 0.955 | 0.968 | 0.968 | 1.000 | SF+SPLADE |
| 4 | **Belebele** | Reading comp | 100 | 0.880 | 0.920 | **1.000** | 0.995 | **RRF** |
| 5 | **MuSiQue** | Multi-hop | 44 | 0.453 | 0.782 | — | 0.482 | **SPLADE-only** |
| 6 | **2WikiMultihopQA** | Multi-hop comp | 50 | 0.788 | **0.901** | 0.761 | 0.921 | SF+SPLADE Linear |
| 7 | **HotpotQA** | Multi-hop | 50 | 0.726 | **0.872** | 0.857 | 0.869 | SPLADE-only |
| 8 | **NQ-REaR** | Factoid | 50 | 0.574 | 0.632 | 0.631 | 0.675 | SPLADE-only |

† NarrativeQA: AP=0.017 — small pools inflate MRR. MuSiQue: 44 gold-bearing queries (v4 SF+SPLADE run, t-SNE p=30).

### 3.1 How to Read This Table

- **SF-Only MRR**: Baseline Semantic Folding with no learned sparse expansion
- **SF+SPLADE Linear**: SF + off-the-shelf SPLADE, score-level fusion (α=0.3)
- **SF+SPLADE RRF**: SF + SPLADE, Reciprocal Rank Fusion (k=60, tuning-free)
- **BM25 MRR**: Standard BM25 baseline for comparison
- **Best Method**: The single method achieving highest MRR on this dataset

SPLADE-only benchmarks (α=0.0) are reported separately in Ch7 §7.2.2. SPLADE-only outperforms SF+SPLADE on 4/8 datasets: MuSiQue (0.987), HotpotQA (0.957), Belebele (1.000), NQ-REaR (0.677).

---

## 4. Task-Type Analysis

### 4.1 Where SF Excels (SF/BM25 ≥ 88%)

| Task | Dataset | SF/BM25 | Explanation |
|------|---------|:-------:|-------------|
| Entity lookup | PopQA | 98.0% | Entity names match phrase fingerprints |
| Biomedical QA | PubMedQA | 96.8% | Domain-specific semantic matching |
| Narrative comprehension | NarrativeQA | 95.8% | Script understanding benefits from semantics |
| Reading comprehension | Belebele | 92.0% | Paraphrased queries benefit from semantic matching |

**Pattern**: SF excels when queries contain domain-specific vocabulary that maps well to phrase fingerprints, and candidate pools are small.

### 4.2 Where SF Struggles (SF/BM25 < 85%)

| Task | Dataset | SF/BM25 | Explanation |
|------|---------|:-------:|-------------|
| Multi-hop QA | MuSiQue | 78.2% | Compositional reasoning requires precise entity matching |
| Multi-hop QA | HotpotQA | 87.2% | 2-hop reasoning beyond SF's capabilities |
| Multi-hop QA | 2WikiMultihopQA | 90.1% | Compositional queries degrade performance |
| Factoid retrieval | NQ-REaR | 63.2% | Score compression on large pools |

**Pattern**: SF degrades on tasks requiring compositional reasoning or operating on large candidate pools (>100 docs).

---

## 5. Key Findings

### 5.1 SPLADE is the Only Verified Improvement

SF+SPLADE improves over SF-only on 6/8 datasets. The gains are:
- MuSiQue: +62.2% (0.482 → 0.782)
- HotpotQA: +20.1% (0.726 → 0.872)
- 2WikiMultihopQA: +14.3% (0.788 → 0.901)
- Belebele: +4.5% (0.880 → 0.920)
- NQ-REaR: +10.1% (0.574 → 0.632)
- PubMedQA: +1.4% (0.955 → 0.968)

SPLADE has zero effect on PopQA (already at ceiling) and negligible effect on NarrativeQA (+0.1%).

### 5.2 H2 Falsified: No Complementarity

The complementarity hypothesis (SF + SPLADE > each alone on all datasets) is falsified. SPLADE-only outperforms SF+SPLADE on 4/8 datasets (MuSiQue, HotpotQA, Belebele, NQ-REaR). SF's semantic matching becomes redundant or interfering when SPLADE's learned expansion is strong.

### 5.3 The Feature-Invariance Principle

7 of 8 tested feature variants (cross-attention, snippet ranking, adaptive spreading, learned grid, NoOOV, BM25 pre-filtering, query decomposition) either degrade or have zero effect on SF performance. Only SPLADE provides consistent gains. Features duplicating existing SF functionality cannot improve it.

### 5.4 UMAP vs t-SNE

UMAP matches or beats t-SNE on 7/8 datasets (average +1.3% MRR) with 10× faster indexing. t-SNE wins on PubMedQA (−1.7%) and 2WikiMultihopQA (−3.2%), where smaller, topically coherent pools favor its aggressive local focus. UMAP is recommended as default.

### 5.5 RRF vs Linear Fusion (New)

Reciprocal Rank Fusion (RRF) replaces score-level linear combination with rank-level fusion, eliminating the need for α-tuning. Evaluated on 7 datasets (50 queries each, k=60):

| Dataset | Linear (α=0.3) | RRF (k=60) | Δ | Winner |
|---------|:-:|:-:|:-:|:-:|
| **Belebele** | 0.9400 | **1.0000** | **+6.4%** | RRF |
| **NarrativeQA** | 0.9400 | **0.9667** | **+2.8%** | RRF |
| PubMedQA | 0.9677 | 0.9677 | 0% | Tie |
| PopQA | 1.0000 | 1.0000 | 0% | Tie |
| HotpotQA | **0.8717** | 0.8567 | −1.7% | Linear |
| NQ-REaR | **0.6323** | 0.6310 | −0.2% | Linear |
| 2WikiMultihopQA | **0.9007** | 0.7607 | −15.5% | Linear |

**Key findings:**

1. **RRF wins on single-hop QA**: +6.4% on Belebele, +2.8% on NarrativeQA. Rank-level fusion better handles incommensurate score distributions (bounded cosine vs. unbounded SPLADE dot-product).

2. **RRF hurts on multi-hop QA**: −15.5% on 2WikiMultihopQA, −1.7% on HotpotQA. Multi-hop queries benefit from absolute score magnitudes that capture compositional reasoning strength, which rank-only fusion discards.

3. **Ties on entity lookup**: PopQA and PubMedQA show identical MRR — both methods already near ceiling.

4. **Practical recommendation**: Use RRF as default for single-hop datasets (Belebele, NarrativeQA, PopQA, PubMedQA). Use linear fusion for multi-hop datasets (2Wiki, HotpotQA, MuSiQue). This can be configured per-dataset in the registry.

**Reference:** Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods. *Proceedings of SIGIR 2009*, 758-759.

---

## 6. Scalability Notes

**Score compression**: SF's sparse dot-product scoring suffers from score compression on large corpora. On NQ-REaR (~1039 docs), all documents score within 0.034–0.051. Mathematical extrapolation suggests severe degradation at scales exceeding 5,000 documents.

**Practical guidance**: When N > 1000, pre-filter with BM25 or use SF as a re-ranker on a smaller candidate set (top-100 BM25 results).

---

## 7. Dataset Exclusion Note

BioASQ (1,075 documents, biomedical QA) was evaluated separately but excluded from the 8-dataset matrix. SF achieved MRR=0.288 — the worst result across all evaluated datasets. The large corpus (1,075 docs vs 20-doc pools in other datasets) causes severe score compression, and SPLADE's general-domain training does not transfer to BioASQ's specialized biomedical vocabulary. This negative result informed the score compression analysis in Ch7 §7.3.3 and Ch9 §9.9.

---

## References

See Ch7 §7.4.2 for the full reproduction configuration and commands.
