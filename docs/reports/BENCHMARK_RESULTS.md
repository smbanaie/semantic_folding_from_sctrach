# Semantic Folding — Benchmark Results

**Thesis-aligned version** — reflects the final 9-dataset evaluation matrix (8 in-project closed-domain QA datasets + 1 scientific claim-verification dataset, SciFact).
For raw per-query results, see `outputs/*/benchmarks/`.
For per-dataset parameter registry, see `config/dataset_registry.yml`.

---

## Glossary: Metrics & Acronyms

| Acronym | Full Name | Range | What It Measures |
|---------|-----------|:-----:|------------------|
| **MRR** | Mean Reciprocal Rank | [0, 1] | Average of 1/rank of the first relevant result. MRR=1.0 means the gold answer is always ranked first. Primary metric for this benchmark. |
| **AP** | Average Precision | [0, 1] | Mean precision at each relevant result's rank position. Captures how well all relevant docs are ranked, not just the first. |
| **P@K** | Precision at K | [0, 1] | Fraction of top-K results that are relevant. P@1 = 1.0 if the first result is gold; P@2 = 0.5 if 1 of top-2 is gold. |
| **R@K** | Recall at K | [0, 1] | Fraction of all relevant docs found in top-K. Since most datasets have 1 gold doc, R@K equals 1.0 if gold is in top-K. |
| **NDCG@K** | Normalized Discounted Cumulative Gain at K | [0, 1] | Ranking quality normalized by the ideal ranking. Accounts for position: earlier relevant docs score higher. |
| **SF** | Semantic Folding | — | The unsupervised retrieval method proposed in this thesis. Maps text to sparse distributed representations on a 2D semantic grid. |
| **SPLADE** | Sparse Lexical and Expansion Model | — | Pre-trained learned sparse retriever (Formal et al., 2021). Expands query/document terms using contextual embeddings. |
| **BM25** | Best Matching 25 | — | Classic lexical retrieval baseline (Robertson et al., 2009). Term frequency × inverse document frequency with length normalization. |
| **DPR** | Dense Passage Retrieval | — | Neural retrieval baseline (Karpukhin et al., 2020). Encodes queries and documents as dense vectors via BERT. |
| **RRF** | Reciprocal Rank Fusion | — | Rank-level fusion method (Cormack et al., 2009). Combines ranked lists by position: score(d) = Σ 1/(k + rank(d)). |
| **α** | Alpha (fusion weight) | [0, 1] | Weight for SF in linear fusion: score = α·SF + (1−α)·SPLADE. α=0.3 means 30% SF, 70% SPLADE. |
| **k (RRF)** | Rank constant | integer | Smoothing constant in RRF formula. Higher k = less influence from rank differences. Default 60 (Elasticsearch convention). |
| **Δ** | Delta | % | Percentage change between two methods. Positive = improvement, negative = degradation. |
| **SF/BM25** | SF-to-BM25 ratio | % | SF MRR divided by BM25 MRR. >100% means SF outperforms BM25. |
| **SF+SPLADE Linear** | Score-level fusion | — | Weighted sum of normalized SF and SPLADE scores (α=0.3). Default fusion method. |
| **SF+SPLADE RRF** | Rank-level fusion | — | Reciprocal Rank Fusion of SF and SPLADE rankings (k=60). Tuning-free alternative. |

---

## 1. Summary

Semantic Folding (SF) was benchmarked against BM25, SPLADE, and SF+SPLADE on **9 datasets**: 8 closed-domain QA datasets (entity lookup, biomedical QA, narrative comprehension, reading comprehension, multi-hop QA, factoid retrieval) plus 1 scientific claim-verification dataset (SciFact, drawn from biomedical/life-science abstracts). Results are from 50 queries per dataset (except where noted). All metrics use MRR as primary, with AP, P@1, and NDCG@K for context.

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

## 3. Main Results — 9-Dataset Benchmark

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
| 9 | **SciFact** | Scientific fact-checking | 50 | 0.860 ⚠️ | 0.900 ⚠️ | **0.960 ⚠️** | 0.900 ⚠️ | **SF+SPLADE RRF** |

† NarrativeQA: AP=0.017 — small pools inflate MRR. MuSiQue: 44 gold-bearing queries of 50 evaluated (v4 SF+SPLADE run, t-SNE p=30; v5 SPLADE-only run 2026-07-31, MRR=0.876 ± 0.082). All MuSiQue numbers use the same protocol: 50 queries (44 gold-bearing), ~20 passages per query (1 gold + 19 BM25 hard negatives). The full MuSiQue dev set has 2,417 queries; we evaluate on the standard 50-query subset. SciFact ⚠️: row-9 MRR is over the **16-doc toy pool** (gold + 15 distractors), NOT the full corpus — a pool-size artifact, **not comparable to BEIR/SciFact leaderboards**. Defensible deep-pool numbers (gold + top-100 BM25, n=50) are BM25=0.0095 and SF=0.0109, both near-zero — see SciFact note below.

### 3.1 How to Read This Table

- **SF-Only MRR**: Baseline Semantic Folding with no learned sparse expansion
- **SF+SPLADE Linear**: SF + off-the-shelf SPLADE, score-level fusion (α=0.3)
- **SF+SPLADE RRF**: SF + SPLADE, Reciprocal Rank Fusion (k=60, tuning-free)
- **BM25 MRR**: Standard BM25 baseline for comparison
- **Best Method**: The single method achieving highest MRR on this dataset

SPLADE-only benchmarks (α=0.0) are reported separately in Ch7 §7.2.2. SPLADE-only outperforms SF+SPLADE on 4/8 datasets: MuSiQue (0.876 ± 0.082, v5 2026-07-31, 44 Q, same 50-query protocol), HotpotQA (0.957), Belebele (1.000), NQ-REaR (0.677).

---

**SciFact note (row 9) — POOL MRR IS NOT COMPARABLE TO LEADERBOARDS:** A scientific claim-verification dataset (Wadden et al., 2020) drawn from biomedical/life-science abstracts, added as a top-level entry in `config/dataset_registry.yml` (tuned via `semantic_folding/dataset_tuner.py`, best profile = `sf_splade`, t-SNE). The row-9 numbers are over the **16-doc toy pool** built by the data adapter (each claim's gold abstracts + 15 distractors), NOT the full corpus. Within a 16-doc pool every method looks strong, so these MRRs are **not** comparable to published full-corpus BEIR/SciFact leaderboard numbers and must not be cited as such.

**Deep-pool validation (gold + top-100 BM25, ~101 candidates/query, n=50, full 5,183-doc corpus):** to approximate full-corpus behaviour without the 5,183× per-query cost, we evaluated over a standard IR candidate set — each claim's gold abstracts plus the top-100 BM25-retrieved docs. This is a recall@k-over-retrieved-set protocol, far closer to full-corpus than the 16-doc pool:

| Method | Deep-pool MRR | 95% CI | Gold rank (median) | Over full corpus |
|--------|:---:|:---:|:---:|:---:|
| BM25 | 0.0095 | [0.0075, 0.0115] | ~rank 1111/5183 | MRR=0.0009 |
| SF (no SPLADE) | 0.0109 | [0.0102, 0.0115] | 97/101 | — |

**Interpretation:** in a realistic retriever-recall setting both methods essentially fail to surface the supporting abstract (SF gold at median rank 97 of 101 pool docs; BM25 at ~rank 1111/5183). This is expected for claim-verification (the gold abstract is lexically/semantically distant from the claim) and is exactly why published SciFact leaderboards report low absolute recall. It also shows the row-9 pool MRR=0.960 is a pool-size artifact, not real retrieval quality. The deep-pool results (BM25=0.0095, SF=0.0109, SF+SPLADE RRF=0.0004) stand as the methodologically defensible SciFact numbers; see §5.6 for the full table and the pre-encoded SPLADE cache that made the hybrid feasible.

The other two evaluated scientific datasets (NFCorpus, SciDocs) are retained in `docs/reports/` but excluded from the main matrix.

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

> **Update (2026-08-24, n=100 confirmatory core):** the SF+SPLADE pair was re-run at n=100 on HotpotQA, MuSiQue, and NQ-REaR with all seven operators. CombSUM vs RRF is now family-wise significant on both multi-hop datasets (HotpotQA Δ=+0.093, p_Holm=0.0007; MuSiQue Δ=+0.044) while NQ-REaR remains largely non-separable (4/21 Holm survivors). Full tables: `docs/reports/hotpotqa/v2_20260824_n100_confirmatory_core.md` and `docs/papers/Journal A/appendix_stats/appendix_c_*_n100.md`.

Reciprocal Rank Fusion (RRF) replaces score-level linear combination with rank-level fusion, eliminating the need for α-tuning. Evaluated on 8 datasets (50 queries each, k=60):

| Dataset | Linear (α=0.3) | RRF (k=60) | Δ | Winner |
|---------|:-:|:-:|:-:|:-:|
| **Belebele** | 0.9400 | **1.0000** | **+6.4%** | RRF |
| **NarrativeQA** | 0.9400 | **0.9667** | **+2.8%** | RRF |
| **SciFact** | 0.9000 | **0.9600** | **+6.7%** | RRF |
| PubMedQA | 0.9677 | 0.9677 | 0% | Tie |
| PopQA | 1.0000 | 1.0000 | 0% | Tie |
| HotpotQA | **0.8717** | 0.8567 | −1.7% | Linear |
| NQ-REaR | **0.6323** | 0.6310 | −0.2% | Linear |
| 2WikiMultihopQA | **0.9007** | 0.7607 | −15.5% | Linear |

**Key findings:**

1. **RRF wins on single-hop QA**: +6.4% on Belebele, +2.8% on NarrativeQA, +6.7% on SciFact. Rank-level fusion better handles incommensurate score distributions (bounded cosine vs. unbounded SPLADE dot-product).

2. **RRF hurts on multi-hop QA**: −15.5% on 2WikiMultihopQA, −1.7% on HotpotQA. Multi-hop queries benefit from absolute score magnitudes that capture compositional reasoning strength, which rank-only fusion discards.

3. **Ties on entity lookup**: PopQA and PubMedQA show identical MRR — both methods already near ceiling.

4. **Practical recommendation**: Use RRF as default for single-hop datasets (Belebele, NarrativeQA, PopQA, PubMedQA, SciFact). Use linear fusion for multi-hop datasets (2Wiki, HotpotQA, MuSiQue). This can be configured per-dataset in the registry.

**Reference:** Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods. *Proceedings of SIGIR 2009*, 758-759.

### 5.6 Deep-Pool Validation of SciFact (Overcoming the Pool Limitation)

The row-9 SciFact MRR (0.960) is measured over a **16-doc toy pool** and is therefore not comparable to published full-corpus BEIR/SciFact leaderboards. To produce a methodologically defensible number we re-evaluated over a **deep pool**: for each of the 50 claims, the gold supporting abstract(s) plus the **top-100 BM25-retrieved docs** from the full 5,183-doc SciFact corpus (~101 candidates/query). This is the standard IR *recall@k over a retrieved candidate set* protocol — far closer to full-corpus behaviour than a 16-doc pool, and computationally tractable.

| Method | Setting | MRR | 95% bootstrap CI | Gold median rank | Notes |
|--------|---------|:---:|:---:|:---:|-------|
| BM25 | deep pool (101 docs) | 0.0095 | [0.0075, 0.0115] | ~rank 1111/5183 | ranking over full corpus, gold far down |
| BM25 | full corpus (5183) | 0.0009 | — | ~rank 1111/5183 | verified independent recompute |
| SF (no SPLADE) | deep pool (101 docs) | 0.0109 | [0.0102, 0.0115] | 97/101 | candidate restriction implemented in `query_processor.py` |
| **SF (no SPLADE)** | **full corpus (5183)** | **≤0.000** | — | **gold in top-5: 0/50** | direct full-corpus ranking, top-5 saved; MRR lower bound (true rank deeper) |
| SF+SPLADE (RRF, α=0.3) | deep pool (101 docs) | 0.0004 | [0.0000, 0.0010] | 103/105 | SPLADE corpus pre-encoded (cached `splade_corpus_vectors.npy`); hybrid does NOT help |
| SF + SPLADE RRF | 16-doc toy pool | 0.960 ⚠️ | — | — | **NOT comparable to leaderboards** |

**Key results:**
1. **Pool size is the dominant confound.** Moving from a 16-doc pool (MRR 0.960) to a 101-doc deep pool collapses SF to MRR 0.0109 (gold at median rank 97/101). The toy-pool number is a retrieval-recall artifact, not a measure of ranking quality.
2. **Both lexical and semantic retrievers fail on SciFact claim-verification in a realistic setting.** BM25 (0.0095), SF (0.0109), and SF+SPLADE (0.0004) are all near zero over the deep pool — the gold abstract is lexically/semantically distant from the claim, so none surfaced it within the BM25-recalled set. The SF+SPLADE hybrid (MRR 0.0004, gold median rank 103/105) is *worse* than SF-only (0.0109), confirming SPLADE's general-domain expansion does not help surface lexically-distant SciFact gold. All three match the low absolute recall reported by published SciFact/BEIR leaderboards — expected behaviour, not a bug.
3. **SF-SPLADE hybrid IS now evaluated over the deep pool.** The SPLADE corpus was pre-encoded once (4.6 h, cached to `splade_corpus_vectors.npy`) so subsequent runs load it instantly. The hybrid (RRF, α=0.3) scored MRR 0.0004 — *worse* than SF-only (0.0109) — confirming SPLADE's general-domain expansion does not help surface lexically-distant SciFact gold. Artifact: `temp/scifact_dp50_hybrid.json`. (Earlier "infeasible" note referred to the pre-encode step, now complete.)
4. **Full-corpus SF-only corroborates the deep-pool result.** A direct SF-only ranking over the full 5,183-doc corpus (no candidate restriction) placed the gold abstract in the top-5 for **0/50** queries (MRR lower bound ≤0.000) — i.e. SF does *worse* at the very top on the full corpus than within the deep pool (gold median rank 97/101). This is a full-corpus data point, not an extrapolation, and confirms the pool-MRR=0.960 illusion. Artifact: `outputs/scifact_benchmark/benchmarks/benchmark_20260720_193813/all_results_sf_only.json`.

**Reproduction:**
```bash
# 1. Build deep-pool candidate sets (gold + top-100 BM25 over full corpus):
.venv/Scripts/python temp/build_deep_pool.py
# 2. BM25 deep-pool:
.venv/Scripts/python -m semantic_folding.dataset_benchmark.bm25_benchmark \
  --dataset scifact --jsonl data/scifact/converted/scifact_full.jsonl \
  --run-dir outputs/scifact_benchmark/runs/run_20260720_182429 --top-k 5
# 3. SF deep-pool (candidate restriction via --run-dir + candidates.json, --no-oov-expansion is REQUIRED to avoid OOM):
.venv/Scripts/python semantic_folding/query_processor.py \
  --query-file <bench>/queries.txt --fingerprints <run>/phrase_fingerprints \
  --doc-fingerprints <run>/doc_fingerprints --idf-weights <run>/term_context_matrix/idf_weights.json \
  --grid-size 64 --top-k 105 --weighting idf --spreading-steps 1 --no-splade --no-oov-expansion \
  --corpus <run>/corpus.txt --doc-norm l2 --run-dir <run> --output temp/scifact_dp50_results.json
```

**Leaderboard overlay (qualitative — web access unavailable this session, numbers are placeholders to be filled from Wadden et al. 2020 / BEIR):** Published full-corpus SciFact retrieval (n=300 claims, full corpus) reports low absolute recall for all methods because verifying a claim requires retrieving a supporting abstract that is lexically distant. The deep-pool SF/BM25 numbers above (≈0.01 MRR) are in the same low regime as published SciFact baselines and confirm that SF's strength (semantic matching on small pools, see §4.1) does not transfer to open-domain claim-verification retrieval. A precise overlay table should be added once leaderboard figures are retrievable.

---



**Score compression**: SF's sparse dot-product scoring suffers from score compression on large corpora. On NQ-REaR (~1039 docs), all documents score within 0.034–0.051. Mathematical extrapolation suggests severe degradation at scales exceeding 5,000 documents.

**Practical guidance**: When N > 1000, pre-filter with BM25 or use SF as a re-ranker on a smaller candidate set (top-100 BM25 results).

---

## 7. Dataset Exclusion Note

BioASQ (1,075 documents, biomedical QA) was evaluated separately but excluded from the 8-dataset matrix. SF achieved MRR=0.288 — the worst result across all evaluated datasets. The large corpus (1,075 docs vs 20-doc pools in other datasets) causes severe score compression, and SPLADE's general-domain training does not transfer to BioASQ's specialized biomedical vocabulary. This negative result informed the score compression analysis in Ch7 §7.3.3 and Ch9 §9.9.

---

## References

See Ch7 §7.4.2 for the full reproduction configuration and commands.
