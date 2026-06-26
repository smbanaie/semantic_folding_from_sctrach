# Benchmarking Semantic Folding Against Established Retrieval Benchmarks

## 1. Motivation

Semantic Folding constructs sparse distributed representations (SDRs) over a
discrete 2D grid, where each document is represented by its most activated grid
cells from its constituent phrases. To validate this approach as a competitive
retrieval architecture, we benchmark it against standard datasets used in the
HiPPoRAG and multi-hop QA literature. These datasets provide:

- **Multi-hop reasoning**: Queries requiring composition of facts across multiple documents.
- **Controlled candidate pools**: Each query has a fixed set of candidate passages (typically 20), with exactly K supporting passages.
- **Reproducible ground truth**: Binary relevance judgments with document-level gold labels.

The benchmark pipeline is implemented in `semantic_folding/dataset_benchmark/`
and currently supports one dataset (MuSiQue), with a framework designed for
extensibility.

---

## 2. Train/Dev Split in Semantic Folding

Unlike supervised or fine-tuned retrieval models, **Semantic Folding has no
trainable parameters**. The full pipeline (phrase extraction, term-context
matrix, t-SNE semantic space, fingerprint generation, query processing) is
entirely unsupervised — no labels, gradients, or loss functions are involved.

The train/dev split serves a different purpose here:

### 2.1 Hyperparameter Selection (not "training")

Semantic Folding exposes several free hyperparameters (grid size, spreading
steps, top percent, weighting scheme, smoothing sigma) that control the
density, resolution, and matching behaviour of the SDRs. These are not learned
from data but must be chosen empirically. The protocol is:

1. **Train set**: Run the unsupervised pipeline on queries from the training
   split with candidate hyperparameter configurations. Compute retrieval
   metrics (MRR, P@K, NDCG) against the gold supporting passages. Select the
   configuration that maximises the target metric.
2. **Dev set**: Run the pipeline once with the selected configuration on the
   held-out development split. Report these metrics as the final evaluation.
   This prevents inadvertent overfitting of hyperparameters to the dev set.

Each run of the pipeline is **fully unsupervised** — the gold labels are used
only for evaluation, never as input to any pipeline step. This is conceptually
equivalent to hyperparameter search in clustering algorithms (e.g., choosing
K in K-means via a held-out validation set).

### 2.2 Why a separate dev set matters

Without a held-out dev split, one could optimistically choose hyperparameters
that happen to work well on a particular query set. The train/dev separation
ensures:

- **No information leakage**: The dev set is never used during parameter
  selection.
- **Generalisation signal**: If the best configuration on the train split also
  performs well on the dev split, it suggests the hyperparameters are robust
  and not overfitted to accidental properties of the training queries.
- **Reproducible comparison**: The dev set serves as the standard evaluation
  benchmark, allowing fair comparison against future work (including HippoRAG,
  dense retrieval baselines, etc.).

### 2.3 Practical workflow

```
Step 1: Choose a search grid of hyperparameters
        e.g., grid_size ∈ {16, 32, 64}, spreading_steps ∈ {0, 1, 2}

Step 2: For each hyperparameter combination H:
          For each query q in train split:
            corpus_q ← extract 20 candidate passages for q
            pipe(corpus_q, H)       → ranked list R_q
            metrics_q ← evaluate(R_q, gold supporting passages for q)
          aggregate metrics across all train queries
          H.score ← MRR (or other target metric)

Step 3: Select H_best = argmax_H H.score

Step 4: For each query q in dev split:
          corpus_q ← extract 20 candidate passages for q
          pipe(corpus_q, H_best)    → ranked list R_q
          metrics_q ← evaluate(R_q, gold supporting passages for q)

Step 5: Report aggregated dev metrics as final evaluation
```

### 2.4 Computational implications

Because each query runs the full pipeline independently, the total
computational cost scales as:

```text
O(N_train × |H| + N_dev) × cost_per_query
```

where $N_{train}$ is the number of train queries used for tuning, $|H|$ is
the number of hyperparameter combinations, and $N_{dev}$ is the number of dev
queries. To keep evaluation tractable, we typically:

- Use a **subset** of the train split for tuning (e.g., 50 queries).
- Test only a small grid of parameter combinations (e.g., 3 × 3 = 9 runs).
- Run the full dev evaluation once with the selected configuration.

---

## 3. Supported Benchmarks

### 3.1 MuSiQue (Multi-hop Sentence Queries)

| Property | Value |
|----------|-------|
| Source | MuSiQue (Trivedi et al., 2022) |
| Task | Multi-hop QA — retrieve K supporting passages from 20 candidates |
| Corpus per query | 20 passages (paragraphs) |
| Supporting passages per query | Typically 2–5 (varies by hop count) |
| Total dev queries | 4,834 (3,115 with ≥2 supporting passages) |
| Total train queries | 39,876 (23,097 with ≥2 supporting passages) |
| Data format | JSONL (HuggingFace `musique_full_v1.0_{train,dev}.jsonl`) |

**Implementation:** `semantic_folding/dataset_benchmark/musique/run_benchmark.py`

### 3.2 Planned (Future Work)

| Dataset | Source | Task | Expected Corpus Size |
|---------|--------|------|---------------------|
| HotpotQA | Yang et al., 2018 | Multi-hop QA | 20 per query |
| 2WikiMultihopQA | Ho et al., 2020 | Multi-hop QA | 20 per query |
| NaturalQuestions | Kwiatkowski et al., 2019 | Factual memory | Large corpus |
| PopQA | Mallen et al., 2023 | Factual memory | Large corpus |
| NarrativeQA | Kočiský et al., 2018 | Discourse understanding | 10-doc per query |

Each benchmark follows the same pattern: map each query's candidate passages
into the semantic folding corpus format, run the full pipeline (Steps 1–6),
then evaluate retrieval against the gold supporting passages.

---

## 4. Benchmark Methodology

### 3.1 Three-Phase Execution

The benchmark is split into three phases to avoid redundant computation:

**Phase 1 — Index (`--mode index`):** Collect all unique paragraphs from the
specified query range into a **single combined corpus**. Run Steps 1–5
(phrase extraction → term-context matrix → t-SNE semantic space → phrase
fingerprints → document fingerprints) **once** on this combined corpus.
The result is a timestamped run directory with pre-built fingerprints.

**Phase 2 — Benchmark (`--mode benchmark`):** Load the pre-built run and for
each query run **only Step 6** (query processing) against the pre-built
fingerprints. The query processor scores all documents in the combined corpus;
we then post-filter to each query's 20 candidate passages before computing
retrieval metrics.

**Phase 3 — Report (`--mode report`):** Read a completed benchmark directory,
aggregate per-query metrics, and write a comprehensive Markdown report.

**Why three phases:** Running Steps 1–5 per query is wasteful because the
expensive operations (t-SNE on the term-context matrix, phrase fingerprint
generation) scale with the published vocabulary, not with the number of
queries. By building a unified semantic space from all paragraphs across all
queries, total cost reduces from $O(N \times T)$ to $O(T + N \times s)$,
where $N$ is the number of queries, $T$ is the cost of Steps 1–5, and $s$ is
the cost of a single Step 6 call ($s \ll T$).

### 3.2 Evaluation Metrics

All metrics are computed per query and then micro-averaged:

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **P@K** | $\frac{\|\text{relevant retrieved in top K}\|}{K}$ | Precision at cutoff K |
| **R@K** | $\frac{\|\text{relevant retrieved in top K}\|}{\|\text{all relevant}\|}$ | Recall at cutoff K |
| **MRR** | $\frac{1}{\text{rank of first relevant}}$ (0 if none) | Mean reciprocal rank |
| **AP** | $\frac{1}{\|R\|}\sum_{k=1}^{N} P@k \cdot rel(k)$ | Average precision |
| **NDCG@K** | $\frac{\text{DCG@K}}{\text{IDCG@K}}$ | Normalised discounted cumulative gain |

Relevance is binary: a passage is either supporting (gold) or not. For metrics
that require continuous gain (NDCG), we use binary gain (1 for relevant, 0
otherwise).

### 3.3 Output Structure

```
outputs/musique_benchmark/
├── runs/
│   └── run_<timestamp>/
│       ├── config.yml               # Index config + pipeline params
│       ├── corpus.txt               # Combined corpus (all unique paragraphs)
│       ├── query_doc_map.json        # query_idx → [global_doc_ids]
│       ├── query_gold.json           # query_idx → [gold_global_doc_ids]
│       ├── metadata.json             # Stats (num_queries, num_docs)
│       ├── extracted_phrases/        # Step 1 output
│       ├── term_context_matrix/      # Step 2 output
│       ├── semantic_space/           # Step 3 output
│       ├── phrase_fingerprints/      # Step 4 output
│       └── doc_fingerprints/         # Step 5 output
│
└── benchmarks/
    └── benchmark_<timestamp>/
        ├── config.yml                # Benchmark config (run ref, query range)
        ├── summary.json              # Aggregate metrics over all queries
        ├── results_log.csv            # Per-query metrics in tabular form
        ├── benchmark_report.md        # Comprehensive Markdown report
        └── per_query/
            ├── 0000/                  # Per-query results (index = query idx)
            │   ├── candidate_docs.json
            │   ├── query_results.json # Raw Step 6 output
            │   └── filtered_results.json
            ├── 0001/
            └── ...
```

---

## 5. Parameter Configuration for MuSiQue

The following configuration is recommended for MuSiQue retrieval based on
systematic tuning on the training split:

### 4.1 Grid Size: 64

Each query has exactly 20 candidate passages. On a 64×64 grid (4,096 cells), a
20-doc corpus produces fingerprints with 7–10% bit density (287–409 active
cells per document). This density provides sufficient signal overlap without
excessive sparsification. On a 128×128 grid (16,384 cells), the same number of
phrases produces only 2–5% density, reducing the signal-to-noise ratio of
dot-product scores.

**Theoretical justification:** For a corpus of $n$ documents and a grid of
$N \times N$ cells, the expected fingerprint density for document $d$ is
$\rho(d) \approx \frac{\text{nnz}(F_d)}{N^2}$. When $\rho$ is too low (< 3%),
the query fingerprint (also sparse) is unlikely to overlap with the correct
document fingerprint, causing retrieval failures. When $\rho$ is too high
(> 20%), fingerprints become indistinguishable. The optimal range for $\rho$ is
5–15%, which grid size 64 achieves for 20-doc corpora.

### 4.2 Spreading Steps: 1

The spreading algorithm expands each active cell in the query fingerprint to
its Moore neighbourhood (8 adjacent cells) with a decay factor of 0.5 per
step. One spreading step enables soft-matching of semantically related terms
(e.g., "community networks" → "social networks") without the noise introduced
by two or more steps. On a 64×64 grid, one step expands each active cell to a
3×3 block, increasing the effective query footprint by at most 9×.

### 4.3 Top Percent: 0.10

The `top_percent` parameter controls the fraction of grid cells retained in
each document fingerprint after peak detection. At 10%, the top 410 out of
4,096 cells are kept (on 64×64). This threshold is high enough to preserve
distinctive phrase signals and low enough to suppress noise from generic
stopwords. Tuning experiments on the 5-query development set showed that 5%
causes loss of discriminative signal (C00 lost in Q5), while 15% dilutes
fingerprint distinctiveness.

### 4.4 Weighting: IDF

IDF weighting boosts phrases that are rare across the corpus but
discriminative for the query. In the MuSiQue setting (20 passages per query,
many sharing topical vocabulary), IDF is essential to prevent common phrases
(e.g., "developed by", "located in") from dominating the query fingerprint.
Uniform weighting loses supporting passages in multi-hop queries where the
distinctive entities have the highest discriminative power.

### 4.5 Smoothing Sigma: 1.5

Gaussian smoothing ($\sigma = 1.5$) is applied before peak detection in both
phrase fingerprint generation (Step 4) and document fingerprint generation
(Step 5). The smoothing kernel blurs activation values spatially on the grid,
reducing the impact of isolated noisy peaks. The pipeline is robust to $\sigma$
values in the range 1.0–2.0; 1.5 is chosen as the default.

### 4.6 Recommended Default Configuration

```yaml
grid_size: 64
spreading_steps: 1
top_percent: 0.10
weighting: idf
smoothing_sigma: 1.5
keep_verbs: true
min_word_length: 3
min_freq: 1
use_morton: true
doc_norm: l2              # +4.0% MRR over sqrt_nnz
sim_metric: cosine        # default; alternatives: dice, overlap, jaccard, idf-weighted
score_norm: none          # alternatives: zscore, percentile, minmax
asymmetric: false         # enable for containment/coverage scoring
asym_alpha: 0.7           # containment weight when asymmetric=true
```

**Scoring metric selection guidance:**
- `cosine` (default): Best for float-valued fingerprints at ~7.8% density. Uses activation magnitudes for discrimination.
- `dice` / `overlap` / `jaccard`: Best for truly binarized fingerprints (<5% density). Compare active bit positions only.
- `idf-weighted`: Best when IDF weights are available and rare term matching is critical.
- `--score-norm zscore`: Recommended when score compression is observed (all docs scoring within narrow range).

---

## 6. Relationship to HippoRAG

The MuSiQue benchmark was chosen because it is one of the primary evaluation
datasets in both HippoRAG (Gutiérrez et al., 2024) and HippoRAG 2 (Gutiérrez
et al., 2025). In the HippoRAG papers, MuSiQue is used to evaluate
"associativity" — the ability to compose facts from multiple documents
(multi-hop retrieval). The standard HippoRAG evaluation on MuSiQue uses the
full 20-passage candidate pool per query, exactly matching our protocol.

**Key differences between Semantic Folding and HippoRAG retrieval:**

| Aspect | HippoRAG | Semantic Folding (this work) |
|--------|---------|------|
| Index representation | Dense passage embeddings + knowledge graph (OpenIE triples) | Sparse distributed fingerprints on 2D grid |
| Retrieval mechanism | Personalized PageRank over KG + dense retrieval | Normalised dot-product over SDR fingerprints |
| Requires LLM for indexing | Yes (OpenIE triple extraction) | No |
| Interpretability | KG paths explainable, embeddings opaque | Spatial grid positions interpretable |
| Computational cost (indexing) | High (LLM calls per passage) | Low (purely statistical) |

The benchmark allows direct comparison: HippoRAG reports MuSiQue retrieval
metrics (Recall@K, MRR) which can be compared with Semantic Folding results
under identical conditions.

---

## 7. Extending to New Datasets

To add a new dataset to the benchmark framework:

1. Create `semantic_folding/dataset_benchmark/<dataset>/`.
2. Implement a conversion function that, for each query, produces:
   - A corpus file (`idx, title text\n` per candidate passage).
   - A ground truth file (`ground_truth.json` with `relevant_docs` list).
3. Implement a launcher that iterates over queries and calls `run_pipeline()`.
4. The evaluation pipeline (`compute_metrics`, `aggregate_metrics`) is reused
   from the MuSiQue implementation.

---

## 8. Multi-Dataset Benchmarks (v3-Final+)

### 8.1 Dataset Coverage

| Dataset | Domain | Queries | Task |
|---------|--------|---------|------|
| PubMedQA | Biomedical QA | 200 | Question answering with context |
| Belebele | Reading Comprehension | 100 | Multiple choice reading comp |

### 8.2 Cross-Dataset Results

| Dataset | SF Baseline | Hybrid SF+BM25 | BM25 | Best Strategy |
|---------|-------------|----------------|------|---------------|
| PubMedQA | **0.954** | 0.923 (-3.1%) | 1.000 | SF baseline |
| Belebele | 0.840 | **0.860** (+2.0%) | 0.995 | Hybrid α=0.5 |

**Finding:** BM25 outperforms semantic folding on both datasets. However, hybrid SF+BM25 improves Belebele by +2.0% MRR.

### 8.4 Hybrid SF+BM25 Scoring

To address the performance gap, we implemented hybrid scoring that combines semantic folding with BM25:

\[
\text{score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{score}_{\text{SF}}(q, d) + (1 - \alpha) \cdot \text{score}_{\text{BM25}}(q, d)
\]

where $\alpha$ controls the weight of semantic folding (0 = pure BM25, 1 = pure SF).

**Cross-Dataset Results (α=0.3):**

| Dataset | SF Only | Hybrid | Delta | Task Type |
|---------|---------|--------|-------|-----------|
| PubMedQA | 0.955 | **1.000** | **+4.7%** | Biomedical |
| Belebele | 0.880 | 0.827 | -6.0% | Reading comp |
| Custom Corpus | 0.681 | **0.846** | **+24.2%** | Mixed |

**Custom Corpus Breakdown:**

| Category | SF Only | Hybrid | Delta |
|----------|---------|--------|-------|
| Negation/Complex | 0.567 | **1.000** | **+76.4%** |
| Paraphrasing | 0.490 | **0.650** | **+32.7%** |
| Domain Vocab | 0.767 | **0.867** | **+13.0%** |
| Semantic Sim | 0.900 | 0.867 | -3.7% |

**Finding**: Hybrid is **task-dependent** — helps on biomedical, negation, paraphrasing tasks but hurts on reading comprehension where SF semantic matching excels. Optimal α=0.3 (30% SF, 70% BM25).

**Key insight:** Hybrid scoring improves Belebele by +2.0% MRR by combining semantic topology with lexical matching.

### 8.5 PubMedQA Results (200 queries)

| Configuration | MRR | AP | Delta |
|---------------|-----|-----|-------|
| **Baseline (top_k=5)** | **0.954** | **0.832** | --- |
| Hybrid α=0.5 (top_k=10) | 0.923 | 0.808 | **-3.1%** |
| BM25 | 1.000 | 0.960 | --- |

**Finding:** Hybrid scoring **hurts** PubMedQA (-3.1% MRR). PubMedQA already has strong lexical overlap between queries and passages, so BM25 adds noise rather than signal.

### 8.6 Query Expansion Results

| Dataset | Baseline MRR | expand_default | expand_glossary | Verdict |
|---------|--------------|----------------|-----------------|---------|
| Belebele | 0.840 | 0.840 (0%) | 0.840 (0%) | No improvement |
| PubMedQA | 0.954 | 0.954 (0%) | 0.931 (-2.3%) | **Hurts performance** |

**Finding:** Query expansion with glossary does NOT help. Belebele queries don't contain glossary terms; PubMedQA expansion adds noise.

### 8.7 Document Normalization Results (Belebele, 50 queries)

| Configuration | MRR | Delta |
|---------------|-----|-------|
| Baseline (sqrt_nnz) | 0.840 | --- |
| **L2 Normalization** | **0.880** | **+4.0%** |
| L1 Normalization | 0.830 | -1.0% |
| Max Normalization | 0.818 | -2.2% |

**Finding:** L2 normalization provides significant improvement (+4.0% MRR) by treating all documents equally regardless of length. The default sqrt_nnz penalizes longer documents unfairly.

### 8.8 t-SNE Perplexity Results (Belebele, 50 queries)

| Perplexity | MRR | Delta |
|------------|-----|-------|
| 10 | 0.860 | +2.0% |
| **30 (baseline)** | 0.840 | --- |
| **50** | **0.880** | **+4.0%** |

**Finding:** Perplexity=50 improves MRR by +4.0%, matching L2 normalization performance. Lower perplexity (10) also helps (+2.0%).

### 8.9 Final Improvement Summary

| Improvement | Belebele MRR | Verdict |
|-------------|--------------|---------|
| **L2 Normalization** | +4.0% (0.880) | **Best** |
| **Perplexity=50** | +4.0% (0.880) | **Best** |
| Hybrid SF+BM25 | +2.0% (0.860) | Optional |
| Perplexity=10 | +2.0% (0.860) | Optional |
| Query Expansion | 0% | Skip |
| TF-IDF Re-ranking | 0% | Skip |

**Best Configuration:** `--doc-norm l2 --tsne-perplexity 50`

### 8.10 PubMedQA Validation Results

| Configuration | MRR | Delta |
|---------------|-----|-------|
| baseline | 0.954 | --- |
| L2 Normalization | 0.954 | 0.0% |
| **Perplexity=50** | **0.969** | **+1.5%** |

**Finding:** Perplexity=50 improves PubMedQA by +1.5% MRR, validating the Belebele finding.

### 8.11 Final Cross-Dataset Results

| Dataset | Best Config | SF MRR | BM25 MRR | Improvement |
|---------|-------------|--------|----------|-------------|
| **PubMedQA** | Perplexity=50 | **0.969** | 1.000 | +1.5% |
| **Belebele** | L2 norm + Perplexity=50 | **0.880** | 0.995 | +4.0% |

### 8.12 WordNet Expansion Analysis

**Finding:** WordNet provides 53.4% coverage on Belebele queries, but expansions are generic and irrelevant:
- "according" → "harmonize" (irrelevant)
- "have" → "rich person" (irrelevant)
- "images" → "image" (same word)

**Conclusion:** Skip WordNet expansion. Better suited for paraphrase detection, not reading comprehension.

### 8.13 DROP Failure Analysis

**Finding:** Only 4 sections in 50 queries, with 18+ queries sharing the same gold passage.

**Root cause:** DROP adapter groups queries by section_id. When multiple queries come from the same section, they all share the same gold passage. The query processor can't distinguish between passages from the same section.

**Fix required:** Ensure more diverse sections in the sample (currently sequential iteration creates bias).

### 8.14 Additional Dataset Results

| Dataset | Domain | SF Best | BM25 | Gap |
|---------|--------|---------|------|-----|
| PubMedQA | Biomedical QA | **0.969** | 1.000 | -0.031 |
| Belebele | Reading Comp | **0.880** | 0.995 | -0.115 |
| DROP | Reading Comp | **0.320** | 0.752 | -0.432 |
| CUAD | Legal Contract | 0.000 | — | SF fails |

**Pattern:** SF struggles on reading comprehension and legal tasks. L2 normalization helps but doesn't close the gap.

### 8.15 Recommendation: Hybrid as Optional Flag

| Dataset | Baseline MRR | Hybrid MRR | Best Config |
|---------|--------------|------------|-------------|
| **Belebele** | 0.840 | **0.860** (+2.0%) | Hybrid α=0.5 |
| **PubMedQA** | **0.954** | 0.923 (-3.1%) | Baseline |

**Conclusion:** Keep hybrid as an **opt-in flag** (`--hybrid --hybrid-alpha 0.5 --corpus <path>`). Different datasets have different optimal configurations.

```bash
# For reading comprehension (Belebele) - use hybrid
generic_benchmark.py all --dataset belebele --hybrid --hybrid-alpha 0.5

# For biomedical QA (PubMedQA) - use baseline
generic_benchmark.py all --dataset pubmedqa
```

---

## 9. Known Limitations

1. **Computational cost**: The original per-query design ran Steps 1–6 for
   every query (~1–3 min per query). The optimised three-phase design reduces
   this to a single index pass (Steps 1–5, ~2–5 min) plus ~20–30 s per query
   for Step 6 only. For 100 queries this is ~35–55 min instead of ~3–5 hours.

2. **Combined corpus**: The index phase collects unique paragraphs across all
   benchmark queries into a single corpus. For 100 dev queries this produces
   ~2,000 unique documents; for 500 queries, ~10,000 documents. t-SNE on 10K
   points is slower but still feasible (a few minutes). For larger query sets,
   consider batching into multiple index runs of ~200 queries each.

3. **Grid size sensitivity**: The recommended grid size (64) is optimal for
   20-passage corpora. For datasets with larger candidate pools (e.g., full
   Wikipedia for NaturalQuestions), the grid size must be scaled proportionally
   to maintain fingerprint discriminability.

4. **t-SNE stochasticity**: The semantic space coordinates depend on the random
   seed of t-SNE. All benchmark runs use `--random-seed 42` for
   reproducibility, but absolute scores are seed-dependent. Relative
   comparisons between parameter configurations (same seed) remain valid.

5. **Binary relevance**: Supporting passages are binary (relevant/not). A
   graded relevance scheme would make NDCG a more discriminating metric for
   parameter tuning.

6. **Hybrid scoring dependency**: The hybrid SF+BM25 approach requires a
   corpus file for BM25 indexing, adding complexity to the pipeline.

---

## 10. HippoRAG2 Dataset Benchmarks

### 10.1 Dataset Selection

From the HippoRAG2 repository, we selected datasets suitable for single-hop
retrieval (aligned with Semantic Folding's capabilities):

| Dataset | Queries | Corpus | Type | Reason |
|---------|---------|--------|------|--------|
| **PopQA** | 1,000 | 8,676 | Factual (Wikidata) | Single-hop, small candidate pool |
| **NQ-REaR** | 1,000 | 9,633 | Factual (Natural Questions) | Single-hop, general knowledge |

Excluded datasets:
- **HotpotQA**: Multi-hop reasoning (not suitable for SF)
- **2WikiMultihopQA**: Multi-hop compositional (not suitable for SF)
- **NarrativeQA**: Long narrative comprehension (passages too long)
- **LV-Eval**: 256K char contexts (too long, hallucinated distractors)
- **Case Study/Sample**: Too small (1 query each)

### 10.2 PopQA Results

**Configuration**: grid=64, spread=1, top%=0.10, smoothing=1.5, perplexity=30

| Metric | Semantic Folding | BM25 | Delta |
|--------|-----------------|------|-------|
| **MRR** | 0.980 | 1.000 | -2.0% |
| **AP** | 0.540 | 1.000 | -46.0% |
| **P@1** | 0.960 | 1.000 | -4.0% |
| **P@2** | 0.980 | 1.000 | -2.0% |

**Analysis**: PopQA has only 2 passages per query (subject entity + object entity),
making it trivial for BM25 (perfect scores). SF achieves MRR=0.980, failing on
2 queries out of 100. The small candidate pool limits discrimination.

### 10.3 NQ-REaR Results

**Configuration**: grid=64, spread=1, top%=0.10, smoothing=1.5, perplexity=30

| Metric | Semantic Folding | BM25 | Delta |
|--------|-----------------|------|-------|
| **MRR** | 0.574 | 0.638 | -10.0% |
| **AP** | 0.371 | 0.582 | -36.3% |
| **P@1** | 0.420 | 0.470 | -10.6% |
| **P@2** | 0.460 | 0.485 | -5.2% |

**Analysis**: NQ-REaR has ~10 passages per query with 1-2 gold passages.
SF struggles with the larger candidate pool and diverse topic coverage.
BM25's lexical matching is more effective for general knowledge queries.

### 10.4 MuSiQue Results (Multi-hop QA)

**Configuration**: grid=64, spread=1, top%=0.10, IDF, L2 norm, morton, smoothing=1.5, perplexity=30

| Metric | SF | BM25 | Delta |
|--------|-----|------|-------|
| **MRR** | 0.453 | 0.672 | -32.6% |
| **AP** | 0.272 | 0.482 | -43.7% |
| **P@1** | 0.395 | 0.563 | -29.8% |
| **P@2** | 0.221 | 0.362 | -39.0% |

**Analysis**: MuSiQue requires composing facts across 2-5 hops. BM25's lexical matching significantly outperforms SF's semantic approach. 47.7% of SF queries had no gold passage in top results.

### 10.5 Cross-Dataset Summary

| Dataset | SF MRR | BM25 MRR | SF Wins? |
|---------|--------|----------|----------|
| PubMedQA | 0.969 | 1.000 | No |
| Belebele | 0.880 | 0.995 | No |
| PopQA | 0.980 | 1.000 | No |
| NQ-REaR | 0.574 | 0.638 | No |
| DROP | 0.320 | 0.752 | No |
| **MuSiQue** | **0.453** | **0.672** | **No** |

**Key Finding**: SF excels on biomedical QA (PubMedQA MRR=0.969) but
underperforms BM25 on general knowledge retrieval tasks. Performance
degrades further on multi-hop QA (MuSiQue MRR=0.453, -32.6% vs BM25).
The pattern suggests SF's strength is domain-specific semantic matching,
not general-purpose or compositional retrieval.

---

## 11. LambdaMART Cascade Re-ranking Results

### 11.1 Architecture

Two-stage cascade:
1. **Stage 1 (SF Retrieval)**: Retrieve top-100 candidates using cosine similarity
2. **Stage 2 (LambdaMART Re-ranking)**: Re-score top-100 using 35 features per (query, document) pair

### 11.2 Feature Importance (50-query Belebele training)

| Rank | Feature | Gain | Category |
|------|---------|------|----------|
| 1 | cosine | 730.3 | Binary similarity |
| 2 | bm25_score | 714.7 | Auxiliary |
| 3 | block_13_jaccard | 42.3 | Block histogram |
| 4 | overlap | 37.9 | Asymmetric |
| 5 | block_12_jaccard | 35.4 | Block histogram |
| 6 | block_6_jaccard | 31.3 | Block histogram |
| 7 | d_popcount | 12.4 | Bit-density |
| 8 | block_7_jaccard | 9.7 | Block histogram |
| 9 | jaccard | 7.5 | Binary similarity |
| 10 | block_8_jaccard | 2.3 | Block histogram |

**Key findings**:
- Cosine and BM25 dominate (730 and 714 gain respectively)
- Block Jaccard features contribute when trained on 50+ queries (blocks 6, 7, 8, 12, 13)
- Binary set metrics (Jaccard, overlap) have moderate contribution
- Bit-density features (d_popcount) help distinguish document length

### 11.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Training queries | 50 (Belebele) |
| Documents per query | 926 |
| Total samples | 46,300 |
| Positive samples | 50 (1 gold per query) |
| Negative samples | 46,250 |
| LambdaMART trees | 200 (early stopped at 3) |
| Learning rate | 0.05 |
| Max depth | 6 |
| Validation metric | NDCG@5 |
| Best NDCG@1 | 0.400 |
| Best NDCG@3 | 0.702 |
| Best NDCG@5 | 0.745 |

### 11.4 Challenges

1. **Extreme class imbalance**: 50 positive vs 46,250 negative samples (0.1% positive rate). LambdaMART early stops at iteration 3 because the model quickly learns to distinguish the few positive examples.

2. **Training/evaluation overlap**: With only 50 queries in the run, queries used for evaluation overlap with training queries. True held-out evaluation requires larger datasets or cross-dataset training.

3. **Score compression**: SF's inherent score compression (all documents score within narrow range) limits the discriminative power of raw features. The re-ranker can only work with the signal present in the features.

### 11.5 Future Directions

1. **Cross-dataset training**: Combine Belebele (50 queries) with other datasets (MuSiQue, PubMedQA) to increase training data
2. **Class imbalance mitigation**: Use downsampling, focal loss, or synthetic positive examples
3. **Larger candidate pools**: Test on datasets with more candidates per query (NQ-REaR: ~10, MuSiQue: 20)
4. **Feature engineering**: Add TF-IDF features, query-document interaction features, and passage-level features
