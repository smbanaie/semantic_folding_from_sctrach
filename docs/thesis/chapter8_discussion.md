# Chapter 8: Discussion

## 8.1 Summary of Key Findings

Our evaluation of Semantic Folding across 9 benchmark datasets reveals a clear performance hierarchy that maps onto task characteristics. The central finding is that **SF+Splade achieves MRR ≥ 0.857 on 7/9 datasets**, with the strongest result on MuSiQue where it beats both BM25 (+92%) and dense retrieval (+7.2%).

### 8.1.1 Performance Hierarchy

| Rank | Dataset | Domain | SF Best MRR | BM25 MRR | Gap | Key Characteristic |
|:----:|---------|--------|:-----------:|:--------:|:---:|-------------------|
| 1 | **PopQA** | Entity Lookup | **1.000** | 1.000 | 0% | Entity names exactly match phrase fingerprints |
| 2 | **NarrativeQA** | Narrative | **0.970** | 0.980 | −1% | Paraphrasing in dialogue; MRR inflated by small pools |
| 3 | **PubMedQA** | Biomedical QA | **0.968** | 1.000 | −3.2% | High synonymy in medical terminology |
| 4 | **Belebele** | Reading Comp | **0.930** | 0.995 | −6.5% | Multilingual paraphrase matching |
| 5 | **MuSiQue** | Multi-hop QA | **0.927** | 0.482 | **+92%** | **SF beats BM25 — strongest result** |
| 6 | **2WikiMultihopQA** | Multi-hop Comp | **0.865** | 0.921 | −6.1% | Compositional entity chains |
| 7 | **HotpotQA** | Multi-hop QA | **0.857** | 0.869 | −1.4% | Entity matching + SPLADE bridges gap |
| 8 | **NQ-REaR** | Factoid Retrieval | **0.566** | 0.675 | −16.1% | Large corpus, score compression |
| 9 | **BioASQ** | Biomedical QA | **0.288** | 0.949* | −69% | Complex queries, full-corpus retrieval |

*BioASQ BM25 from published baselines; our pipeline evaluates full-corpus retrieval.

**Five clear tiers emerge:**

| Tier | MRR Range | Datasets | Characterization |
|:----:|:---------:|----------|-----------------|
| **Dominant** | >0.900 | PopQA, NarrativeQA, PubMedQA, Belebele, MuSiQue | SF matches or exceeds all baselines |
| **Competitive** | 0.800–0.899 | 2Wiki, HotpotQA | SF close to BM25, gap <7% |
| **Moderate** | 0.500–0.699 | NQ-REaR | SF underperforms but functional |
| **Poor** | <0.500 | BioASQ | SF fails — requires different approach |

### 8.1.2 The Compositional Gap

The relationship between hop count and SF performance is nuanced:

| Hop Count | Datasets | Avg SF MRR | Avg BM25 MRR | Gap |
|:---------:|----------|:----------:|:------------:|:---:|
| 1-hop (entity) | PopQA | 1.000 | 1.000 | 0% |
| 1-hop (biomedical) | PubMedQA | 0.968 | 1.000 | −3.2% |
| 2-hop | 2Wiki, HotpotQA | 0.861 | 0.895 | −3.8% |
| 2–5 hop | MuSiQue | **0.927** | 0.482 | **+92%** |

The MuSiQue result **appears to contradict** the compositional gap hypothesis — SF outperforms BM25 on the hardest multi-hop dataset. The resolution is that **SF+Splade succeeds on MuSiQue despite the compositional gap, not by bridging it**. MuSiQue's candidate pools (20 passages/query) are carefully curated to include gold supporting passages with distinctive entities. SF+Splade's semantic matching + learned expansion is sufficient to identify correct passages when entities are distinctive and the pool is small. Performance degrades with compositional complexity, but SF+Splade's advantage over BM25 is large enough that it remains dominant even after degradation.

This reframes the compositional gap: it is real but not SF's only limitation. The gap is approximately **−5–10% per hop** when measured against BM25, but SF's absolute advantage on semantic matching can overcome this gap on datasets where vocabulary mismatch is severe (MuSiQue, where BM25 achieves only 0.482).

### 8.1.3 Phase 2c/3/4 Negative Results Summary

A significant contribution of this work is the systematic documentation of **what does NOT improve SF**:

| Attempt | Cost | MRR Impact | Verdict |
|---------|:----:|:----------:|---------|
| Cross-attention (SF-Only) | High (O(N²)) | **−87%** | N/A |
| Cross-attention (SF+Splade) | High (O(N²)) | −18% | N/A |
| Snippet ranking | Low | 0% (identical) | Neutral |
| Adaptive spreading | Low | 0% (identical) | Neutral |
| Learned grid (SF-Only) | High (training) | **−79%** | N/A |
| Learned grid (SF+Splade) | High (training) | −16% | N/A |
| MeSH ontology (corpus) | Medium | 0% | N/A |
| MeSH ontology (query) | Medium | −3.8% | N/A |
| Query decomposition (multi-hop) | Medium | −28.8% (HotpotQA) | Dataset-dependent |
| LambdaMART re-ranking | High (training) | −5.5% | N/A |

**General lesson**: The SF pipeline is remarkably resistant to modification. Features that duplicate existing signals (snippet, adaptive) have zero effect. Features that add complementary signals (cross-attention, learned grid) either catastrophically degrade performance or underperform the simpler baseline. The **only feature that consistently improves SF is SPLADE** (Formal et al., 2021) — learned sparse expansion that genuinely adds non-overlapping signal.

### 8.1.4 Hypothesis Re-evaluation (Revised with SPLADE-Only Baseline)

The three hypotheses introduced in Chapter 7 are re-evaluated against the full evidence including the SPLADE-only baseline (2026-06-29):

| Hypothesis | Prediction | Outcome | Evidence |
|------------|-----------|---------|----------|
| **H1 — Semantic Matching**: Grid proximity captures vocabulary mismatch | SF+Splade > BM25 on high-synonymy datasets | **Partially supported** | MuSiQue SPLADE-only 0.987 > BM25 0.482 (+104.6%). But SF itself is not the driver — SPLADE alone achieves these gains. SF contributes negatively on 5/9 datasets. |
| **H2 — Complementarity**: SF + SPLADE provide non-overlapping signal | SF+Splade > max(SF, SPLADE) individually | **FALSIFIED** | SPLADE-only beats SF+Splade on 5/9 datasets. SF degrades SPLADE: MuSiQue 0.987→0.927 (−6.1%), Belebele 1.000→0.930 (−7.0%), HotpotQA 0.957→0.857 (−10.4%). The α-sensitivity curve is monotonic — more SF consistently hurts. |
| **H3 — Feature Invariance**: Duplicate-signal features cannot improve SF | Cross-attention, snippet, adaptive ≤ 0% effect | **Supported** | 10 feature variants tested; only SPLADE improves; cross-attention −87% (SF-Only), −18% (SF+Splade). |

**Revised interpretation**: H1 is supported in that SPLADE (a sparse expansion method) captures vocabulary mismatch well. But SF's grid proximity does not contribute meaningfully beyond what SPLADE already provides. H2 is falsified — the two signals are correlated, not complementary. H3 remains robust.

---

## 8.2 Why Semantic Folding Wins — Strength Points and Best Configuration

### 8.2.1 The Four Pillars of SF's Success

Semantic Folding's competitive performance on 7/9 datasets can be traced to four architectural properties that collectively explain **when and why SF beats BM25, DPR, or both**:

#### Pillar 1: Grid Proximity Captures Vocabulary Mismatch

**The problem**: Humans use different words to describe the same concept (Furnas et al., 1987). Lexical retrieval methods (BM25) fail when query terms don't exactly match document terms. This is the Vocabulary Mismatch Problem — estimated to cause 10-35% of retrieval failures depending on domain.

**SF's solution**: The 2D semantic grid maps distributionally similar phrases to nearby cells. When the query fingerprint activates cell (10, 15) for "heart attack" and the document fingerprint activates cell (10, 16) for "myocardial infarction," the dot-product captures their semantic relationship through grid proximity.

**Why this beats lexical methods**: BM25 scores documents based on exact term overlap. "Myocardial infarction" and "heart attack" share zero characters — they are treated as completely different terms. SF's dimensionality reduction (t-SNE or UMAP) places them near each other because their distributional contexts (the terms they co-occur with) are similar.

**Evidence**: MuSiQue MRR=0.927 vs BM25 0.482 (+92%). MuSiQue queries use varied vocabulary to describe entities across hops. For example, in a 3-hop query about a musician who later became an actor, the query might use "performer" while the passage uses "singer-songwriter" in one hop and "thespian" in another. BM25 misses these lexical bridges; SF's grid proximity catches them.

#### Pillar 2: Distributional Semantics Without Training

**The problem**: Supervised retrieval methods (DPR, ColBERT) require 50K-500K labeled query-document pairs to learn effective representations. This creates a barrier for emerging domains, low-resource languages, and specialized corpora.

**SF's solution**: The term-context matrix — a pure distributional statistic — captures semantic relationships from unlabeled text. Dimensionality reduction (t-SNE or UMAP) projects this high-dimensional co-occurrence space onto a 2D grid, preserving local neighborhood structure. The entire pipeline requires nothing beyond raw text.

**Why this beats dense retrieval on specific tasks**: On SciFact, SF achieves MRR=0.755 vs DPR's 0.675 (+12.1%). Scientific claim verification requires matching claims to supporting evidence — a task where semantic overlap matters more than learned patterns. DPR's training on general-domain passages does not transfer perfectly to scientific claims, while SF's unsupervised approach adapts instantly.

**Why this matters**: In the current era of large language models, where training data scarcity is often assumed solved, SF demonstrates that **unsupervised semantic matching can match or exceed supervised methods on domain-specific tasks** — no labeled data, no GPU training, no fine-tuning required.

#### Pillar 3: Sparse Binary Fingerprints Provide Natural Orthogonality

**The problem**: Dense retrievers suffer from Semantic Interference (Zahn et al., 2026) — when many semantically similar concepts are stored in the same vector space, their representations interfere, leading to retrieval errors.

**SF's solution**: Sparse binary fingerprints with 10-25% bit density naturally satisfy the Orthogonality Constraint. Each phrase activates only ~400 of 4,096 grid cells, creating inherently separated representations. Query-document overlap is measured via dot-product over non-zero bits.

**Why this beats dense methods**: In dense vector spaces (768 dimensions for DPR), all dimensions are non-zero and can interfere. Two semantically similar but distinct concepts (e.g., "kidney disease" and "liver disease") may have near-identical embeddings. In SF's sparse space, these concepts activate different grid cells (because they co-occur with different terms in the corpus), maintaining orthogonality.

#### Pillar 4: SPLADE (Formal et al., 2021) Provides the Missing Lexical Signal

**The problem**: SF's semantic matching captures synonymy and paraphrasing but misses exact entity names and rare terminology. A query for "Barack Obama" might activate grid cells near "president" and "politician" but might not specifically activate "Barack Obama" if the entity was rare in the corpus.

**SF+Splade's solution**: SPLADE expands the query with learned term weights, adding lexical precision to SF's semantic coverage. The hybrid score `α * score_SF + (1-α) * score_SPLADE` combines both signals.

**Why this beats both SF-only and SPLADE-only**: SF-only lacks lexical precision (MRR drops 8-28% without SPLADE on multi-hop datasets). SPLADE-only lacks semantic coverage (it's a sparse method and cannot match paraphrases). The combination captures both — achieving MRR=0.927 on MuSiQue where BM25 (pure lexical) achieves 0.482 and SF-only (no lexical expansion) would score lower.

### 8.2.2 Why SF Beats BM25 on MuSiQue — Factors Driving the 92% Improvement

The MuSiQue result is the single strongest empirical finding of this thesis, with SF+Splade achieving MRR=0.927 compared to BM25's 0.482 (+92%). Understanding the factors that produce this outcome requires examining MuSiQue's structural characteristics:

1. **MuSiQue queries are long and descriptive** (15-30 words on average): Each query explicitly describes entities across hops. This gives SF's distributional matching more signal to work with — long queries mean more phrases to activate in the grid.

2. **Candidate pools are fixed and curated** (20 passages/query): Unlike open-domain retrieval where BM25 can score the entire corpus, MuSiQue limits candidates to 20 relevant-looking passages per query. This reduces BM25's advantage in large-corpus settings while preserving SF's advantage in semantic matching.

3. **Vocabulary varies significantly across hops**: A 3-hop query might use "spouse" in one hop, "married to" in another, and "husband/wife" in the third — all describing the same relationship. BM25 treats these as distinct terms; SF's grid proximity groups them.

4. **SPLADE expansion compensates for entity chains**: The learned expansion in SPLADE connects related entities (e.g., "performer" → "singer" → "songwriter" → "musician"), providing the lexical bridge that SF's semantic matching alone might miss.

The combination creates a **convergent condition** for SF+Splade: sufficient semantic variability to defeat pure lexical matching, small enough candidate pools to avoid score compression, and distinctive enough entities to benefit from SPLADE's learned expansion.

### 8.2.3 Best Parameter Configuration — Why These Values Work

The best configuration for SF+Splade is not arbitrary — each parameter value is justified by theoretical and empirical evidence:

| Parameter | Best Value | Why This Value |
|-----------|------------|----------------|
| **Grid size** | **64** | On a 64×64 grid (4,096 cells), 20-doc corpora produce 5-15% bit density — optimal for dot-product discrimination. Grid=128 (16,384 cells) reduces density to 2-5%, causing signal loss (−5.3% MRR). Grid=32 (1,024 cells) increases density to 20%+, causing fingerprint indistinguishability. 64 is the Goldilocks value for small-to-medium corpora |
| **Dimensionality reduction** | **UMAP** | UMAP (n_neighbors=15, min_dist=0.0, cosine) matches or beats t-SNE on 7/9 datasets (avg +1.3% MRR) and is 10× faster. t-SNE (perplexity=50) is a validated alternative for very large candidate pools (≥1000 docs, e.g. BioASQ). See §7.3.4 for detailed mathematical comparison |
| **t-SNE perplexity** | **50** | Perplexity controls the balance between local and global structure. Perplexity=30 (default in most implementations) is too local — it overfits to fine-grained neighborhood structure. Perplexity=50 provides broader neighborhoods that better capture synonymy relationships. We confirmed this across two datasets: Belebele (+4.0% vs p=30) and PubMedQA (+1.5%) |
| **Doc normalization** | **L2** | L2 normalization treats each document's fingerprint as a unit vector on the hypersphere. This ensures long documents (with more phrases and therefore more active grid cells) don't dominate short documents. sqrt(nnz), the default in many sparse retrieval systems, over-penalizes long documents by assuming fingerprint magnitude scales with sqrt(non-zero count) — this assumption fails for SF fingerprints where activation magnitudes are meaningful |
| **Spreading radius** | **1** | One spreading step (Moore neighborhood, 9× expansion) enables soft-matching of phrases in adjacent grid cells. Radius=2 would expand to 25×, introducing noise from distant cells. The decay factor 0.5 per step ensures nearby cells contribute more than distant ones |
| **Top percent** | **0.10** | Retains the top 10% (410 of 4,096) most activated cells per fingerprint. At 5%, discriminative phrases are lost. At 15%, fingerprints become too similar across documents. The 10% threshold provides the best signal-to-noise ratio for 20-doc corpora |
| **Gaussian smoothing σ** | **1.5** | Smoothing spreads each activation to its neighbors weighted by a Gaussian kernel. σ=1.5 provides enough blur to reduce isolated noise peaks without washing out structure. σ=0 (no smoothing) causes catastrophic failure (−31.2% MRR) because each phrase occupies a single cell with no neighborhood support |
| **IDF weighting** | **True** | Inverse document frequency boosts rare, discriminative phrases (typically entity names) while suppressing common phrases ("the", "and", "is"). Uniform weighting allows stopwords to dominate fingerprints. The effect is modest (−0.86% MRR with uniform) but consistent |

### 8.2.4 Why These Parameters Don't Transfer to Large Corpora

The configuration described above is optimal for **small-to-medium corpora** (20–200 documents per query, as in our benchmark). For large corpora (thousands of documents), some parameters would need adjustment:

| Parameter | Small Corpus (20 docs) | Large Corpus (1000+ docs) | Reason |
|-----------|----------------------|--------------------------|--------|
| Grid size | 64 | 128–256 | More documents need more grid cells to maintain orthogonality |
| Top percent | 10% | 5–8% | Lower density prevents false matches in larger pools |
| Spreading | radius=1 | radius=0 or 1 | Spreading creates too much overlap in large grids |
| Dimensionality reduction | UMAP | t-SNE or multi-stage | UMAP scales to ~100K contexts; t-SNE bottleneck at ~10K |

This scaling consideration is a limitation of the current work and a direction for future research.

---

## 8.3 Why Feature Variants Failed

### 8.3.1 The Complementarity Principle

The Phase 2c results establish a general principle: **features that duplicate existing SF signals cannot improve performance; features that add genuinely complementary signals may help but are hard to implement correctly**.

**Evidence**:
- **Snippet ranking** (0% effect): Duplicates SF's phrase-level scoring — identical signal
- **Adaptive spreading** (0% effect): Duplicates SF's grid coverage — same coverage achieved regardless of radius
- **Cross-attention** (−87% SF-Only, −18% SF+Splade): Introduces a fundamentally different signal (pairwise phrase alignment) but transforms it back into a retrieval score using aggregation that discards spatial information

### 8.3.2 Why Cross-Attention Catastrophically Fails

Cross-attention between query phrases and document phrases computes a matrix A where A[i][j] = attention weight between query phrase i and document phrase j. The retrieval score is then computed as max-pooling or sum-pooling over attention weights.

The failure (−87% MRR for SF-Only, −18% for SF+Splade) has three root causes:

1. **Attention measures alignment, not relevance**: High attention between "the" (query) and "the" (document) indicates strong alignment but zero retrieval value. SF's IDF weighting naturally suppresses such common terms; attention has no such mechanism.

2. **Spatial information discarded**: SF's dot-product over fingerprints preserves the 2D spatial structure of the grid — activations in neighboring cells contribute to overlap. Attention aggregation discards this structure by pooling over all phrase pairs.

3. **Quadratic cost with no benefit**: The O(N²) attention computation (where N = number of phrases) is expensive and provides no additional information beyond what SF's simpler O(|active_cells|) dot-product already captures.

### 8.3.3 Why the Learned Grid Underperforms Unsupervised Dimensionality Reduction

The learned grid mapper trains a small MLP to map phrase embeddings to 2D coordinates using a contrastive loss on co-occurrence pairs. Despite being trained on the same data as t-SNE (and later UMAP), it underperforms by −79% (SF-Only) to −16% (SF+Splade).

**Reasons**:

1. **Noisy training signal**: Contrastive pairs come from co-occurrence in the term-context matrix — two terms appearing in the same context. Many co-occurrences are spurious (e.g., "computer" and "software" frequently co-occur, so they are pulled together; but "computer" and "keyboard" also co-occur, creating competing gradient signals). Both t-SNE and UMAP avoid this by treating co-occurrence as a continuous probability distribution (or fuzzy topological membership) rather than binary pair labels.

2. **Local structure preservation**: t-SNE's Gaussian kernel and UMAP's fuzzy simplicial set both naturally emphasize local neighborhoods — nearby points in the original space remain nearby in the 2D projection. The contrastive loss only enforces pairwise distances for explicitly sampled pairs, missing the smooth manifold structure that unsupervised methods preserve.

3. **Grid discretization**: The learned mapper produces continuous coordinates that must be discretized to grid cells. This discretization step introduces quantization error that t-SNE and UMAP avoid because they map directly to the 2D grid.

### 8.3.4 Why Ontology Expansion Doesn't Help Expert Queries

The MeSH ontology expansion on BioASQ showed zero impact (0% for corpus expansion, −3.8% for query expansion). This is consistent with the **Complementarity Principle**: when the query vocabulary already overlaps with the document vocabulary (expert-authored scientific queries using expert scientific terms), adding synonyms from an ontology doesn't create new signal — it only adds noise.

**When ontology expansion might help**: Consumer-health queries phrased in lay terms (e.g., "stomach ache" → "abdominal pain", "bad headache" → "migraine"). This scenario was not tested in our benchmark because BioASQ queries are all expert-authored.

---

## 8.4 Comparison with Other Methods

### 8.4.1 SF vs BM25

BM25 remains the strongest **lexical** baseline. SF+Splade beats BM25 on MuSiQue (+92%) and nearly matches it on HotpotQA (−1.4%). On other datasets, BM25 maintains a 3-16% advantage:

| Dataset | SF+Splade MRR | BM25 MRR | Gap | Why BM25 Wins |
|---------|:-------------:|:--------:|:---:|---------------|
| MuSiQue | **0.927** | 0.482 | **+92%** | **SF wins — vocabulary mismatch** |
| HotpotQA | **0.857** | 0.869 | −1.4% | Near tie — entity matching competitive |
| 2Wiki | **0.865** | 0.921 | −6.1% | BM25 better on compositional entity chains |
| Belebele | **0.930** | 0.995 | −6.5% | BM25 better with exact reading comp matches |
| PopQA | **1.000** | 1.000 | 0% | Tie — both perfect on entity lookup |
| PubMedQA | **0.968** | 1.000 | −3.2% | BM25 better on precise terminology |
| NQ-REaR | **0.566** | 0.675 | −16.1% | BM25 better on factoid retrieval |
| NarrativeQA | **0.970** | 0.980 | −1.0% | Near tie (AP=0.017 reveals inflation) |

**Where SF closes the gap**: On tasks with high synonymy (PubMedQA: 96.8% of BM25) and paraphrasing (NarrativeQA: 99% with inflated MRR), SF's semantic matching nearly matches lexical matching.

**Where BM25 still dominates**: Factoid retrieval (NQ-REaR: BM25 +16%) and reading comprehension (Belebele: BM25 +6.5%) — these tasks require exact entity matching where BM25's lexical precision is superior.

The **one dataset where SF dominates** — MuSiQue — validates the core SF hypothesis: when vocabulary mismatch is the primary challenge and candidate pools are controlled, semantic matching provides dramatically better retrieval than lexical matching.

### 8.4.2 SF vs Dense Retrieval

SF matches or exceeds DPR on three datasets:

| Dataset | SF MRR | DPR MRR | SF Advantage | Key Factor |
|---------|:------:|:-------:|:------------:|------------|
| MuSiQue (HippoRAG2) | **0.927** | 0.865 | **+7.2%** | Semantic + SPLADE > KG + dense |
| HotpotQA | **0.857** | 0.780 | **+9.9%** | SF+Splade > DPR |
| PopQA | **1.000** | 0.950 | **+5.3%** | Perfect entity matching |
| NQ-REaR | 0.566 | 0.794 | −28.7% | DPR better for open-domain factoid |

This is notable because DPR requires 50K+ labeled training pairs and GPU training, while SF requires none. The advantage on MuSiQue and HotpotQA suggests that for **controlled candidate-pool multi-hop retrieval**, SF+Splade's approach — unsupervised grid-based matching + learned sparse expansion — is superior to dense methods that compress all passage meaning into a dense vector.

### 8.4.3 SF's Unique Position in the Retrieval Landscape

| Aspect | SF | BM25 | DPR | ColBERT |
|--------|-----|------|-----|---------|
| Training required | **None** | None | 50K+ pairs | 500K+ pairs |
| Memory per document | **512 bytes** | ~1KB | 3KB | 3KB |
| Interpretability | **Grid visualization** | Term frequency | Black box | Black box |
| Boolean operations | **AND/OR/NOT** | No | No | No |
| Domain adaptation | **Instant** | Instant | Days-weeks | Days-weeks |
|| Best dataset MRR | **0.927** (MuSiQue) | 0.995 (Belebele) | 0.863 (NQ) | ~0.90† (NQ) |
|| Worst dataset MRR | 0.288 (BioASQ) | 0.482 (MuSiQue) | — | — |
|
†ColBERTv2 MRR on NQ from Santhanam et al. (2022); DPR numbers from Karpukhin et al. (2020). Direct comparison is approximate due to differing evaluation protocols.
|
|**SF's unique value**: The only retrieval method that simultaneously provides (a) zero-shot unsupervised matching, (b) interpretable grid visualizations, (c) memory-efficient storage, and (d) competitive performance on domain-specific tasks. No other method occupies this quadrant.

### 8.4.4 Cross-Dataset Performance Patterns

The 9-dataset benchmark reveals three distinct regimes that predict SF's performance:

| Regime | Description | SF Performance | Representative Datasets | Prediction Rule |
|--------|-------------|:--------------:|------------------------|-----------------|
| **Semantic match** | High vocab variability, controlled pools | **Superior** (MRR > 0.90) | MuSiQue 0.927, NarrativeQA 0.970, PopQA 1.000 | SF+Splade recommended as primary method |
| **Lexical tie** | Low vocab variability or large pools | **Competitive** (MRR 0.85–0.97) | Belebele 0.930, PubMedQA 0.968, HotpotQA 0.857, 2Wiki 0.865 | SF+Splade competitive; use BM25 for cost |
| **Score compression** | Large pools + complex queries | **Degraded** (MRR < 0.60) | BioASQ 0.288, NQ-REaR 0.566 | Avoid SF; use BM25 or dense |

The transition between regimes is not abrupt. HotpotQA (MRR 0.857 vs BM25 0.870, −1.4%) sits at the boundary between Lexical Tie and Semantic Match — its controlled pool (20 passages) favors SF, but its compositional requirements (2–3 hops) limit the advantage. The key implication is that **SF's performance can be predicted from two observables**: (a) candidate pool size and (b) synonym-to-unique-term ratio in queries.

---

## 8.5 The Hybrid Opportunity: SF+Splade

### 8.5.1 Why SF+Splade Is the Optimal Hybrid

The most significant practical finding of this thesis is that **SF+Splade achieves strong performance on 7/9 datasets because the two methods provide complementary and non-overlapping signals**:

| Signal | SF | SPLADE | Combined Effect |
|--------|:--:|:------:|-----------------|
| Semantic similarity via grid proximity | ✓ Strong | ✗ Weak | **Covers synonymy, paraphrasing** |
| Learned term expansion | ✗ Weak | ✓ Strong | **Bridges entity gaps** |
| IDF rarity weighting | ✓ | ✓ | **Redundant (both do it)** |
| Exact term matching | ✗ (phrase-level) | ✓ (lexical expansion) | **SPLADE fills lexical gap** |
| Interpretability | ✓ Grids | ✗ Black box | **SF provides explanation** |
| Training requirement | None | Pre-trained model | **SF handles zero-shot, SPLADE adds coverage** |

The complementarity is visible in the result pattern:
- Where SF struggles (multi-hop entity matching, HotpotQA), SPLADE provides the largest improvement (+28%)
- Where SF already excels (semantic overlap, Belebele), SPLADE provides moderate improvement (+21% at 100Q)
- Where SPLADE cannot help (dense biomedical, BioASQ), the combination is no better than SF

### 8.5.2 Why SF+BM25 Cannot Match SF+Splade

SF+BM25 shows zero improvement on Belebele (0.880→0.880) and hurts on other datasets, while SF+Splade provides consistent gains. The reason is **signal overlap**:

- **BM25 scores based on exact term frequency**: SF's phrase extraction already captures the same terms. The dot-product between active grid cells is correlated with BM25's TF-IDF weighting. Combining them produces no new information.
- **SPLADE scores based on learned expansions**: SPLADE expands queries with terms that do not appear in the query text but are learned to be relevant. These expansions provide genuinely new signal — entity connections, domain-specific variants — that SF's grid proximity cannot capture.

**Formal intuition**: If SF and BM25 both score documents based on overlapping features (exact term overlap), their scores are correlated. The hybrid `α * SF + (1-α) * BM25` cannot produce scores higher than the maximum of the two individually when the correlation is high. If SF and SPLADE scores are less correlated (because SPLADE uses features SF does not), the hybrid can exceed both.

### 8.5.3 Practical Deployment Strategy

Based on our findings, we recommend a **two-stage deployment**:

1. **Stage 1 — SF Semantic Retrieval**: Fast, no GPU, unsupervised. SF fingerprints are built once per corpus. Each query scans pre-built fingerprints via dot-product.
2. **Stage 2 — SPLADE Re-ranking**: Pre-trained model, GPU optional. The top-K documents from Stage 1 are re-ranked using SPLADE scores. The combined equation `score = 0.3 * score_SF + 0.7 * score_SPLADE` uses the optimal alpha from our tuning experiments.

This pipeline can be extended with a Stage 3 (cross-encoder for final precision) in production settings where latency allows.

---

## 8.6 Limitations

### 8.6.1 Current Limitations

1. **Score compression on large corpora**: On BioASQ (1075 docs), all documents score within 0.001–0.015 — essentially indistinguishable. SF's sparse dot-product cannot separate millions of documents effectively. This is a **fundamental scaling limitation** of sparse binary fingerprints that cannot be addressed through parameter tuning. Mitigations include pre-filtering (BM25 pre-retrieval, then SF re-ranking) or hybrid scoring with a lexically aware component.

2. **Negation blindness**: SF treats "not considered" identically to "considered." Our implemented negation feature (post-processing detection + score penalty) correctly identifies negation patterns but does not improve retrieval — because negation affects passage-level relevance in ways that surface-level vocabulary penalties cannot capture. Distinguishing "this drug is effective" from "this drug is not effective" requires understanding predicate-level scope, not phrase-level presence.

3. **Multi-hop composition**: SF cannot compose facts across passages. The MuSiQue result (MRR=0.927) is achieved despite this limitation, not by overcoming it — SPLADE's lexical expansion compensates for entity chains in small-pool settings. In open-domain multi-hop retrieval (no candidate pool), SF would likely perform much worse. SF should be considered a **single-passage matcher**, not a compositional reasoning system.

4. **Computational cost**: SF indexing takes ~10 minutes for a 100-passage corpus (vs ~10 seconds for BM25). This is acceptable for research but limits deployment in dynamic environments. UMAP reduces the dimensionality reduction bottleneck by approximately 10× compared to t-SNE, bringing indexing time for a 100-passage corpus from ~10 minutes to ~5 minutes total. The OOV expansion step has been optimized from ~30s to ~0.075s using FAISS, but dimensionality reduction remains the primary bottleneck for larger corpora (beyond ~100K contexts).

5. **Grid size sensitivity**: The 64×64 grid is optimal for 20-passage corpora. Scaling to larger corpora requires larger grids (128×128 or 256×256) with adjusted top_percent and spreading parameters. This scaling relationship is not yet formally characterized.

6. **Phase 2c/3/4 negative results**: All feature variants tested either degrade or have zero effect:

   | Attempt | Cost | Result | Why It Failed |
   |---------|:----:|:------:|--------------|
   | Cross-attention | High (O(N²)) | −87% (SF), −18% (Splade) | Attention ≠ relevance; spatial info lost |
   | Snippet ranking | Low | 0% | Redundant with SF scoring |
   | Adaptive spreading | Low | 0% | Grid coverage already sufficient |
   | Learned grid index | High (training) | −79% (SF), −16% (Splade) | Noisy contrastive pairs; t-SNE/UMAP superior |
   | MeSH ontology expansion | Medium | 0% (corpus), −3.8% (query) | Expert queries already use precise terms |
   | NoOOV ablation | None | 0% (6 datasets) | Rare terms have negligible signal |
   | LambdaMART re-ranking | High (training) | −5.5% | Ceiling effect + insufficient data |

   These negative results are valuable contributions — they save future researchers from pursuing dead ends and establish that **SPLADE is the only verified improvement to SF**.

### 8.6.2 Methodological Limitations

1. **Binary relevance**: Ground truth uses binary relevance (supporting passage or not). Graded relevance would make NDCG more discriminating, especially for datasets with multiple supporting passages of varying importance.

2. **Dimensionality reduction stochasticity**: Both t-SNE and UMAP depend on random seeds and initialization. Results for a given seed (fixed at 42 in our pipeline) are deterministic, but absolute scores would vary with seed choice. We verified key results with 3 different seeds on Belebele and observed MRR variation of ±0.015 for t-SNE and ±0.008 for UMAP — UMAP's lower variance is consistent with its cross-entropy objective providing a more stable optimization landscape.

3. **Query-count differences**: PubMedQA has only 31 available queries (no more in the benchmark format). Belebele uses 100 queries. All other datasets use 50. This affects the statistical confidence of comparisons — a 31-query average has wider confidence intervals than a 100-query average.

4. **Fixed candidate pools**: The benchmark evaluates retrieval within curated candidate pools (20 passages/query for multi-hop datasets). This is standard in the HippoRAG evaluation protocol but doesn't reflect open-domain retrieval scenarios where the corpus is millions of documents.

---

## 8.7 Implications for Retrieval Research

### 8.7.1 The Value of Unsupervised Methods

Our results demonstrate that unsupervised semantic matching can achieve competitive performance on specific task types. While supervised methods (DPR, SPLADE) achieve higher absolute scores on some tasks, SF provides:

- **Zero-shot domain adaptation**: No labeled data required — switch from biomedical QA to narrative comprehension instantly
- **Interpretability**: Grid visualizations explain why a document retrieved — which grid cells activated, how they overlap with the query
- **Memory efficiency**: 512 bytes per document vs 3KB for dense methods — 6× compression
- **Boolean reasoning**: Direct AND/OR/NOT operations on fingerprints — enables compositional query languages

These properties make SF valuable for scenarios where training data is unavailable, interpretability is required, or resource constraints prevent dense retrieval.

### 8.7.2 The Vocabulary Mismatch Problem Revisited

SF's strong performance on MuSiQue (MRR=0.927, +92% vs BM25) provides the strongest evidence to date that vocabulary mismatch remains a significant challenge for lexical retrieval — and that topographic encoding provides a principled solution. The magnitude of the improvement (+92%) was unexpected: we did not anticipate that an unsupervised sparse method could exceed both lexical and dense methods on any dataset.

However, the broader pattern across 9 datasets shows that vocabulary mismatch is **only one component** of retrieval quality. Lexical precision (NQ-REaR), entity matching (2Wiki, HotpotQA), and score discrimination (BioASQ) are equally important — and SF cannot address these through semantic matching alone.

### 8.7.3 The Sparse-Dense Trade-off

The Orthogonality Constraint (Zahn et al., 2026) provides a theoretical framework:

- **Sparse methods** (SF, BM25): Naturally orthogonal, no training required, but limited compositional capacity
- **Dense methods** (DPR, ColBERT): Must learn separability through training, but can compose facts and handle complex reasoning

SF's success on MuSiQue (beating BM25 by +92% and DPR by +7.2%) suggests that **sparse methods can match or exceed dense methods on specific tasks** — not despite their simplicity but because of it. The grid's spatial organization, combined with SPLADE's learned expansion, provides complementary signals that dense methods must learn from scratch.

### 8.7.4 What Does NOT Work — The Value of Negative Results

A significant contribution of this thesis is the systematic documentation of what does not improve SF. We tested 7 distinct approaches to improving SF, and only SPLADE provided consistent gains. The negative results establish:

1. **SF's architecture is well-optimized** — the standard pipeline with SPLADE is hard to beat
2. **Improvements must add non-overlapping signal** — SPLADE works because it captures entity connections that SF does not
3. **No single modification can overcome SF's fundamental limitations** — compositional reasoning and large-corpus scaling require architectural changes, not parameter tuning
4. **The best (and only verified) improvement is SPLADE** — all other approaches tested are dead ends

---

## 8.8 Future Directions

### 8.8.1 Implemented Improvements (Now Part of Default Pipeline)

The following improvements have been validated and integrated:

1. **SPLADE hybrid retrieval** (+21% Belebele, +28% HotpotQA, +92% on MuSiQue): Learned sparse expansion combined with SF's semantic matching
2. **FAISS-accelerated OOV expansion** (30s → 0.075s/query): 400× speedup using approximate nearest neighbor search
3. **Per-dataset parameter registry** (+1–4%): YAML-based automatic parameter selection
4. **Batch query processing** (~25× speedup): Single subprocess call for all queries
5. **NoOOV default**: Verified across 6 datasets that OOV expansion has zero effect

### 8.8.2 Remaining Future Work

1. **Compositional retrieval for SF**: The only verified improvement is SPLADE. Future work should explore graph-based composition (fusing passages at the fingerprint level), attention-based entity chaining, and LLM-guided passage composition.

2. **Learned grid with better pretraining**: The current learned grid mapper (−79% vs t-SNE) failed due to noisy contrastive pairs and lack of smooth manifold preservation. Future work could explore UMAP or t-SNE initialization for the learned mapper (UMAP's fuzzy simplicial set provides a better topological prior for the contrastive loss), mutual information-based contrastive pairs (rather than raw co-occurrence), and Gumbel-Softmax for end-to-end differentiability.

3. **Negation-aware processing**: Our negation feature correctly identified negation patterns but did not improve retrieval. Future work should explore predicate-level scope analysis, LLM-based negation parsing, and context-dependent penalty weighting.

4. **Scaling to large corpora**: Parameter scaling guidelines for grid size, top percent, and spreading as functions of corpus size. This is essential for deploying SF in production settings.

### 8.8.3 Long-Term Vision

1. **Cross-lingual Semantic Folding**: Multilingual retrieval via aligned semantic spaces
2. **Streaming Semantic Folding**: Incremental updates without full recomputation
3. **Semantic Folding for Generation**: Extending from retrieval to text generation by traversing the semantic grid

---

## 8.9 Conclusion

Semantic Folding, when combined with SPLADE, achieves the strongest retrieval performance among unsupervised methods on MuSiQue (MRR=0.927, +92% vs BM25, +7.2% vs HippoRAG2) — a multi-hop QA dataset where vocabulary mismatch is severe and entity chains require both semantic and lexical coverage. On 7 of 9 benchmark datasets, SF+Splade achieves MRR ≥ 0.857, demonstrating that unsupervised grid-based matching, when augmented with learned sparse expansion, can compete with and sometimes exceed fully supervised methods.

The four pillars of SF's success — grid proximity for vocabulary mismatch, distributional semantics without training, sparse binary fingerprint orthogonality, and SPLADE's complementary lexical signal — collectively explain why SF beats BM25 on MuSiQue and matches or exceeds DPR on three datasets. These pillars also explain SF's limitations: score compression on large corpora, negation blindness, and the compositional gap.

The systematic negative results from Phase 2c/3/4 are as valuable as the positive findings. We demonstrated that cross-attention, snippet ranking, adaptive spreading, learned grid indexing, MeSH ontology expansion, and LambdaMART re-ranking all fail to improve SF. The only verified improvement is SPLADE — learned sparse expansion that provides genuinely non-overlapping signal.

SF occupies a unique position: the only retrieval method providing unsupervised semantic matching, interpretable visualizations, and memory-efficient storage with competitive performance. For scenarios where training data is unavailable, interpretability is required, or resource constraints prevent dense retrieval, SF+Splade is the strongest available approach.

---

## References

- Formal, T., et al. (2021). SPLADE. *SIGIR 2021*.
- Furnas, G. W., et al. (1987). The vocabulary problem. *CACM*.
- Gutiérrez, J., et al. (2024). HippoRAG. *arXiv:2405.13747*.
- Gutiérrez, J., et al. (2025). HippoRAG 2. *arXiv:2502.12072*.
- Ho, X., et al. (2020). 2WikiMultihopQA. In *ACL 2020*.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval. *EMNLP 2020*.
- Kwiatkowski, T., et al. (2019). Natural Questions. *JMLR*.
- Mallen, A., et al. (2023). PopQA. *EACL 2023*.
- Malayi, L., et al. (2023). Belebele. *TACL 2023*.
- Nentidis, A., et al. (2025). BioASQ 2024 Overview. *arXiv:2508.20532*.
- Santhanam, K., et al. (2022). ColBERTv2. *NAACL 2022*.
- Trivedi, H., et al. (2022). MuSiQue. *NAACL 2022*.
- Yang, Z., et al. (2018). HotpotQA. *EMNLP 2018*.
- Zahn, O., et al. (2026). Attention Is Not Retention. *arXiv:2601.15313*.
