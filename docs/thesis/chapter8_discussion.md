# Chapter 8: Discussion

## 8.1 Summary of Key Findings

Our evaluation of Semantic Folding (SF) across 9 benchmark datasets reveals a nuanced performance pattern that depends critically on task characteristics, corpus size, and the presence of vocabulary mismatch.

### 8.1.1 Performance Hierarchy

*For the complete cross-dataset performance tables, see Chapter 7, Table 7.1 and Table 7.2.*

The central finding is that **SPLADE-only outperforms SF-only on 5/9 datasets**, and the SF+SPLADE hybrid is beneficial on only 2/9 datasets (2WikiMultihopQA and PubMedQA). This contradicts the earlier claim that "SF+SPLADE is the best configuration on 7/9 datasets" — that assessment did not measure SPLADE-only performance.

**Performance tiers** (based on SF+SPLADE MRR):

| Tier | MRR Range | Datasets | Characterization |
|:----:|:---------:|----------|-----------------|
| **Dominant** | >0.900 | PopQA, NarrativeQA, PubMedQA, Belebele, MuSiQue | SF matches or exceeds all baselines |
| **Competitive** | 0.800–0.899 | 2WikiMultihopQA, HotpotQA | SF close to BM25, gap <7% |
| **Moderate** | 0.500–0.699 | NQ-REaR | SF underperforms but functional |
| **Poor** | <0.500 | BioASQ | SF fails — requires different approach |

### 8.1.2 The Compositional Gap

The relationship between hop count and SF performance is nuanced. As shown in Chapter 7 (Table 7.1), SF+SPLADE outperforms BM25 on MuSiQue (2–5 hops, MRR=0.782 vs 0.482, +62%), but underperforms on 2-hop datasets like HotpotQA (MRR=0.857 vs 0.869, −1.4%).

The MuSiQue result appears to contradict the compositional gap — SF succeeds on the hardest multi-hop dataset. The resolution is that **SF+SPLADE succeeds on MuSiQue despite the compositional gap, not by bridging it**. MuSiQue's candidate pools (20 passages/query) are carefully curated to include gold supporting passages. SF+SPLADE's semantic matching, combined with SPLADE's learned expansion, is sufficient to identify correct passages when entities are distinctive and the pool is small. The compositional gap is real but masked by SPLADE's lexical expansion in small-pool settings.

### 8.1.3 Negative Results Summary

A significant contribution of this work is the systematic documentation of **what does NOT improve SF**. As detailed in Chapter 7 (§7.2.3–§7.2.6), we tested 8 distinct approaches:

| Attempt | MRR Impact | Verdict |
|---------|:----------:|---------|
| Cross-attention (SF-Only) | **−87%** | Catastrophic |
| Cross-attention (SF+SPLADE) | −18% | Degrades |
| Snippet ranking | 0% (identical) | Neutral |
| Adaptive spreading | 0% (identical) | Neutral |
| Learned grid (SF-Only) | **−79%** | Severely degrades |
| Learned grid (SF+SPLADE) | −16% | Degrades |
| MeSH ontology expansion | 0% to −3.8% | No benefit |
| LambdaMART re-ranking | −5.5% | Underperforms |

**General lesson — The Complementarity Principle**: Features duplicating existing SF signals (snippet ranking, adaptive spreading) cannot improve performance. Features adding complementary signals may help but are difficult to implement correctly (cross-attention degenerates). The **only verified improvement** is SPLADE's learned sparse expansion, which provides genuinely non-overlapping signal.

### 8.1.4 Hypothesis Re-evaluation

The three research hypotheses (introduced in Chapter 1, evaluated in Chapter 7) are re-assessed in light of the full 9-dataset evidence:

| Hypothesis | Prediction | Outcome | Assessment |
|------------|-----------|---------|------------|
| **H1 — Semantic Matching** | SF captures vocabulary mismatch better than BM25 | Partially supported | SPLADE (not SF) drives vocabulary mismatch gains. SF contributes negatively on 5/9 datasets. |
| **H2 — Complementarity** | SF + SPLADE provide non-overlapping signals | **Falsified** | SPLADE-only beats SF+SPLADE on 5/9 datasets. SF degrades SPLADE's performance. The α-sensitivity curve is monotonic — more SF consistently hurts. |
| **H3 — Feature Invariance** | Duplicate-signal features ≤ 0% effect | **Supported** | 7 feature variants tested; only SPLADE improves. Cross-attention catastrophically fails. |

**Revised interpretation**: H1 is partially supported in that sparse methods (SPLADE is sparse) capture vocabulary mismatch. But SF's grid proximity does not contribute meaningfully beyond SPLADE. H2 is falsified — the two signals are correlated, not complementary. H3 remains robust and establishes an empirical ceiling for SF-based architectures.

---

## 8.2 Why Semantic Folding Wins — Interpreting the Results

### 8.2.1 The Four Pillars of SF's Success (Revised)

Semantic Folding's competitive performance on 7/9 datasets can be traced to four architectural properties. However, the third pillar requires revision in light of the SF fingerprint correlation analysis (Chapter 5, §5.2.3):

#### Pillar 1: Grid Proximity Captures Vocabulary Mismatch

**Mechanism**: SF maps phrases to 2D grid positions via dimensionality reduction (t-SNE or UMAP). Distributionally similar phrases map to nearby cells, creating a semantic manifold where vocabulary-mismatched terms cluster together.

**Evidence**: MuSiQue (MRR=0.782 vs BM25 0.482, +62.2%), though SPLADE-only (0.876) shows the gain is largely SPLADE's learned expansion, not SF's grid proximity.

**Limitation**: This pillar explains SF's advantage only when vocabulary mismatch is the primary challenge. On datasets where exact term matching dominates (Belebele, NQ-REaR), grid proximity provides no advantage over BM25.

#### Pillar 2: Distributional Semantics Without Training

**Mechanism**: The term-context matrix captures semantic relationships from unlabeled text. No labeled training data required.

**Evidence**: SciFact (MRR=0.755 vs DPR 0.675, +12.1%). Domain-specific tasks where DPR's general-domain training does not transfer; SF's unsupervised approach adapts instantly.

**Implication**: SF is the only unsupervised method achieving competitive performance against supervised baselines on domain-specific tasks.

#### Pillar 3: Sparse Binary Fingerprints — Limited Orthogonality Due to Spatial Correlation

**Original claim**: "Sparse binary fingerprints provide natural orthogonality."

**Revised claim**: Independent random sparse binary vectors are nearly orthogonal, but **SF fingerprints are spatially correlated by design** (Gaussian smoothing σ=1.5, Morton encoding, IDF-weighted aggregation). The actual pairwise cosine similarity distribution has higher mean and higher variance than the random-SDR prediction.

**Evidence**: As shown in Chapter 5 (§5.2.3), SF fingerprints are **not** orthogonal. Terms with similar distributional contexts map to nearby grid cells, intentionally increasing overlap. This is the desired mechanism for semantic matching, but it simultaneously reduces orthogonality.

**Implication**: The Orthogonality Constraint (Zahn et al., 2026) applies strictly to independent random SDRs. SF's spatially correlated fingerprints do not satisfy the constraint, which explains why SF cannot match dense methods on tasks requiring fine-grained discriminability (e.g., BioASQ score compression).

#### Pillar 4: SPLADE Provides the Missing Lexical Signal

**Mechanism**: SF captures semantic similarity but misses exact entity names. SPLADE's learned sparse expansion adds lexical precision.

**Evidence**: SF+SPLADE vs SF-only on 2WikiMultihopQA (+8.5% MRR). SPLADE improves SF on 7/9 datasets.

**Caveat**: As shown in Chapter 7 (§7.2.2), SPLADE-only outperforms SF+SPLADE on 5/9 datasets. The "missing lexical signal" from SPLADE is so effective that SF's semantic matching becomes redundant or even interfering.

### 8.2.2 When and Why SF Succeeds

SF's success is not uniform — it follows a clear pattern determined by task characteristics:

| Condition | SF Performance | Example Dataset |
|-----------|:--------------:|-----------------|
| High vocabulary mismatch + small candidate pool | **Excellent** (MRR > 0.90) | MuSiQue |
| Low vocabulary mismatch + small pool | **Competitive** (MRR 0.85–0.90) | HotpotQA, Belebele |
| Large candidate pool + complex queries | **Poor** (MRR < 0.60) | BioASQ, NQ-REaR |

**Predictive rule**: SF excels when (a) query and document vocabularies differ substantially, and (b) the candidate pool is small enough to avoid score compression (< 100 docs). When either condition reverses, SF's advantage diminishes or disappears.

---

## 8.3 Why Feature Variants Failed

### 8.3.1 The Complementarity Principle

The Phase 2c/3/4 results establish a general principle: **features that duplicate existing SF signals cannot improve performance**. This explains why snippet ranking and adaptive spreading had zero effect — they compute the same phrase-level overlap that SF's dot-product already captures.

Cross-attention introduced a genuinely different signal (pairwise phrase alignment) but failed because attention scoring discards the spatial structure that makes SF effective. The O(N²) attention computation provides no information beyond SF's O(active_cells) dot-product.

### 8.3.2 Why the Learned Grid Underperforms

The learned grid mapper (Phase 3) was trained to predict 2D coordinates from phrase embeddings using a contrastive loss. Despite being trained on the same term-context matrix as t-SNE/UMAP, it underperformed by −79% (SF-only) to −16% (SF+SPLADE).

**Root causes**:
1. **Noisy training signal**: Contrastive pairs from co-occurrence include many spurious relationships
2. **No local structure preservation**: t-SNE's Gaussian kernel and UMAP's fuzzy simplicial set naturally emphasize local neighborhoods; the contrastive loss does not
3. **Discretization error**: Continuous coordinates must be rounded to grid cells, losing precision

**Future direction**: A learned grid initialized from UMAP coordinates (rather than trained from scratch) might avoid these issues.

---

## 8.4 Comparison with Other Methods

### 8.4.1 SF vs BM25

SF+SPLADE beats BM25 on MuSiQue (+62%) and approaches parity on HotpotQA (−1.4%). On other datasets, BM25 maintains a 3–16% advantage.

**Where SF wins**: High synonymy tasks (MuSiQue, PubMedQA)
**Where BM25 wins**: Exact entity matching (PopQA, Belebele), factoid retrieval (NQ-REaR)

### 8.4.2 SF vs Dense Retrieval

SF+SPLADE matches or exceeds DPR on two datasets (HotpotQA +9.9%, PopQA +5.3%) without any training data; on MuSiQue it beats BM25 (+62.2%) but trails HippoRAG2's dense baseline (0.865, −9.6%). This is notable because DPR requires 50K+ labeled pairs.

**Key advantage**: Zero-shot domain adaptation. SF can switch from biomedical QA to narrative comprehension instantly; DPR requires retraining.

### 8.4.3 SF's Unique Position

SF occupies a unique quadrant in the retrieval landscape:

| Aspect | SF | BM25 | DPR | ColBERT |
|--------|-----|------|-----|---------|
| Training required | None | None | 50K+ pairs | 500K+ pairs |
| Memory per document | 512 bytes | ~1KB | 3KB | 3KB |
| Interpretability | Grid visualization | Term frequency | Black box | Black box |
| Best dataset MRR | 0.782 (MuSiQue) | 0.995 (Belebele) | 0.863 (NQ) | ~0.90 (NQ) |

No other method provides unsupervised semantic matching, interpretable grids, and memory-efficient storage simultaneously.

---

## 8.5 Limitations

### 8.5.1 Current Limitations

1. **Score compression on large corpora**: On BioASQ (1075 docs), all documents score within 0.001–0.015. SF's sparse dot-product lacks dynamic range for large corpora. **Mitigation**: Use BM25 pre-retrieval + SF re-ranking.

2. **Negation blindness**: SF treats "not considered" identically to "considered." Predicate-level scope analysis is needed.

3. **Compositional gap**: SF cannot compose facts across passages. The MuSiQue result is achieved despite this limitation, not by overcoming it.

4. **Computational cost**: Indexing takes ~5 minutes for 100 passages (with UMAP). Acceptable for research; limits dynamic deployment.

5. **Grid size sensitivity**: The 64×64 grid is optimal for 20-doc corpora. Scaling to larger corpora requires larger grids with re-tuned parameters.

### 8.5.2 Methodological Limitations

1. **Binary relevance**: Ground truth uses binary relevance. Graded relevance would improve NDCG discrimination.
2. **Dimensionality reduction stochasticity**: t-SNE and UMAP results vary with random seed. We verified key results with 3 seeds; MRR variation is ±0.015 (t-SNE) to ±0.008 (UMAP).
3. **Query-count differences**: PubMedQA has 31 queries; Belebele has 100; others have 50. Affects statistical confidence.
4. **Statistical power for operator separation (journal extension)**: the n=50 confirmatory protocol with Holm–Bonferroni correction shows that single-gold-per-query MRR requires larger samples to separate individual operator pairs — only one of 63 pairwise comparisons is family-wise significant. Claims therefore rest on replicated orderings across datasets, checkpoints, and perturbation conditions rather than on any single test.
5. **Checkpoint coverage**: two learned sparse checkpoints (SPLADE-cocondenser, SPLADE-v3) agree in operator ordering; DPR is tested at one checkpoint, so the geometry-conditioning claim for dense signal B rests on score-scale arguments plus the four-pair matrix rather than checkpoint replication.

---

## 8.6 Implications for Retrieval Research

### 8.6.1 The Value of Unsupervised Methods

Our results demonstrate that unsupervised semantic matching can achieve competitive performance on specific tasks. SF provides zero-shot domain adaptation, interpretability, and memory efficiency — properties valuable for scenarios where training data is unavailable.

### 8.6.2 The Vocabulary Mismatch Problem Revisited

SF's strong performance on MuSiQue (+62% vs BM25) provides evidence that vocabulary mismatch remains a significant challenge for lexical retrieval. However, the broader 9-dataset pattern shows that vocabulary mismatch is only one component of retrieval quality. Lexical precision, entity matching, and score discrimination are equally important.

### 8.6.3 The Sparse-Dense Trade-off

The Orthogonality Constraint (Zahn et al., 2026) provides a theoretical framework: sparse methods are naturally orthogonal but limited in compositional capacity; dense methods can compose facts but suffer from Semantic Interference. SF's success on specific tasks suggests that sparse methods can match or exceed dense methods — not despite simplicity, >- **The "Interference Wall"**: As technical domains become more specialized, dense models inevitably suffer from Semantic Interference because they cluster related facts too tightly. SF's high-dimensional bit-space avoids this wall by mathematical construction. The SciFact result (SF 0.755 vs DPR 0.675) validates this theory.

**SPLADE backbone note.** The results in this thesis use the standard SPLADE checkpoint (naver/splade-cocondenser-ensembledistil) from 2021. Given the rapid progress in learned sparse models, replacing this with the latest variants could further push the performance ceiling. **Mistral-SPLADE** (Echo-Mistral-SPLADE) uses a decoder-only LLM (Mistral-7B) with echo embeddings to achieve state-of-the-art performance on BEIR benchmark (avg NDCG@10 = 55.07), outperforming all SPLADE variants. The hybrid architecture is model-agnostic — any learned sparse model can serve as the SPLADE component, allowing immediate adoption of Mistral-SPLADE for improved performance.

### 8.6.4 A Diagnostic Theory of Hybrid Retrieval

Our diagnostic framework provides a comprehensive diagnostic framework that recontextualizes our experimental results:

#### The Taxonomy of Hybrid Failures
All hybrid retrieval failures fall into three categories independent of retriever choice:

```
Hybrid Failure
├── Signal Failure          (components carry no exploitable information)
│     ├── True redundancy         — rank correlation τ ≈ 1
│     └── Locality-induced feature ceiling — feature adds nothing beyond base signal
├── Operator Failure        (operator discards information task needs)
│     ├── Scale mismatch          — score spaces incommensurate
│     └── Magnitude destruction   — rank-only operator discards magnitude
└── Representation Failure  (signal's encoding has structural ceiling)
      ├── Compositional gap       — no relational algebra to bind facts
      └── Score concentration     — dynamic range collapses with pool size
```

Our SF+SPLADE experiments map directly to this taxonomy:
- **Belebele, NarrativeQA**: Signal Failure → Complementarity Illusion (true redundancy)
- **2WikiMultihopQA, HotpotQA**: Operator Failure → Magnitude Destruction (RRF discards compositional confidence)
- **MuSiQue**: Signal + Operator Failure → SPLADE-only outperforms hybrid
- **NQ-REaR**: Signal Failure → True Redundancy (τ=0.82, no RRF recovery)
- **BioASQ, NQ-REaR, SciFact (deep pool)**: Representation Failure → Score Concentration / Deep-Pool Collapse

**Refinements from the journal-extension evidence** (Chapter 7, §§7.2.9–7.2.12): the Operator Failure branch is now quantified across all seven operators — magnitude-preserving operators (CombSUM/CombMNZ) lead on every multi-hop/factoid dataset at n=50, with the single family-wise-significant pairwise separation being Borda vs CombMNZ on MuSiQue. The Magnitude Destruction mechanism is now established *causally*: rank-preserving magnitude transforms leave RRF's fused ranking bit-identical while reordering score-space fusion, and rank destruction collapses RRF maximally — separating the two information channels experimentally rather than by definition. The Score Concentration branch is likewise refined: controlled pool growth from 20 to 494 documents moves no operator's MRR in either pairing; concentration effects bind only at full-corpus scale (SciFact 5,183), not gradually.

#### Score Geometry and Operator Information Preservation
The **Score Geometry** framework provides a coordinate system $\mathcal{G}_R(q) = (\pi, \mathbf{s}, \mu_S, \sigma_S^2)$ that captures exactly what fusion operators act upon. The proven claim that **RRF preserves only ordering while linear interpolation preserves both ordering and magnitude** is a theoretical result from operator definitions, not empirical observation.

This explains all our fusion phenomena:
- **Complementarity Illusion** (Belebele, NarrativeQA): High τ + scale mismatch → RRF restores performance
- **Magnitude Destruction** (2Wiki, HotpotQA): Multi-hop needs magnitude → Linear preserves it
- **True Redundancy** (NQ-REaR): High τ + no RRF recovery → drop weaker signal

#### Hybrid Compatibility Profile
The **Hybrid Compatibility Profile** $(\tau, \mathrm{RRF\text{-}recoverable}, T)$ provides a pre-fusion diagnostic to choose the operator before fusion. The decision rule is retrospectively consistent across 9 datasets but not yet validated out-of-sample or on other retriever pairs.

#### Deep-Pool Collapse
Small-pool MRR scores are upper bounds of re-ranking conditioned on a strong first-stage retriever. Full-corpus evaluation (SciFact 5,183 docs) shows near-random MRR for SF (0.0109), BM25 (0.0095), and SF+SPLADE RRF (0.0004). This aligns with standard IR benchmarks like BEIR.

#### Locality-Induced Feature Ceiling
For SDRs with Morton-ordering, any feature built from spatial overlap is informationally equivalent to the dot product. This explains why snippet ranking, adaptive spreading, OOV, BM25 filtering, and query decomposition showed zero effect — they compute the same overlap. Only locality-breaking features (learned grid, cross-attention) change performance, and they degrade it.

#### Score Concentration
The bounded dynamic range of SF scores ($\mathbb{E}[s] \approx 41$, $\sigma[s] \approx 6$) explains score compression on large pools. BM25's unbounded range (mean 5.2, std 4.1 on same corpus) maintains discrimination.

---

## 8.7 Future Directions

### 8.7.1 Implemented Improvements

The following are now part of the default pipeline:
1. **SPLADE hybrid retrieval** (α=0.3)
2. **FAISS-accelerated OOV expansion** (400× speedup)
3. **Per-dataset parameter registry** (YAML-based automatic selection)
4. **Batch query processing** (~25× speedup)

### 8.7.2 Remaining Challenges

1. **Compositional retrieval**: Graph-based composition, attention-based entity chaining
2. **Learned grid with better pretraining**: UMAP-initialized mapper
3. **Negation-aware processing**: Predicate-level scope analysis
4. **Scaling to large corpora**: Parameter scaling guidelines

---

## 8.8 Conclusion

Semantic Folding provides unsupervised semantic matching that is competitive with supervised methods on specific task types. The key findings are:

1. **SPLADE-only outperforms SF-only on 5/9 datasets**. SF's contribution is positive on only 2/9 datasets.
2. **The complementarity hypothesis (H2) is falsified**. SF and SPLADE signals are correlated, not complementary.
3. **The only verified improvement to SF is SPLADE**. All other feature variants either degrade or have zero effect.
4. **SF excels when vocabulary mismatch is high and candidate pools are small**. These conditions predict SF's performance across datasets.

SF occupies a unique position: the only retrieval method providing unsupervised semantic matching, interpretable grid visualizations, and memory-efficient storage. For scenarios where training data is unavailable or interpretability is required, SF+SPLADE is the strongest available approach.

---


## 8.4 The Information-Preservation Framework (Journal V4 Findings)

The journal extension of this thesis reframes hybrid fusion as an information-preservation problem rather than solely a choice among aggregation formulas. Rank-only operators such as RRF preserve ordinal information while discarding score magnitude; score-space operators preserve additional information but are consequently sensitive to calibration and score geometry.

### 8.4.1 Confirmatory Evidence at n=100

The confirmatory core demonstrates that CombSUM significantly outperforms RRF after Holm correction within the 21-comparison operator family on both multi-hop datasets: HotpotQA (Δ=+0.093, p_Holm=0.0007) and MuSiQue (Δ=+0.044, p_Holm<0.0001). The effect is conditional — NQ-REaR remains largely non-separable (4/21 comparisons survive), consistent with its large-pool factoid profile.

### 8.4.2 Cross-Signal Scale as the Locus of Magnitude Utility

A critical finding from the rank-conditioned magnitude analysis is that within a single signal, magnitude adds no gold-discrimination information beyond ordinal position (ΔAUC ≤ 0 with bootstrap CIs spanning zero). This null is informative rather than contradictory: per-signal scores are monotone in their own ranks. The utility of magnitude arises at the **cross-signal level** — when two heterogeneous-scale signals are fused, relative magnitudes across signals change which document wins. Relevance-bearing magnitude is thus a property of the pair geometry, not of either component alone.

### 8.4.3 Practical Guidance for Closed-Domain QA

**1. Diagnose before choosing fusion.** Measure score dispersion, top-rank margins, rank agreement, calibration, and gold-score separation when labels are available.

**2. Use RRF when rank is the reliable information source**, especially when component scores are incomparable or poorly calibrated.

**3. Use score-space fusion when magnitude is demonstrably informative**, especially when one component has stable relevance-bearing score separation.

**4. Do not assume "multi-hop implies CombSUM."** Instead: multi-hop + magnitude-bearing sparse signal + complementary geometry → investigate score-space fusion.

**5. Never expect fusion to fix candidate-generation failure.**

### 8.4.4 What Would Falsify Our Interpretation?

Our reading would be weakened if rank-preserving magnitude interventions consistently failed to alter score-space fusion rankings, if magnitude provided relevance information beyond rank after conditioning (it does not within a single signal), or if operator × retriever-pair interactions disappeared under larger confirmatory samples.


## References

- Formal, T., et al. (2021). SPLADE. *SIGIR 2021*.
- Furnas, G. W., et al. (1987). The vocabulary problem. *CACM*.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval. *EMNLP 2020*.
- Zahn, O., et al. (2026). Attention Is Not Retention. *arXiv:2601.15313*.
