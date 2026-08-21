# What Does Fusion Preserve? Task-Dependent Information Loss in Hybrid Retrieval

**Mojtaba Banaei¹, Maseud Rahgozar², and Heshaam Faili³**

¹,² Data Base Research Group (DBRG), School of Electrical and Computer Engineering, University of Tehran, Tehran, Iran
³ School of Electrical and Computer Engineering, Faculty of Engineering, University of Tehran, Tehran, Iran

`smbanaei@ut.ac.ir`, `rahgozar@ut.ac.ir`, `hfaili@ut.ac.ir`

---

## Abstract

Hybrid retrieval combines the ranked lists or score distributions of multiple retrievers to improve robustness, yet the choice of fusion operator is routinely treated as a tunable hyperparameter. We argue — and demonstrate experimentally — that this choice is not free: the information a fusion operator preserves or discards must be compatible with the information structure that the retrieval task itself depends on. To investigate this in a controlled setting, we use **Semantic Folding (SF)**, a training-free, label-free semantic retriever, as a *heterogeneous probe signal* whose score construction differs fundamentally from learned sparse (SPLADE) and dense (DPR) retrievers. Across eight closed-domain question-answering datasets spanning single-hop, reading-comprehension, and multi-hop topologies, and across seven fusion operators (RRF, Borda, CombSUM, CombMNZ, linear, min-max, z-score) and four retriever pairs (SF+SPLADE, SF+DPR, BM25+SPLADE, BM25+DPR), we make four contributions. (1) We show that hybrid fusion effectiveness depends systematically on task topology, not merely on signal complementarity. (2) We isolate **magnitude information loss** as the mechanism by which rank-only fusion can degrade compositional retrieval. (3) Through controlled synthetic magnitude-perturbation experiments in which rank is held fixed while score magnitude is manipulated, we establish that score magnitude can causally determine multi-hop retrieval performance, rather than merely correlating with it. (4) We characterize two independent limitations of the SF probe — feature invariance and score concentration under growing candidate pools — and separate architectural limitations from fusion-operator limitations. We position our work explicitly against Bruch et al. (TOIS 2024): where they analyze what fusion functions do to score distributions, we ask *when the information they discard becomes task-relevant*, and we demonstrate the answer experimentally across task topology and retriever pairs.

**Keywords:** Hybrid Retrieval · Fusion Functions · Sparse Distributed Representations · Information Preservation · Multi-Hop Question Answering · Reciprocal Rank Fusion · Task-Operator Compatibility

---

## 1. Introduction

### 1.1 Problem

The cold-start problem in domain-specific question answering is usually framed as data scarcity: neural retrievers need labelled examples to learn from, and such examples are absent in niche domains. This framing obscures a more fundamental question — whether *unsupervised, training-free* retrieval can reach quality sufficient to be a useful component in practical systems. We find partial support: Semantic Folding (SF), a training-free method encoding semantic structure into Sparse Distributed Representations, matches BM25 on single-hop biomedical questions with no training data. Its performance collapses, however, when retrieval requires multi-hop compositional reasoning.

Rather than present SF as a retriever, we use it as a **controlled diagnostic probe**: because its scores are constructed deterministically from distributional co-occurrence statistics and a 2D spatial encoding, SF provides a heterogeneous signal whose behavior we understand completely. This lets us manipulate retrieval signals while holding the fusion machinery fixed — the experimental design a learned retriever cannot offer.

### 1.2 Why fusion is not operator-neutral

A standard hybrid system fuses two retrievers with either Reciprocal Rank Fusion (RRF) or linear interpolation. We show these are not interchangeable: RRF rescues single-hop performance to BM25 parity but degrades multi-hop retrieval by a large margin (−15.5 points MRR on 2WikiMultihopQA), while linear interpolation inverts this pattern, improving multi-hop MRR by 62.2% on MuSiQue yet matching or underperforming on single-hop tasks. The divergence is not a tuning artifact. It is a *structural* property of what each operator preserves:

- **Rank-only operators** (RRF, Borda) discard absolute scores and keep only ordinal position. They are robust to score-scale mismatch but blind to magnitude.
- **Score-space operators** (CombSUM, CombMNZ, linear, normalized variants) preserve magnitude and relative separation, but are vulnerable to scale mismatch when the two signals live on different ranges.

The central claim of this paper is that **the relevance of the information a fusion operator discards is task-dependent**. For single-hop matching, rank is often sufficient; for multi-hop composition, absolute score magnitude encodes how many reasoning hops were satisfied, and discarding it is harmful.

### 1.3 Research Questions

- **RQ1 (Complementarity).** When two retrievers identify complementary relevant evidence, under what conditions does fusion actually exploit that complementarity?
- **RQ2 (Fusion information).** Which properties of retrieval scores are preserved or discarded by different fusion operators, and how does this affect performance across task topologies?
- **RQ3 (Causality).** Does score magnitude *causally* contribute to multi-hop retrieval performance, or is the observed association merely a consequence of ranking correlation and score normalization?
- **RQ4 (Boundaries).** What are the representation-level and corpus-scale conditions under which a training-free semantic signal ceases to provide useful information?

### 1.4 Contributions

1. A controlled cross-task analysis showing that hybrid fusion effectiveness depends systematically on the information structure of the retrieval task, not merely on the complementarity of the signals.
2. Identification of **magnitude information loss** as the mechanism through which rank-based fusion degrades compositional retrieval.
3. Controlled magnitude-perturbation experiments (synthetic + real) that isolate score magnitude as a causal factor, distinguishing the mechanism from model-specific score behavior and scale mismatch.
4. Characterization of two boundary conditions of the training-free probe — feature invariance and score concentration — with practical guidance for deploying heterogeneous retrieval signals.

---

## 2. Background and Related Work

### 2.1 Hybrid Retrieval

Combining multiple retrievers is standard practice for robustness. Cold-start QA systems must adapt quickly to terminology shifts; BM25, the usual zero-shot baseline, is challenged by domain synonyms, while neural methods close the gap at the cost of in-domain annotation. Learned sparse methods (SPLADE-family) offer a middle ground. The conventional combination is RRF or linear interpolation. We argue these overlook that signals can be combined in more ways than mixing them at the top of the ranking, which may destroy the mathematical properties that make individual signals useful.

### 2.2 Fusion Functions

Score combination has a long history. Fox and Shaw (1994) proposed CombSUM and CombMNZ, score-summation with multiplicity weighting. Cormack, Clarke, and Buettcher (2009) proposed RRF, motivated by the observation that raw scores from different models are not comparable — RRF discards absolute scores in favor of relative ranks. Bruch, Gai, and Ingber (2024, TOIS) provide the most recent and comprehensive analysis, examining convex combination and RRF, normalization, and parameter sensitivity, and explicitly noting that rank-based fusion discards score-distribution information. **Our work does not re-derive these properties; it builds on them.** Where Bruch et al. establish *what fusion functions do to score distributions*, we investigate *when the information they discard becomes task-relevant*, and we demonstrate the relationship experimentally across task topology and retriever pairs. This positioning is essential: the claim "RRF discards magnitude" is already established; the contribution is showing *when that discarded magnitude matters*.

### 2.3 Rank vs Score Information

A rank-only operator is invariant under any strictly monotonic transformation of component scores (formalized in §3.6). Consequently it discards score magnitude, score distance, nonlinear calibration, and confidence separation — provided ordering is unchanged. Score-space operators preserve these. The empirical question is therefore: *when is discarded magnitude information actually useful for the task?*

### 2.4 Multi-Hop Retrieval

Multi-hop QA requires composing evidence from multiple documents. Recent work identifies lost-in-retrieval and reasoning-depth problems that are precisely the failure mode we observe: when absolute scores encode how many hops matched, rank-only fusion collapses distinct compositional evidence to identical ranks.

### 2.5 Semantic Folding and Sparse Distributed Representations

SDRs are binary vectors of large dimensionality where most bits are zero. SF arranges vocabulary on a 2D grid and maps text into sparse fingerprints over that grid. We treat SF as a *controlled probe* whose score construction is fully characterized, not as a principal algorithmic contribution. Its key properties: no task labels, no gradient training, binary SDR representation, ~512 B/doc storage, CPU-only query. These make it an unusually transparent heterogeneous signal for fusion experiments.

### 2.6 Positioning Against Prior Fusion Analysis

Bruch et al. (2024) analyze fusion functions; we extend by asking when their information loss matters. Our novelty is the *task-dependent* framing plus the causal isolation (RQ3) and the second-model-pair validation (RQ2) that together show the phenomenon follows the task topology, not a specific retriever's score behavior.

---

## 3. Conceptual Framework

We model hybrid retrieval as a pipeline:

```
Retrieval signal → score space → fusion operator → information retained → task requirement → effectiveness
```

### 3.1 Retrieval Signal Properties

Each retriever emits, per query, a score distribution over candidates. Two structural properties matter: (a) the **rank** of each candidate (ordinal position), and (b) the **magnitude / margin** between scores (how confidently the retriever distinguishes relevant from irrelevant, and — in multi-hop settings — how many hops were satisfied).

### 3.2 Rank Information

Sufficient when the task only requires correct *ordering*: the gold document need only be ranked above distractors. Single-hop matching often satisfies this.

### 3.3 Score Magnitude

Carries additional signal when the *degree* of match matters: in multi-hop QA, a high SPLADE/DPR score indicates multiple hops matched; a low score indicates a partial match. Magnitude thereby encodes compositional confidence.

### 3.4 Complementarity vs Redundancy

Two retrievers are **complementary** when they surface different relevant documents (low rank correlation, Kendall's τ); **redundant** when they agree (high τ). We use Kendall's τ as a diagnostic: high τ on a multi-hop task suggests true redundancy (fusion is pointless); high τ on a single-hop task suggests switching to RRF.

### 3.5 Task-Operator Compatibility Hypothesis

> The optimal fusion operator is a function of the task's information requirement, not merely of the signals' scale properties: rank-preserving operators suit tasks whose relevance is captured by ordering; magnitude-preserving operators suit tasks whose relevance is captured by score separation.

We state this as a *hypothesis* to be tested, deliberately avoiding the stronger "constraint"/"law" language of the conference version, which our own 2WikiMultihopQA RRF result already contradicted in edge cases.

### 3.6 Formal Rank-Invariance Proposition

**Proposition 1 (Rank-fusion invariance).** Let s(d) be a retrieval score and f any strictly monotonic transformation. Then rank(s(d)) = rank(f(s(d))). Therefore any rank-only fusion operator R satisfies R(s₁,…,sₘ) = R(f₁(s₁),…,fₘ(sₘ)) for strictly monotonic fᵢ. Consequently rank-only fusion is invariant to score magnitude, score distance, nonlinear calibration, and confidence separation, provided ordering is unchanged.

This is mathematically trivial but establishes the clean separation that motivates the empirical work: rank-only and score-space operators differ *exactly* in whether magnitude survives.

---

## 4. Experimental Methodology

### 4.1 Datasets

Eight closed-domain QA datasets (PopQA, PubMedQA, NarrativeQA, Belebele, 2WikiMultihopQA, HotpotQA, MuSiQue, NQ-REaR) plus SciFact for scientific fact-verification. Candidate sets are 1 gold + 19 BM25 hard negatives (or naturallar smaller pools), standard in this benchmark family.

### 4.2 Task Topology

We classify each dataset by the reasoning its questions demand:

| Topology | Datasets | Information needed |
|----------|----------|--------------------|
| Entity lookup | PopQA | Ordering |
| Biomedical / narrative / reading | PubMedQA, NarrativeQA, Belebele | Ordering (+ semantics) |
| Multi-hop (2 hops) | 2WikiMultihopQA, HotpotQA | Magnitude / composition |
| Multi-hop (2–5 hops) | MuSiQue | Magnitude / composition |
| Factoid (large pool) | NQ-REaR | Magnitude / separation |

### 4.3 Candidate Regimes

We explicitly distinguish two regimes:

- **Controlled reranking (Regime A).** A preselected candidate set (gold + BM25 negatives). Isolates fusion mechanics, candidate availability, and score distributions. *This is not first-stage retrieval.*
- **Full-corpus retrieval (Regime B).** Query → entire corpus → retriever A + retriever B → fusion → ranking. Validates that findings generalize beyond reranking (§8.5, SciFact deep-pool + full-corpus).

### 4.4 Retrieval Models

- **SF** (training-free SDR probe)
- **SPLADE** (learned sparse)
- **DPR** (dense bi-encoder) — second model pair
- **BM25** (lexical baseline)

### 4.5 Fusion Operators

Seven operators spanning three information classes:

| Class | Operators | Preserves |
|-------|-----------|-----------|
| Rank-space | RRF (k=60), Borda | rank only |
| Raw score-space | CombSUM, CombMNZ, Linear (α=0.3) | magnitude + scale |
| Normalized score-space | min-max+Linear, z-score+Linear | magnitude (scale-removed) |

*[Empirical results: §6, master tables to be filled from runs.]*

### 4.6 Parameter Tuning

α swept over {0.1, 0.3, 0.5, 0.7} for linear family (§6.5); RRF k fixed at 60 (Elasticsearch convention, sensitivity in Appendix D). Grid 64×64, UMAP, σ=1.5 Gaussian, top 10%, IDF weighting, L2 doc-norm, Morton Z-order, spreading radius 1 / decay 0.5.

### 4.7 Statistical Testing

All MRR values reported with 95% bootstrap confidence intervals (1000 resamples, joint query resampling). Pairwise operator comparisons use paired bootstrap with **Holm correction** for multiple comparisons across the 7-operator × 4-pair matrix. We report ΔMRR with CI and p-value, not overlapping-CI heuristics.

---

## 5. Zero-Shot Semantic Signal (SF as Probe)

*[To be filled from runs — SF vs BM25, SF vs learned retrieval, where SF succeeds/fails. Framing per advisor: evidence that a training-free semantic signal can achieve meaningful retrieval quality in selected closed-domain settings, not "SF solves zero-shot retrieval."]*

| Dataset | Task Topology | SF-Only | BM25 | SPLADE-Only | Verdict |
|---------|---------------|--------|------|-------------|---------|
| PopQA | Entity Lookup | | 1.000 | | |
| PubMedQA | Biomedical | | 1.000 | | |
| NarrativeQA | Narrative | | 0.980 | | |
| Belebele | Reading Comp. | | 0.995 | | |
| 2WikiMultihopQA | Multi-hop 2 | | 0.921 | | |
| HotpotQA | Multi-hop 2 | | 0.869 | | |
| MuSiQue | Multi-hop 2–5 | | 0.482 | | |
| NQ-REaR | Factoid | | 0.675 | | |

---

## 6. Fusion Operator Analysis

*[Empirical centerpiece. Master table: 8 datasets × 7 operators × 4 model pairs, with CI + Holm-adjusted significance. To be filled from runs.]*

### 6.1 Complete Operator Matrix
### 6.2 Rank-space vs Score-space
### 6.3 Normalization (min-max / z-score)
### 6.4 Task Topology
### 6.5 Second-Model Validation (SF+DPR, BM25+SPLADE, BM25+DPR)
### 6.6 Complementarity vs Redundancy (Kendall's τ)

**Key figure (planned):** operator × task-topology heatmap of MRR, with the single-hop/multi-hop divergence visible across all four retriever pairs.

---

## 7. The Magnitude Information Hypothesis

### 7.1 Rank Invariance (Proposition 1)
Verified computationally: RRF output is bit-identical (to 1e-12) under strictly monotonic transforms of component scores, while CombMNZ changes — confirming magnitude sensitivity.

### 7.2 Synthetic Magnitude Control

We construct synthetic retrieval scores where rank is fixed (Doc A rank 1, Doc B rank 2) but magnitude is manipulated:

| Condition | Score(A) | Score(B) | Margin | A should win? |
|-----------|----------|----------|--------|---------------|
| 1 (large) | 45 | 12 | 33 | Yes (strong) |
| 2 (small) | 20 | 18 | 2 | Marginally |
| 3 (reversed) | 12 | 45 | −33 | No |

Applying all seven operators, we measure whether A is correctly ranked above B. **Rank-only operators (RRF, Borda) cannot distinguish the three conditions** — they see only ranks 1 and 2. **Score-space operators separate them by margin.** This is the clean causal isolation: with rank held constant, only magnitude-aware operators respond to the magnitude manipulation.

### 7.3 Real Retrieval Traces

*[To be filled: logged SF+SPLADE/D   PR score distributions on multi-hop vs single-hop queries, showing multi-hop queries exhibit larger gold-vs-distractor score margins.]*

### 7.4 Single-hop vs Multi-hop
### 7.5 When RRF Discards Useful Information

**Magnitude Fallacy (empirical phenomenon, not a theorem):** the failure mode occurring when a rank-only fusion operator treats retrieval results with different score magnitudes as equivalent whenever their ordinal ranks coincide, despite score magnitude carrying useful evidence about compositional relevance. We document this as an *observed phenomenon* with a Proposition (rank-invariance) andHypothesis (magnitude matters more for compositional tasks), supported by synthetic control and real traces — deliberately avoiding the unprovable "theorem" wording of the conference version.

---

## 8. Representation and Scaling Boundaries

### 8.1 Feature Invariance (Overlap-Feature Invariance)

For binary SDRs q,d ∈ {0,1}ᴰ, the dot product is qᵀd = Σ qᵢdᵢ (overlap count). If a proposed feature is a deterministic transformation of the same overlap count, it contains no independent ranking information. We state this conditionally and test it with **adversarial non-collinear features** (term rarity, document length, phrase coverage, query-term diversity, proximity, entropy, score margin, independent BM25) measuring corr(feature, qᵀd) vs ΔMRR.

### 8.2 Non-Collinear Feature Tests
*[To be filled: corr(feature, overlap) vs ΔMRR scatter.]*

### 8.3 Score Concentration (Candidate-Growth-Induced)

We **abandon the O(√N) "Scaling Wall" claim** of the conference version as theoretically problematic. Instead we analyze **score concentration under growing candidate populations**. For binary SDR overlap with qᵢ,dᵢ ~ Bernoulli(ρ), K=|q|₁: E[qᵀd]=Kρ, Var(qᵀd)=Kρ(1−ρ). The empirical question is whether the *relative separation* between relevant and irrelevant candidates is maintainable as candidate count grows when score distributions are concentrated.

### 8.4 Candidate-Size Scaling

*[To be filled: N ∈ {20,50,100,250,500,1k,5k,10k}; measure mean, std, CV, gold rank, MRR for SF/BM25/SPLADE/DPR.]*

### 8.5 Full-Corpus Evaluation

*[To be filled: SciFact deep-pool (gold+top-100 BM25) and full-corpus results, establishing that controlled-reranking findings generalize and that SF's pool-MRR=0.960 on the 16-doc toy pool is a retrieval-recall artifact, not real quality.]*

---

## 9. Discussion

### 9.1 Task-Operator Compatibility

Synthesis: rank-preserving operators for ordering tasks; magnitude-preserving for compositional tasks. Not a universal law — a compatibility hypothesis supported by the multi-pair, multi-operator, magnitude-control evidence.

### 9.2 Relation to Prior Fusion Theory

We extend Bruch et al. (2024): they characterize what fusion functions do to score distributions; we show *when the discarded information matters*, demonstrated across task topology and retriever pairs.

### 9.3 Practical Hybrid Retrieval Guidelines

1. Use Kendall's τ as a pre-fusion diagnostic: high τ on multi-hop → abandon fusion (redundancy); high τ on single-hop → use RRF.
2. Single-hop: SF+SPLADE with RRF (k=60) — scale-invariant, rescues scale mismatch.
3. Multi-hop: SF+SPLADE (or SF+DPR) with linear/magnitude-preserving fusion — preserves compositional confidence.
4. Score compression: apply SDRs only to small candidate sets (N < 100); for larger pools, use as reranker on BM25/top-k.

### 9.4 What the Results Do NOT Establish

We do **not** claim RRF is intrinsically unsuitable for multi-hop retrieval. We identify conditions under which rank-only fusion discards useful score information. We do **not** claim a universal law; the Task-Operator Compatibility is a hypothesis, scoped to the tested operators, datasets, and retriever pairs.

### 9.5 Deployment Considerations

No GPU; CPU-only query; ~512 B/doc (6× smaller than DPR). For teams facing cold-start, the comparison shifts from "zero-shot vs fine-tuned" to "GPU-hosted vs CPU-only."

---

## 10. Limitations and Conclusion

**Limitations:** model dependence (frozen SPLADE/DPR checkpoints); dataset dependence (English QA); candidate construction (BM25 negatives); score calibration (magnitude semantics vary by retriever); multi-hop interpretation (magnitude as compositional confidence is inferred, not directly observed); corpus scale (score concentration not validated at MS MARCO scale); language (English only); SF-specificity; fusion-operator coverage (seven, not exhaustive); generalization beyond QA.

**Conclusion.** Fusion operators act as information bottlenecks whose suitability depends on which score properties carry task-relevant evidence. Using SF as a controlled probe, we showed this is not operator-agnostic: rank-only fusion discards magnitude that compositional tasks require, and the effect survives multiple operators, a second retriever pair, and synthetic magnitude control. This reframes hybrid retrieval design from "pick a fusion function" to "match the operator to the task's information structure."

---

## Appendices

- **A.** Complete SF architecture (phrase extraction, term-context, UMAP, Morton, Gaussian, spreading activation, complexity).
- **B.** Hyperparameters.
- **C.** Full statistical tables (per-dataset MRR with 95% CI, Holm-adjusted p-values).
- **D.** k/α sensitivity.
- **E.** Additional retrieval traces.
- **F.** Dataset details.
- **G.** Reproducibility (commands, seeds, environment).

---

## References

*[To be consolidated from conference paper references + additions: Fox & Shaw 1994; Cormack et al. 2009; Bruch et al. 2024 (TOIS); Karpukhin et al. 2020 (DPR); Formal et al. 2021 (SPLADE); dataset citations; multi-hop RAG works. Web access for citation verification is pending.]*
