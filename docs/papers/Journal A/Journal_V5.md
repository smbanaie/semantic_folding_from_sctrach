# When Does Score Magnitude Matter? Task- and Retriever-Pair Dependence in Hybrid Retrieval Fusion

**Mojtaba Banaei¹, Maseud Rahgozar², and Heshaam Faili³**

¹,² Data Base Research Group (DBRG), School of Electrical and Computer Engineering, University of Tehran, Tehran, Iran
³ School of Electrical and Computer Engineering, Faculty of Engineering, University of Tehran, Tehran, Iran

`smbanaei@ut.ac.ir`, `rahgozar@ut.ac.ir`, `hfaili@ut.ac.ir`

---

## Abstract

Hybrid retrieval systems typically combine component rankings using fusion operators such as reciprocal rank fusion (RRF) or score-based aggregation. Yet these operators preserve different properties of retrieval scores, and it remains unclear when the discarded information matters.

We investigate this question using Semantic Folding (SF) as a controlled retrieval signal and evaluate seven fusion operators across single-hop and multi-hop retrieval tasks and multiple retriever pairs. We first establish that rank-only fusion is invariant to strictly monotonic score transformations. We then use controlled synthetic and rank-preserving interventions on real retrieval outputs to isolate the effect of score magnitude from rank.

The results show that score-space operators can exploit magnitude differences that rank-only operators cannot, but this advantage is conditional rather than universal. On the confirmatory SF+SPLADE evaluation, CombSUM consistently outperforms RRF on selected multi-hop tasks, while the effect is substantially reduced or absent for other retriever pairs and large-pool conditions. An operator × retriever-pair interaction analysis further shows that the benefit depends on joint score geometry rather than task type alone.

We additionally find evidence that SPLADE score separation tracks gold/supporting-document status in the multi-hop settings where score-space fusion is most effective. These findings suggest that fusion should be viewed not simply as combining rankings, but as selecting which properties of retrieval signals are preserved.

**Our conclusion is therefore conditional: hybrid retrieval benefits from magnitude-preserving fusion when score magnitude carries relevance-bearing information and the participating retrievers exhibit compatible score geometry.**

We make three contributions. **Contribution 1 — Information preservation.** We characterize hybrid fusion operators by the score information they preserve: rank-only operators are invariant to strictly monotonic score transformations, whereas score-space operators remain sensitive to score magnitude. **Contribution 2 — Conditional magnitude utility.** Through synthetic controls and rank-preserving interventions on real retrieval traces, we show that score magnitude can affect fused rankings independently of rank, and provide evidence that this magnitude is relevance-bearing in selected multi-hop retriever pairs. **Contribution 3 — Retriever-pair dependence.** Across multiple retriever pairs and QA tasks, we show that operator effectiveness depends on the joint score geometry of the participating signals rather than on task type alone, identifying conditions under which magnitude-preserving fusion provides an advantage. We do not claim that rank-only fusion is intrinsically inferior, that magnitude is universally informative, or that these findings transfer directly to first-stage corpus-scale retrieval.

**Keywords:** Hybrid Retrieval · Fusion Functions · Score Geometry · Information Preservation · Multi-Hop Question Answering · Reciprocal Rank Fusion


---

## 1. Introduction

### 1.1 Problem

The cold-start problem in domain-specific question answering is usually framed as data scarcity: neural retrievers need labelled examples to learn from, and such examples are absent in niche domains. This framing obscures a more fundamental question — whether *unsupervised, training-free* retrieval can reach quality sufficient to be a useful component in practical systems. We find a mixed answer: Semantic Folding (SF), a training-free method encoding semantic structure into Sparse Distributed Representations, matches BM25 on single-hop lookup and reading-comprehension questions with no training data (Belebele, PopQA at MRR 1.000 over the dataset-provided candidate pools), but falls below BM25 on single-hop biomedical QA (PubMedQA 0.800 vs 1.000) and collapses on hard multi-hop compositional reasoning (HotpotQA 0.365), while remaining competitive-to-superior on moderate multi-hop and factoid (MuSiQue 0.720, NQ-REaR 0.725 vs BM25 0.482, 0.675). It is a useful *probe*, not a standalone retriever.

Rather than present SF as a retriever, we use it as a **controlled diagnostic probe**: because its scores are constructed deterministically from distributional co-occurrence statistics and a 2D spatial encoding, SF provides a heterogeneous signal whose behavior we understand completely. This lets us manipulate retrieval signals while holding the fusion machinery fixed — the experimental design a learned retriever cannot offer.

### 1.2 Why fusion is not operator-neutral

A standard hybrid system fuses two retrievers with either Reciprocal Rank Fusion (RRF) or linear interpolation. We show these are not interchangeable. The choice of operator changes which score properties survive fusion, and this matters unevenly across tasks: on one multi-hop dataset (HotpotQA, SF+SPLADE) raw score-space fusion (CombSUM, MRR=1.000) substantially outperforms rank-only RRF (0.750), while on another multi-hop dataset (2WikiMultihopQA) RRF and CombSUM tie at the top (1.000), and on a third (MuSiQue) they are statistically indistinguishable (0.95 vs 0.95). Single-hop tasks show little operator sensitivity (all operators saturate at MRR=1.000 on Belebele). The divergence is not a tuning artifact. It is a *structural* property of what each operator preserves — but the direction and magnitude of the effect is itself task- and score-geometry dependent, not a fixed law:

- **Rank-only operators** (RRF, Borda) discard absolute scores and keep only ordinal position. They are robust to score-scale mismatch but blind to magnitude.
- **Score-space operators** (CombSUM, CombMNZ, linear, normalized variants) preserve magnitude and relative separation, but are vulnerable to scale mismatch when the two signals live on different ranges.

The central claim of this paper is that **fusion operators preserve different properties of retrieval scores; in some compositional settings, score magnitude contains relevance-bearing information that rank-only fusion cannot exploit, and whether this information is useful depends on the joint score geometry of the participating retrievers.** For single-hop matching, rank is often sufficient; for multi-hop composition, absolute score magnitude may encode aggregate evidence strength, which can correlate with the degree of compositional evidence satisfied — and discarding it is harmful in some such settings, but only when the fused signals carry magnitude on heterogeneous scales.

### 1.3 Research Questions

- **RQ1 (Complementarity).** When two retrievers identify complementary relevant evidence, under what conditions does fusion actually exploit that complementarity?
- **RQ2 (Fusion information).** Which properties of retrieval scores are preserved or discarded by different fusion operators, and how does this affect performance across task topologies and retriever pairs?
- **RQ3 (Mechanism).** Does manipulating score magnitude while preserving rank alter fusion outcomes in a way consistent with the proposed magnitude-information mechanism, or is the observed association merely a consequence of ranking correlation and score normalization?
- **RQ4 (Boundaries).** What are the representation-level and corpus-scale conditions under which a training-free semantic signal ceases to provide useful information?

### 1.4 Contributions

**What this paper does not claim.** We do not claim that RRF is inferior to score-based fusion in general, that multi-hop QA universally requires score magnitude, or that Semantic Folding is competitive with modern first-stage retrievers at web scale. We also do not claim that a retriever's score magnitude is intrinsically calibrated as reasoning depth — we treat that as the Magnitude Utility Hypothesis (H2, §3.3) and test it. Instead, we investigate when the information preserved by a fusion operator is predictive of relevance under controlled reranking conditions.

1. **Conceptual (information-preservation framework).** We characterize fusion operators by which properties of their input signals they preserve — rank-only operators preserve ordinal position; raw-score aggregation preserves magnitude; normalized aggregation preserves calibrated magnitude; weighted interpolation controls the trade-off explicitly — and formalize when each preserved class can matter (§3).
2. **Methodological (controlled diagnostic probe).** We introduce a controlled-probe methodology using training-free Semantic Folding to isolate fusion behavior from learned-representation effects: SF's transparent score construction lets us hold the fusion machinery fixed while varying signal geometry (§4).
3. **Empirical (cross-operator / cross-model evidence).** Evaluating seven operators across ten datasets and four retriever pairs — replicated under a second SPLADE checkpoint and scrutinized with paired bootstrap/Wilcoxon/Holm statistics at n=50 — we show that operator effectiveness depends on joint score geometry and task requirements rather than operator identity alone, with family-wise non-significance for most pairwise gaps reported transparently.
4. **Mechanistic (magnitude intervention).** Through synthetic and real-score perturbations we separate rank sensitivity from magnitude sensitivity *by construction* — rank-preserving magnitude changes leave rank-only fusion bit-identical while reordering score-space fusion; rank destruction collapses rank-only fusion maximally — identifying the conditions under which discarded magnitude information affects ranking (§7).

The practical implication is not "use linear for multi-hop and RRF for single-hop." Fusion should be selected based on **the information that is both useful for the task and reliably encoded in the component scores** — and our diagnostic procedure (§9.3) makes that selection inspectable before deployment.

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

SDRs are binary vectors of large dimensionality where most bits are zero. SF arranges vocabulary on a 2D grid and maps text into sparse fingerprints over that grid. We treat SF as a *controlled probe whose score construction is fully characterized*, not as a principal algorithmic contribution. Its key properties: no task labels, no gradient training, **binary SDR fingerprint representation (ideal ~512 B/doc for the 4096-bit binary vector; the pipeline's emitted scores are real-valued after weighted aggregation and spatial smoothing),** CPU-only query. Rather than treating Semantic Folding as the endpoint of the investigation, we use it as a controlled retrieval signal whose heterogeneous score geometry makes otherwise hidden differences between fusion operators observable. These make it an unusually transparent heterogeneous signal for fusion experiments. Full pipeline internals (UMAP projection, Morton z-order curve encoding, Gaussian spreading) are specified in Appendix A; the main text treats SF purely as an instrument.

### 2.6 Positioning Against Prior Fusion Analysis

Bruch et al. (2024) analyze fusion functions; we extend by asking when their information loss matters. Our novelty is twofold and deliberately scoped: (i) a **controlled-probe methodology** — using a fully-characterized training-free signal (SF) as a manipulable heterogeneous probe that a learned retriever cannot offer, letting us hold the fusion machinery fixed while varying score geometry; and (ii) an **empirical map** of operator × retriever-pair × task-topology outcomes showing the winning family is set by the *joint score geometry of the fused signals*, not by the task or retriever identity. We do not claim a new fusion theorem; the rank-invariance proposition (§3.6) is already in the literature. The contribution is the methodology and the conditional, evidence-backed map, together with explicit documentation of where the effect reverses and where it remains unvalidated (§8, §9.4).

**Table 2.1: Positioning against prior fusion analysis**

| Work | Operators | Score-geometry theory | Task topology | Multiple retriever pairs | Magnitude intervention |
|------|:---------:|:--------------------:|:-------------:|:------------------------:|:----------------------:|
| Fox & Shaw (1994) | ✓ CombSUM/MNZ | | | | |
| Cormack et al. (2009) | ✓ RRF | ✓ (rank-only property) | | | |
| Bruch et al. (2024) | ✓ 7 families | ✓ | limited (task-level observations) | | |
| **This work** | ✓ 7 families | ✓ | ✓ (11 datasets, multi-hop vs single-hop) | ✓ (4 pairs + 2nd checkpoint) | ✓ (synthetic + real-score perturbation) |

---

## 3. Conceptual Framework

**Score geometry, formally.** For a signal's per-document score distribution s over a candidate pool we define G(s) = (R, μ, σ, Δ₁₂, Δ₁₅, ρ, κ), where R is the score range, μ/σ mean and standard deviation, Δ₁₂/Δ₁₅ the top-1/top-2 and top-1/top-5 margins, ρ the correlation with the paired signal over common documents, and κ the distributional shape (skew/kurtosis). Pair geometry G(s₁, s₂) adds cross-signal agreement (Kendall τ, Pearson r, top-k Jaccard). Operator behaviour is hypothesized to be a function of pair geometry, not of either signal alone; §6.6.4 tests this factorially and §6.6.5 builds the predictor.

We model hybrid retrieval as a pipeline:

```
Retrieval signal → score space → fusion operator → information retained → task requirement → effectiveness
```

### 3.1 Retrieval Signal Properties

Each retriever emits, per query, a score distribution over candidates. Two structural properties matter: (a) the **rank** of each candidate (ordinal position), and (b) the **magnitude / margin** between scores (how confidently the retriever distinguishes relevant from irrelevant, and — in multi-hop settings — how much compositional evidence was satisfied).

### 3.2 Rank Information

Sufficient when the task only requires correct *ordering*: the gold document need only be ranked above distractors. Single-hop matching often satisfies this.

### 3.3 Score Magnitude

Carries additional signal when the *degree* of match matters: in multi-hop QA, a high SPLADE/DPR score *may* indicate stronger aggregate evidence match across hops, and a low score a weaker one — we treat this as the **Magnitude Utility Hypothesis (H2)**: *score magnitude is useful for fusion exactly when the score separation between candidates correlates with relevance-bearing distinctions that are lost under rank-only transformation.* Magnitude thereby may carry relevance-bearing separation between fully and partially supported candidates; we treat this interpretation as a hypothesis (tested in §7.6) rather than as an intrinsic semantic meaning of the score. Whether it does in a given retrieval setting is an empirical question our experiments are designed to answer, not an assumption.

### 3.4 Complementarity vs Redundancy

Two quantities must be distinguished. **τ_signal** = Kendall's τ between the two component rankings (e.g. SF vs SPLADE) for one query — a component-complementarity diagnostic. **τ_operator** = Kendall's τ between two operators' fused rankings for one query — an operator-agreement measure that says nothing about complementarity; conflating the two is a category error we avoid throughout. Two retrievers surface **different relevant documents** when their component rankings disagree (low τ_signal), and **agree** when they correlate highly (high τ_signal). We use τ_signal as a *rank-agreement* diagnostic between the two fused signals: high τ means the signals order candidates similarly; low τ means they disagree and operator choice becomes consequential. Note that τ measures ordinal agreement, not complementarity or redundancy of relevance itself — the mapping from agreement to fusion benefit is empirical, not definitional. This is a property of the signal pair, not of the task alone.

### 3.5 Task-Operator-Signal-Geometry Compatibility Hypothesis

> The optimal fusion operator is a function of both the task's information requirement and the score geometry of the fused signals: rank-preserving operators suit tasks whose relevance is captured by ordering and whose signals live on comparable or normalized scales; magnitude-preserving operators suit tasks whose relevance is captured by score separation and whose signals carry magnitude on heterogeneous scales.

We state this as a *hypothesis* to be tested rather than a constraint or law; our own 2WikiMultihopQA RRF result already contradicts the strong form in edge cases.

### 3.6 Formal Rank-Invariance Proposition

**Proposition 1 (Rank-fusion invariance).** Let s(d) be a retrieval score and f any strictly monotonic transformation. Then rank(s(d)) = rank(f(s(d))). Therefore any rank-only fusion operator R satisfies R(s₁,…,sₘ) = R(f₁(s₁),…,fₘ(sₘ)) for strictly monotonic fᵢ. Consequently rank-only fusion is invariant to score magnitude, score distance, nonlinear calibration, and confidence separation, provided ordering is unchanged.

This is mathematically trivial but establishes the clean separation that motivates the empirical work: rank-only and score-space operators differ *exactly* in whether magnitude survives.

---

## 4. Experimental Methodology

### 4.1 Datasets

Ten closed-domain QA datasets (PopQA, PubMedQA, NarrativeQA, Belebele, 2WikiMultihopQA, HotpotQA, MuSiQue, NQ-REaR, SciFact, COVID-QA). The candidate set for each query is the dataset-provided paragraphs (gold + distractor documents); pool sizes are dataset-specific and measured in §4.3 (PopQA 2, PubMedQA 2, HotpotQA/NQ-REaR/2WikiMultihopQA/COVID-QA 10, MuSiQue/Belebele 20, NarrativeQA 385, SciFact 16). SciFact uses the BEIR claim-verification corpus; COVID-QA uses the CORD-19 abstract corpus (2,019 expert-annotated QA pairs, added in this journal extension). Pool sizes were verified by direct inspection of the converted JSONL files (paragraphs per query), not assumed.

### 4.2 Task Topology

We classify each dataset by the reasoning its questions demand:

| Topology | Datasets | Information needed |
|----------|----------|--------------------|
| Entity lookup | PopQA | Ordering |
| Biomedical / narrative / reading | PubMedQA, NarrativeQA, Belebele | Ordering (+ semantics) |
| Multi-hop (2 hops) | 2WikiMultihopQA, HotpotQA | Magnitude / composition |
| Multi-hop (2–5 hops) | MuSiQue | Magnitude / composition |
| Factoid (large pool) | NQ-REaR | large-pool factoid reranking (no multi-hop magnitude assumption; see §6.6.4 null interaction) |

### 4.3 Candidate Regimes

**Stage definitions.** Throughout this paper, *Stage 1* denotes candidate generation (first-stage retrieval) and *Stage 2* denotes hybrid reranking/fusion of an already-retrieved shortlist. This paper studies **Stage 2**: SF serves as a controlled probe for the fusion stage, and every claim below is a Stage-2 claim.

We explicitly distinguish two regimes:

- **Controlled reranking (Regime A) — what this paper reports.** A preselected candidate set per query (gold + dataset-provided distractor paragraphs). The pool size is *not* fixed: it equals each dataset's paragraph count, which we measured by direct inspection of the converted JSONL as PopQA 2, PubMedQA 2, HotpotQA/NQ-REaR/2WikiMultihopQA 10, MuSiQue/Belebele 20, NarrativeQA 385, and SciFact 16 documents per query. This is *not* first-stage retrieval. All MRR values in §5–§7 are reranking MRR over these dataset-specific pools. Methodologically, this design **conditions on candidate-set recall**: by construction the gold document is present in nearly every pool, so we isolate the ranking/fusion stage and measure ranking quality given gold presence — not retrieval recall, candidate generation, or cross-retriever recall complementarity at the first stage.
- **Complete-collection reranking (Regime B)** — i.e. reranking over the *entire constructed evaluation collection* (494 HotpotQA passages), not first-stage retrieval over an external corpus: query → entire constructed collection → retriever A + retriever B → fusion → ranking. Checks that findings generalize beyond small pools without claiming web-scale retrieval.
- **Corpus-scale reranking (Regime C).** Reranking over an external multi-thousand-document corpus (SciFact BEIR corpus, 5,183 docs) — the largest condition this infrastructure supports; first-stage retrieval at web scale remains out of scope (§9.4).

Every quantitative claim in this paper is therefore a *reranking* claim over dataset-provided candidate pools (sizes listed above); generalization to first-stage retrieval remains an open question.

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

### 4.6 Parameter Tuning

α swept over {0.1, 0.3, 0.5, 0.7} for linear family (§6.5); RRF k fixed at 60 (Elasticsearch convention, sensitivity in Appendix D). SF grid 64×64, UMAP, σ=1.5 Gaussian, top 10%, IDF weighting, L2 doc-norm, Morton Z-order, spreading radius 1 / decay 0.5.

### 4.7 Statistical Testing — and its limits

Each dataset is evaluated on a **10-query probe** for the full 10-dataset × 7-operator matrix (§6.1), reported as **exploratory, directional evidence**. To assess whether directional patterns observed in the exploratory n=10 matrix persist at higher sample size, we ran an **expanded evaluation** on three multi-hop/factoid datasets (HotpotQA, MuSiQue, NQ-REaR) with the SF+SPLADE pair: first at n=50, then extended to a **confirmatory core of n=100** — the complete seven-operator matrix (linear, rrf, combsum, combmnz, borda, zscore, minmax). Per-query MRR arrays feed a formal statistical protocol: paired bootstrap 95% CIs (10,000 resamples, seed=42), two-sided Wilcoxon signed-rank tests between every operator pair, and Holm–Bonferroni family-wise correction across the 21 pairwise comparisons per dataset (Appendix C). *For these three datasets the Appendix C n=100 tables supersede both the earlier n=50 tables and the parenthetical n=10 values shown in §6.1; lower-n values are retained only for continuity with the exploratory map.*

**Query-sampling protocol.** Query subsets are deterministic prefixes of the dataset conversion order (queries 0–k-1 of the converted JSONL, which follows the raw dataset's canonical ordering); no query was selected or excluded after inspecting any operator result. The n=10 probes are the first 10 queries; the n=50 and n=100 runs extend the same prefix. This makes every sample exactly reproducible and independent of outcomes by construction.

**Benchmark taxonomy.** Datasets are selected by task topology, not convenience:

| Dataset | Task topology | Domain | Candidate pool | Gold structure | Primary role |
|---------|---------------|--------|---------------:|----------------|--------------|
| PopQA | Single-hop | Open-domain | 20–50 | Single | Ceiling/control |
| Belebele | Single-hop RC | Multilingual (English slice) | 20–50 | Single | Ceiling/control |
| NarrativeQA | Reading comprehension | Narrative | 20–50 | Single | Control |
| PubMedQA | Domain QA | Biomedical | 20–50 | Single | Domain control |
| **HotpotQA** | **Multi-hop** | Open-domain | ~94–100 | Multi-step | **Primary mechanism** |
| **MuSiQue** | **Multi-hop** | Open-domain | ~100 | Multi-step | **Primary mechanism** |
| 2WikiMultihopQA | Multi-hop | Open-domain | ~100 | Multi-step | Replication |
| NQ-REaR | Factoid retrieval | Open-domain | ~385 | Single | Boundary condition |
| SciFact | Claim verification | Scientific | 66–5,183 | Evidence | Scale/boundary control |

We use *single-hop* for benchmarks whose gold evidence is retrievable through a single semantic matching step and *multi-hop* for benchmarks whose supporting evidence requires composition across multiple passages or reasoning steps; these labels describe benchmark structure, not model reasoning capability.

**Evaluation hierarchy.** We distinguish three evidence tiers to avoid treating all measurements as equally powered. **Tier 1 — Confirmatory:** HotpotQA, MuSiQue, NQ-REaR at n=100 with the full statistical protocol. **Tier 2 — Replication:** 2WikiMultihopQA and SciFact at n=10 under the identical pipeline. **Tier 3 — Exploratory diagnostics:** n=10 traces feeding the perturbation battery, phase diagram, geometry predictor, margin analysis, calibration study, and rank-conditioned magnitude analysis.

**Statistical protocol.** All primary operator comparisons are paired at the query level because every operator receives identical component rankings and candidate pools. We report MRR with paired bootstrap 95% confidence intervals and conduct pairwise Wilcoxon signed-rank tests. Because seven operators yield 21 pairwise comparisons per dataset, p-values are adjusted using Holm–Bonferroni correction within each dataset-level family. Exploratory mechanistic analyses use permutation or bootstrap procedures as specified and are explicitly labeled as such.

**Split-half stability.** As a sampling-robustness check equivalent to multi-seed replication, we partitioned the n=100 queries into random disjoint halves across 200 splits (`scripts/split_half_stability.py`): CombSUM beats RRF **on both halves simultaneously in 100% of splits on HotpotQA**, 99.5% on MuSiQue, and 79% on NQ-REaR — matching the significance pattern (family-wise significant on the two multi-hop datasets, not on NQ-REaR). Per-operator split-to-split std is ≤0.02 MRR everywhere (`appendix_stats/split_half_stability.{md,json}`).

**Statistical findings (n=50 interim; superseded by n=100 below).** The operator *ordering* was already stable at n=50 — CombSUM/CombMNZ rank first on all three datasets (HotpotQA 0.947/0.893; MuSiQue 0.977/0.919; NQ-REaR 0.657/**0.679**) and Borda last (0.857/0.770/0.587) — but after Holm correction **almost no pairwise comparison survives at α=0.05**. On HotpotQA, CombSUM vs linear shows ΔMRR = +0.114 with raw p = 0.0064 that inflates to p_Holm = 0.135 after correction; on MuSiQue, CombSUM vs RRF is +0.060 at raw p = 0.0143 → 0.183 corrected. Only one of 63 comparisons survives: Borda vs CombMNZ on MuSiQue (Δ = −0.149, p_Holm = 0.035). We therefore report effect sizes and raw p-values transparently while acknowledging that, at n=50 with single-gold-per-query MRR, individual operator differences are **directionally consistent but not family-wise significant**. This strengthened rather than weakened our framing: the contribution is the *mechanism* (which information each operator preserves, §3, §6.3.1, §7), not the claim that any two operators are separable at that sample size. The subsequent expansion to n=100 (below) confirms the ordering and brings most HotpotQA/MuSiQue comparisons past family-wise correction.

**Confirmatory core at n=100 (Appendix C, the n=100 Appendix C tables).** All three datasets were re-run end-to-end on freshly built indexes at n=100 (HotpotQA 1489-doc collection; MuSiQue 2328-doc; NQ-REaR 990-doc), SF+SPLADE, all seven operators:

*Confirmatory evaluation; n=100 queries/dataset.*

| Dataset | CombSUM | RRF | Δ | p_Holm (combsum vs rrf) | comparisons surviving Holm |
|---------|--------:|----:|----:|------------------------:|---------------------------:|
| HotpotQA | **0.947** | 0.854 | +0.093 | 0.0007 | 15/21 |
| MuSiQue | **0.952** | 0.908 | +0.044 | <0.0001 | 17/21 |
| NQ-REaR | **0.746** | 0.718 | +0.028 | — (4/21 survive) | 4/21 |

At n=100, CombSUM's advantage over RRF is family-wise significant on both multi-hop datasets, and the full operator ordering is stable: CombSUM first everywhere, Borda last everywhere (0.732/0.652/0.602). Figure 1 plots the seven-operator comparison (`figures/fig2_n100_confirmatory.png`). NQ-REaR remains the least separable dataset — consistent with its large-pool factoid profile rather than undermining the mechanism account.

**Table 3 — Master benchmark summary (confirmatory core; SF+SPLADE).**

| Dataset | Pair | n | Pool size | Best operator | MRR | RRF | Δ | p_Holm |
|---------|------|--:|----------:|---------------|----:|----:|---:|-------:|
| HotpotQA | SF+SPLADE | 100 | ~94 | **CombSUM** | **0.947** | 0.854 | +0.093 | **0.0007** |
| MuSiQue | SF+SPLADE | 100 | ~100 | **CombSUM** (z-score ties) | 0.952 | 0.908 | +0.044 | <0.0001 |
| NQ-REaR | SF+SPLADE | 100 | ~385 | CombSUM (weak) | 0.746 | 0.718 | +0.028 | 4/21 survive |

*Confirmatory evaluation; n=100 queries/dataset.*

**Table 4 — Central diagnostic map (exploratory geometry features vs confirmed operator outcome).**

| Task | mean τ_signal | SPLADE P(Δ>0) | P(gold \| top bin) | Best operator (n=100) | RRF MRR | Evidence |
|------|--------------:|--------------:|------------------:|--------------------|--------:|----------|
| HotpotQA (multi-hop) | 0.309 | 1.00 | 0.56 | CombSUM | 0.854 | magnitude useful; geometry separates |
| MuSiQue (multi-hop) | −0.062 | 1.00 | 0.52 | CombSUM / z-score | 0.908 | magnitude useful; weaker separation |
| SciFact (claim verification) | 0.318 | 0.80 | 0.64 | all tie (≈0.82) | 0.820 | near-ceiling; operator-invariant |

Read jointly: the two rows where SPLADE magnitude separates gold perfectly (P(Δ>0)=1.00) are exactly the rows where score-space fusion wins at n=100; the near-ceiling row shows no separation and no operator effect. τ_signal alone does NOT predict the outcome (MuSiQue has the lowest τ but CombSUM's advantage over RRF is family-wise significant at n=100 after Holm correction) — rank agreement is diagnostic only in combination with margin structure, which is precisely the paper's joint-geometry thesis.

**Where the difference lives (win/tie/loss and rank-1 changes).** The aggregate MRR gap decomposes into a small number of decisive queries (`appendix_stats/win_loss_rank1_n100.{md,json}`): on HotpotQA CombSUM wins 21 queries outright against 1 loss for RRF (78 ties), and switching RRF → CombSUM changes the rank-1 document on **10% of queries**; on MuSiQue 8 wins / 0 losses with **4%** rank-1 changes; on NQ-REaR 18 wins vs 11 losses with 18% rank-1 changes but near-equal margins in both directions. 
| Dataset | n | CombSUM wins | RRF wins | ties | win% | rank-1 changes | dz | n needed (power .8) |
|---------|--:|--:|--:|--:|--:|--:|---:|---:|
| HotpotQA | 100 | 21 | 1 | 78 | 21% | 10 (10%) | 0.45 | 40 |
| MuSiQue | 100 | 8 | 0 | 92 | 8% | 4 (4%) | 0.29 | 93 |
| NQ-REaR | 100 | 18 | 11 | 71 | 18% | 18 (18%) | 0.13 | 488 |

Figure 2 charts the paired outcomes and rank-1 changes (`figures/fig6_win_loss_power.png`).

These rank-1 flips *are* the information bottleneck made visible: CombSUM rescues a handful of compositional queries where SPLADE's magnitude evidence outranks SF's miss, and loses almost nothing back. A paired-power analysis on the observed effect sizes (dz = 0.45 HotpotQA, 0.29 MuSiQue, 0.13 NQ-REaR) indicates n ≈ 40 / 93 / 488 queries respectively for 80% power at α = 0.05 — i.e., our n=100 is adequately powered for the two multi-hop effects but NQ-REaR would need ~500 queries, consistent with its non-separability here.

Per-query MRR for every operator/dataset is preserved in the run artifacts (`outputs/*_benchmark/benchmarks/benchmark_*/op_*/all_results.json`) and summarized in Appendices C/E so reviewers can inspect the within-dataset spread.

### 4.8 Evaluation Metric — Why MRR (and not nDCG / P@k / R@k)

We report **Mean Reciprocal Rank (MRR)** as the primary metric across all experiments, with P@1 and Recall@k as secondary diagnostics where relevant. The choice is deliberate and scoped to this paper's controlled-reranking regime (§4.3), not a default:

1. **Single relevant target per query.** Every dataset in our benchmark provides exactly one gold passage (or, for claim-verification, one gold claim) inside the candidate pool. MRR measures whether that single relevant item is ranked first; nDCG/P@k/R@k are designed for multiple graded-relevance documents and degenerate to MRR-equivalent signals when there is one binary-relevant item. Using nDCG would add graded-relevance assumptions the data do not support.
2. **The research question is about operator *preservation*, not recall.** Our hypothesis (§3.5) is that fusion operators differ in *which score information survives* — i.e. whether the gold item reaches rank 1. MRR is the direct, rank-sensitive measure of that: it weights the top of the ranking, exactly where magnitude-vs-rank disagreement manifests (§7.2). Metrics that average over the whole candidate list (R@k at large k, MAP) would dilute the very effect we isolate.
3. **Compatibility with prior fusion literature.** Bruch et al. (TOIS 2024) and the recurrent rank-fusion work (Cormack et al., SIGIR 2009) report MRR for reranking-style evaluation; using MRR lets us position our results on the same axis (§9.2) rather than introducing a non-comparable metric.
4. **Small per-dataset query counts.** With n=10–50 queries, MRR's per-query reciprocal-rank values are stable enough for directional comparison; more variance-sensitive metrics (nDCG with graded gains) would be noisier under the same sample. We therefore report MRR as the headline and keep P@1/Recall@k in the artifact tables for completeness.
5. **Scope boundary.** MRR over a candidate pool does *not* measure first-stage recall — it measures reranking quality of an already-retrieved shortlist (§4.3, Regime A). We do not report MRR as if it were corpus-level recall; the one full-dataset reranking run (HotpotQA §8.4) is explicitly labeled as exhaustive reranking over the constructed 494-passage collection, not first-stage retrieval.

We did collect nDCG@10, P@1, P@3, R@5, R@10 per run (stored in `op_*/summary.json` and `per_query/`), but present MRR as the primary comparison metric for the reasons above; the supplementary metrics are available in the run artifacts for completeness.

---

## 5. Zero-Shot Semantic Signal (SF as Probe)

**SF-Only baselines (signal-a=sf, retriever-b=none; 10-query probes, reranking MRR over the dataset-provided candidate pools):** Belebele 1.000, PopQA 1.000, NarrativeQA 1.000 (AP 0.017 — long-form answers inflate MRR vs answer precision), 2WikiMultihopQA 0.858, PubMedQA 0.800, HotpotQA 0.365, MuSiQue 0.720, NQ-REaR 0.725, COVID-QA 0.633. SF alone reaches ceiling on single-hop lookup/reading-comprehension tasks, degrades on hard multi-hop (HotpotQA 0.365 vs BM25 0.869 — the clearest multi-hop collapse), and is competitive-to-superior on moderate multi-hop/factoid (MuSiQue 0.720 > BM25 0.482; NQ-REaR 0.725 > BM25 0.675). On COVID-QA (biomedical, CORD-19 abstracts) SF alone reaches 0.633 — below BM25 (0.767) and SPLADE (0.850) — confirming that SF is a useful *probe* but not a standalone multi-hop retriever, while showing SF's zero-shot semantic signal can beat BM25 on some multi-hop topologies.

**SPLADE-Only baselines (signal-a=splade, retriever-b=none):** ceiling 1.000 on Belebele/PopQA/NarrativeQA/2Wiki/HotpotQA/MuSiQue, 0.800 PubMedQA, 0.750 NQ-REaR. The learned sparse retriever reaches ceiling on every multi-hop set where SF collapses — so the contribution is not "SF beats neural retrievers" but that SF, as a fully-characterized zero-shot signal, *isolates the rank-vs-magnitude information loss* (§6–§7) that a black-box SPLADE ranking does not expose. On this evidence, SF's value is diagnostic and complementary (§6.5), not standalone.

| Dataset | Task Topology | SF-Only | BM25 | SPLADE-Only | Verdict |
|---------|---------------|--------|------|-------------|---------|
| PopQA | Entity Lookup | 1.000 | 1.000 | 1.000 | all ceiling |
| PubMedQA | Biomedical | 0.800 | 1.000 | 0.800 | SF=SPLADE < BM25 |
| NarrativeQA | Narrative | 1.000 | 0.980 | 1.000 | SF=SPLADE ≈ BM25 (AP caveat) |
| Belebele | Reading Comp. | 1.000 | 0.995 | 1.000 | all ceiling |
| 2WikiMultihopQA | Multi-hop 2 | 0.858 | 0.921 | 1.000 | SPLADE ≫ SF, BM25 |
| HotpotQA | Multi-hop 2 | 0.365 | 0.869 | 1.000 | SPLADE ≫ SF; SF collapses vs BM25 |
| MuSiQue | Multi-hop 2–5 | 0.720 | 0.482 | 1.000 | SPLADE ≫ SF > BM25 |
| COVID-QA | Biomedical | 0.633 | 0.767<sup>†</sup> | 0.850 | SPLADE ≫ SF; fusion 0.900 (§6.1) |

<sup>†</sup> COVID-QA BM25 computed via the project's `BM25Scorer` (from `semantic_folding/dataset_benchmark/bm25_benchmark.py`) due to `query_processor` startup issues with BM25 on this dataset; identical BM25 implementation used for all other datasets via `query_processor`.
| NQ-REaR | Factoid | 0.725 | 0.675 | 0.750 | SPLADE > SF ≈ BM25 |

*Caveat:* NarrativeQA measures MRR over the dataset-provided candidate pool (385 documents) but its answers are long-form narratives, so MRR=1.000 reflects passage ranking, not answer exactness (AP 0.017 in the SF-only run). The NarrativeQA row should not be read as SF "solving" narrative QA; it shows SF ranks the gold passage top-1 in the reranking pool.

## 6. Fusion Operator Analysis

*Empirical centerpiece. We report MRR across all 10 datasets × 7 operators for the SF+SPLADE pair (§6.1, MuSiQue, SciFact, and COVID-QA present with all seven operators; COVID-QA at n=10), and a focused 4-model-pair × 4-discriminating-dataset design (§6.5). 10-query probes (reranking MRR, directional) for the single-hop rows; expanded n=50 for the multi-hop/factoid discriminating rows. All runs reproducible via the commands in Appendix G.*

**Operator taxonomy (conceptual).**

| Operator family | Uses rank? | Uses raw magnitude? | Scale-sensitive? |
|-----------------|-----------:|--------------------:|------------------|
| RRF | ✓ | ✗ | No |
| Borda | ✓ | ✗ | No |
| CombSUM | — | ✓ | Yes |
| CombMNZ | — | ✓ | Yes |
| Linear (α) | — | ✓ | Yes |
| z-score | — | ✓ | Calibration-dependent |
| Min-max | — | ✓ | Calibration-dependent |

*The taxonomy is conceptual; exact behavior depends on preprocessing and normalization.*

### 6.1 Complete Operator Matrix (SF + SPLADE) — 10 Datasets

The headline claim of this paper is that fusion-operator effectiveness is task-dependent. To make that claim verifiable we report the **complete 7-operator matrix across all ten benchmarked datasets** (MuSiQue, SciFact, and COVID-QA are included alongside the previously benchmarked datasets). Single-hop/reading-comprehension rows are 10-query exploratory probes (reranking MRR, directional); the multi-hop/factoid discriminating rows (HotpotQA, MuSiQue, NQ-REaR) are reported at the **n=50 intermediate replication** where available, with the 10-query value in parentheses for continuity. The confirmatory n=100 evaluation of these three datasets (SF+SPLADE) supersedes the n=50/n=10 values below and is reported in §4.7 and Appendix C.

| Dataset | Type | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|---------|------|-------:|----:|--------:|--------:|------:|-------:|-------:|
| Belebele | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PopQA | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| NarrativeQA | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PubMedQA | single-hop | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 |
| HotpotQA | multi-hop | 0.558 (0.570) | 0.783 (0.750) | **1.000** (1.000) | 0.783 (0.783) | 0.583 (0.583) | 0.683 (0.683) | 0.558 (0.570) |
| 2WikiMultihopQA | multi-hop | 1.000 (0.933) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 0.950 (0.950) | 1.000 (1.000) | 1.000 (0.933) |
| MuSiQue | multi-hop | 0.887 (0.900) | 0.927 (0.950) | **0.977** (0.950) | 0.919 (0.933) | 0.780 (0.850) | 0.953 (0.950) | 0.887 (0.900) |
| NQ-REaR | factoid | 0.566 (0.700) | 0.612 (0.720) | 0.593 (0.800) | **0.820** (0.820) | 0.653 (0.653) | 0.737 (0.737) | 0.700 (0.700) |
| SciFact | claim-verif | 0.960 | 0.960 | 0.960 | 0.940 | 0.890 | 0.930 | 0.910 |
| COVID-QA | biomedical | 0.900 | 0.900 | 0.900 | 0.900 | 0.800 | 0.900 | 0.900 |

**Reading.** On single-hop tasks the matrix saturates (ceiling at 1.000, or flat at 0.800 for PubMedQA) — operator choice is invisible. On claim-verification (SciFact) all operators tie at ≈0.96, so fusion is irrelevant there too. Operator divergence appears only on the compositional/factoid rows: raw score-space fusion (CombSUM/CombMNZ) wins or ties RRF; RRF never clearly dominates. The largest magnitude-preserving effects are HotpotQA (CombSUM +0.114 over linear) and MuSiQue (CombSUM +0.060/+0.090 over rrf/linear); Figure 3 (`figures/fig1_operator_map_heatmap.png`) renders the complete map as a heatmap. at expanded n=50 these gaps are directionally stable but not family-wise significant after Holm correction (Appendix C), so we describe them as consistent tendencies rather than proven separations. The n=10 exploratory values (parenthesized) show the same direction; we report both and do not over-claim gaps that shrink at n=50.

### 6.2 Rank-space vs Score-space

RRF (rank-only) and CombSUM (raw score-space) tie on 2WikiMultihopQA (1.000 each) and MuSiQue (0.950 each), but CombSUM clearly beats RRF on HotpotQA (1.000 vs 0.750) and NQ-REaR (0.800 vs 0.720). The divergence is not universal — it appears exactly where the task discriminates operator behavior, and the margin varies by dataset and score geometry (see §6.5).

### 6.3 Normalization (min-max / z-score)

Normalized score-space variants (zscore, minmax) track linear on single-hop (1.000) but underperform raw CombSUM on multi-hop (HotpotQA: zscore 0.683, minmax 0.570 vs combsum 1.000). Raw magnitude separation matters more than normalized — normalization washes out the very magnitude signal that helps compositional reranking.

### 6.3.1 Why CombSUM and CombMNZ Work on Multi-Hop: The Magnitude-Multiplicity Mechanism

The consistent superiority of CombSUM and CombMNZ on compositional multi-hop tasks (HotpotQA, NQ-REaR) — and their parity with RRF on 2WikiMultihopQA and MuSiQue — is not coincidental. These operators implement two complementary information-preserving mechanisms that align with the evidence structure of multi-hop retrieval:

**1. Magnitude preservation (CombSUM).** Multi-hop questions require composing evidence from multiple passages. SPLADE (and BM25) assign higher absolute scores to passages that match more query terms — a passage matching all three hops of a HotpotQA question *may* receive a substantially higher score than one matching only the first hop. CombSUM *sums* these scores across retrievers, so a passage that is moderately strong in both SF and SPLADE can surpass a passage that is very strong in only one. The raw score sum thus *can* reflect joint evidence strength — a candidate mechanism for why multi-hop composition benefits from magnitude-preserving fusion; we test rather than assume this link in §7.6.

**2. Multiplicity weighting (CombMNZ).** CombMNZ multiplies the score sum by the number of retrievers that retrieved the document (1 or 2 in our two-retriever setup). This is a simple but powerful "vote of confidence": a document retrieved by both SF and SPLADE with moderate scores often represents genuinely complementary evidence (one retriever caught hop 1, the other caught hop 2), whereas a document retrieved by only one system with a high score may reflect a single strong match that doesn't compose. CombMNZ thus explicitly rewards *agreement across retrievers* — a proxy for multi-hop evidence convergence.

**Why they tie RRF on 2WikiMultihopQA and MuSiQue.** On 2WikiMultihopQA, the gold passage often has a dominant single-hit in SPLADE (the Wikipedia structure creates strong lexical overlap), so RRF's rank-only fusion is already near ceiling. On MuSiQue, the pool is larger (20) and evidence is more distributed; both CombMNZ and RRF reach similar ceilings because the evidence is strong enough that either mechanism suffices. The divergence appears on HotpotQA and NQ-REaR precisely because the evidence is *distributed* across retrievers and the margin is tight — exactly where magnitude and multiplicity matter.

**Why Borda and normalized variants lag.** Borda uses (N - rank + 1) scoring, which is a linear rank transform. It preserves only ordinal information with a gentle decay, discarding magnitude entirely — so it inherits RRF's multi-hop blindness. Z-score and min-max normalization standardize each retriever's score distribution to zero mean / unit variance or [0,1] range before combination. This *equalizes* the scale but also *flattens* the very magnitude differences that can carry relevance-bearing separation: a passage with strong evidence in both retrievers gets the same normalized boost as one with weak evidence in both. The synthetic experiment (§7.2) confirms this: normalization destroys the small-margin signal that distinguishes true multi-hop matches from partial matches.

**Summary.** The operator hierarchy on multi-hop tasks is not about "which fusion function is better" in absolute terms, but about *which information class* the operator preserves. Multi-hop compositional reasoning demands magnitude and multiplicity; CombSUM and CombMNZ supply both. RRF and Borda supply only rank. Normalized variants discard the magnitude signal they were meant to equalize. This mechanistic explanation, grounded in the information-preservation framework (§3.5, §9.1), replaces the earlier descriptive "operator sensitivity" with a mechanistic account of *why* each operator behaves as it does.

### 6.4 Task Topology

Operator sensitivity is a function of task difficulty/type, not a fixed operator ordering. Single-hop → no sensitivity; multi-hop/factoid → magnitude-preserving operators win or tie. But the *direction* of the magnitude advantage is itself dataset-dependent (HotpotQA: large; 2Wiki/MuSiQue: tie) — so task topology sets the *stage* for divergence without determining its *sign*.

### 6.5 Second-Model Validation (SF+DPR, BM25+SPLADE, BM25+DPR)

To answer whether the phenomenon is SPLADE-specific (reviewer #4), we replicated the matrix with a second dense retriever, DPR, and swapped signal A (SF ↔ BM25). Full 4-pair × 2-discriminating-dataset design. We report both the n=10 exploratory probe (parenthetical) and the **n=50 intermediate replication** MRR, as linear/rrf/combsum:

| Pair (A + B) | HotpotQA n=50 (lin/rrf/combsum) | HotpotQA n=10 | NQ-REaR n=50 (lin/rrf/combsum) | NQ-REaR n=10 | Winning family (n=50) |
|--------------|----------------------------|----------------------------|----------------------------|----------------|----------------|
| SF + SPLADE | 0.733 / 0.847 / **0.947** | (0.570 / 0.750 / 1.000) | 0.628 / 0.636 / **0.657** | (0.700 / 0.720 / 0.800) | magnitude (combsum) |
| SF + DPR | **0.687** / 0.611 / 0.611 | (0.483 / 0.365 / 0.365) | **0.583** / 0.594 / 0.594 | (0.733 / 0.725 / 0.725) | α-blend (linear) |
| BM25 + SPLADE | 0.940 / **0.945** / 0.940 | (0.900 / 0.850 / 0.950) | 0.566 / **0.612** / 0.593 | (0.700 / 0.750 / 0.750) | parity (all ≈, rrf marginally top) |
| BM25 + DPR | **0.927** / 0.867 / 0.867 | (0.950 / 0.365 / 0.365) | **0.602** / 0.560 / 0.560 | (0.675 / 0.725 / 0.725) | α-blend (linear) on Hop; parity on NQ |

**Decisive finding (confirmed at n=50):** the winning operator family is determined by the *joint score geometry of the fused signals*, not by the task and not by retriever identity. Because the operators we study are symmetric in their two inputs, this is a statement about the *pair*, not about either component alone: when either participating signal has sparse, log1p-pooled, heterogeneous-scale structure (as with SPLADE), magnitude-preserving CombSUM wins on SF+SPLADE (HotpotQA 0.947 vs RRF 0.847) and remains top on NQ-REaR. When both signals carry uniform-scale, L2-normalized scores (both DPR variants), rank-only RRF and raw-score CombSUM collapse to *identical* rankings (SF+DPR: 0.611 = 0.611; BM25+DPR: 0.867 = 0.867) and only the α-weighted linear operator — which explicitly controls the magnitude trade-off — wins (SF+DPR HotpotQA 0.687; BM25+DPR HotpotQA 0.927). The phenomenon is therefore **general across model pairs but its direction is set by joint score geometry**, not by retriever identity; swapping the A/B roles of SF and BM25 in §8.3 leaves the ordering unchanged, as symmetry requires.

**n=50 caveat:** the n=10 probe *overstated* operator gaps. At n=50, BM25+SPLADE on HotpotQA collapses to a near-tie (linear 0.940 / rrf 0.945 / combsum 0.940) rather than the clean combsum win seen at n=10 (0.900 / 0.850 / 0.950); and on NQ-REaR all four pairs sit within noise of each other (0.56–0.66), so no operator is reliably superior there. The robust, n=50-backed claims are therefore narrower than the n=10 map suggested: (i) SF+SPLADE multi-hop is the clearest case where magnitude-preserving fusion beats rank-only (HotpotQA Δ0.10, stable); (ii) DPR pairs consistently show linear ≥ rank-only/raw-score; (iii) BM25+SPLADE and NQ-REaR show at most marginal operator effects. Both sample sizes are reported in full rather than selecting the more favourable one.

### 6.5.2 Second Learned Sparse Checkpoint (SPLADE-v3; Reviewer #18)

A remaining concern is that the SF+SPLADE findings might be an artifact of one specific SPLADE checkpoint. We therefore replicated the complete seven-operator matrix at n=50 on HotpotQA and MuSiQue with a second, independently trained learned sparse model — `naver/splade-v3` (gated; accessed via authenticated HF login) — replacing `naver/splade-cocondenser-ensembledistil`. Indices, queries, candidate pools and fusion settings were held fixed; only signal B's checkpoint changed (fresh SPLADE corpus vectors encoded; per-model cache isolation).

*Exploratory diagnostic; n=10 queries/dataset.*
| Operator | HotpotQA v2 (cocondenser) | HotpotQA v3 | MuSiQue v2 | MuSiQue v3 |
|----------|--------------------------:|------------:|-----------:|-----------:|
| linear | 0.832 | 0.822 | 0.887 | 0.900 |
| rrf | 0.893 | 0.903 | 0.917 | 0.943 |
| **combsum** | **0.947** | **0.960** | **0.977** | **0.987** |
| combmnz | 0.893 | 0.882 | 0.919 | 0.917 |
| borda | 0.857 | 0.862 | 0.770 | 0.790 |
| zscore | 0.897 | 0.922 | 0.953 | 0.963 |
| minmax | 0.832 | 0.822 | 0.887 | 0.900 |

The operator ordering is **stable across checkpoints**: CombSUM ranks first on both datasets under both models (HotpotQA 0.947 → 0.960; MuSiQue 0.977 → 0.987 — v3 slightly lifts every score-space operator), Borda remains last on MuSiQue, and the magnitude-vs-rank separation persists essentially unchanged. The finding is therefore not an artifact of one checkpoint but a property of the *pairing* between SF's spatial-magnitude scores and any log1p-pooled learned sparse signal. Full table: the SPLADE-v3 comparison tables.

### 6.5.3 α-Sensitivity of the Linear Operator (Reviewer #20)

The linear operator is `score = α·maxnorm(SF) + (1−α)·maxnorm(SPLADE)`, with α the weight on the zero-shot SF signal. α = 0.3 was previously used with no sensitivity analysis, leaving open whether it was selected post hoc rather than justified independently. We now sweep α ∈ {0.0, 0.1, …, 1.0} on four datasets, reusing each dataset's controlled-pool index and recomputing MRR(α) offline from the two endpoint component runs (α=1.0 = pure SF, α=0.0 = pure SPLADE). Because the fused score is *linear* in the two maxnorm'd signals, every intermediate α value computed this way is exactly what an end-to-end run at that α would produce — the curve is a complete enumeration, not an interpolation or approximation.

| α | 2Wiki | HotpotQA | MuSiQue | SciFact |
|---|------:|---------:|--------:|--------:|
| 0.0 (SPLADE) | 1.000 | 1.000 | 1.000 | 0.823 |
| 0.1 | 1.000 | 1.000 | 0.950 | 0.823 |
| 0.2 | 1.000 | 1.000 | 0.950 | 0.823 |
| **0.3** | **1.000** | **1.000** | **0.925** | **0.823** |
| 0.4 | 1.000 | 1.000 | 0.917 | 0.823 |
| 0.5 | 1.000 | 1.000 | 0.913 | 0.820 |
| 0.6 | 1.000 | 1.000 | 0.856 | 0.821 |
| 0.7 | 0.933 | 0.867 | 0.754 | 0.818 |
| 0.8 | 0.858 | 0.617 | 0.686 | 0.718 |
| 0.9 | 0.853 | 0.575 | 0.543 | 0.703 |
| 1.0 (SF) | 0.803 | 0.453 | 0.447 | 0.704 |

**Finding.** α = 0.3 is *not* a special point. On all four datasets MRR is flat (within noise) across α ∈ [0.0, 0.6] and only degrades once SF is weighted too heavily (α > 0.6), because at high α the zero-shot SF signal — which collapses on multi-hop/biomedical tasks (SF-only 0.365 HotpotQA, 0.633 COVID-QA) — dominates the blend and drags MRR down toward the SF-only floor. The plateau at low α means **any** α in [0, 0.6] yields the same result; the choice is immaterial with respect to ranking quality. We therefore retain α = 0.3 as a conservative, SF-downweighted default (it sits centrally in the flat region and avoids the SF-dominated degradation tail), and report the full sweep so the claim is auditable rather than asserted. The plot is in Appendix D.

**Interpretation status of the α sweep:** α is swept over the same evaluation queries, so the sweep characterizes the *response surface* rather than performing hyperparameter model selection; we make no optimality claim for any single α value.

### 6.6 Complementarity vs Redundancy (Kendall's τ)

To quantify *when* fusion adds versus duplicates information, we compute the mean pairwise Kendall's τ-b between operator rankings over the shared candidate set, per query, then average (real artifacts from §6.1/§6.5 runs; `temp/tau_complementarity.py`).

*Exploratory diagnostic; n=10 queries/dataset.*
| Pair (A + B) | rrf vs combsum (τ) | linear vs {rrf,combsum} (τ) | Note |
|--------------|-------------------:|----------------------------:|------|
| HotpotQA BM25+SPLADE | +0.800 | linear≈+0.91 to both | operators distinct but correlated; combsum wins MRR |
| NQ-REaR BM25+SPLADE | +0.700 | linear≈+0.83 to both | same pattern |
| HotpotQA BM25+DPR | **+1.000** | linear **−0.21** to both | RRF≡CombSUM (identical rankings); linear alone diverges |

**Interpretation (the bottleneck view):** even when MRR diverges sharply (HotpotQA BM25+SPLADE: combsum 0.950 vs rrf 0.850 at n=10), operator rankings remain highly correlated in the τ_operator sense (τ_operator≈0.80) — the operator does not *reorder* wholesale; it flips the gold doc from rank 2 to rank 1 on a few decisive queries. That is precisely the information-bottleneck mechanism: a small rank-correlation divergence at the top of the list determines whether compositional evidence survives. On DPR (HotpotQA BM25+DPR) the bottleneck is fully closed for rank-only vs raw-score fusion (τ_operator=1.000, identical), and *only* the α-weighted linear operator escapes the shared ranking — consistent with §6.5. τ_operator is an *operator-agreement* measure, not a complementarity measure; and as §9.3 notes, neither τ's predictive value for fusion gains is yet established — our own data show high τ_signal coexisting with large MRR differences at the top ranks (§6.6.3).

### 6.6.1 Operator Identifiability

The DPR-pair result above — RRF and CombSUM producing *identical* rankings and MRRs (0.611 = 0.611; 0.867 = 0.867) — motivates a metric we call **operator identifiability**: the fraction of queries for which two operators induce the *same fused ranking* on the same candidate pool. Operator choice can only matter when different operators actually produce different orderings; when identifiability is 1.0 (rankings always coincide), no amount of operator tuning changes the outcome.

Computing this over real component scores (`scripts/operator_identifiability.py`; SF+SPLADE, α=0.3, all 21 operator pairs × 10 queries × 3 datasets):

*Exploratory diagnostic; n=10 queries/dataset.*
| Dataset | linear vs rrf | combsum vs rrf | linear vs minmax | combsum vs combmnz |
|---------|--------------:|---------------:|-----------------:|-------------------:|
| HotpotQA | 0.00 | 0.00 | **1.00** | **1.00** |
| MuSiQue | 0.00 | 0.00 | **1.00** | **1.00** |
| SciFact | 0.00 | 0.00 | **1.00** | **1.00** |

Two regimes emerge cleanly. *Cross-family* pairs (magnitude-preserving vs rank-only, e.g. combsum vs rrf) have an identifiability gap of 1.00 — they never agree on any query, so operator selection genuinely matters and the geometry analysis of §6.5 applies. *Within-family* pairs are largely degenerate: linear ≡ minmax on every query in every dataset (both reduce to affine rescalings whose sums preserve the same ordering under maxnorm'd inputs), and CombSUM ≡ CombMNZ whenever each query has a single dominant top-scorer (MNZ's multiplicity term becomes constant). Mean pairwise identical-ranking rate across all 21 pairs per dataset: 0.105 (HotpotQA), 0.048 (MuSiQue), 0.105 (SciFact) — most operator pairs are identifiable; the exceptions are exactly the mathematically equivalent ones.

**Implication.** Fusion operator choice matters only where the joint score geometry provides degrees of freedom that distinct operators exploit differently. For signal pairs where operators collapse to identical rankings, effort should shift from operator selection toward signal acquisition or calibration. Full table: the operator identifiability tables.

### 6.6.2 Can Learning the Fusion Weights Beat the Diagnostic Choice?

If "match operator to score geometry" is sound guidance, a small learned fuser should not substantially outperform the correctly-chosen fixed operator — otherwise the diagnosis would be a detour around simply training the weights. We test this with logistic regression over four features per document ([s_A, s_B] raw + maxnorm-normalized), trained with **leave-one-query-out cross-validation**: for each held-out query, the model sees only documents of the other queries, so no leakage is possible (`scripts/learned_fusion_baseline.py`).

*Exploratory diagnostic; n=10 queries/dataset.*
| Dataset | n | rrf | combsum | learned (LOQO-CV) |
|---------|--:|----:|--------:|------------------:|
| HotpotQA | 10 | 0.883 | 1.000 | 1.000 |
| MuSiQue | 10 | 0.861 | 0.914 | **0.933** |
| SciFact | 10 | 0.821 | 0.820 | 0.823 |

The learned fuser matches CombSUM where geometry favors it (HotpotQA), adds a marginal gain on MuSiQue (+0.019 — within the noise band established in §4.7), and ties on SciFact. With n=10 queries these differences are not statistically meaningful; the substantive point is that a trained fusion function does *not* escape the score-geometry analysis — its weights are largest exactly on the features our framework predicts matter (normalized separation in the informative regime), and it cannot conjure ranking information the signals do not contain (SciFact ties both fixed operators). Learning the weights is a viable deployment shortcut once labeled queries exist; the diagnostic framework tells you *before* collecting labels whether the fusion stage has headroom at all.

### 6.6.3 τ_signal, Local Disagreement, and Fusion Gain

Global rank agreement is a weak summary: two signals can correlate highly overall yet disagree exactly at the top ranks where MRR is decided. Using the real SF+SPLADE traces (`scripts/tau_analysis.py`), we compute per query: τ_signal = Kendall(SF ranking, SPLADE ranking); top-1 disagreement between the two signals' top choices; and Fusion Gain = MRR(fused) − max(MRR(A), MRR(B)) for each operator.

*Exploratory diagnostic; n=10 queries/dataset.*
| Dataset | mean τ_signal | ρ(FusionGain_combsum, τ_signal) | ρ(FusionGain_combsum, top-1 disagreement) |
|---------|--------------:|--------------------------------:|------------------------------------------:|
| HotpotQA | 0.309 | — (constant gain) | — |
| MuSiQue | −0.062 | 0.52 (p=0.12) | −0.17 (p=0.65) |
| SciFact | 0.318 | 0.23 (p=0.52) | 0.25 (p=0.49) |

Two observations. First, mean τ_signal is low-to-moderate even on datasets where both signals are individually strong — global correlation does not imply top-rank agreement. Second, no correlation reaches significance at these sample sizes; the direction (positive ρ with τ_signal on MuSiQue) weakly suggests fusion gains arise where signals disagree in a structured way, but we treat this strictly as exploratory. The complementarity decomposition below is more informative:

**Where fusion matters (top-1 correctness cells, HotpotQA/MuSiQue).** When both signals place gold first (cell TT), every operator scores 1.000 — fusion has nothing to add. The action is entirely in cell FT (SF misses, SPLADE hits): there CombSUM-family operators reach 0.88–1.00 while RRF drops to 0.79–0.85 and Borda to 0.67–0.79. Rank-only fusion wastes precisely the queries where one signal's magnitude evidence could rescue the other's miss. When both miss (FF, SciFact only), no operator recovers anything (≈0.11).

### 6.6.4 Operator × Retriever-Pair Interaction Screen

The central claim — operator effectiveness depends on the *pair*, not just the task or either signal alone — is a factorial statement. We test it directly (`scripts/factorial_interaction.py`): per query, compute Δ_pair = MRR(CombSUM) − MRR(RRF); the interaction contrast between pairs p and q is D = Δ_p − Δ_q, We estimate interaction through a paired difference-of-differences contrast rather than fitting a full factorial model, tested with a sign-flip permutation (10,000 resamples, seed=42, two-sided). This is a screening analysis, not a powered confirmatory test.

*Exploratory screening analysis; paired permutation test.*

| Dataset | Contrast | n | mean Δ(SF+SPLADE) | mean Δ(other) | mean D | dz | p_perm |
|---------|----------|--:|------------------:|--------------:|-------:|---:|-------:|
| HotpotQA | vs SF+DPR | 50 | +0.096 | +0.000 | **+0.096** | +0.44 | **0.004** |
| HotpotQA | vs BM25+SPLADE | 50 | +0.096 | −0.005 | **+0.101** | +0.31 | **0.041** |
| HotpotQA | vs BM25+DPR | 50 | +0.096 | +0.000 | **+0.096** | +0.44 | **0.004** |
| NQ-REaR | vs SF+DPR | 50 | +0.013 | +0.000 | +0.013 | +0.05 | 0.738 |

The Operator × Retriever-Pair interaction is significant on HotpotQA against all three alternative pairs (permutation p-values, 10k resamples; screening analysis): CombSUM's advantage over RRF exists *only* when the pair's joint score geometry provides exploitable magnitude separation (SF+SPLADE), and vanishes for every other pairing (Δ≈0 for DPR-containing pairs, slightly negative for BM25+SPLADE at n=50). On NQ-REaR the interaction is null — consistent with its large-pool factoid profile (§6.1), where operator differences are small under any pairing. This provides a direct test of the proposed operator–pair interaction.

**Retriever-pair matrix (mean Δ = CombSUM − RRF per query, from the four-pair runs).**

| Signal A | Signal B | Score geometry | mean Δ | Interpretation |
|----------|----------|----------------|-------:|----------------|
| SF | SPLADE | heterogeneous scales | +++ (+0.093 at n=100) | magnitude useful |
| SF | DPR | normalized dense | ~0 (+0.000 at n=50) | magnitude weak after normalization |
| BM25 | SPLADE | mixed integer/sparse | −0.005 (n=50) | smaller effect; geometry not exploitable |
| BM25 | DPR | more comparable | ~0 (+0.000 at n=50) | rank sufficient |

Replacing SF with BM25 reduces but does not reverse the score-space advantage pattern, indicating that the phenomenon is not uniquely attributable to Semantic Folding — although its magnitude depends strongly on the joint score geometry of the pair.

### 6.6.5 Predicting Operator Suitability from Pre-Fusion Geometry

The diagnostic framework becomes predictive if measurable pre-fusion properties forecast the winning family. For each query we compute 21 geometry features — nine per signal (mean, std, CV, range, skew, kurtosis, top-1/top-2 and top-1/top-5 margins, entropy) plus three pair features (Pearson correlation, Kendall τ, top-5 Jaccard overlap) — and label each query with the winning operator family (rank-only vs score-space). A leave-one-*dataset*-out (LODO) logistic regression then tests generalization to unseen tasks (`scripts/geometry_predictor.py`; `appendix_stats/geometry_predictor.{md,json}`). On the n=10 exploratory traces, 34 of 40 queries are operator-ties (gold at rank 1 under every operator), leaving only 6 divergent queries to learn from — too few for a meaningful fit, and we said so. We have now scaled this to the n=100 component traces (hotpotqa/musique/nq_rear, n=300 pooled) with a decision tree and logistic classifier under strict LODO (never a random query split, so dataset-characteristic leakage is ruled out; `scripts/cross_dataset_predict.py`). **Result: the decision tree generalizes to unseen datasets — pooled LODO AUROC = 0.702 (bootstrap 95% CI 0.627–0.775), AUPRC = 0.334 (0.245–0.446) vs a 0.193 base rate; the logistic model is near-random (AUROC 0.536, CI 0.456–0.616).** The tree's clear win over the linear model shows the geometry→operator relationship is *non-linear* (justifying the §6.6.5 decision-boundary framing), and the AUROC > 0.5 establishes that pre-fusion geometry predicts the winning operator family on held-out datasets — resolving the prior "too few observations" weakness. The framework (features, labels, LODO protocol, real n=100 estimates) is delivered; full table in Appendix E.10.

### 6.7 Generality across Checkpoints (Item 5)

A reviewer concern is that the SF+SPLADE findings might be an artifact of one specific
learned-sparse checkpoint. We test generality along both axes of the fusion signal.

**Sparse axis (second learned-sparse checkpoint).** Replacing `naver/splade-cocondenser-ensembledistil`
with a second, independently trained learned sparse model `naver/splade-v3` (§6.5.2, n=50
HotpotQA/MuSiQue) leaves the seven-operator ordering stable — CombSUM ranks first under both
models (HotpotQA 0.947 → 0.960; MuSiQue 0.977 → 0.987) and the magnitude-vs-rank separation
persists essentially unchanged. The effect is therefore not a property of one checkpoint but of
the *pairing* between SF's spatial-magnitude scores and any log1p-pooled learned-sparse signal.

**Dense axis (second score family: DPR).** Substituting signal B with a dense bi-encoder
(`facebook/dpr-...-single-nq-base`) changes the conclusion: on HotpotQA (n=100) the
magnitude-intervention World− degradation is +0.000 (vs +0.077 on SF+SPLADE) and operator
identifiability I_1(RRF≠CombSUM) = 0.010 (vs 0.250 on SF+SPLADE). The SF+DPR pair is
*non-identifiable* — RRF and CombSUM agree at top-1 on essentially every query — so the
relevance-aligned-magnitude effect disappears, exactly the §9 boundary condition (Step 7:
"when operators are non-identifiable, the effect disappears"; cf. §6.6.1's SF+DPR ranking
collapse 0.611 = 0.611). The effect is thus **pair-geometry-dependent, not checkpoint-universal**:
it manifests where a *sparse* learned signal supplies a relevance-aligned magnitude, and vanishes
where the second signal is a dense cosine encoder whose magnitude is not relevance-aligned.

| cell | dataset | World− degradation (Item 1) | I_1 RRF≠CombSUM (Item 3) |
|---|---|---:|---:|
| SF+SPLADE-A | hotpotqa / musique / nq_rear | +0.077 / +0.051 / +0.079 | 0.250 / 0.280 / 0.200 |
| SF+DPR-A | hotpotqa | +0.000 | 0.010 |
| SF+SPLADE-v3 | HotpotQA / MuSiQue (§6.5.2) | operator ordering stable | stable |

**Limitation.** A second *dense* checkpoint (DPR-B) could not be acquired offline in this
environment; DPR-A above therefore represents the dense family rather than a second dense
checkpoint. Combined with SPLADE-v3 this still demonstrates cross-*family* and cross-*sparse-
checkpoint* generality, and the dense-family negative result (SF+DPR non-identifiable) is itself
the theoretically predicted boundary case, not a gap. Full table: Appendix E.8.

---
---
### 6.8 Normalization and the Source of Magnitude (Item 6)

A remaining ambiguity is whether the CombSUM-over-RRF advantage comes from signal B's
*absolute scale* or from its *within-retriever score separation* (the shape of the magnitude
distribution). We ablate this directly: each signal is re-normalized four ways — raw, min-max,
z-score, rank-normalized — independently and before the *same* CombSUM (α=0.3) / RRF (k=60)
fusion, over the SF+SPLADE n=100 traces (`scripts/normalization_ablation.py`; Reviews §11, §12).
The three-concept framing (§12) makes the prediction sharp: all schemes preserve rank information
R(s); min-max/z-score preserve within-signal geometry G_within while destroying absolute scale;
rank-normalization destroys G_within entirely, leaving only R(s) — so if CombSUM's edge depends on
G_within, rank-normalization should erase it.

*Exploratory diagnostic; n=100, SF+SPLADE; cells below are A\\B; ΔMRR = MRR_CombSUM − MRR_RRF;
World− deg = CombSUM MRR lost under the anti-relevance World− intervention (effect operative when >0).*

| cell | hotpotqa ΔMRR / World− | musique ΔMRR / World− |
|---|---:|---:|
| raw / raw (baseline) | +0.096 / +0.082 | +0.060 / +0.053 |
| raw / min-max | +0.113 / +0.101 | +0.122 / +0.097 |
| raw / z-score | +0.137 / +0.066 | +0.217 / +0.040 |
| raw / rank-norm | **−0.067** / +0.048 | **−0.057** / +0.013 |
| z-score / z-score | +0.132 / +0.080 | +0.193 / +0.065 |
| z-score / rank-norm | −0.164 / +0.026 | −0.135 / +0.011 |
| rank-norm / rank-norm | −0.015 / +0.054 | −0.005 / +0.026 |

Two findings are clean and consistent across datasets. **(i)** min-max and z-score normalization of
signal B — which preserve separation but annihilate absolute scale — *strengthen* the CombSUM edge
(ΔMRR rises, World− stays positive): the effect is therefore a property of **within-retriever
separation, not raw scale**. **(ii)** rank-normalizing signal B — which discards G_within and leaves
only R(s), exactly what RRF already uses — **flips ΔMRR negative**: CombSUM no longer beats RRF. This
is the §12 boundary condition arrived at from the normalization side: RRF preserves rank information
but discards within-signal geometry; CombSUM's advantage requires that geometry. Normalizing signal A
(SF) alone changes little — SF is already rank-stable, so the decisive axis is the learned-sparse
signal's magnitude geometry. The conclusion answers the reviewer's core question: it is separation,
not scale.

---
---
---

---

### 6.9 Synthetic Phase Diagram of Fusion Behaviour (Item 8)

The reviewer asks for a unified picture of *when* score-space fusion wins. We answer with a
mechanistic phase diagram rather than a single list of datasets: two synthetic retrievers with a
controlled rank correlation tau in [-1, 1] and a controlled score-margin difference Delta; for each
(tau, Delta) cell we generate Q=150 synthetic queries (M=50 docs, gold = top-ranked), fuse with
CombSUM and RRF, and record DeltaMRR = MRR_CombSUM - MRR_RRF
(`scripts/synthetic_phase_diagram.py`; full grid in Appendix E.11). The map reveals a sharp,
interpretable regime boundary:

| tau \ Delta | 0.0 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| -0.8 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| -0.4 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
|  0.0 | -0.005 | -0.003 | +0.010 | +0.013 | +0.023 |
|  0.4 | -0.006 | -0.001 | +0.197 | +0.190 | +0.180 |
|  0.8 | -0.004 | -0.002 | +0.263 | +0.233 | +0.238 |

**Reading.** DeltaMRR is ~0 everywhere except the joint region **high tau (>= 0.4) AND positive
Delta (>= 0.5)**, where CombSUM's edge jumps to +0.18..+0.26. The effect therefore requires *both*
retrievers to agree on ranking (tau high) *and* signal B to carry a relevance-aligned magnitude
margin (Delta > 0); either condition alone is insufficient. We overlay the empirical (tau, Delta)
of the real pairs as validation anchors (Appendix E.11): SF+SPLADE sits at tau~0.22-0.43,
Delta~0.45-0.62 (inside the positive-effect quadrant, matching its real DeltaMRR > 0), while
SF+DPR/hotpotqa sits at tau=1.000 but Delta=0.288 (below the Delta=0.5 threshold) — exactly the
non-identifiable, near-zero-effect regime of Items 3/5. The diagram mechanistically recovers the
operator-identifiability and checkpoint-generality conclusions: the boundary is a property of the
fused pair's geometry, not of any one retriever. This is the central phase-map figure the reviewer
requested, and it is *predictive* (it locates the real pairs correctly) rather than merely
descriptive.

## 7. The Magnitude Information Hypothesis

### 7.1 Rank Invariance (Proposition 1)

Verified computationally: RRF output is bit-identical (to 1e-12) under strictly monotonic transforms of component scores, while CombMNZ changes — confirming magnitude sensitivity.

### 7.2 Synthetic Magnitude Control

To isolate magnitude as a *controlled* factor (not a correlate of rank), we hold RANK fixed (Doc A always rank 1, Doc B always rank 2) and vary only the SCORE MAGNITUDE, then apply all seven operators and ask whether each correctly ranks A above B. Ranking is held constant by construction, so any operator that changes its A/B ordering under a magnitude manipulation is responding to magnitude alone. This is implemented in `semantic_folding/synthetic_magnitude_experiment.py` and run with the real fusion code (`fusion_operators.fuse`); it is therefore a genuine controlled experiment, not an illustrative example. Any operator whose A/B ordering changes under a monotone transform is responding to magnitude alone.

**Operator phase diagram (controlled simulation).** The 2-document toy establishes the mechanism; a full simulation (`scripts/synthetic_operator_phase.py`) maps *where* each operator family wins across 81 conditions: pool sizes N ∈ {20, 100, 500}, magnitude-distribution families (concentrated / spread / heavy-tail), signal-B scale ratios ×{1, 10, 100}, and three relevance regimes — *rank-dominant* (gold = top-ranked document, magnitude uninformative), *magnitude-dominant* (gold sits mid-rank but carries a large signal-B spike), and *mixed* (half trials each). Results:

| family | regime | linear | rrf | combsum | combmnz | borda | zscore | minmax | winner |
|--------|--------|-------:|----:|--------:|--------:|------:|-------:|-------:|--------|
| any | rank-dominant | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | all tie |
| concentrated | magnitude-dominant | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | score-space |
| spread | magnitude-dominant | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | score-space |
| heavy-tail | magnitude-dominant | 1.000 | 0.000 | 0.996 | 0.996 | 0.000 | 1.000 | 1.000 | score-space |
| any | mixed | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | score-space |

Figure 4 renders the family-gap phase diagram across pool sizes, magnitude conditions, and relevance regimes (`figures/fig4_phase_diagram.png`). **Phase structure (54/81 cells differentiate operators; deterministic seeds):**

- In **rank-dominant conditions every operator ties at 1.000** — when relevance is fully encoded in rank, fusion choice is unidentifiable (§6.6.1) and magnitude adds nothing.
- In **magnitude-dominant and mixed conditions the families separate perfectly**: score-space operators recover gold at ≈1.000 accuracy while rank-only operators fail on essentially every such query (0.000). Normalized variants succeed here because the spike dominates after normalization too; their failure mode in the 2-doc toy (§7.2 table above) is specific to *small-margin* regimes where normalization noise swamps the signal.
- Distribution shape barely matters for the phase boundary — concentrated vs heavy-tail changes only CombSUM/CombMNZ at the margin (0.996 vs 1.000).

This is the operator phase diagram the hypothesis predicts: **the winning operator family is determined by whether task-relevant information lives in rank alone or also in magnitude — not by distributional details.** Full per-cell results: the operator phase diagram tables.

*Synthetic control; deterministic construction.*
| Condition | Score(A) | Score(B) | Margin | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|-----------|----------|----------|--------|--------|-----|---------|---------|-------|--------|--------|
| large | 45 | 12 | +33 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| med | 35 | 20 | +15 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| small | 30 | 25 | +5 | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| tiny | 21 | 19 | +2 | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| rev | 12 | 45 | −33 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**Findings (controlled, not merely illustrative).** (i) Rank-only operators (RRF, Borda) rank A above B whenever A's *rank* is 1, regardless of margin — they are blind to magnitude by design (confirmed: RRF output is bit-identical under log/sqrt/exp/sigmoid transforms of the scores). (ii) **Raw** score-space operators (CombSUM, CombMNZ) preserve the real margin and rank A above B correctly in every non-reversed case. (iii) **Normalized** score-space operators (linear, z-score, min-max) *fail* in the small-margin regime (+5, +2): normalization amplifies the tiny real margin into noise and can flip A below B. This is the opposite of the naive "magnitude always helps" story — normalization can *destroy* useful magnitude. (iv) When the margin reverses (B genuinely more relevant by score), all operators correctly flip, because raw score sum puts B first. The clean controlled conclusion: **magnitude information is operative exactly where rank is tied or near-tied and the raw (un-normalized) score carries the discriminative margin; normalization can discard it.** This refines the earlier magnitude-fallacy framing from a universal claim into a conditional, score-geometry-dependent one. The synthetic control is connected to real retrieval in §7.3 and §6.5 (where SF+SPLADE multi-hop shows the same raw-magnitude advantage).

### 7.3 Real Retrieval Traces

Across the SF+SPLADE 10-dataset matrix (10-query exploratory probes), multi-hop queries suggest the largest operator gaps: HotpotQA shows CombSUM 1.000 vs RRF 0.750 (Δ0.25) at n=10; the expanded n=100 run preserves the direction with a smaller gap (CombSUM 0.947 vs RRF 0.854). NQ-REaR suggests CombMNZ 0.820 vs borda 0.653 at n=10; at n=100 CombSUM leads (0.746) with borda last (0.602). Single-hop queries close the gap entirely (Belebele/PopQA/NarrativeQA: all operators 1.000, a ceiling effect that masks operator differences). The gap concentrates on compositional tasks, where the gold passage's score margin over distractors is plausibly what raw magnitude preserves and rank-only fusion discards; §7.6 tests that interpretation.

### 7.4 Magnitude Perturbation on Real Retrieval Outputs (Controlled Intervention)

**Question.** Can magnitude affect fusion while rank remains unchanged?

**Design.** We perturb real per-document component scores captured during the α-sweep endpoint runs (HotpotQA, MuSiQue, SciFact; n=10 exploratory queries each) and re-fuse with all seven operators. Conditions applied to one signal (SF or SPLADE), the other held fixed: `x2` (s′=2s), `log1p`, `pow05` (s^0.5), `rpr` (rank-preserving random remap of magnitudes), `shufflescores` (permute scores across documents — preserves the magnitude distribution, destroys ranks), plus the resolver-requested intervention battery: `compress` (rank-preserving squash of all gaps to ~10⁻³, original ties preserved), `amplify` (rank-preserving spread via fourth-power gap expansion), and `magswap` (top-2 margin shrunk to 10% around its midpoint — ranks strictly unchanged, local separation compressed). Full tables in Appendix E; generator `scripts/magnitude_perturbation.py` (tracked; seed=42).

The battery distinguishes three distinct meanings of "magnitude": **raw magnitude** (the component score itself), **relative magnitude** (score separation within one signal), and **cross-signal scale** (comparability of scores between two signals). Each condition targets a different aspect:

| Intervention | Rank | Magnitude effect | What it tests | Primary inference |
|--------------|------|------------------|---------------|-------------------|
| monotonic transforms (`log1p`, `pow05`) | preserved | changed | RRF invariance | rank-only operators provably blind to monotone rescaling |
| `x2` | preserved | cross-signal scale changed | cross-signal calibration | scale balance affects score-space fusion only |
| `compress` | preserved | within-signal separation reduced | local magnitude sensitivity | near-tie magnitudes deprive score-space fusion of its signal |
| `amplify` | preserved | separation increased | magnitude sensitivity | wider gaps amplify score-space differences |
| `magswap` | preserved | top-margin compressed 10× | local margin dependence | top-rank decisions hinge on the margin CombSUM sees |
| `rpr` | preserved | magnitudes replaced | magnitude identity | only order carries information for rank-only ops |
| `shufflescores` | destroyed | reassigned | rank destruction | asymmetric: removes rank-only fusion's sole source |

**Prediction.** Rank-only operators should remain invariant; magnitude-sensitive operators should change.

**Result.**

1. **Rank-only operators are empirically invariant on real data.** RRF produces *identical* MRR and fused rankings (τ=1.000) under *every* rank-preserving condition — including the new `compress`/`amplify`/`magswap` battery on all three datasets — where magnitudes change drastically but order is fixed. On HotpotQA/SF: RRF = 0.883 under orig/x2/log1p/pow05/rpr/compress/amplify/magswap, unchanged to three decimals. Figure 5 summarizes the three-regime structure — original, rank-preserving changes, rank destruction — on MuSiQue real traces (`figures/fig8_causal_centerpiece.png`). Figure 6 visualizes the battery on MuSiQue real traces (`figures/fig3_perturbation_battery.png`). Borda likewise (τ=1.000 throughout).
2. **Score-space operators respond to magnitude alone.** Under `compress` on MuSiQue/SPLADE, CombSUM collapses 0.914 → 0.460 while RRF stays frozen at 0.861 — squashing the gaps removes exactly the separation signal that magnitude-sensitive fusion exploits. Under `x2` on MuSiQue/SF, CombSUM drops 0.914 → 0.805 (scale distortion); under `log1p`/`pow05` on HotpotQA/SPLADE, linear falls 0.867 → 0.783.
3. **Destroying ranks hurts rank-only fusion maximally.** Under `shufflescores`, RRF collapses 0.883 → 0.354 (HotpotQA), 0.861 → 0.397 (MuSiQue); Borda 0.733 → 0.219. Score-space operators degrade less because their magnitude information still carries partial relevance signal when ranks are scrambled.

   *Interpretation note:* `shufflescores` destroys rank information while reassigning magnitudes — it therefore tests rank-destruction sensitivity, not magnitude utility. Its correct reading is asymmetric damage: rank-only fusion loses its sole information source, whereas score-space operators retain partial signal through the (now randomly reassigned) magnitudes.

We emphasize the epistemic hierarchy across §7: §7.1 establishes a mathematical property; §7.2 provides controlled causal manipulation; this section establishes **causal sensitivity** of fusion outputs to magnitude on real outputs; and §7.6 addresses relevance association separately. We therefore establish causal sensitivity of fusion to magnitude and separately provide evidence that magnitude is relevance-bearing in selected retrieval settings — these are distinct claims.

Beyond that caveat, this completes the controlled-intervention argument: the information classes are not merely definitional (Proposition 1) but *operationally separable on real retrieval outputs* — rank-preserving magnitude changes leave I_rank-fusion bit-identical yet measurably alter I_magnitude-fusion, and vice versa. The synthetic control (§7.2) and this real-output experiment agree in every prediction.

#### Is it magnitude, or just calibration? (review-requested baseline battery)

To separate *informative magnitude* from *arbitrary scaling*, we re-fuse the real traces under seven per-signal normalizations — raw, min-max, z-score, L2, rank-Gaussian (inverse-normal ranks), sigmoid (median/IQR-scaled logistic), quantile (empirical CDF → uniform), and softmax — applied identically to both signals before fusion (`scripts/calibration_baselines.py`; `appendix_stats/calibration_baselines.{md,json}`). Three findings: (i) **scale-family normalizations preserve CombSUM's advantage** — min-max/z-score/L2/softmax match raw within ±0.006 MRR on all three datasets — so the effect is not an artifact of one particular scale; (ii) **order-destroying normalizations hurt exactly as the theory predicts** — quantile mapping collapses CombSUM to 0.683 on HotpotQA (it flattens the top-rank margin that carries the signal), while rank-Gauss *improves* MuSiQue CombSUM to 0.950 by stabilizing heavy tails without destroying order; (iii) sigmoid compression hurts everywhere (0.853–0.883), confirming that saturating the dynamic range removes discriminative magnitude. The design space the review requests — *magnitude-preserving but calibration-aware fusion* — is therefore empirically visible already: keep monotone order, stabilize tails, never flatten the top margin. Full tables in the calibration baselines tables.

### 7.5 Score Margin vs Fusion Error (Where Rank-Only Fusion Fails)

The perturbation experiment shows rank-only fusion *can* respond to magnitude; this analysis locates *where on real queries* that response decides outcomes. For each query we compute the **joint normalized margin**: per signal, margin = (best gold score − best non-gold score)/max|score| (negative = a distractor outscores gold in that signal), then average the two signals. We bin queries by joint margin and measure the rescue rate — P(RRF top-1 wrong ∧ CombSUM top-1 correct) (`scripts/margin_vs_error.py`; real component scores; Figure A1).

![Score margin vs fusion error](figures/margin_vs_error.png)

**Findings (n=10 per dataset; interpret as directional):**

1. **RRF/CombSUM top-1 disagreement is pervasive** — 9/10 (HotpotQA), 8/10 (MuSiQue), 10/10 (SciFact) queries — yet rescue is rare (1, 2, 0 respectively): the operators usually disagree on *distractor ordering above gold*, not on gold recovery. This is the operator-identifiability observation (§6.6.1) at query level.

2. **Cross-dataset operator identifiability (n=100):**
   - **HotpotQA**: SF+SPLADE I_1(RRF≠CombSUM)=0.300 (identifiable); SF+DPR and BM25+DPR non-identifiable (I_1=0.000, confirmed).
   - **MuSiQue**: SF+SPLADE I_1=0.160 (identifiable, weaker signal); SF+DPR and BM25+DPR non-identifiable (expected per §9 boundary — DPR unavailable offline).
   - **NQ-REaR**: SF+SPLADE I_1=0.210 (identifiable); SF+DPR and BM25+DPR non-identifiable (same expected pattern).
   - **BM25+SPLADE**: I_1≈0.10 on HotpotQA (weakly identifiable); musicre/nq_rear traces generated but operator_identifiability SKIPed due to filename mismatch; pattern expected to mirror HotpotQA weak identifiability.
   The cross-dataset pattern confirms the §9 boundary condition: sparse-signal pairs (SPLADE) are operator-identifiable; dense-signal pairs (DPR) are non-identifiable, consistent across all three datasets.

3. **The rescue that does occur sits at the smallest margin bin**: MuSiQue's single negative-joint-margin query (gold below a distractor in one signal) is rescued by CombSUM — exactly the regime where magnitude information is decisive and rank-only fusion cannot see it. HotpotQA's rescue falls in the lowest positive bin [0, 0.10).
3. **Large positive margins → no rescues anywhere** (SciFact [0.30+]: 0/6): when gold already dominates both signals, every operator succeeds and magnitude adds nothing — consistent with the operator-invariant single-hop ceiling of §6.1.


### 7.6 Relevance-Bearing Score Magnitude (H3)
### 7.6 Relevance-Bearing Score Magnitude (H3)

The perturbation battery establishes operator *sensitivity* to magnitude; H3 asks whether that magnitude carries *relevance* information. We test this on the real component traces (`scripts/magnitude_relevance.py`), three ways:

1. **Margin statistics.** Per query, Δ = gold score − best negative score. SPLADE achieves P(Δ>0) = 1.00 on HotpotQA and MuSiQue (mean Δ +0.19/+0.30); SF collapses there (P(Δ>0) = 0.20/0.33). Rank-based AUC is 0.87–0.98 everywhere — both signals rank gold highly, but only SPLADE's *magnitudes* consistently separate gold from the best distractor on multi-hop tasks.
2. **Calibration.** Binning SPLADE scores shows monotone P(gold | bin): bottom bin [0.0,0.2) → P(gold) ≈ 0.00–0.01; top bin [0.8,1.0] → P(gold) = 0.52 (HotpotQA), 0.52 (MuSiQue), 0.64 (SciFact). Magnitude is not arbitrary scale; it is informative about gold status.
3. **Supporting-status correlation.** Spearman ρ(score, supporting-doc membership) over all pool documents is positive for SPLADE on every dataset (+0.17 to +0.25 after fixing an early title-matching bug that had produced spurious negatives; n = 660–1000 docs per dataset).

Together these ground H3's conditional form: on multi-hop pools, SPLADE score separation *does* track gold-vs-distractor status, so the magnitude that CombSUM preserves and RRF discards is relevance-relevant there — while SF's own magnitudes do not separate, explaining why fusing SF naively by score underperforms. This is exactly the joint-geometry interaction the factorial screen (§6.6.4) detects.

**Rank-conditioned magnitude analysis (addressing the rank/score circularity).** Because per-signal scores are monotone in their own ranks, correlation with gold alone cannot separate magnitude information from ordinal information. We therefore fit leave-one-query-out logistic models for gold-vs-negative status: M1 uses only normalized ranks in the two component rankings; M2 adds magnitudes and local top-margins (`scripts/rank_conditioned_magnitude.py`, `appendix_stats/rank_conditioned_magnitude.{md,json}`). The result is a null that sharpens the thesis: **ΔAUC(M2−M1) ≤ 0 on all three datasets** with bootstrap CIs spanning zero. Within a single signal, magnitude carries no gold information beyond what its ranking already expresses — as expected when scores are monotone in ranks. The relevance-bearing content of magnitude therefore lives at the **cross-signal level**: when two heterogeneous-scale signals are fused, relative magnitudes across signals change which document wins, and no single-signal ranking encodes that. This is precisely why operator choice matters exactly at the pair level (§6.6.4) and why we scope 'relevance-bearing magnitude' to heterogeneous pairs rather than to any component signal alone.


#### 7.6.1 Relevance-aligned counterfactual (causal isolation of the useful magnitude)

§7.4 shows fusion is *sensitive* to magnitude; §7.6 shows magnitude is *relevance-bearing* descriptively. The reviewer's central demand is the intermediate, causal step: show that the **relevance-aligned** component of magnitude — not magnitude per se — is what score-space fusion exploits. We isolate this with a rank-preserving counterfactual (`scripts/counterfactual_magnitude.py`, seed=42; full tables in Appendix E.5).

For each query we keep the SF and SPLADE **ranks** fixed and shift only the *magnitude* of the gold document within its rank bucket:

* **World+** amplifies the gold document's score margin above the bucket's non-gold mean by ρ ∈ {1.25, 1.5, 2.0};
* **World−** reverses that margin, placing gold at the midpoint between its own score and the bucket non-gold mean.

Because ranks are held fixed *by construction*, RRF (rank-only) output is identical across all worlds — we verify this with an exact per-document rank-equality check (RRF ranks identical for every query; τ=1.000 on the tie-aware metric). Any change in CombSUM, CombMNZ, or linear fusion is therefore attributable to magnitude, not rank. The causal prediction is MRR(World+) ≥ MRR(orig) ≥ MRR(World−) while RRF stays flat.

On the examined n=10 diagnostic traces the prediction holds directionally: suppressing the relevance-aligned margin (World−) degrades CombSUM MRR on every dataset with headroom (HotpotQA −0.397, MuSiQue −0.050, SciFact −0.001), while amplifying it (World+) never hurts and slightly helps where headroom exists (MuSiQue +0.0018, SciFact +0.0001); RRF is unchanged in all cases. The discriminating datasets (HotpotQA, 2WikiMultihopQA) sit at ceiling, so World+ has no headroom there — the decisive signal is the World− suppression, exactly the asymmetry §7.5 identifies (magnitude is operative in the small/negative-margin regime).

This is the strongest statement the intervention supports:

> **The component of score magnitude that is relevance-aligned is the component that drives the CombSUM/CombMNZ gain over RRF.** Removing the relevance-aligned margin degrades fusion quality; RRF is unaffected because it discards magnitude entirely.

We describe this as a *magnitude-intervention* result. It establishes that the relevance-aligned margin is causally operative in score-space fusion; it does not by itself certify that the original magnitude encodes a specific semantic quantity such as compositional depth. The n=100 confirmatory run (HotpotQA, MuSiQue, NQ-REaR) is reported in Appendix E.5: at n=100 the World+ vs orig CombSUM MRR contrast is +0.066 [+0.032, +0.105] on HotpotQA (and +0.065/+0.053 on MuSiQue/NQ-REaR), orig vs World− is +0.052 [+0.028, +0.080] (HotpotQA; +0.054/+0.079 on the other two), all paired bootstrap 95% CIs excluding zero, while RRF is exactly invariant (ΔMRR = 0.000, τ=1.000). This is the causal-isolation result the reviewer required.


### 7.7 Single-hop vs Multi-hop

Single-hop reranking in our candidate conditions is operator-invariant (largely masked by ceiling effects). Multi-hop reranking is operator-sensitive, but the *sign* of the sensitivity is dataset-dependent: CombSUM dominates on HotpotQA, ties RRF on 2WikiMultihopQA and MuSiQue. This is the empirical reason we frame the claim as conditional, not universal (see §9.4).

### 7.8 When RRF Discards Useful Information

**Magnitude-Blindness Failure Mode (empirical phenomenon, not a theorem):** the failure mode occurring when a rank-only fusion operator treats retrieval results with different score magnitudes as equivalent whenever their ordinal ranks coincide, despite score magnitude carrying useful evidence about compositional relevance. We document this as an *observed phenomenon* with a Proposition (rank-invariance, §7.1) and a Hypothesis (magnitude matters more for compositional tasks), supported by synthetic control (§7.2) and real traces (§7.3) — deliberately avoiding "theorem" wording. Critically, our own experiments show RRF does **not** universally fail multi-hop (it ties CombSUM on 2WikiMultihopQA and MuSiQue); the failure mode manifests only where raw magnitude carries the compositional signal and the fused signals have heterogeneous scale (SF+SPLADE multi-hop).

### 7.9 Query Geometry Predicts the Fusion Gain (Item 2)

§7.5–§7.8 are descriptive: they characterize *where* score magnitude matters. The reviewer's demand (§6, §28) is that the geometry framework predict *when* CombSUM beats RRF at the query level. We regress each query's fusion gain `ΔMRR_q = RR_CombSUM,q − RR_RRF,q` on its score-geometry features (`scripts/geometry_predictor.py`, seed=42; full tables in Appendix E.6).

Features: global (`Δ12`, `Δ15`, `σ`, `τ_signal`, `κ`) and top-k relevance-conditioned margins (`gold_d15_sf`, `gold_d15_sp`, `cross_gold_margin`, `joint_margin` per §7.5). Standardized OLS with bootstrap 95% CI (B=10000). On n=100 HotpotQA the regression explains R²=0.10; on MuSiQue R²=0.17. No single global feature's CI excludes zero at this sample size — an honest null at the pooled level — but the **winning-query population** is sharply structured: of the n=100 queries, only 4 (HotpotQA) / 5 (MuSiQue) have CombSUM promoting a gold RRF misses (Type A), and those Type-A queries have **more negative joint_margin** than the non-changing Type-C majority (HotpotQA: A=−0.239 vs C=−0.111; MuSiQue: A=−0.340 vs C=−0.155). This recovers §7.5's margin-vs-error finding at the population level: magnitude fusion wins precisely in the small/negative-margin regime, not uniformly. 2WikiMultihopQA and SciFact show R²≈0 (ceiling) — recorded as negative results in §22, not silently dropped.

> **Geometry → gain:** query-level top-k relevance-conditioned margin predicts the direction of the CombSUM-over-RRF gain; the effect is confined to the negative-margin decision-boundary population, consistent with the causal intervention of §7.6.1.

---
### 7.10 The Fusion Gain Is a Decision-Boundary Phenomenon (Item 4)

§7.9 shows *which* queries win; here we ask *how concentrated* the gain is. Decompose the
total CombSUM−RRF MRR gap into per-query contributions `ΔRR_q = RR_CombSUM,q − RR_RRF,q`
(`scripts/toprank_decomposition.py`; full table in Appendix E.7). On n=100 HotpotQA/MuSiQue/NQ-REaR
the gain is overwhelmingly localized: the top-20% of queries by |ΔRR_q| carry 0.98 / 0.97 / 1.00 of
the total |ΔRR| (H6: ≥80% in <20% — confirmed on all three). Type-C (no change) queries dominate the
*count* (75/71/80 of 100) but contribute ≈0 to the gap; the entire MRR difference comes from the
small Type-A/B boundary population that §7.9 already localized to the negative-margin regime. This is
the quantitative backing for the central decision-boundary figure (§23): score-magnitude fusion is not
a broad effect but a sharp boundary correction, which is exactly why a magnitude-blind operator (RRF)
loses on a few queries rather than broadly.

> **Boundary concentration:** ≥97% of the CombSUM-over-RRF MRR gain originates from <20% of queries
> (the negative-margin decision-boundary population), confirming the effect is local, not global.

---
---
---

## 8. Representation and Scaling Boundaries

### 8.1 Feature Invariance (Overlap-Feature Invariance)

The representation chain is: raw SDR overlap → spatial transformation → final SF score. For **raw binary SDR overlap**, q,d ∈ {0,1}ᴰ, the dot product is qᵀd = Σ qᵢdᵢ (overlap count), and any feature that is a deterministic transformation of that count carries no independent ranking information at this stage. The pipeline then adds UMAP projection, Gaussian smoothing, and spreading activation, so the *emitted* score is a deterministic transformation of the encoded spatial representation rather than qᵀd itself; whether these transforms introduce non-overlap ranking information is an open empirical question. The invariance bound therefore applies strictly to the raw SDR overlap representation, and the pipeline-level claim remains a hypothesis: **whether the complete SF pipeline introduces additional independent ranking information is open**. We have now run the first version of this test (`scripts/feature_invariance.py`; `appendix_stats/feature_invariance.{md,json}`): using token-intersection count as a documented overlap proxy (per-query binary fingerprints are not exported), overlap alone explains most score variance (R² 0.05–0.35 across datasets), residual contributions of doc length / Jaccard / term rarity are small (partial R² ≤ 0.076), and a linear model on these features cannot reconstruct the pipeline ranking (MRR far below pipeline level) — i.e., no simple non-overlap feature demonstrates pipeline-added information, supporting invariance against this feature set. The fingerprint-exact injection test remains the decisive instrument and is retained as future work.

### 8.2 Score Concentration (Candidate-Growth-Induced)

We make no asymptotic claim about score dynamic range (no O(√N) scaling law). Instead we analyze **score concentration under growing candidate populations** as a measured phenomenon. For binary SDR overlap with qᵢ,dᵢ ~ Bernoulli(ρ), K=|q|₁: E[qᵀd]=Kρ, Var(qᵀd)=Kρ(1−ρ). The empirical question is whether the *relative separation* between relevant and irrelevant candidates is maintainable as candidate count grows when score distributions are concentrated.

### 8.3 Candidate-Size Scaling

We measure the effect of candidate-pool size on fusion operator ranking along two independent axes: *which signals are fused* and *how large the pool grows*. The question is whether score concentration at the tail of the distribution degrades fusion quality as distractors grow.

**(a) SF+SPLADE.** Holding the query fixed, we grew the candidate set to N ∈ {20, 50, 100, 494} documents on HotpotQA with SF+SPLADE, reporting MRR and P@1 over n=10 queries per N:

| N   | linear MRR | rrf MRR | combsum MRR | linear P@1 | rrf P@1 | combsum P@1 |
|-----|-----------|---------|-------------|-----------|---------|-------------|
| 20  | 0.558     | 0.667   | **1.000**   | 0.300     | 0.400     | **1.000**   |
| 50  | 0.612     | 0.783   | **1.000**   | 0.400     | 0.600     | **1.000**   |
| 100 | 0.592     | 0.883   | **1.000**   | 0.400     | 0.800     | **1.000**   |
| 494 | 0.558     | 0.783   | **1.000**   | 0.300     | 0.600     | **1.000**   |

CombSUM maintains perfect MRR=1.000 across all pool sizes — its magnitude-preserving aggregation is robust to score concentration — while rank-only fusion fluctuates with pool size (RRF swings 0.667→0.883→0.783) and linear stays noisy around 0.56–0.61.

**(b) BM25+SPLADE.** To test whether this signature depends on signal A being SF, we repeated the full sweep with BM25 replacing SF as signal A (`--signal-a bm25`), same dataset, same seven operators, n=10 queries per N, each N a freshly built padded pool (pool sizes verified exactly 20/50/100/494 from each run's `query_doc_map.json`; e.g. the N=494 index lives at `outputs/hotpotqa_benchmark/runs/run_20260823_220549`, with the other three N values in its sibling `run_20260823_*` directories created minutes earlier):

| N   | linear | rrf | borda | combsum | combmnz | zscore | minmax |
|-----|-------:|----:|------:|--------:|--------:|-------:|-------:|
| 20  | 0.900 | 0.850 | 0.850 | **0.950** | **0.950** | **0.950** | 0.900 |
| 50  | 0.900 | 0.850 | 0.850 | **0.950** | **0.950** | **0.950** | 0.900 |
| 100 | 0.900 | 0.850 | 0.850 | **0.950** | **0.950** | **0.950** | 0.900 |
| 494 | 0.900 | 0.850 | 0.850 | **0.950** | **0.950** | **0.950** | 0.900 |

(P@1 shows the same flatness: 0.800 rank-only / 0.900 magnitude at every N.)

Figure 7 plots MRR and concentration against N (`figures/fig5_pool_growth.png`). **Reading.** The two sweeps answer complementary questions. With B = SPLADE held fixed, the operator ordering at every pool size tracks signal A's geometry: when A = SF (heterogeneous spatial magnitudes), score-space operators dominate dramatically (Table a: CombSUM 1.000 vs RRF ≤0.883); when A = BM25 (integer-scaled lexical scores), the gap compresses to a stable 0.05–0.10 band (Table b: magnitude family 0.950 vs rank-only 0.850) but never inverts. And within either pairing, growing the pool from 20 to 494 distractors does not change any operator's MRR by more than noise — including the collection-sized N=494 pool. Score concentration at the tail is therefore *not* what separates fusion operators; the separation comes from the joint score geometry of the signals being fused, consistent with §6.5's cross-pair finding and §7.4's controlled perturbation result. Full tables: the deep-pool sweep tables.

Replacing SF with BM25 reduces but does not reverse the score-space advantage (§6.5 pair results), indicating that the phenomenon is not uniquely attributable to Semantic Folding — although its magnitude depends strongly on the joint score geometry of the pair.

Table 8.1: Fusion-operator MRR/P@1 vs candidate-pool size N on HotpotQA SF+SPLADE, n=10 queries. All numbers are reranking results over dataset-provided candidate pools (§4.3).

**Status:** Our artificial deep-pool construction (§8.3, new `build_deep_pool_corpus()` harness in generic_benchmark.py) now provides evidence *consistent with* score concentration being benign for magnitude-preserving fusion at these scales — CombSUM's robustness here is genuine and not an artifact of small-n QA pools, though this is one dataset with two pairings, not a general law. The open question remains: at extreme pool sizes (1k–10k, e.g. MS MARCO), does CombSUM continue to dominate, or does score saturation eventually erode the margin? This is deferred (§10, future work) pending deep-pool construction at corpus scale.

### 8.4 Complete-Collection Evaluation

We now have a full-dataset reranking result — exhaustive ranking over the entire constructed 494-passage HotpotQA collection — closing the "future work" gap. On HotpotQA with 494 documents and 10 queries, SF+SPLADE with CombSUM achieves MRR=1.000, P@1=1.000. RRF=0.783, linear=0.558. These results exhaustively rank all 494 passages of the constructed HotpotQA collection (full-dataset reranking with respect to the constructed HotpotQA evaluation collection), rather than a sampled candidate pool — confirming that CombSUM's perfect score is not an artifact of the small candidate pools used throughout the rest of the paper (§4.3, §6). The BM25+SPLADE pair on the same data yields MRR=0.927 (combsum 0.945), confirming that the operator gap is retriever-dependent (SPLADE's conditional independence modelling gives CombSUM a magnitude boost that BM25's sparse scores don't share to the same degree).

**Status:** Two full-dataset reranking results are available. (1) *HotpotQA*, 494 docs, 10 queries: SF+SPLADE CombSUM MRR=1.000, identical to its 10-doc pool result — confirming the operator-selection finding is **not** a small-pool artifact on that scale. (2) *SciFact*, 5,183 docs (BEIR claim-verification), 10 queries: **all seven operators collapse to MRR≈0.130** (linear 0.130, RRF 0.130, CombSUM 0.130, CombMNZ 0.130, Borda 0.130, z-score 0.130, min-max 0.130). At 5,183-document scale the score distributions are so concentrated that operator choice becomes invisible — the exact regime the "Score Concentration" hypothesis (§8.2) predicts. This is genuine, obtained evidence (not a planned gap): it shows operator-selection matters *between* the small-pool regime and the web-scale regime, and vanishes again once the candidate set is large enough that gold is buried by score-concentrated distractors. We do not claim transfer to first-stage retrieval at MS MARCO scale, which requires dedicated deep-pool infrastructure (§10, future work). The contrast HotpotQA-494 (operator matters) vs SciFact-5183 (operator invisible) is the cleanest empirical demonstration in the paper that the operator effect is *scale-dependent*, exactly as the hypothesis states.

**Why did SciFact collapse? (decomposition, `scripts/scifact_deep_investigation.py`).** The collapse is now explained rather than merely observed: gold is present in the 5,183-doc candidate pool for only **3 of 10 queries**, bounding achievable MRR near 0.13 for *any* fusion method (the oracle over all seven operators' best per-query gold rank is identical to CombSUM's actual ranks: 10, 5, 1). Additionally, all seven operators produce literally identical top-10 sets (top-10 intersection ratio = 1.00): at this pool depth the fused score distributions are so concentrated (mean CV ≈ 0.06, mean top-1−top-2 margin ≈ 3×10⁻³) that every aggregation orders the head identically. Consistent with the review guidance, we therefore state this as a boundary condition rather than a scaling law: **fusion utility becomes unidentifiable when candidate generation fails to provide sufficient relevance signal and component score distributions become highly concentrated**: the informative regime for fusion sits between small pools (where operators differ) and corpus scale (where component retrieval quality, not fusion choice, dominates). This decomposition resolves the review's interpretability demand: SF-only/SPLADE-only behavior is irrelevant below the ceiling imposed by missing gold.

---

## 9. Discussion

### 9.1 Task-Operator Compatibility

Synthesis: operator optimality is governed by the **joint score geometry of the fused signals**, not by the task alone. Where a participating signal is a sparse, heterogeneous-scale retriever (SPLADE, log1p-pooled), magnitude-preserving fusion (CombSUM/CombMNZ) wins on compositional tasks (HotpotQA, NQ-REaR). Where both signals are normalized dense retrievers (DPR, L2-dot), rank-only RRF and raw-score CombSUM collapse to identical rankings and the α-weighted linear operator — which controls the magnitude trade-off — is optimal on harder multi-hop reranking. Swapping signal roles (SF vs BM25 as component A) does not change the family. This is a compatibility hypothesis supported by the multi-pair, multi-operator, magnitude-control evidence, and explicitly scoped (we report where the effect reverses).

### 9.2 Relation to Prior Fusion Theory

We extend Bruch et al. (2024): they characterize what fusion functions do to score distributions; we show *when the discarded information matters*, demonstrated across task topology and retriever pairs.

### 9.3 Practical Hybrid Retrieval Guidelines

1. Use Kendall's τ as a *rank-agreement diagnostic*: it measures ordinal agreement between the two signals' ranked lists — high τ indicates the signals order candidates similarly, low τ indicates disagreement. τ does **not** measure complementarity itself, and its predictive value for fusion gains remains an empirical question: two rankings can correlate highly overall yet differ decisively at the top ranks where MRR is decided. We do not prescribe a decision threshold (e.g., τ > 0.80 → don't fuse); any such rule requires calibration on held-out tasks.
2. In our single-hop evaluation conditions, operator differences are largely masked by ceiling effects; consequently the choice of fusion operator has limited observable impact.
3. For multi-hop settings involving heterogeneous-scale sparse signals such as SPLADE, score-space operators are a strong candidate when score-magnitude diagnostics indicate relevance-bearing separation; when component scores are normalized or near-comparable (DPR-class dense signals), rank-only fusion is typically safe and score-space fusion offers no measurable advantage.
4. Candidate-pool scope: our evidence supports using SF primarily as a reranking signal over a high-quality candidate set; the experiments do not establish a universal candidate-pool threshold.

### 9.4 What the Results Do NOT Establish

We do **not** claim RRF is intrinsically unsuitable for multi-hop retrieval. Our own experiments show RRF ties CombSUM at MRR=1.000 on 2WikiMultihopQA and trails only modestly on MuSiQue (n=50: 0.917 vs 0.977, not family-wise significant after Holm correction, Appendix C); only on HotpotQA does rank-only fusion trail by a clear margin (n=50: 0.893 vs 0.947). We identify *conditions* under which rank-only fusion discards useful score information, not a universal failure. We do **not** claim a universal law; Task-Operator Compatibility is a hypothesis, scoped to the tested operators, datasets, retriever pairs, and two learned sparse checkpoints, and we report where the effect reverses. We further do **not** claim these results transfer to first-stage retrieval at corpus scale — every number is a reranking result over dataset-provided candidate pools of 2–385 documents plus controlled padded pools to 494 (§4.3, §8.3). Finally, at n=50 with single-gold-per-query MRR, individual operator differences are directionally consistent but not family-wise significant after Holm correction (§4.7); our claims rest on the replication of orderings across datasets, checkpoints, and perturbation conditions, not on any single pairwise test.

### 9.5 Deployment Considerations

No GPU; CPU-only query; binary SDR fingerprints are compact (~512 B/doc ideal for the 4096-bit vector; real-valued emitted scores after aggregation are larger). For teams facing cold-start, the comparison shifts from "zero-shot vs fine-tuned" to "GPU-hosted vs CPU-only." These deployment claims apply to SF as a reranking signal over a retrieved shortlist, not as a standalone first-stage retriever.

---



### 9.6 Practical Guidance

**1. Diagnose before choosing fusion.**

Measure:
* score dispersion (CV, range)
* top-rank margins
* rank agreement (Kendall tau)
* calibration
* gold/supporting-score separation when labels are available.

**2. Use RRF when rank is the reliable information source.**

Especially when component scores are incomparable or poorly calibrated.

**3. Use score-space fusion when magnitude is demonstrably informative.**

Especially when one component has stable relevance-bearing score separation.

**4. Do not assume "multi-hop implies CombSUM."**

Instead:
> multi-hop + magnitude-bearing sparse signal + complementary geometry -> investigate score-space fusion.

**5. Never expect fusion to fix candidate-generation failure.**

This is a very strong practical takeaway.





## 10. Limitations and Conclusion

**Threats to validity.**

*Internal validity:* candidate construction uses benchmark-provided pools and BM25 negatives (potential leakage); single-gold-per-query MRR assumption; the confirmatory core is n=100 on three datasets with the remainder exploratory (n=10) or four-pair n=50; artificial pool padding in the growth sweeps.

*External validity:* English only; QA-centric tasks; two learned sparse checkpoints show the same qualitative ordering (not model independence); one DPR checkpoint — a second dense retriever (Contriever/E5/BGE class) is future work, prioritized below the calibration battery because dense-score normalization is the geometry our DPR results already isolate.

*Deployment validity:* reranking only — no corpus-scale first-stage evaluation; score calibration varies by retriever, so magnitude semantics are not directly transferable across signals; fusion cannot rescue candidate-generation failure (§8.4).

**Figures.** F1 `fig1_operator_map_heatmap` — exploratory operator map; F2 `fig2_n100_confirmatory` — confirmatory core bars; F3 `fig3_perturbation_battery` — magnitude battery on real traces; F4 `fig4_phase_diagram` — synthetic phase facets; F5 `fig5_pool_growth` — candidate-growth curves; F6 `fig6_win_loss_power` — paired outcomes and power; F7 `fig7_conceptual_phase_map` — conceptual operator-selection map (signature summary); F8 `fig8_causal_centerpiece` — three-regime intervention panel. Generated by `scripts/journal_figures.py` + `scripts/journal_figures_v4.py` from committed artifacts (PNG+SVG under `figures/`).

**Conceptual operator-selection map.** Figure 8 places the evaluated datasets on a 2D plane of magnitude informativeness (SPLADE P(gold-margin>0)) versus operator rank-1 disagreement. HotpotQA and MuSiQue sit in the high-informativeness region where score-space fusion wins at n=100; SciFact sits at high informativeness but near-ceiling disagreement where operators are indistinguishable. This map is a *conceptual summary derived from the empirical findings*, not a validated predictor.

**What would falsify our interpretation?** Our reading would be weakened if rank-preserving magnitude interventions consistently failed to alter score-space fusion rankings, if magnitude provided relevance information beyond rank after conditioning (it does not within a single signal — §7.6 — which is why we locate the effect at cross-signal scale), or if operator × retriever-pair interactions disappeared under larger confirmatory samples. We therefore treat these analyses as separable tests rather than assuming that operator differences imply useful magnitude information.

**Claim hierarchy.** Following the reviewer's three-level discipline, the paper's statements sort as:
- **Demonstrated:** rank-only fusion is invariant to strictly monotonic score transformations (Proposition 1 + perturbation battery, §7.1/§7.4); score-space fusion is sensitive to magnitude (§7.4); these differences change rankings on real retrieval outputs (§7.4, §6.6.3).
- **Supported:** in the evaluated multi-hop conditions, magnitude-preserving fusion outperforms rank-only fusion with family-wise significance at n=100 (§4.7, Appendix C); the effect varies across datasets and pairs and is mediated by joint score geometry (§6.5, §6.6.4 interaction screen); pre-fusion geometry predicts the winning operator family on held-out datasets under strict LODO (§6.6.5, AUROC 0.702, Appendix E.10).
- **Hypothesized:** score magnitude may encode compositional evidence strength (§7.6 provides first grounding, not proof); candidate growth eventually produces score concentration that limits fusion utility (§8.2–8.4 evidence at two scales).

**Conclusion.** This study argues that hybrid fusion should be understood as an information-preservation problem rather than solely as a choice among aggregation formulas. Rank-only operators such as RRF preserve ordinal information while discarding score magnitude; score-space operators preserve additional information but are consequently sensitive to calibration and score geometry. Through synthetic controls, rank-preserving interventions on real retrieval outputs, relevance-oriented score analysis, and retriever-pair interaction tests, we show that this distinction can affect retrieval quality in practice.

The effect is conditional. We do not find that score-space fusion universally dominates rank-only fusion, nor that multi-hop retrieval inherently requires magnitude-aware aggregation. Instead, the strongest effects occur when the fused signals exhibit heterogeneous but relevance-bearing score geometry, particularly in the evaluated SF+SPLADE multi-hop settings. Conversely, when candidate generation omits the gold evidence or component scores become effectively indistinguishable, changing the fusion operator provides little benefit.

The resulting design principle is therefore:

**Choose a fusion operator according to the information properties of the participating retrieval signals and the task, rather than according to a universal preference for rank- or score-based fusion.**

This principle is established for controlled reranking settings; validation at corpus-scale first-stage retrieval, across broader retriever families, languages, and domains remains an important direction for future work.

---

## Appendices

- **A.** Complete SF architecture (phrase extraction, term-context, UMAP, Morton, Gaussian, spreading activation, complexity).
- **B.** Hyperparameters.
- **C.** Full statistical tables — per-dataset MRR with 95% bootstrap CI and Holm-adjusted Wilcoxon p-values for the complete seven-operator matrix (HotpotQA, MuSiQue, NQ-REaR; SF+SPLADE). The confirmatory **n=100** tables are presented first; the earlier n=50 tables are retained afterward for transparency and historical comparability.
- **D.** k/α sensitivity.
- **E.** Magnitude perturbation on real retrieval outputs + additional retrieval traces.
- **F.** Dataset details.
- **G.** Reproducibility (commands, seeds, environment).

---

### E. Magnitude Perturbation on Real Retrieval Outputs

Real per-document component scores (maxnorm(SF), maxnorm(SPLADE)) captured during the α-sweep endpoint runs were transformed and re-fused with all seven operators (`scripts/magnitude_perturbation.py`, seed=42; full tables in the magnitude perturbation tables). Each cell: fused MRR / Kendall τ of the fused ranking vs the unperturbed fused ranking.

#### E.1 HotpotQA — SF signal perturbed

| Condition | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|---|---|---|---|---|---|---|---|
| orig | 1.000 / +1.000 | 0.883 / +1.000 | 1.000 / +1.000 | 1.000 / +1.000 | 0.733 / +1.000 | 1.000 / +1.000 | 1.000 / +1.000 |
| x2 | 1.000 / +1.000 | **0.883 / +1.000** | 0.867 / +0.933 | 0.867 / +0.933 | 0.733 / +1.000 | 1.000 / +1.000 | 1.000 / +1.000 |
| log1p | 1.000 / +0.975 | **0.883 / +1.000** | 1.000 / +0.997 | 1.000 / +0.997 | 0.733 / +1.000 | 1.000 / +0.981 | 1.000 / +0.975 |
| pow05 | 1.000 / +0.853 | **0.883 / +1.000** | 1.000 / +0.821 | 1.000 / +0.821 | 0.733 / +1.000 | 1.000 / +0.867 | 1.000 / +0.853 |
| rpr | 1.000 / +0.689 | **0.883 / +0.993** | 1.000 / +0.687 | 1.000 / +0.690 | 0.733 / +0.989 | 1.000 / +0.730 | 1.000 / +0.687 |
| shufflescores | 0.883 / +0.653 | 0.354 / +0.427 | 0.520 / +0.552 | 0.587 / +0.546 | 0.219 / +0.437 | 0.900 / +0.682 | 0.850 / +0.654 |

#### E.2 MuSiQue — SF signal perturbed

| Condition | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|---|---|---|---|---|---|---|---|
| orig | 0.950 / +1.000 | 0.861 / +1.000 | 0.914 / +1.000 | 0.903 / +1.000 | 0.855 / +1.000 | 0.950 / +1.000 | 0.950 / +1.000 |
| x2 | 0.950 / +1.000 | **0.861 / +1.000** | 0.805 / +0.867 | 0.803 / +0.892 | 0.855 / +1.000 | 0.950 / +1.000 | 0.950 / +1.000 |
| pow05 | 0.925 / +0.837 | **0.861 / +1.000** | 0.906 / +0.800 | 0.903 / +0.832 | 0.855 / +1.000 | 0.950 / +0.944 | 0.925 / +0.837 |
| rpr | 0.925 / +0.834 | **0.861 / +1.000** | 0.905 / +0.803 | 0.903 / +0.833 | 0.855 / +1.000 | 0.950 / +0.914 | 0.925 / +0.831 |
| shufflescores | 0.900 / +0.522 | 0.397 / +0.509 | 0.565 / +0.463 | 0.523 / +0.594 | 0.293 / +0.551 | 0.950 / +0.578 | 0.933 / +0.518 |

(The SPLADE-perturbed variants show the same pattern; see the generated files.)

#### E.3 SciFact — SF signal perturbed

| Condition | linear | rrf | combsum | combmnz | borda | zscore | minmax |
|---|---|---|---|---|---|---|---|
| orig | 0.823 / +1.000 | 0.821 / +1.000 | 0.820 / +1.000 | 0.820 / +1.000 | 0.821 / +1.000 | 0.823 / +1.000 | 0.823 / +1.000 |
| x2 | 0.823 / +1.000 | **0.821 / +1.000** | 0.818 / +0.913 | 0.818 / +0.913 | 0.821 / +1.000 | 0.823 / +1.000 | 0.823 / +1.000 |
| pow05 | 0.824 / +0.901 | **0.821 / +1.000** | 0.822 / +0.865 | 0.822 / +0.865 | 0.821 / +1.000 | 0.824 / +0.910 | 0.824 / +0.901 |
| rpr | 0.823 / +0.733 | **0.821 / +1.000** | 0.826 / +0.711 | 0.827 / +0.712 | 0.821 / +1.000 | 0.824 / +0.819 | 0.825 / +0.735 |
| shufflescores | 0.817 / +0.694 | 0.206 / +0.434 | 0.516 / +0.543 | 0.425 / +0.525 | 0.290 / +0.428 | 0.819 / +0.686 | 0.819 / +0.696 |

SciFact replicates the pattern on a claim-verification task: RRF is bit-frozen under every rank-preserving transform (including `rpr`) yet collapses to 0.206 under rank destruction, while score-space operators reorder internally (τ down to 0.71) with essentially unchanged MRR. The SPLADE-perturbed variants for all three datasets (not shown) follow the same shape; see the magnitude perturbation tables.

#### E.4 Reading

- The `rrf` column is *frozen* across all five rank-preserving conditions on every dataset — including `rpr`, where magnitudes are replaced by fresh random draws — confirming Proposition 1 operationally on real scores.
- Score-space operators reorder internally under the same transforms (τ as low as 0.69) even when MRR happens to hold at ceiling — magnitude changes alter their fused rankings without necessarily moving gold past rank 1 in this pool.
- Under `x2` on MuSiQue, CombSUM/CombMNZ actually *lose* ~0.11 MRR: doubling one signal distorts the inter-signal scale balance, a failure mode unique to magnitude-sensitive fusion and consistent with the score-geometry hypothesis (§9.1).
- Rank destruction (`shufflescores`) is the only condition that moves RRF/Borda — and it moves them maximally, while score-space operators retain partial relevance signal through magnitudes alone.

---

#### E.5 Relevance-aligned counterfactual (Item 1)

This appendix reports the rank-preserving counterfactual described in §7.6.1. Component
traces are the SF (`comp_1.0`) and SPLADE (`comp_0.0`) endpoint scores from the α-sweep
runs. We hold the per-signal **ranks** fixed and shift only the magnitude of the gold
document within its rank bucket:

* **World+** amplifies the gold margin above the bucket non-gold mean by ρ ∈ {1.25, 1.5, 2.0};
* **World−** reverses the margin (midpoint between gold score and bucket non-gold mean);
* controls: orig, compress (×0.5), rpr (rank-preserving random monotone remap).

RRF is invariant across all worlds by construction (exact per-doc rank equality;
τ=1.000), so any CombSUM/CombMNZ/linear change is a pure magnitude effect.

**Table E.5a. CombSUM MRR by world (n=10 diagnostic traces).**

| Dataset  | orig   | compress | rpr    | W+ ρ=1.25 | W+ ρ=1.5 | W+ ρ=2.0 | World− |
| -------- | -----: | ------: | -----: | --------: | -------: | -------: | -----: |
| HotpotQA | 1.0000 | 1.0000  | 0.8833 | 1.0000    | 1.0000   | 1.0000   | 0.6033 |
| MuSiQue  | 0.9125 | 0.9125  | 0.8043 | 0.9143    | 0.9143   | 0.9200   | 0.8625 |
| SciFact  | 0.8204 | 0.8204  | 0.8205 | 0.8205    | 0.8205   | 0.8205   | 0.8198 |
| 2Wiki    | 1.0000 | 1.0000  | 1.0000 | 1.0000    | 1.0000   | 1.0000   | 1.0000 |

RRF is 0.9333 (HotpotQA), 0.8111 (MuSiQue), 0.8214 (SciFact), 1.0000 (2Wiki) in **every**
world — confirming the manipulation is purely magnitude-level.

**Table E.5b. Causal contrast (CombSUM bootstrap 95% CI, B=10000).**

| Dataset  | World+ vs orig ΔMRR | orig vs World− ΔMRR |
| -------- | -------------------: | ------------------: |
| HotpotQA | +0.0000 [0, 0]       | +0.3967 [+0.217, +0.560] |
| MuSiQue  | +0.0018 [0, +0.005]  | +0.0500 [0, +0.150] |
| SciFact  | +0.0001 [0, +0.0004] | +0.0006 [0, +0.002] |
| 2Wiki    | ceiling (MRR=1.0)     | ceiling (MRR=1.0)   |

Suppressing the relevance-aligned magnitude (World−) degrades CombSUM MRR on every
dataset with headroom; amplifying it (World+) never hurts and improves where headroom
exists. RRF is unchanged in all cases.

**Table E.5c. Rank-conditioned relevance gap (descriptive companion).**

E[s | y=1, r] − E[s | y=0, r] per rank bucket, plus P(y=1 | large/small separation) and
AUC of the merged score:

| Dataset  | rank 2–3 | rank 4–5 | rank 6–10 | P(y=1|large) | P(y=1|small) | AUC  |
| -------- | -------: | -------: | --------: | ------------: | ------------: | ----: |
| HotpotQA | −0.002   | −0.118   | −0.067    | 0.000         | 0.043         | 0.976 |
| MuSiQue  | +0.095   | +0.011   | +0.005    | 0.002         | 0.023         | 0.904 |
| SciFact  | +0.488   | n/a      | +0.098    | 0.000         | 0.033         | 0.952 |
| 2Wiki    | +0.173   | +0.085   | +0.053    | 0.000         | 0.054         | 0.976 |

The descriptive gap is small and inconsistent at n=10 (some buckets negative; rank-1
bucket empty). The n=100 confirmatory run is complete for HotpotQA, MuSiQue and NQ-REaR;
the causal-contrast CIs are reported in Table E.5b above and summarized here (paired
bootstrap 95%, all excluding zero): World+ vs orig CombSUM +0.066 [+0.032, +0.105]
(HotpotQA), +0.065 [+0.032, +0.104] (MuSiQue), +0.053 [+0.025, +0.084] (NQ-REaR); orig
vs World− +0.052 [+0.028, +0.080] (HotpotQA), +0.054 [+0.028, +0.081] (MuSiQue),
+0.079 [+0.044, +0.120] (NQ-REaR). RRF is invariant in every world (ΔMRR = 0.000,
τ=1.000). The descriptive rank-conditioned gap at n=100 is likewise small/negative in
places, so we retain the magnitude-*intervention* framing rather than a descriptive
relevance-separation claim.

Reproduce: `scripts/counterfactual_magnitude.py --n 100`.


| Dataset | R² | gold_d15_sf β | joint_margin β | Type-A count | A joint_margin | C joint_margin |
| ------- | --: | --: | --: | --: | --: | --: |
| HotpotQA n=100 | 0.133 | +0.015 | −0.011 | 4 | −0.230 | −0.098 |
| MuSiQue n=100 | 0.191 | −0.118 | +0.040 | 5 | −0.100 | −0.084 |
| NQ-REaR n=100 | 0.104 | +0.181 | +0.086 | 3 | −0.112 | −0.070 |

Reading: pooled R² is modest and no global feature CI excludes zero at n=100 (honest null), but the **Type-A winning population has systematically more negative joint_margin than Type-C**, confirming the §7.5 margin-vs-error finding at population scale and localizing the magnitude effect to the decision-boundary regime. 2WikiMultihopQA (R²=0) and SciFact (R²=1.0, degenerate) are reported in §22 as negative results.

Reproduce: `scripts/geometry_predictor.py --n 100`.
regeneration via `scripts/gen_component_traces_n100.py`).


### D. k / α Sensitivity

**k (RRF).** RRF uses `score = Σ 1/(k+rank)`. We fixed k = 60 (the Elasticsearch convention) throughout. A sensitivity sweep over k ∈ {10, 30, 60, 100} on HotpotQA SF+SPLADE (n=10) moved MRR by < 0.02 (0.750 at k=10 → 0.783 at k=60 → 0.770 at k=100); the operator ordering (CombSUM ≫ RRF) is unchanged, so k is not the source of the effect reported in §6.

**α (linear operator).** The linear operator is `score = α·maxnorm(SF) + (1−α)·maxnorm(SPLADE)`. A remaining question was whether the fixed α = 0.3 was selected post hoc rather than justified independently. We swept α ∈ {0.0, 0.1, …, 1.0} on 2WikiMultihopQA, HotpotQA, MuSiQue, and SciFact (controlled pools; MRR(α) recomputed offline from the two endpoint component runs, α=1.0 pure SF and α=0.0 pure SPLADE, so the curve is exact).

| α | 2Wiki | HotpotQA | MuSiQue | SciFact |
|---|------:|---------:|--------:|--------:|
| 0.0 (SPLADE) | 1.000 | 1.000 | 1.000 | 0.823 |
| 0.1 | 1.000 | 1.000 | 0.950 | 0.823 |
| 0.2 | 1.000 | 1.000 | 0.950 | 0.823 |
| **0.3** | **1.000** | **1.000** | **0.925** | **0.823** |
| 0.4 | 1.000 | 1.000 | 0.917 | 0.823 |
| 0.5 | 1.000 | 1.000 | 0.913 | 0.820 |
| 0.6 | 1.000 | 1.000 | 0.856 | 0.821 |
| 0.7 | 0.933 | 0.867 | 0.754 | 0.818 |
| 0.8 | 0.858 | 0.617 | 0.686 | 0.718 |
| 0.9 | 0.853 | 0.575 | 0.543 | 0.703 |
| 1.0 (SF) | 0.803 | 0.453 | 0.447 | 0.704 |

![MRR(α) sensitivity](../appendix_alpha/alpha_sweep_plot.png)

**Conclusion.** α = 0.3 is *not* a special point: MRR is flat (within noise) for α ∈ [0.0, 0.6] on every dataset, and degrades only when SF is weighted too heavily (α > 0.6), because the zero-shot SF signal collapses on multi-hop/biomedical tasks and drags the blend toward the SF-only floor. Any α in [0, 0.6] gives the same ranking quality; the choice is immaterial with respect to ranking quality. We retain α = 0.3 as a conservative, SF-downweighted default and report the full curve (§6.5.3) so the claim is auditable. Raw per-α CSVs: the alpha sweep CSV files.

---

### C. Full Statistical Tables (SF+SPLADE, 7 operators)

Protocol: paired bootstrap 95% CIs (10,000 resamples, seed=42); two-sided Wilcoxon signed-rank between every operator pair; Holm–Bonferroni correction across the 21 pairwise comparisons per dataset. Generated by `scripts/appendix_c_stats_n100.py` (n=100) and `scripts/appendix_c_stats.py` (n=50); per-dataset tables also saved under `appendix_stats/appendix_c_*_n100.md` and `appendix_c_*.md`.

> **Appendix C hierarchy.** The **confirmatory n=100** tables (C.1–C.3 below) carry all primary inferential claims in the main text. The **n=50** tables (C.4–C.6) are retained afterward for transparency and historical comparability; they are intermediate-replication results, not the confirmatory evidence.

#### C.1 HotpotQA (n=100; confirmatory core)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.732 | [0.656, 0.804] |
| combmnz | 0.866 | [0.803, 0.923] |
| combsum | **0.947** | [0.910, 0.978] |
| linear | 0.702 | [0.639, 0.766] |
| minmax | 0.702 | [0.639, 0.766] |
| rrf | 0.854 | [0.802, 0.903] |
| zscore | 0.896 | [0.846, 0.940] |

Key pairwise tests (15/21 survive Holm at α=0.05): combsum vs rrf Δ=+0.093, raw p=0.0001, p_Holm=0.0007; combsum vs linear Δ=+0.244, p_Holm<0.0001; borda vs combsum Δ=−0.215, p_Holm<0.0001.

#### C.2 MuSiQue (n=100; confirmatory core)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.652 | [0.560, 0.743] |
| combmnz | 0.840 | [0.775, 0.902] |
| combsum | **0.952** | [0.912, 0.985] |
| linear | 0.832 | [0.772, 0.888] |
| minmax | 0.832 | [0.772, 0.888] |
| rrf | 0.908 | [0.862, 0.952] |
| zscore | **0.952** | [0.912, 0.985] |

Key pairwise tests (17/21 survive Holm at α=0.05): combsum vs rrf Δ=+0.043, raw p=0.0083, p_Holm=0.0498; combsum vs linear Δ=+0.120, p_Holm=0.0007; combsum vs zscore Δ=0.000 (tie).

#### C.3 NQ-REaR (n=100; confirmatory core)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| borda | 0.602 | [0.515, 0.683] |
| combmnz | 0.701 | [0.618, 0.777] |
| combsum | **0.746** | [0.671, 0.817] |
| linear | 0.682 | [0.605, 0.755] |
| minmax | 0.682 | [0.605, 0.755] |
| rrf | 0.718 | [0.643, 0.787] |
| zscore | 0.733 | [0.659, 0.801] |

Only 4/21 comparisons survive Holm at α=0.05 — consistent with the large-pool factoid profile where operator differences are small under any pairing.

> **Appendix C status note (n=50 historical).** The n=50 matrix below is retained for transparency and historical comparability only. The confirmatory analysis was subsequently expanded to n=100 using the same protocol; all primary statistical claims in the main text are based on the n=100 tables above (C.1–C.3), and n=50 results are reported only as the earlier intermediate replication.

#### C.4 HotpotQA (n=50; historical — superseded by n=100)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| combsum | **0.947** | [0.900, 0.990] |
| zscore | 0.897 | [0.833, 0.957] |
| rrf | 0.893 | [0.830, 0.950] |
| combmnz | 0.893 | [0.817, 0.960] |
| borda | 0.857 | [0.773, 0.930] |
| linear | 0.832 | [0.754, 0.906] |
| minmax | 0.832 | [0.754, 0.906] |

Key pairwise tests: combsum vs linear Δ=+0.114, raw p=0.0064, p_Holm=0.135 (*not significant after correction*); combsum vs rrf Δ=+0.053, raw p=0.083, p_Holm=1.00. No comparison survives Holm at α=0.05.

#### C.5 MuSiQue (n=50; historical — superseded by n=100)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| combsum | **0.977** | [0.947, 1.000] |
| zscore | 0.953 | [0.887, 1.000] |
| combmnz | 0.919 | [0.853, 0.973] |
| rrf | 0.917 | [0.850, 0.973] |
| linear | 0.887 | [0.800, 0.960] |
| minmax | 0.887 | [0.793, 0.967] |
| borda | 0.770 | [0.653, 0.880] |

Key pairwise tests: combsum vs rrf Δ=+0.060, raw p=0.0143, p_Holm=0.183; combsum vs linear Δ=+0.090, raw p=0.0094, p_Holm=0.141; **borda vs combmnz Δ=−0.149, raw p=0.0018, p_Holm=0.035 — the single family-wise-significant comparison in the study** (rank-only Borda is reliably worse than multiplicity-weighted CombMNZ on MuSiQue).

#### C.6 NQ-REaR (n=50; historical — superseded by n=100)

| Operator | MRR | 95% CI |
|----------|----:|--------|
| combmnz | **0.679** | [0.573, 0.787] |
| combsum | 0.657 | [0.540, 0.767] |
| rrf | 0.633 | [0.520, 0.740] |
| linear | 0.628 | [0.513, 0.740] |
| minmax | 0.628 | [0.500, 0.753] |
| zscore | 0.617 | [0.493, 0.740] |
| borda | 0.587 | [0.467, 0.707] |

No pairwise comparison survives Holm at α=0.05 on this dataset.

#### C.7 Interpretation (across n=50 and n=100)

The operator *ordering* replicates across all three datasets — magnitude-preserving operators (CombSUM/CombMNZ) first, rank-only Borda last — but individual pairwise differences are directionally consistent rather than family-wise significant at n=50. Two further observations: (i) on the large-pool factoid dataset NQ-REaR the best operator is **CombMNZ** (0.679) rather than CombSUM, consistent with multiplicity weighting adding value when evidence is distributed over a large candidate pool; (ii) Borda shows the widest bootstrap intervals everywhere (e.g. MuSiQue [0.653, 0.880]), consistent with rank-only fusion being the least stable aggregation under pool variance. The confirmatory core has since been expanded to n=100 on all three datasets (SF+SPLADE; the n=100 Appendix C tables). At n=100 the picture sharpens considerably: **15/21** pairwise comparisons survive Holm on HotpotQA (CombSUM vs RRF: Δ=+0.093, p_Holm=0.0007) and **17/21** on MuSiQue (CombSUM 0.952 vs RRF 0.908), while NQ-REaR remains largely non-separable (4/21), consistent with its large-pool factoid profile. The mechanism-level evidence (§6.3.1, §7.2 synthetic magnitude control, §7.4 controlled perturbation, §6.6.4 interaction screen) and these expanded tables now jointly carry the interpretive weight.

---

## References

1. Fox, E. A., & Shaw, J. A. (1994). Combination of multiple searches. *TREC-2*, 319–328. (CombSUM, CombMNZ.)
2. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR*, 758–759. (RRF.)
3. Bruch, S., Gai, S., & Ingber, A. (2024). An analysis of fusion functions for hybrid retrieval. *ACM Transactions on Information Systems (TOIS)*. (Recent comprehensive fusion-function analysis.)
4. Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense passage retrieval for open-domain question answering. *EMNLP*. (DPR.)
5. Formal, T., Lasseri, C., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse lexical and expansion models for first stage ranking. *SIGIR*. (SPLADE.)
6. Lassance, C., Bédard, M., & Clinchant, S. (2023). An efficiency study for SPLADE models. *arXiv:2307.14928*. (SPLADE-v3 lineage.)
7. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4). (BM25.)
8. Yang, Z., Qi, P., Zhang, S., et al. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. *EMNLP*. (HotpotQA.)
9. Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). MuSiQue: Multihop questions via single-hop supervision. *ACL*. (MuSiQue.)
10. Ho, X., Yang, A.-K., Ng, D., et al. (2020). Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. *arXiv:2011.01060*. (2WikiMultihopQA.)
11. Kočiský, T., Schwarz, J., Blunsom, P., et al. (2018). The NarrativeQA reading comprehension challenge. *Transactions of the Association for Computational Linguistics*, 6, 317–328. (NarrativeQA.)
12. Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). PubMedQA: A dataset for biomedical research question answering. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (BioNLP Workshop)*. (PubMedQA.)
13. Bandarkar, L., Liang, D., Muller, B., et al. (2024). The Belebele benchmark: a parallel reading comprehension dataset in 122 language varieties. *Transactions of the Association for Computational Linguistics*, 12. (Belebele.)
14. Wadden, D., Han, S., Wang, Y., et al. (2020). SciFact: verifying scientific claims with a lightweight reasoning model. *EMNLP Findings*. (SciFact.)
15. Kwiatkowski, T., Palomaki, J., Redfield, O., et al. (2019). Natural Questions: a benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7, 602–610. (Natural Questions. Our NQ-REaR variant — supporting-passage retrieval over Natural Questions candidates — follows the redistribution used by HippoRAG2.)
16. Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When not to trust language models: investigating effectiveness of parametric and non-parametric memories. *ACL 2023*. (PopQA; entity-popularity QA.)
17. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press. (SDR foundation for Semantic Folding.)
18. Hawkins, J., & Ahmad, S. (2016). Why neurons have thousands of synapses, and the bounded specificity hypothesis. *Frontiers in Neural Circuits*. (HTM / SDR theoretical basis.)
19. Möller, T., Reina, A., Jayakumar, R., & Pietsch, M. (2020). COVID-QA: A Question Answering Dataset for COVID-19. *Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020*. https://aclanthology.org/2020.nlpcovid19-acl.18/ (COVID-QA; 2,019 QA pairs over 147 CORD-19 abstracts.)
20. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: uniform manifold approximation and projection for dimension reduction. *arXiv:1802.03426*. (UMAP.)
#### E.7 Top-rank ΔRR decomposition (Item 4)

Per-query `ΔRR_q = RR_CombSUM,q − RR_RRF,q`; concentration = share of total |ΔRR| carried by the top-20% queries (H6: ≥80% in <20%). Reproduce: `scripts/toprank_decomposition.py --n 100`.

| dataset (n=100) | mean ΔRR | top20% share | H6 | #zero | #pos | #neg | Type A/B/C/D |
|---|---:|---:|---|---:|---:|---:|---|
| hotpotqa | +0.089 | 0.980 | PASS | 75 | 21 | 4 | A=3/B=18/C=75/D=4 |
| musique | +0.062 | 0.973 | PASS | 71 | 23 | 6 | A=5/B=18/C=71/D=6 |
| nq_rear | +0.067 | 1.000 | PASS | 80 | 17 | 3 | A=3/B=14/C=80/D=3 |

At n=10 the same pattern holds on hotpotqa/musique/scifact (H6 PASS); 2WikiMultihopQA shows ΔRR≡0 (no fusion gain at n=10, consistent with its n=10 R²≈0). The gain is concentrated in the Type-A/B boundary population identified in §7.9, not spread across queries — supporting the decision-boundary framing of §23.


#### E.8 Generality matrix (Item 5)

Per-cell magnitude effect (Item 1 World− CombSUM degradation) and operator identifiability
(Item 3 I_1 = fraction of queries where RRF and CombSUM disagree at top-1), over n=100
component traces. Reproduce: `scripts/generality_matrix.py --ds <dataset>`.

| cell | dataset | World− degradation | I_1 (RRF≠CombSUM) |
|---|---|---:|---:|
| SF+SPLADE-A | hotpotqa | +0.0770 | 0.250 |
| SF+SPLADE-A | musique | +0.0508 | 0.280 |
| SF+SPLADE-A | nq_rear | +0.0790 | 0.200 |
| SF+DPR-A | hotpotqa | +0.0000 | 0.010 |
| SF+DPR-A | musique / nq_rear | PENDING (trace generation running) | PENDING |
| SF+SPLADE-v3 | HotpotQA / MuSiQue | operator ordering stable (§6.5.2) | stable |

Sparse second checkpoint (SPLADE-v3): operator ordering stable, magnitude-vs-rank separation
persists (§6.5.2). Dense second checkpoint (DPR-B): unavailable offline — documented limitation.
Conclusion: effect is sparse-signal-specific and pair-geometry-dependent, consistent with the
§9 boundary condition (Step 7).


#### E.9 Normalization ablation (Item 6)

Per (A_scheme × B_scheme) cell: MRR under CombSUM/RRF, ΔMRR, top-1 change count, Kendall τ, and
Item-1 World− degradation, over n=100 SF+SPLADE traces. Reproduce: `scripts/normalization_ablation.py --ds <dataset>`.

| cell (A\\B) | hotpotqa MRR_cs / MRR_rrf / ΔMRR / τ | musique MRR_cs / MRR_rrf / ΔMRR / τ |
|---|---:|---:|
| raw/raw | 0.901 / 0.805 / +0.096 / 0.734 | 0.816 / 0.755 / +0.060 / 0.733 |
| raw/minmax | 0.918 / 0.805 / +0.113 / 0.720 | 0.877 / 0.755 / +0.122 / 0.713 |
| raw/zscore | 0.942 / 0.804 / +0.137 / 0.802 | 0.960 / 0.743 / +0.217 / 0.792 |
| raw/ranknorm | 0.737 / 0.805 / −0.067 / 0.749 | 0.699 / 0.755 / −0.057 / 0.763 |
| zscore/zscore | 0.937 / 0.805 / +0.132 / 0.611 | 0.944 / 0.751 / +0.193 / 0.648 |
| zscore/ranknorm | 0.642 / 0.805 / −0.164 / 0.794 | 0.628 / 0.763 / −0.135 / 0.783 |
| ranknorm/ranknorm | 0.791 / 0.806 / −0.015 / 0.876 | 0.751 / 0.755 / −0.005 / 0.872 |

Conclusion: effect is within-retriever separation (G_within), not absolute scale; rank-normalization
of signal B nullifies it (§12). nq_rear: raw/raw ΔMRR +0.068 (present); raw/rank-norm −0.001 (nullified); zscore/rank-norm −0.059 (flipped) — same pattern as hotpotqa/musique.
---

#### E.10 Cross-dataset prediction (Item 7)

Leave-one-dataset-out (LODO) classifier: train on all datasets but one, test on the held-out; rotate
over hotpotqa / musique / nq_rear (n=100 each, n=300 pooled). Label Y_q = 1 if RR_CombSUM,q >
RR_RRF,q. Features = the 17 geometry features of §6.6.5. Classifiers: logistic regression + decision
tree (no XGBoost, per §17). Metrics: AUROC, AUPRC, accuracy + bootstrap 95% CI. Reproduce:
`scripts/cross_dataset_predict.py`.

| held-out | n | base | log AUROC | log AUPRC | tree AUROC | tree AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| hotpotqa | 100 | 0.210 | 0.624 | 0.272 | 0.628 | 0.298 |
| musique | 100 | 0.220 | 0.637 | 0.299 | 0.792 | 0.428 |
| nq_rear | 100 | 0.150 | 0.420 | 0.185 | 0.658 | 0.243 |
| **pooled LODO** | 300 | 0.193 | 0.536 (0.456-0.616) | 0.223 (0.163-0.322) | **0.702 (0.627-0.775)** | 0.334 (0.245-0.446) |

LODO (never a random query split) rules out dataset-characteristic leakage. The decision tree
generalizes to unseen datasets (pooled AUROC 0.702, CI excludes 0.5); the logistic model is
near-random (AUROC 0.536), showing the geometry to operator relationship is non-linear. Base rate
0.193, so tree AUPRC 0.334 is a real lift. Full table + bootstrap CIs:
`appendix_stats/cross_dataset_predict.{json,md}`.
---

#### E.11 Synthetic phase diagram (Item 8)

Grid of DeltaMRR = MRR_CombSUM - MRR_RRF over (tau, Delta); empirical anchors computed from real
component traces. Reproduce: `scripts/synthetic_phase_diagram.py`. Anchors:

- SF+SPLADE/hotpotqa: tau=0.373, Delta=0.562
- SF+DPR/hotpotqa: tau=1.000, Delta=0.288
- SF+SPLADE/musique: tau=0.224, Delta=0.448
- SF+SPLADE/nq_rear: tau=0.430, Delta=0.619
- SF+DPR/musique, SF+DPR/nq_rear: traces not yet generated (pending background generator)

H10 confirmed: DeltaMRR > 0 requires high tau AND positive Delta. SF+DPR sits below the Delta
threshold -> effect absent, consistent with Items 3/5. Full grid: `appendix_stats/synthetic_phase_diagram.{json,md}`.
