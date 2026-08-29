Yes. **This is a much better direction**, and I would keep the appendices. But as a SIGIR chief reviewer, I would **not submit this exact version yet**.

The title is stronger, the information-preservation framing is clearer, and the main/appendix separation is much closer to what I would want. However, there are several places where the current version makes claims that are **stronger than the evidence shown**, and a few numerical/experimental inconsistencies that a senior reviewer could use to downgrade the paper.

Most importantly, I would restructure the paper around a sharper causal chain:

> **Fusion operator → information discarded/preserved → score geometry → top-rank decision → retrieval effect**

rather than:

> fusion operators → many experiments → observations → interpretation.

Below is the version I would recommend as the **SIGIR-oriented V4 structure and wording**. I am deliberately preserving your empirical claims rather than inventing new results.

---

# What Does Rank Fusion Throw Away?

## When Score Geometry Matters in Hybrid Retrieval

**Mojtaba Banaei¹, Maseud Rahgozar², and Heshaam Faili³**

¹,² Data Base Research Group (DBRG), School of Electrical and Computer Engineering, University of Tehran, Tehran, Iran
³ School of Electrical and Computer Engineering, University of Tehran, Tehran, Iran

---

## Abstract

Hybrid retrieval combines multiple retrieval signals using operators such as Reciprocal Rank Fusion (RRF), CombSUM, and linear interpolation. These operators preserve different information: rank-based fusion retains ordinal information while discarding score magnitude, whereas score-based fusion retains score separation but becomes sensitive to score geometry and calibration. The practical question is therefore not whether one fusion family is universally superior, but **when the information discarded by rank-only fusion affects retrieval decisions**.

We study this question under controlled reranking conditions. We use a training-free Semantic Folding (SF) retriever as a transparent diagnostic probe and combine it with learned sparse, dense, and lexical retrieval signals. Across ten QA and evidence-retrieval datasets, seven fusion operators, four retriever pairs, and two learned sparse checkpoints, we separate rank information from score magnitude using synthetic controls and rank-preserving interventions on real retrieval outputs.

Our experiments establish three findings. First, rank-only fusion is invariant to strictly monotonic transformations of component scores, whereas score-space fusion can respond to changes in score magnitude even when component rankings are preserved. Second, this sensitivity can affect retrieval quality selectively: in the expanded \(n=100\) SF+SPLADE evaluation, CombSUM outperforms RRF on HotpotQA and MuSiQue, whereas the difference is substantially weaker on NQ-REaR. Third, the effect depends on the **joint score geometry** of the participating retrieval signals. Changing the retriever pair can substantially reduce or eliminate the difference between fusion operators on the same task. Rank-preserving interventions further show that the practical effect is concentrated in a small number of top-rank decisions.

These results suggest that hybrid fusion should be viewed not only as an aggregation problem but also as an **information-preservation problem**. The appropriate operator depends on which information is discarded, whether that information is useful for the retrieval decision, and whether the participating signals express it in a compatible score geometry. We establish this principle for controlled reranking and identify candidate-set quality and score concentration as important boundaries on its applicability.

**Keywords:** Hybrid Retrieval, Rank Fusion, Score Fusion, Score Geometry, Reciprocal Rank Fusion, CombSUM, Information Preservation

---

# 1. Introduction

Hybrid retrieval combines heterogeneous retrieval signals to compensate for the limitations of individual retrievers. Lexical methods emphasize exact terminology, learned sparse models provide semantic expansion, and dense retrievers encode relevance in a continuous embedding space. Combining these signals is therefore a standard strategy for improving retrieval robustness.

The fusion stage, however, is often treated primarily as a choice among aggregation formulas. Given component rankings and scores, a system may use Reciprocal Rank Fusion (RRF), Borda-style aggregation, CombSUM, CombMNZ, linear interpolation, or normalized score combinations. These operators do not merely implement different formulas. **They preserve different information about the underlying retrieval signals.**

RRF uses only ordinal position. If two documents have the same rank under a component retriever, their original score magnitudes do not affect the RRF contribution. Consequently, strictly monotonic transformations of component scores leave the resulting ranking unchanged.

Score-based operators make a different choice. CombSUM, for example, combines the numerical scores themselves:

$$
F_{\mathrm{SUM}}(d)=s_1(d)+s_2(d).
$$

It can therefore distinguish documents with similar ranks but different score separation. This additional information may be useful when score margins correspond to meaningful relevance distinctions. It may also be harmful when the component scores are poorly calibrated or have incompatible distributions.

This creates a more fundamental question than *which fusion operator performs best?*:

> **When does the information discarded by rank-only fusion affect the retrieval decision?**

We approach this question by separating three levels of claim.

First, there is a **mathematical property**: rank-only fusion discards score magnitude.

Second, there is an **interventional property**: if rankings are held fixed while score magnitudes are changed, rank-only fusion is invariant whereas score-based fusion can change.

Third, there is an **empirical retrieval question**: whether the magnitude information that score-based fusion retains is actually useful for identifying relevant documents.

The third question is conditional. It cannot be inferred simply from task topology. A multi-hop task may create situations in which distinguishing stronger from weaker evidence is useful, but this information must also be expressed by the component retrieval signals in a form that the fusion operator can exploit.

Our experiments therefore examine the interaction among **task, retriever pair, and score geometry**. We use Semantic Folding (SF) as a transparent diagnostic probe rather than as the primary algorithmic contribution. SF is combined with SPLADE, DPR, and BM25 to create heterogeneous retriever pairs. We then evaluate seven fusion operators and perform rank-preserving interventions on real retrieval scores.

The resulting evidence is deliberately narrower than a claim that score fusion is generally superior. In the confirmatory \(n=100\) evaluation, CombSUM provides a substantial advantage over RRF on HotpotQA and a smaller advantage on MuSiQue, while NQ-REaR shows much weaker separation. The retriever-pair experiments further show that the effect can disappear when the component signals change.

We therefore argue for the following principle:

> **A fusion operator should be evaluated according to the information it preserves, the information the task requires, and whether that information is reliably expressed by the participating retrieval signals.**

### Contributions

This paper makes four contributions:

1. **Information-preservation formulation.**
   We characterize common fusion operators according to whether they preserve ordinal information, raw score separation, or normalized score information.

2. **Controlled intervention methodology.**
   We use synthetic controls and rank-preserving transformations of real retrieval outputs to separate score magnitude from ranking.

3. **Conditional empirical evidence.**
   In the expanded \(n=100\) SF+SPLADE evaluation, CombSUM significantly outperforms RRF on HotpotQA and MuSiQue, while the effect is substantially weaker on NQ-REaR.

4. **Retriever-pair and decision-boundary analysis.**
   We show that fusion effects depend on joint score geometry and are concentrated in a small number of top-rank decisions.

We explicitly do **not** claim that RRF is inferior to score-based fusion in general, that multi-hop retrieval universally requires magnitude-aware fusion, or that the observed effects directly establish first-stage corpus-scale retrieval gains.

---

# 2. What Information Does Fusion Preserve?

## 2.1 Rank and score are different information objects

For query \(q\), let retriever \(i\) assign scores

$$
s_i(q,d)
$$

to candidate document \(d\).

These scores contain at least two conceptually distinct forms of information:

* **Ordinal information:** which candidate ranks above another.
* **Metric information:** how far apart their scores are and how score differences are distributed.

For example,

$$
[0.90,0.89,0.10]
$$

and

$$
[0.51,0.50,0.49]
$$

induce the same ranking but have very different score separation.

A rank-only operator treats these two vectors as equivalent. A score-space operator does not.

This distinction motivates the central experimental design of this work.

---

## 2.2 Rank-only fusion

RRF computes

$$
F_{\mathrm{RRF}}(d)
=
\sum_i\frac{1}{k+r_i(d)},
$$

where \(r_i(d)\) is the rank of \(d\) under retriever \(i\).

### Proposition 1 — Rank invariance

Let \(s(d)\) be a retrieval score and \(f\) a strictly monotonic function. Then

$$
\operatorname{rank}(s(d))
=
\operatorname{rank}(f(s(d))).
$$

Therefore any fusion operator that depends only on component ranks produces the same fused ranking before and after applying \(f\).

This proposition is elementary, but it provides an important experimental control:

> **If score magnitudes change while ranks remain fixed, any change in a fusion result cannot be attributed to rank information.**

---

## 2.3 Score-space fusion

CombSUM computes

$$
F_{\mathrm{SUM}}(d)=s_1(d)+s_2(d).
$$

Unlike RRF, its result depends directly on score separation.

This gives score-space fusion access to information that RRF discards. However, access to additional information does not imply that the information is useful.

For example, if one retriever produces scores on a much larger numerical scale than another, CombSUM may effectively weight the first signal more heavily. Thus score-space fusion introduces a dependency on **score geometry and cross-signal comparability**.

The key distinction is therefore:

> **Magnitude retention is a capability, not a guarantee of retrieval utility.**

---

## 2.4 Joint score geometry

We characterize a retrieval signal using

$$
G(s)=
(R,\mu,\sigma,\Delta_{12},\Delta_{15},\rho,\kappa),
$$

where \(R\) denotes score range, \(\mu\) and \(\sigma\) describe central tendency and dispersion, \(\Delta_{12}\) and \(\Delta_{15}\) describe local top-rank margins, \(\rho\) captures cross-signal association, and \(\kappa\) describes distributional shape.

For pairs of retrieval signals, we additionally examine Kendall's \(\tau\), Pearson correlation, top-\(k\) overlap, relative score scale, and local score margins.

The complete definitions and extraction procedure are given in **Appendix C**.

Importantly, we do not assume that these variables form a validated predictive model. They are used as **diagnostic descriptors of the score geometry under which operators act**.

---

# 3. Experimental Design

## 3.1 Controlled reranking setting

Our experiments study **Stage-2 candidate-set fusion/reranking**, rather than unrestricted first-stage corpus retrieval.

Each query is associated with a candidate set containing a gold document and distractors. This design conditions on candidate-set recall and allows the fusion stage to be studied independently of candidate generation.

This distinction is important: the results answer the question of how fusion behaves **given a candidate set**, not whether a fusion method improves end-to-end first-stage retrieval.

Candidate-pool scaling experiments examining this boundary are reported in **Appendix F**.

---

## 3.2 Retrieval signals

We evaluate four retrieval signals:

* **Semantic Folding (SF):** a deterministic, training-free semantic signal used as a transparent diagnostic probe;
* **SPLADE:** learned sparse retrieval;
* **DPR:** dense bi-encoder retrieval;
* **BM25:** lexical retrieval.

SF is not presented as the principal algorithmic contribution. Its role is to provide a controllable and interpretable retrieval signal that can be combined with heterogeneous learned and lexical systems.

---

## 3.3 Fusion operators

We evaluate seven operators:

* RRF;
* Borda;
* CombSUM;
* CombMNZ;
* Linear;
* z-score;
* min-max.

These span rank-based, raw-score, and normalized-score fusion.

RRF uses \(k=60\). Linear fusion uses \(\alpha=0.3\). Sensitivity results are reported in **Appendix D.1**.

---

## 3.4 Datasets and evaluation hierarchy

We evaluate ten QA and evidence-retrieval datasets.

To avoid conflating exploratory screening with statistical confirmation, we distinguish three levels of evaluation:

1. **Confirmatory expanded evaluation:** HotpotQA, MuSiQue, and NQ-REaR, \(n=100\).
2. **Expanded diagnostic experiments:** selected datasets and retriever pairs evaluated at \(n=50\).
3. **Exploratory probes:** smaller samples used to identify and illustrate candidate phenomena.

The complete benchmark matrix is provided in **Appendix A**, while the statistical hierarchy and complete tests are provided in **Appendix B**.

---

## 3.5 Statistical protocol

For the \(n=100\) expanded evaluation, we report paired bootstrap 95% confidence intervals and paired statistical comparisons with Holm correction across the operator-comparison family.

Full pairwise tests, effect sizes, and win/tie/loss counts are provided in **Appendix B**.

---

# 4. Confirmatory Evidence: When Does Magnitude Matter?

The central result is not that one fusion operator wins everywhere. Rather, **operator separation occurs selectively**.

### Table 1. Expanded \(n=100\) evaluation for SF+SPLADE

| Dataset  | Best operator         |  Best MRR | RRF MRR |   ΔMRR | Holm-adjusted \(p\) |
| -------- | --------------------- | --------: | ------: | -----: | ------------------: |
| HotpotQA | **CombSUM**           | **0.947** |   0.854 | +0.093 |          **0.0007** |
| MuSiQue  | **CombSUM / z-score** | **0.952** |   0.908 | +0.044 |          **0.0498** |
| NQ-REaR  | **CombSUM**           |     0.746 |   0.718 | +0.028 |                   — |

Complete confidence intervals and all seven operators are reported in **Appendix B.1–B.3**.

On HotpotQA, CombSUM exceeds RRF by \(0.093\), with the comparison surviving Holm correction. On MuSiQue, the advantage is smaller, and CombSUM ties z-score at the observed MRR. NQ-REaR provides an important boundary condition: the numerical difference is smaller and the overall operator separation is considerably weaker.

This pattern argues against a universal rule of the form:

> *multi-hop task → score fusion.*

Instead, the result suggests that the usefulness of magnitude depends on whether the participating signals expose discriminative score geometry.

---

# 5. Intervening on Score Magnitude

The preceding section establishes association between fusion choice and retrieval performance. It does not by itself establish that score magnitude is responsible.

We therefore intervene directly on the scores.

## 5.1 Synthetic control

We construct candidate pairs with fixed ordinal rankings and manipulate score separation.

Three regimes are examined:

1. **Rank-dominant:** score changes do not alter the preferred decision.
2. **Magnitude-sensitive:** score separation changes the preferred fused candidate.
3. **Small-margin:** score differences are insufficient to produce a stable magnitude-based distinction.

The complete construction and parameter ranges are given in **Appendix D.2**.

The expected behavior follows directly from the information preserved by the operators: RRF remains invariant when rank remains fixed, whereas score-space operators can change.

---

## 5.2 Real-score intervention

We next apply rank-preserving transformations to actual retrieval traces, including scaling, compression, power transformations, rank-preserving remapping, and randomized score replacement.

The critical control is the rank-preserving condition.

If the component ranking remains unchanged:

* RRF should remain unchanged;
* Borda should remain unchanged;
* score-space operators may change.

This prediction is observed in the real traces.

For example, on the examined MuSiQue traces, compressing SPLADE score gaps reduces CombSUM MRR from 0.914 to 0.460 while RRF remains unchanged.

The complete intervention matrix is reported in **Appendix D.2**.

These experiments therefore establish a stronger statement than observational correlation:

> **Score-space fusion is operationally sensitive to score magnitude even when the component ranking is held fixed.**

We use "sensitivity" deliberately. The intervention establishes that magnitude affects the fusion computation; it does **not** imply that the original magnitude necessarily represents compositional confidence or any particular semantic property.

That stronger interpretation would require direct measurement of what the score represents.

---

## 5.3 Relevance-aligned counterfactual

The sensitivity result above shows that *some* magnitude matters. A sharper question is
whether the *useful* part of the magnitude is the part that is **relevance-aligned** — i.e.
whether a gold document sits higher in the fused magnitude space than its rank neighbours
for a reason tied to relevance, rather than by coincidence of scale.

We construct a rank-preserving counterfactual that isolates this. For each query we keep
the SF and SPLADE **ranks** fixed and shift only the *magnitude* of the gold document
within its rank bucket:

* **World+** amplifies the gold document's score margin above the bucket's non-gold mean
  (ρ ∈ {1.25, 1.5, 2.0});
* **World−** reverses that margin, placing the gold document at the midpoint between its
  own score and the bucket non-gold mean.

Because ranks are held fixed by construction, RRF (rank-only) output is identical across
all worlds (verified: Kendall τ = 1.000 across worlds, RRF ΔMRR = 0). Any change in
CombSUM, CombMNZ, or linear fusion is therefore attributable to magnitude, not rank. The
causal prediction is that MRR(World+) ≥ MRR(orig) ≥ MRR(World−) while RRF stays flat.

On the examined traces this prediction holds directionally: suppressing the
relevance-aligned magnitude (World−) degrades CombSUM MRR on every dataset with headroom
(HotpotQA −0.397, MuSiQue −0.050, SciFact −0.001 at n=10), while amplifying it (World+)
never hurts and slightly helps where headroom exists (MuSiQue +0.0018, SciFact +0.0001).
RRF does not move. The complete counterfactual matrix and bootstrap confidence intervals
are reported in **Appendix D.4**.

This is the strongest statement available from the intervention:

> **The component of score magnitude that is relevance-aligned is the component that
> drives the CombSUM/CombMNZ gain over RRF.** Removing the relevance-aligned margin
> degrades fusion quality; RRF is unaffected because it discards magnitude entirely.

We describe this as a *magnitude-intervention* result. It establishes that the
relevance-aligned margin is causally operative in score-space fusion; it does not by
itself certify that the original magnitude encodes a specific semantic quantity such as
compositional depth. The descriptive companion (rank-conditioned relevance gap) is given
in Appendix D.4.

---

# 6. Retriever-Pair Dependence

If task topology were sufficient to explain the observed effect, changing the retriever pair should have a limited influence on the fusion difference within the same task.

The experiments show otherwise.

We evaluate:

* SF + SPLADE;
* SF + DPR;
* BM25 + SPLADE;
* BM25 + DPR.

The complete results are provided in **Appendix E**.

### Table 2. Operator × retriever-pair interaction

| Dataset  | Contrast    | Δ(SF+SPLADE) | Δ(other) | Difference | \(p_{\mathrm{perm}}\) |
| -------- | ----------- | -----------: | -------: | ---------: | --------------------: |
| HotpotQA | SF+DPR      |       +0.096 |   +0.000 |     +0.096 |             **0.004** |
| HotpotQA | BM25+SPLADE |       +0.096 |   −0.005 |     +0.101 |             **0.041** |
| HotpotQA | BM25+DPR    |       +0.096 |   +0.000 |     +0.096 |             **0.004** |

The important observation is not simply that SF+SPLADE favors CombSUM.

Rather:

> **The same task can exhibit substantially different fusion behavior when the participating retrieval signals change.**

In some DPR-containing settings, CombSUM and RRF induce the same ranking. When two operators produce identical fused rankings, downstream ranking metrics must also be identical. We refer to this as **operator identifiability**: an operator comparison has no empirical headroom when the candidate operators cannot produce different decisions.

The broader implication is that operator selection should begin with an examination of whether the operators are even capable of producing different rankings under the observed score geometry.

---

# 7. Where Does the Difference Occur?

The aggregate MRR difference can obscure where the actual decisions occur.

We therefore examine query-level wins and top-rank changes.

### Table 3. Query-level and rank-1 differences

| Dataset  | CombSUM wins | RRF wins | Ties | Rank-1 changes | \(d_z\) |
| -------- | -----------: | -------: | ---: | -------------: | ------: |
| HotpotQA |           21 |        1 |   78 |       10 (10%) |    0.45 |
| MuSiQue  |            8 |        0 |   92 |         4 (4%) |    0.29 |
| NQ-REaR  |           18 |       11 |   71 |       18 (18%) |    0.13 |

On HotpotQA, only 10% of queries change their rank-1 document, yet the resulting MRR difference is substantial.

This is important because global rank agreement is not necessarily informative about the decisions that determine MRR. A fused ranking can remain highly similar overall while differing at the one position that changes a query from reciprocal rank \(1/2\) to \(1\), or from \(1/3\) to \(1\).

We therefore interpret the effect as a **top-rank decision-boundary phenomenon**, rather than as evidence that score fusion completely reconstructs the ranking.

Qualitative examples are provided in **Appendix H**.

---

# 8. Boundary Conditions

## 8.1 Candidate-pool scaling

The observed effect persists across several constructed HotpotQA candidate-pool sizes.

On the examined pools from 20 to 494 candidates, CombSUM maintains MRR = 1.000 in the reported \(n=10\) diagnostic experiment, while RRF varies with candidate-pool size.

However, this should not be interpreted as evidence that score fusion is robust to arbitrary candidate generation.

---

## 8.2 Full-collection boundary

The SciFact full-collection experiment provides an important counterexample.

With 5,183 documents, the seven operators converge to approximately MRR = 0.130. Inspection shows that the gold document is present for only 3 of 10 queries.

Thus, the fusion stage is unable to recover information that candidate generation failed to provide.

The detailed decomposition is given in **Appendix F**.

This establishes an important boundary:

> **Fusion can reorder available evidence; it cannot recover evidence that is absent from the candidate set.**

---

# 9. Discussion

## 9.1 Fusion as information preservation

The conventional question is:

> Which fusion operator should I use?

Our results suggest a more useful question:

> **What information does this operator discard, and is that information useful for the retrieval decision?**

RRF discards score scale and is consequently robust to score-scale incompatibility.

Score-space fusion retains more metric information but becomes dependent on score geometry.

Neither property is universally desirable.

---

## 9.2 Task topology is insufficient

Multi-hop retrieval can create situations where distinguishing partial from stronger evidence is useful.

But our results do not support the stronger claim that multi-hop structure itself predicts the superiority of magnitude-aware fusion.

Instead:

> **Task topology determines what information may be useful; retriever geometry determines whether that information is available to the fusion operator.**

This distinction explains why changing the retriever pair can alter the result on the same task.

---

## 9.3 Practical diagnostic workflow

Our findings suggest the following workflow for practitioners:

1. **Measure score geometry**
   Inspect dispersion, top-rank margins, scale differences, and cross-signal association.

2. **Check operator identifiability**
   Determine whether candidate operators actually generate different fused rankings.

3. **Test magnitude sensitivity**
   Apply rank-preserving transformations to determine whether score-space fusion materially changes decisions.

4. **Test relevance association**
   Where labels are available, examine whether score separation corresponds to useful relevance distinctions.

5. **Check candidate-set quality**
   Verify that the candidate generator provides sufficient relevant evidence before optimizing fusion.

The complete geometry variables and diagnostic calculations are provided in **Appendix C**, while the intervention procedures are detailed in **Appendix D**.

---

# 10. Limitations

Our conclusions are deliberately scoped.

**First**, the principal experiments are controlled reranking experiments. They condition on candidate-set recall and therefore do not establish first-stage corpus-scale retrieval gains.

**Second**, the expanded confirmatory evaluation uses \(n=100\) queries per dataset, while several diagnostic experiments use smaller samples. Those smaller experiments should therefore be interpreted as exploratory evidence.

**Third**, the benchmark is predominantly English and QA/evidence oriented.

**Fourth**, two learned sparse checkpoints provide replication evidence but do not establish model independence across the learned-sparse family.

**Fifth**, only one dense retriever checkpoint is evaluated.

**Sixth**, although score geometry is a useful descriptive framework, we do not yet establish it as a reliable predictive model for selecting fusion operators. The exploratory operator-selection analysis contains too few genuinely divergent observations to support such a claim. Details are reported in **Appendix C**.

---

# 11. Conclusion

Hybrid retrieval is commonly framed as a choice among aggregation formulas. Our results suggest that this framing misses a more fundamental issue: **different fusion operators preserve different information**.

Rank-only operators such as RRF preserve ordinal structure while discarding score magnitude. Score-space operators retain additional metric information but consequently depend on score geometry.

The important question is therefore not whether magnitude is universally useful, but:

> **When does the information discarded by rank-only fusion affect the retrieval decision?**

Our controlled interventions establish that score-space fusion responds to magnitude changes even when component rankings are fixed. The expanded \(n=100\) evaluation then shows that this sensitivity can matter in practice: CombSUM outperforms RRF on HotpotQA and MuSiQue under SF+SPLADE, while the effect is substantially weaker on NQ-REaR. Retriever-pair experiments further demonstrate that changing the participating signals can eliminate the difference, and query-level analysis shows that the practical effect is concentrated in a small number of top-rank decisions.

These results support a conditional design principle:

> **Choose fusion according to the information properties of the participating retrieval signals, rather than according to a universal ranking of fusion formulas.**

For controlled reranking, this means inspecting score geometry, determining whether operators are identifiable, testing magnitude sensitivity, and verifying candidate-set quality.

More broadly, hybrid retrieval can be viewed not only as an aggregation problem, but as an **information-preservation problem**.

---

# Appendices

## Appendix A. Complete Benchmark Results

This appendix reports the complete benchmark matrix for the evaluated fusion operators.

### Table A1. SF + SPLADE operator matrix

| Dataset         | Type               | Linear |   RRF |   CombSUM |   CombMNZ | Borda | z-score | Min-max |
| --------------- | ------------------ | -----: | ----: | --------: | --------: | ----: | ------: | ------: |
| Belebele        | single-hop         |  1.000 | 1.000 |     1.000 |     1.000 | 1.000 |   1.000 |   1.000 |
| PopQA           | single-hop         |  1.000 | 1.000 |     1.000 |     1.000 | 1.000 |   1.000 |   1.000 |
| NarrativeQA     | single-hop         |  1.000 | 1.000 |     1.000 |     1.000 | 1.000 |   1.000 |   1.000 |
| PubMedQA        | single-hop         |  0.800 | 0.800 |     0.800 |     0.800 | 0.800 |   0.800 |   0.800 |
| HotpotQA        | multi-hop          |  0.558 | 0.783 | **1.000** |     0.783 | 0.583 |   0.683 |   0.558 |
| 2WikiMultihopQA | multi-hop          |  1.000 | 1.000 |     1.000 |     1.000 | 0.950 |   1.000 |   1.000 |
| MuSiQue         | multi-hop          |  0.887 | 0.927 | **0.977** |     0.919 | 0.780 |   0.953 |   0.887 |
| NQ-REaR         | factoid            |  0.566 | 0.612 |     0.593 | **0.820** | 0.653 |   0.737 |   0.700 |
| SciFact         | claim verification |  0.960 | 0.960 |     0.960 |     0.940 | 0.890 |   0.930 |   0.910 |
| COVID-QA        | biomedical         |  0.900 | 0.900 |     0.900 |     0.900 | 0.800 |   0.900 |   0.900 |

Exploratory \(n=10\) and expanded \(n=50\) values should be retained in a secondary table rather than mixed into the primary matrix. This prevents the reader from confusing exploratory screening with the \(n=100\) expanded evaluation.

---

# Appendix B. Statistical Analysis

## B.1 HotpotQA — \(n=100\)

| Operator |       MRR | 95% CI         |
| -------- | --------: | -------------- |
| Borda    |     0.732 | [0.656, 0.804] |
| CombMNZ  |     0.866 | [0.803, 0.923] |
| CombSUM  | **0.947** | [0.910, 0.978] |
| Linear   |     0.702 | [0.639, 0.766] |
| Min-max  |     0.702 | [0.639, 0.766] |
| RRF      |     0.854 | [0.802, 0.903] |
| z-score  |     0.896 | [0.846, 0.940] |

CombSUM versus RRF:

$$
\Delta = +0.093,
$$

raw \(p=0.0001\), Holm-adjusted \(p=0.0007\).

Fifteen of the 21 pairwise comparisons survive the stated Holm procedure.

---

## B.2 MuSiQue — \(n=100\)

| Operator |       MRR | 95% CI         |
| -------- | --------: | -------------- |
| Borda    |     0.652 | [0.560, 0.743] |
| CombMNZ  |     0.840 | [0.775, 0.902] |
| CombSUM  | **0.952** | [0.912, 0.985] |
| Linear   |     0.832 | [0.772, 0.888] |
| Min-max  |     0.832 | [0.772, 0.888] |
| RRF      |     0.908 | [0.862, 0.952] |
| z-score  | **0.952** | [0.912, 0.985] |

CombSUM versus RRF:

$$
\Delta=+0.044,
$$

raw \(p=0.0083\), Holm-adjusted \(p=0.0498\).

CombSUM and z-score produce the same reported MRR.

---

## B.3 NQ-REaR — \(n=100\)

| Operator |       MRR | 95% CI         |
| -------- | --------: | -------------- |
| Borda    |     0.602 | [0.515, 0.683] |
| CombMNZ  |     0.701 | [0.618, 0.777] |
| CombSUM  | **0.746** | [0.671, 0.817] |
| Linear   |     0.682 | [0.605, 0.755] |
| Min-max  |     0.682 | [0.605, 0.755] |
| RRF      |     0.718 | [0.643, 0.787] |
| z-score  |     0.733 | [0.659, 0.801] |

Only four of the 21 comparisons survive Holm correction according to the reported analysis.

---

## B.4–B.6 Historical \(n=50\) Results

The historical \(n=50\) experiments are retained for transparency but should **not** be described as confirmatory evidence.

This distinction is important because the paper's inferential hierarchy should be unmistakable:

> **\(n=100\) = primary expanded evaluation**
> **\(n=50\) = expanded diagnostic evidence**
> **\(n=10\) = exploratory probe**

This resolves one of the biggest reviewer concerns in the previous version.

---

# Appendix C. Score-Geometry Analysis

For each retrieval signal we compute:

$$
G(s)=
(R,\mu,\sigma,\Delta_{12},\Delta_{15},\rho,\kappa).
$$

Here:

* \(R\): score range;
* \(\mu\): mean score;
* \(\sigma\): score dispersion;
* \(\Delta_{12}\): top-1/top-2 margin;
* \(\Delta_{15}\): top-1/top-5 margin;
* \(\rho\): cross-signal association;
* \(\kappa\): distributional shape.

Pair-level diagnostics additionally include:

* Kendall's \(\tau\);
* Pearson correlation;
* top-\(k\) overlap;
* relative score scale;
* local top-rank margins.

### C.1 Operator identifiability

We distinguish three cases:

**Global identifiability:** two operators produce different rankings somewhere in the candidate population.

**Top-\(k\) identifiability:** two operators differ within the evaluation depth \(k\).

**Decision identifiability:** two operators differ on a ranking decision that changes the evaluation metric.

This distinction is important because global differences do not necessarily imply meaningful retrieval differences.

### C.2 Exploratory learned selection

We initially formulate operator selection using 21 pre-fusion geometry features.

However, the exploratory traces contain 34/40 queries with operator ties, leaving only six informative observations.

Consequently, we **do not claim predictive performance** for a learned operator-selection model.

This analysis is retained as a methodological direction rather than presented as a validated model.

---

# Appendix D. Intervention and Sensitivity Analysis

## D.1 RRF sensitivity

RRF uses

$$
\frac{1}{k+r}.
$$

We evaluate:

$$
k\in\{10,30,60,100\}.
$$

On the reported HotpotQA SF+SPLADE \(n=10\) diagnostic experiment, MRR varies by less than 0.02 across the tested settings, while the qualitative ordering remains unchanged.

---

## D.2 Linear-fusion sensitivity

Linear fusion is defined as

$$
F(d)=
\alpha\,\operatorname{maxnorm}(s_{\mathrm{SF}})
+
(1-\alpha)\operatorname{maxnorm}(s_{\mathrm{SPLADE}}).
$$

The sweep covers

$$
\alpha\in\{0.0,0.1,\ldots,1.0\}.
$$

The results indicate that \(\alpha=0.3\) is not a uniquely optimal point: performance is relatively flat over a substantial interval and degrades primarily when SF receives excessive weight.

This supports treating \(\alpha=0.3\) as a fixed experimental setting rather than an optimized hyperparameter.

---

## D.3 Rank-preserving transformations

We evaluate:

* \(x2\);
* log compression;
* power transformation;
* rank-preserving remapping;
* randomized score replacement.

For each condition we report:

$$
\text{MRR}
$$

and Kendall's

$$
\tau
$$

between the perturbed and original fused rankings.

The complete matrices should remain here rather than in the main paper.

The central result is:

> **RRF remains invariant when the component ranking remains fixed. Score-space operators need not.**

The randomized-score condition provides the strongest stress test because it destroys the original metric scale while preserving the experimentally imposed rank structure.

---

## D.4 Relevance-aligned counterfactual (Item 1)

This appendix reports the rank-preserving counterfactual described in §5.3. Component
traces are the SF (`comp_1.0`) and SPLADE (`comp_0.0`) endpoint scores from the
controlled reranking setting. We hold the per-signal **ranks** fixed and shift only the
magnitude of the gold document within its rank bucket:

* **World+** amplifies the gold margin above the bucket non-gold mean by ρ ∈ {1.25, 1.5, 2.0};
* **World−** reverses the margin (midpoint between gold score and bucket non-gold mean);
* controls: orig, compress (×0.5), rpr (rank-preserving random monotone remap).

RRF is invariant across all worlds by construction (Kendall τ = 1.000; RRF ΔMRR = 0), so
any CombSUM/CombMNZ/linear change is a pure magnitude effect.

### Table D.4a. CombSUM MRR by world (n=10 diagnostic traces)

| Dataset  | orig   | compress | rpr    | W+ ρ=1.25 | W+ ρ=1.5 | W+ ρ=2.0 | World− |
| -------- | -----: | ------: | -----: | --------: | -------: | -------: | -----: |
| HotpotQA | 1.0000 | 1.0000  | 0.8833 | 1.0000    | 1.0000   | 1.0000   | 0.6033 |
| MuSiQue  | 0.9125 | 0.9125  | 0.8043 | 0.9143    | 0.9143   | 0.9200   | 0.8625 |
| SciFact  | 0.8204 | 0.8204  | 0.8205 | 0.8205    | 0.8205   | 0.8205   | 0.8198 |
| 2Wiki    | 1.0000 | 1.0000  | 1.0000 | 1.0000    | 1.0000   | 1.0000   | 1.0000 |

RRF is 0.9333 (HotpotQA), 0.8111 (MuSiQue), 0.8214 (SciFact), 1.0000 (2Wiki) in **every**
world — confirming the manipulation is purely magnitude-level.

### Table D.4b. Causal contrast (CombSUM bootstrap 95% CI, B=10000)

| Dataset  | World+ vs orig ΔMRR | orig vs World− ΔMRR |
| -------- | -------------------: | ------------------: |
| HotpotQA | +0.0000 [0, 0]       | +0.3967 [+0.217, +0.560] |
| MuSiQue  | +0.0018 [0, +0.005]  | +0.0500 [0, +0.150] |
| SciFact  | +0.0001 [0, +0.0004] | +0.0006 [0, +0.002] |
| 2Wiki    | ceiling (MRR=1.0)     | ceiling (MRR=1.0)   |

Suppressing the relevance-aligned magnitude (World−) degrades CombSUM MRR on every
dataset with headroom; amplifying it (World+) never hurts and improves where headroom
exists. RRF is unchanged in all cases.

### Table D.4c. Rank-conditioned relevance gap (descriptive companion)

E[s \| y=1, r] − E[s \| y=0, r] per rank bucket, plus P(y=1 | large/small separation) and
AUC of the merged score:

| Dataset  | rank 2–3 | rank 4–5 | rank 6–10 | P(y=1\|large) | P(y=1\|small) | AUC  |
| -------- | -------: | -------: | --------: | ------------: | ------------: | ----: |
| HotpotQA | −0.002   | −0.118   | −0.067    | 0.000         | 0.043         | 0.976 |
| MuSiQue  | +0.095   | +0.011   | +0.005    | 0.002         | 0.023         | 0.904 |
| SciFact  | +0.488   | n/a      | +0.098    | 0.000         | 0.033         | 0.952 |
| 2Wiki    | +0.173   | +0.085   | +0.053    | 0.000         | 0.054         | 0.976 |

The descriptive gap is small and inconsistent at n=10 (some buckets negative; rank-1
bucket empty). The n=100 confirmatory run (HotpotQA, MuSiQue, NQ-REaR) is required before
any quantitative effect-size claim; results will be reported here when available.

Reproduce: `scripts/counterfactual_magnitude.py --n 10` (n=100 pending trace
regeneration via `scripts/gen_component_traces_n100.py`).

---

# Appendix E. Retriever-Pair Analysis

### Table E1. Retriever-pair comparison

| Pair          | HotpotQA                  | NQ-REaR                   | Main observation  |
| ------------- | ------------------------- | ------------------------- | ----------------- |
| SF + SPLADE   | 0.733 / 0.847 / **0.947** | 0.628 / 0.636 / **0.657** | CombSUM advantage |
| SF + DPR      | **0.687** / 0.611 / 0.611 | **0.583** / 0.594 / 0.594 | Linear advantage  |
| BM25 + SPLADE | 0.940 / **0.945** / 0.940 | 0.566 / **0.612** / 0.593 | Near parity       |
| BM25 + DPR    | **0.927** / 0.867 / 0.867 | **0.602** / 0.560 / 0.560 | Linear advantage  |

Values are reported as Linear / RRF / CombSUM.

The important conclusion is not that a particular retriever always prefers a particular fusion operator.

Rather:

> **The operator effect changes with the joint geometry of the participating signals.**

---

# Appendix F. Candidate-Pool Scaling

### Table F1. HotpotQA SF+SPLADE

| \(N\) | Linear |   RRF |   CombSUM |
| ----: | -----: | ----: | --------: |
|    20 |  0.558 | 0.667 | **1.000** |
|    50 |  0.612 | 0.783 | **1.000** |
|   100 |  0.592 | 0.883 | **1.000** |
|   494 |  0.558 | 0.783 | **1.000** |

These results are from the \(n=10\) diagnostic pool-scaling experiment.

The SciFact full-collection experiment uses 5,183 documents. The reported MRR is approximately 0.130 across operators, and gold evidence is present for only 3/10 queries.

This provides a direct demonstration of the candidate-set boundary:

$$
\text{fusion} \neq \text{candidate generation}.
$$

---

# Appendix G. Reproducibility

## G.1 Semantic Folding configuration

The SF pipeline uses:

* phrase extraction;
* term-context generation;
* UMAP projection;
* \(64\times64\) grid;
* Morton/Z-order encoding;
* Gaussian spreading activation;
* \(\sigma=1.5\);
* L2 document normalization.

The binary SDR fingerprint representation and emitted real-valued retrieval scores are retained exactly as used by the experiments.

## G.2 Hyperparameters

* top 10% IDF weighting;
* spreading radius = 1;
* decay = 0.5;
* RRF \(k=60\);
* Linear \(\alpha=0.3\).

## G.3 Dataset configuration

The exact query counts and pool sizes used in each experiment are listed here, rather than inferred from the benchmark names.

## G.4 Implementation

Tracked analysis scripts include:

* `scripts/magnitude_perturbation.py`;
* `scripts/appendix_c_stats_n100.py`;
* `scripts/operator_identifiability.py`;
* `scripts/scifact_deep_investigation.py`.

The random seed for the perturbation analysis is 42.

---

# Appendix H. Qualitative Decision-Boundary Examples

### H.1 HotpotQA example

The qualitative examples illustrate cases in which RRF and CombSUM produce different top-ranked documents despite relatively similar component rankings.

For each example, report:

| Feature      |  RRF | CombSUM |
| ------------ | ---: | ------: |
| Gold rank    |    3 |       1 |
| SF rank      |    1 |       1 |
| SPLADE rank  |    4 |       2 |
| SF score     | 0.95 |    0.95 |
| SPLADE score | 0.12 |    0.45 |

The example should be described cautiously:

> CombSUM uses the larger SPLADE score separation in this example, whereas RRF depends only on the ordinal positions.

Avoid saying that SPLADE's magnitude is definitively "compositional evidence" unless you have an independent experiment measuring compositionality.

---

# My chief-reviewer verdict on this V4

**Yes — this is now the right paper.** But I would make **five changes before calling it a 9+ SIGIR submission.**

### 1. The title is substantially better

I strongly prefer:

> **What Does Rank Fusion Throw Away? When Score Geometry Matters in Hybrid Retrieval**

over the previous *What Does Fusion Preserve?*

"Throw away" creates the information-loss question immediately, and **score geometry** gives the paper its technical identity.

I would keep it.

---

### 2. Do NOT call the intervention result "causal sensitivity to magnitude" too aggressively

This was one of the most important issues in your previous review.

Use:

> **"operational sensitivity to score magnitude"**

or

> **"interventional evidence that score magnitude affects score-space fusion."**

Don't write:

> "magnitude encodes compositional confidence."

unless you directly measure that.

That one change makes the paper much more defensible.

---

### 3. The biggest remaining technical issue: your numerical hierarchy needs auditing

There is an important potential inconsistency in the material you pasted.

For example, Appendix A reports HotpotQA CombSUM = **1.000**, whereas the \(n=100\) confirmatory table reports CombSUM = **0.947**.

That may be completely legitimate if they correspond to **different query subsets / \(n\)**, but the manuscript needs to make this absolutely explicit.

A reviewer must never wonder:

> "Which is the actual HotpotQA result?"

I would label every table explicitly:

> **Exploratory \(n=10\)**
> **Expanded \(n=50\)**
> **Expanded \(n=100\)**

and never mix them in one row without an extremely clear notation.

This is probably the **single most important cleanup before submission**.

---

### 4. I would change "confirmatory" to "expanded \(n=100\) evaluation"

This is subtle but important.

Unless the datasets and hypotheses were genuinely pre-specified before examining the exploratory results, calling the \(n=100\) analysis "confirmatory" can invite an HARKing criticism.

Safer:

> **Expanded \(n=100\) evaluation**

Then say:

> "The \(n=100\) evaluation provides the primary statistical evidence."

That is much harder for a reviewer to attack.

If you genuinely preregistered the hypotheses/dataset selection, then "confirmatory" is fine.

---

### 5. The paper's strongest contribution is NOT "CombSUM beats RRF"

Do not let the paper become perceived as:

> **"We ran seven fusion algorithms and found CombSUM works better on HotpotQA."**

That is not SIGIR-9-level novelty.

Your real contribution is:

> **Fusion operators make different information-preservation choices, and the retrieval consequences of those choices depend on joint score geometry and top-rank decision boundaries.**

The experiments are evidence for that proposition.

That framing is considerably more publishable.

---

## The paper's strongest causal ladder

I would make the entire manuscript mentally follow this sequence:

**Proposition**

$$
\boxed{\text{RRF discards score magnitude}}
$$

↓

**Intervention**

$$
\boxed{\text{Hold rank fixed → change magnitude}}
$$

↓

**Mechanistic consequence**

$$
\boxed{\text{RRF unchanged; score fusion changes}}
$$

↓

**Retrieval consequence**

$$
\boxed{\text{Sometimes the change improves MRR}}
$$

↓

**Boundary condition**

$$
\boxed{\text{Only for particular retriever geometries}}
$$

↓

**Decision-level explanation**

$$
\boxed{\text{Effect concentrated in top-rank decisions}}
$$

↓

**General principle**

$$
\boxed{
\text{Choose fusion by information preservation + score geometry}
}
$$

**That is your paper.**

And yes: **with the appendices integrated this way, I think the paper has a credible path toward the 9/10 range.** The remaining work is less about adding experiments and more about **making every claim exactly match the strength of the evidence and eliminating every numerical/experimental ambiguity.**
