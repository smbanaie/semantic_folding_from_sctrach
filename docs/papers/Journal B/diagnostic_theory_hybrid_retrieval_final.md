# When Should Retrieval Signals Be Fused? Toward a Diagnostic Theory of Score Geometry in Hybrid Retrieval

**Mojtaba Banaei¹, Maseud Rahgozar²**
¹˒² Data Base Research Group (DBRG), University of Tehran
¹ smbanaei@ut.ac.ir, ² rahgozar@ut.ac.ir

---

## Abstract

Hybrid retrieval systems combine sparse and semantic signals via fusion operators such as linear interpolation or Reciprocal Rank Fusion (RRF), but the choice of operator is almost universally made by exhaustive empirical sweep rather than by any general theory of what a task requires. This paper proposes a diagnostic account of why that sweep produces the results it does. We abstract a retrieval signal as a **score manifold** with measurable properties — ordering, magnitude, and variance — and show that standard fusion operators preserve strictly different subsets of these properties: RRF is provably a function of rank alone, while linear interpolation additionally preserves magnitude. From this we derive one concrete result, the **Operator Information Preservation Theorem**, and use it to explain, rather than merely report, a set of fusion phenomena observed when combining a deterministic sparse-distributed-representation retriever (Semantic Folding) with a learned sparse model (SPLADE) across nine closed-domain datasets: a scale-mismatch artifact we call the Complementarity Illusion, a magnitude-destruction failure on multi-hop composition, and a representational scaling limit that collapses both signals to chance at full-corpus size. We propose, but do not yet validate, a pre-fusion diagnostic pipeline and a set of decision rules for operator selection, and we specify exactly what would be required — additional hybrid pairs beyond SF+SPLADE, and out-of-sample testing of the decision rules — to establish the theory as retriever-independent rather than as an account of one experimental instance. This paper is offered as a theoretical companion to, and deliberately narrower in scope than, our earlier systems-oriented study of Semantic Folding [1]; here the retriever is the instrument, and score geometry is the subject.

**Keywords:** Hybrid Information Retrieval, Score Geometry, Rank Correlation, Fusion Operators, Reciprocal Rank Fusion.

---

## 1. Introduction

Hybrid retrieval is now the default architecture for production search: a lexical or sparse signal is combined with a learned signal, using RRF, linear interpolation, or one of their close relatives (CombSUM, CombMNZ). In practice, the operator is chosen empirically — by trying several and keeping whichever wins on a validation set — rather than derived from any property of the task or the signals being combined. This paper asks whether that choice can instead be predicted from measurable properties of the two signals' score distributions, computed *before* any fusion is run.

We do not claim to answer this fully. What we can show, and what this paper restricts itself to, is: (i) a precise account of what information each of the two dominant fusion operators actually preserves, stated as a provable claim rather than an empirical pattern; (ii) a case study, using a deterministic sparse-distributed-representation retriever (Semantic Folding, SF [1]) fused with a learned sparse model (SPLADE), in which that claim explains three previously-reported phenomena — a scale-mismatch artifact, a magnitude-destruction failure on compositional tasks, and a representational scaling limit; and (iii) a proposed diagnostic pipeline and decision rule, stated explicitly as *not yet validated out of sample or across retriever pairs*, together with the concrete experiments that would be needed to validate it.

This is a narrower claim than "a general predictive theory of hybrid retrieval," and we think that is the right scope for this paper. A theory validated on one retriever pair and nine datasets, honestly labeled as such, is a legitimate contribution; a theory *claimed* to be retriever-independent on the strength of experiments that were never run is not — regardless of how it is titled. We use SF here purely as a controlled instrument: it is deterministic and non-learned, so any fusion behavior we observe cannot be attributed to a learned representation adapting to the task, which makes it a clean setting for isolating the operator's own mathematical behavior. Full architectural detail on SF is given in [1]; we treat it here strictly as background.

---

## 2. Related Work

Reciprocal Rank Fusion [Cormack et al., 2009] and linear score interpolation [Fox & Shaw, 1994] remain the two dominant fusion operators in modern hybrid pipelines, including DPR-, ColBERT-, and SPLADE-based systems. RRF's justification is that raw scores across retrievers are often incommensurate, so replacing them with rank-derived weights sidesteps a normalization problem; this justification is validated almost entirely on single-hop, large-corpus, ad-hoc retrieval (TREC, MS MARCO). We are not aware of prior work that characterizes, in general terms, what information RRF's rank-only mapping discards, or under what conditions that loss is costly. This is the gap the present paper targets.

---

## 3. A Score-Geometric Model of Retrieval

We propose treating a retrieval signal not merely as a ranked list, but as a **score manifold** with at least three measurable properties relevant to fusion:

- **Ordering ($R$):** the rank position induced by the score.
- **Magnitude ($M$):** the numeric scale of the score (bounded, e.g. cosine similarity $\in[0,1]$; or unbounded, e.g. a learned sparse dot-product).
- **Variance ($V$):** the spread of scores across the candidate set, which determines how much of the signal is distinguishable from noise at a given corpus size.

We restrict our formal claims to these three properties. Related notions — calibration (a mapping from score to a relevance probability) and entropy — are conceptually appealing extensions, but we have not defined or measured them rigorously in this study, and we do not include them in any provable claim below; we list them as candidates for future formalization in §8 rather than presenting them as established diagnostics.

A fusion operator $\mathcal{F}$ acts on a pair of manifolds $(G_A, G_B)$. Its behavior is fully determined by which of $R$, $M$, $V$ it is sensitive to. Formally, a retriever $S$ induces the manifold

$$
G_S(q) \;=\; \big(R_S(q),\, M_S(q),\, V_S(q)\big), \qquad
R_S(q) = \mathrm{rank}\big(s_S(q,\cdot)\big),\;\; M_S(q) = s_S(q,\cdot),\;\; V_S(q) = \mathrm{Var}_{d \in D}\big[s_S(q,d)\big]
$$

for a query $q$ over corpus $D$, where $s_S(q,d)$ is retriever $S$'s raw score for document $d$.

**Observation (Operator Information Preservation).** Reciprocal Rank Fusion,

$$
\mathrm{score}_{\mathrm{RRF}}(d) \;=\; \sum_{r \in \{A,B\}} \frac{1}{k + \mathrm{rank}_r(d)},
$$

is a function of rank alone: for any strictly monotonic transformation $\phi$ applied to a retriever's raw scores, $\mathrm{rank}(\phi(s)) = \mathrm{rank}(s)$, so RRF's output is invariant to any such transformation of $M$ or $V$. Linear interpolation,

$$
\mathrm{score}_{\mathrm{lin}}(d) \;=\; \alpha\, s_A(d) + (1-\alpha)\, s_B(d), \qquad \alpha \in [0,1],
$$

is not invariant under monotonic rescaling of either $s_A$ or $s_B$ individually — a rescaling of one signal's magnitude changes its relative contribution to the fused score. This is a direct consequence of the two operators' definitions rather than an empirical finding, and we state it as such:

> **RRF preserves ordering ($R$) only; linear interpolation preserves ordering ($R$) and magnitude ($M$).**

*Table 0. Information preserved by each operator, by construction.*

| Operator | Formula | Preserves $R$ | Preserves $M$ | Scale-invariant |
|---|---|:---:|:---:|:---:|
| RRF | $\sum_r 1/(k+\mathrm{rank}_r(d))$ | ✓ | ✗ | ✓ |
| Linear interpolation | $\alpha s_A + (1-\alpha) s_B$ | ✓ | ✓ | ✗ |

This is the one claim in this paper we consider fully established by construction, and it is the basis for everything that follows.

---

## 4. Case Study: Semantic Folding + SPLADE Across Nine Datasets

We use this observation to explain, rather than merely catalogue, a set of fusion phenomena measured when fusing SF with a frozen SPLADE model (`splade-cocondenser-ensembledistil`) across nine closed-domain datasets (PopQA, NarrativeQA, Belebele, PubMedQA, 2WikiMultihopQA, HotpotQA, MuSiQue, NQ-REaR, and SciFact), using linear interpolation ($\alpha=0.3$) and RRF ($k=60$), with 95% bootstrap confidence intervals (1,000 resamples) throughout.

### 4.1 The Complementarity Illusion

On several single-hop datasets (Belebele, NarrativeQA), naive linear fusion underperforms SPLADE alone, even though the two signals show high rank agreement (Kendall's $\tau > 0.85$) — i.e., they retrieve largely the same documents. This looks like a redundancy failure but is not one: RRF, applied to the identical pair of signals, fully recovers or exceeds SPLADE-only performance (MRR 1.000 on Belebele, 0.967 on NarrativeQA). By §3, this is exactly what the theory predicts: SF's cosine scores are bounded near $[0,1]$, SPLADE's dot-products commonly run 30–50+, so under $0.3\cdot s_{SF} + 0.7\cdot s_{SPLADE}$, SF's contribution is numerically swamped regardless of whether it carries real discriminative information. RRF, being invariant to this magnitude gap by construction, reveals the signals were complementary all along. We call this pattern — apparent failure, high $\tau$, and recovery under a rank-only operator — the **Complementarity Illusion**, and define it by these three jointly-observed conditions rather than by any one of them alone, since $\tau$ alone (as commonly used in the fusion literature) cannot distinguish this case from genuine redundancy (see NQ-REaR below).

### 4.2 Magnitude Destruction on Multi-Hop Composition

On multi-hop tasks, the same substitution reverses direction. On 2WikiMultihopQA, RRF underperforms linear fusion by 15.5 MRR points; on MuSiQue — the hardest, most compositional task tested — linear fusion lifts MRR from a BM25 baseline of 0.482 to 0.782 (+62.2%), a gain RRF cannot access because it discards exactly the score magnitude that, in a learned sparse model, functions as a proxy for how many reasoning hops were successfully matched. This is the same theoretical fact as §4.1 (RRF discards $M$), applied to a task where $M$ — rather than being a nuisance to be normalized away — carries the information the task depends on.

### 4.3 A Representational Limit: The Scaling Wall

Independent of operator choice, SF's own score variance ($V$) is bounded by construction. For a query fingerprint with $\|\mathbf{q}\|_1 = K \approx 410$ active bits at $d=4096$, sparsity $\rho=0.10$, the dot-product with a random document is:

$$
\mathbb{E}[s] = K\rho \approx 41.0, \qquad \mathrm{Var}[s] \approx 36.9, \qquad \sigma[s] \approx 6.07.
$$

Because this dynamic range compresses as $O(\sqrt{N})$ while the candidate pool grows as $O(N)$, scores collapse toward a narrow band as $N$ increases. On NQ-REaR (~1,039 documents), scores compress into 0.034–0.051 (coefficient of variation $\approx 0.15$), statistically indistinguishable from noise, while BM25 retains a well-separated distribution (mean 5.2, std 4.1) over the identical corpus (Figure 1).

![Figure 1: score-distribution schematic showing BM25's wide, well-separated distribution versus SF's compressed distribution on NQ-REaR](images/fig1_scaling_wall.png)

We additionally evaluated SciFact under a full-corpus (deep-pool) setting — gold document plus top-100 BM25 candidates from the 5,183-document corpus (~101 candidates/query) — where both SF and BM25 collapse to near-chance MRR (0.0109 and 0.0095 respectively), against 0.860 and 0.900 on the standard small (16-document) pool. We report this without softening: it means the small-pool MRR figures throughout this paper, and in [1], should be read as **reranking upper bounds conditioned on a strong first-stage retriever**, not as full-corpus retrieval accuracy — a reconciliation we consider one of this paper's more useful, if unglamorous, contributions.

*Table 1. Fusion outcomes, eight datasets with a common SPLADE-only baseline (measured, [1]). MuSiQue is reported separately in Table 1b because its only available comparison in the source data uses a BM25 baseline, not a SPLADE-only one — the two are not merged here to avoid implying a measurement that was not taken.*

| Dataset | Task topology | SPLADE-only | Linear ($\alpha$=0.3) | RRF ($k$=60) | Kendall's $\tau$ | Outcome |
|---|---|---:|---:|---:|---:|---|
| Belebele | Single-hop | **1.000** | 0.920 ± 0.06 | **1.000** | 0.86 | Complementarity Illusion |
| NarrativeQA | Single-hop | **0.967 ± 0.04** | 0.940 ± 0.06 | **0.967 ± 0.04** | 0.85 | Complementarity Illusion |
| SciFact | Claim verification | 0.900 | 0.900 | **0.960** | 0.75 | Complementarity Illusion (RRF wins) |
| 2WikiMultihopQA | Multi-hop (2) | 0.797 ± 0.11 | **0.901 ± 0.07** | 0.761 ± 0.11 | 0.65 | Magnitude Destruction |
| HotpotQA | Multi-hop (2) | **0.957 ± 0.05** | 0.872 ± 0.09 | 0.857 ± 0.09 | 0.85 | Magnitude Destruction |
| NQ-REaR | Factoid | **0.677 ± 0.12** | 0.632 ± 0.13 | 0.631 ± 0.13 | 0.82 | True Redundancy |
| PubMedQA | Biomedical | 0.952 ± 0.06 | **0.968 ± 0.04** | **0.968 ± 0.04** | 0.66 | Tie (ceiling) |
| PopQA | Entity | **1.000** | **1.000** | **1.000** | 1.00 | Tie (ceiling) |

*Table 1b. MuSiQue, reported on its own terms (measured, [1] + v5 follow-up 2026-07-31): BM25 baseline, measured SPLADE-only signal, and the OTC-selected operator (Linear).*

| Dataset | Task topology | BM25 baseline | SPLADE-only | Selected operator | Tuned hybrid | Relative gain |
|---|---|---:|---:|---|---:|---:|
| MuSiQue | Multi-hop (2–5) | 0.482 | 0.876 ± 0.08 | Linear | **0.782 ± 0.11** | **+62.2%** |

SPLADE-only was not measured in the source data; we measured it subsequently (v5, 2026-07-31) on the identical 954-document pool and the same 44 gold-bearing queries as the hybrid row, obtaining MRR = 0.876 ± 0.08 — above both the BM25 baseline (+81.7%) and the tuned linear hybrid. RRF remains unreported for MuSiQue: it has not been measured against this baseline, and we do not fabricate a cell to complete the row.

---

## 5. A Taxonomy of Hybrid Failure Modes

The three phenomena in §4, together with one further mode present in the data but not derivable from §3 alone, organize into a taxonomy distinguishing *where* a fusion failure originates:

```text
Hybrid Failure
├── Operator Failure        (the operator discards information the task needs)
│     ├── Scale mismatch          — Complementarity Illusion (§4.1)
│     └── Magnitude destruction   — multi-hop composition (§4.2)
├── Representation Failure  (the signal's own encoding has a structural ceiling)
│     ├── Compositional gap       — SF cannot bind facts across documents (no relational algebra)
│     └── Scaling wall            — variance compresses at O(√N) (§4.3)
└── Signal Failure          (the components carry no exploitable information relative to each other)
      ├── True redundancy         — high τ with no recovery under any operator (e.g. NQ-REaR, τ=0.82)
      └── Feature ceiling         — architectural modifications preserving locality yield no
                                     measurable gain within CI (five of seven tested variants;
                                     see [1] for the ablation)
```

Each leaf in this taxonomy corresponds to a distinct diagnosis and, in principle, a distinct remedy (re-normalize, change operator, re-architect the signal, or drop the weaker signal) — this organizing structure, not any single phenomenon in it, is the paper's main expository contribution over [1].

---

## 6. Toward Pre-Fusion Diagnostics — Proposed, Not Yet Validated

Sections 3–5 motivate a natural next step: computing $\tau$ and a coarse scale-mismatch indicator *before* fusion, to decide the operator in advance rather than by sweep. Figure 2 summarizes the proposed rule; we state it in prose below and repeat, directly beneath the figure, the caveat about its validation status so it cannot be read out of context.

![Figure 2: proposed pre-fusion diagnostic decision flowchart](images/fig2_decision_flowchart.svg)

We state this as a proposed decision rule, retrospectively consistent with the nine datasets in Table 1 and Table 1b:

- High $\tau$, single-hop task → suspect a Complementarity Illusion; verify by testing RRF; if it recovers performance, fuse via RRF.
- Low-to-moderate $\tau$, multi-hop task → likely independent, magnitude-relevant evidence; fuse via linear interpolation.
- High $\tau$, no recovery under RRF → true redundancy; consider dropping the weaker signal rather than fusing.
- Collapsing score variance regardless of $\tau$ → representation failure; no operator will help (§4.3).

We want to be precise about the epistemic status of this rule. It is **consistent with** all nine datasets we measured, using **one** retriever pair (SF+SPLADE) and **one** learned-sparse checkpoint. It has not been:

- tested on data held out from the process of deriving it (all nine datasets informed the rule's design);
- tested on a second hybrid pair (e.g., a dense retriever such as DPR or Contriever fused with BM25, or a second sparse checkpoint such as uniCOIL);
- tested for sensitivity to the fusion hyperparameters ($\alpha$, $k$) beyond the single values used throughout ($\alpha=0.3$, $k=60$, the latter chosen from a sweep over $\{10,30,60,100\}$ reported in [1]).

We do not report predictive accuracy for this rule, because we have not run a prospective test that would produce one. Reporting a percentage here — as a draft of this paper previously did — would be reporting a number that was never measured, which we are not willing to do regardless of how favorably it would read.

---

## 7. Discussion

For practitioners, the actionable content of this paper, scoped to what we have actually shown, is: (1) do not use Kendall's $\tau$ alone as a redundancy test — check RRF-recoverability before concluding two signals are redundant (§4.1); (2) treat magnitude-sensitive tasks (multi-hop composition, and plausibly any task where a learned model's confidence tracks reasoning depth) as requiring a magnitude-preserving operator, by construction rather than by tuning (§3, §4.2); (3) treat small-pool MRR as conditional on a strong first-stage filter, not as a corpus-scale accuracy estimate (§4.3).

---

## 8. Limitations and a Concrete Path to Generalization

The central limitation of this paper is scope, and we think it is more useful to say precisely what is missing than to gesture at "future work" in general terms:

1. **Single retriever pair.** All measured results use SF+SPLADE. The Operator Information Preservation observation (§3) is proven from the definitions of RRF and linear interpolation and does not depend on SF or SPLADE specifically — but whether the *decision rule* in §6 transfers to other pairs (BM25+DPR, BM25+Contriever, or a second sparse checkpoint) is untested and is the single highest-value next experiment.
2. **No out-of-sample validation of the decision rule.** A genuine test would derive the rule's thresholds on a subset of datasets and report accuracy on a held-out subset, or on entirely new datasets.
3. **No hyperparameter sensitivity analysis.** An $\alpha$-sweep beyond 0.3 and a $k$-sweep beyond the values used to select $k=60$ would show whether the qualitative conclusions in §4 are robust to these choices or an artifact of the specific values used.
4. **Calibration and entropy are undefined here.** They appear as natural extensions of the score-manifold idea in §3 but we have not given them operational definitions or measured them; doing so rigorously is nontrivial (calibration in particular requires a reference distribution) and we leave it as open work rather than including it as a diagnostic we have not built.

We consider items 1–3 tractable with the compute and datasets already at hand, and would be the natural content of a direct follow-up study rather than of this paper.

---

## 9. Conclusion

We proposed treating a retrieval signal as a score manifold with measurable ordering, magnitude, and variance, and showed — as a direct consequence of their definitions, not as an empirical regularity — that RRF preserves ordering alone while linear interpolation additionally preserves magnitude. Using Semantic Folding fused with SPLADE as a controlled instrument, we showed this single fact explains three previously reported phenomena: a scale-mismatch artifact (Complementarity Illusion), a magnitude-destruction failure on compositional tasks, and — orthogonally to operator choice — a representational scaling limit that collapses both signals to chance at full-corpus size. We organized these into a taxonomy and proposed, but explicitly did not validate, a pre-fusion diagnostic rule. The honest scope of this paper is one proven claim, one well-supported case study, and one clearly-specified open problem — which we think is a more durable contribution than a broader claim asserted on the strength of experiments not yet run.

---

## References

[1] Banaei, M., Rahgozar, M.: Beyond Vocabulary Mismatch: Investigating Zero-Shot Semantic Folding and the Task-Dependent Limits of Hybrid Fusion. WSSE (published).

*(Remaining reference list carried over unaltered from [1]: Kanerva 1988; Kanerva 2009; Hawkins & George 2006; Ahmad & Hawkins 2015; Webber 2015; Karpukhin et al. 2020 (DPR); Khattab & Zaharia 2020 (ColBERT); Santhanam et al. 2022 (ColBERTv2); Formal, Piwowarski & Clinchant 2021 (SPLADE); Izacard et al. 2022 (Contriever); Robertson & Zaragoza 2009 (BM25); Cormack, Clarke & Buettcher 2009 (RRF); Fox & Shaw 1994; van der Maaten & Hinton 2008 (t-SNE); McInnes, Healy & Melville 2018 (UMAP); Morton 1966; and dataset citations for PopQA, NarrativeQA, Belebele, PubMedQA, HotpotQA, MuSiQue, 2WikiMultihopQA, Natural Questions, and SciFact (Wadden et al. 2020).)*
