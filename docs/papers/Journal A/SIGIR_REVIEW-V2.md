Yes. I reviewed **this exact version**—the manuscript titled **“What Does Fusion Preserve? Task- and Score-Geometry Dependent Information Loss in Hybrid Retrieval”**—rather than the earlier draft.

My verdict as a **strict SIGIR/TOIS-style chief reviewer** is:

> **Much better conceptual paper than the previous version, but I would still recommend Major Revision / Weak Reject in its current form.**
>
> The important point is that **the paper is now close to having a publishable core**. The remaining problems are no longer mainly about missing experiments. They are about **claim–evidence alignment, causal validity, experimental consistency, and statistical discipline**.

The good news: I would **not rebuild the paper again**. I would make a targeted final revision.

---

# 1. My overall reviewer score

If this were submitted today:

| Criterion              |          Score | Reviewer view                                                               |
| ---------------------- | -------------: | --------------------------------------------------------------------------- |
| Novelty                |       **7/10** | Interesting framing: information preservation × score geometry × task       |
| Technical significance |       **7/10** | Potentially useful beyond SF                                                |
| Experimental breadth   |     **7.5/10** | Much improved: 7 operators, 4 pairs                                         |
| Experimental rigor     |       **5/10** | n=10 / n=50 remains the major weakness                                      |
| Theoretical rigor      |     **5.5/10** | Rank invariance is solid but trivial; magnitude mechanism is not yet causal |
| Clarity                |       **7/10** | Much better than previous versions                                          |
| Reproducibility        |       **6/10** | Good intentions, but some claims depend on unfinished appendices            |
| Generalization         |       **6/10** | Better, but several conclusions still overreach                             |
| Related work           |       **6/10** | Bruch positioning is much better, but needs deeper engagement               |
| Overall                | **6.2–6.7/10** | **Major Revision / Weak Reject**                                            |

The manuscript has moved from:

> **“interesting but overclaims a law”**

to:

> **“interesting empirical/theoretical framework, but several central claims are still stronger than the evidence.”**

That is a very meaningful improvement.

---

# 2. The biggest positive change

The strongest change is the reframing.

You now explicitly say:

> “the relevance of the information a fusion operator discards is task-dependent AND score-geometry dependent” 

and, importantly, you have abandoned the old “Operator-Topology Constraint is a law” framing.

You also explicitly say:

> “we do not claim a universal law” 

That is exactly the right direction.

Even better, the paper now explicitly positions itself against Bruch et al.:

> “where Bruch et al. establish what fusion functions do to score distributions, we investigate when the information they discard becomes task-relevant” 

**Keep this positioning.**

I would build the entire final version around it.

---

# 3. But there is one major problem: your central causal claim is still not established

This is currently the biggest issue.

Your RQ3 asks:

> “Does score magnitude causally contribute to multi-hop reranking outcomes?” 

And contribution #3 says:

> “Controlled magnitude-perturbation experiments ... isolate score magnitude as a causal factor” 

But the experiment in §7.2 does **not establish causality in the retrieval task**.

You have:

> Doc A rank 1, Doc B rank 2
> scores manipulated
> magnitude-aware operators react differently 

That's a valid **operator sensitivity experiment**.

But it does not establish:

> “magnitude is causally relevant to multi-hop relevance.”

Why?

Because you decide that:

> A “should win”

based on the magnitude condition itself.

For example, if A=45 and B=12, A should win; if A=20 and B=18, A should win; if A=12 and B=45, A should lose.

That's essentially defining the desired answer through magnitude.

A reviewer will say:

> “You have shown that CombSUM uses magnitude and RRF doesn't. You have not shown that magnitude corresponds to relevance.”

This distinction is critical.

### Required change

Replace:

> **causal**

with:

> **controlled evidence for the functional role of magnitude**

throughout the paper.

Your strongest defensible claim is:

> **We establish causal sensitivity of the fusion operator to magnitude, not causal semantic validity of magnitude as a relevance signal.**

That's a much stronger scientific statement because it is actually true.

---

# 4. The synthetic experiment needs one important redesign

Current §7.2 is too simplistic.

You should replace the two-document experiment with a **retrieval-ranking intervention experiment**.

Instead of:

```text
A rank 1
B rank 2

change scores
```

use something like:

### Experiment A — Rank-preserving magnitude intervention

Take real retrieval outputs.

For every query:

1. preserve original ranking;
2. transform scores monotonically;
3. create controlled magnitude gaps;
4. run RRF, Borda, CombSUM, CombMNZ, linear;
5. verify rank-only operators remain invariant;
6. verify score-space operators change.

This validates operator sensitivity.

### Experiment B — Magnitude-swapping intervention

Take actual real retrieval outputs:

```text
Gold:      rank 2, score 42
Distractor rank 1, score 15
```

then manipulate:

```text
Gold:      rank 2, score 50
Distractor rank 1, score 15
```

and observe whether a score-based fusion operator promotes the gold.

Now you are manipulating **real retrieval evidence**, rather than synthetic documents.

### Experiment C — Counterfactual score destruction

Take real scores and:

* preserve rank;
* replace magnitudes with rank-equivalent values.

Then compare:

```text
Original scores
vs.
rank-preserving magnitude-destroyed scores
```

This is much closer to the actual research question.

---

# 5. Your second biggest problem: the “signal B determines the operator” claim is too strong

You currently write:

> “the winning operator family is determined by the score geometry of signal B, not by the task and not by which signal is A.” 

This is **too strong**.

Your own results don't justify it.

Look at your table:

| Pair          | Hotpot n=50   |
| ------------- | ------------- |
| SF + SPLADE   | CombSUM 0.947 |
| SF + DPR      | Linear 0.687  |
| BM25 + SPLADE | RRF 0.945     |
| BM25 + DPR    | Linear 0.927  |

The BM25+SPLADE result is especially damaging to the “signal B determines it” claim.

If B=SPLADE:

* SF+SPLADE → CombSUM
* BM25+SPLADE → RRF

So **B alone cannot determine the winning operator**.

Your later caveat actually admits this:

> “BM25+SPLADE ... collapses to a near-tie” 

### Replace the claim with:

> **Fusion behavior depends jointly on the score geometry of the component signals, their interaction, and the information requirements of the task.**

That is actually more interesting.

Your current paper is trying to reduce a multidimensional interaction to:

> `B → operator`

But your own evidence suggests:

> `task × signal A geometry × signal B geometry × operator → outcome`

That should become the conceptual model.

---

# 6. Your most defensible contribution is actually stronger than your current claim

I would formulate your central contribution as:

> **Fusion operators should not be selected independently of the information geometry of the signals they combine.**

Then:

### Three layers

**Layer 1 — Operator geometry**

RRF:

```text
score magnitude → discarded
rank → preserved
```

CombSUM:

```text
rank → indirectly preserved
magnitude → preserved
```

This is mathematically trivial but useful.

**Layer 2 — Signal geometry**

Different retrieval models produce:

* bounded cosine scores;
* sparse dot products;
* normalized dense similarities;
* lexical BM25 scores.

**Layer 3 — Task information requirement**

Different tasks require:

* ordering;
* separation;
* compositional evidence;
* multi-signal agreement.

Then the empirical question becomes:

> When does the information discarded by an operator correspond to information required by the task?

**That is the paper.**

Not:

> “CombSUM is better than RRF.”

---

# 7. There is a serious problem with your “causal” interpretation of the real traces

You say:

> “The gap widens exactly on compositional tasks, where the gold passage's score margin over distractors is what raw magnitude preserves and rank-only fusion discards.” 

The first half is empirical.

The second half is an **interpretation**.

You have not demonstrated that:

> high SPLADE magnitude = number of successful reasoning hops.

This is acknowledged in your older versions, but the current manuscript needs to make the distinction even sharper.

A reviewer will ask:

> Why can't SPLADE magnitude simply be lexical density, term frequency, expansion density, or document length?

Therefore introduce:

### “Magnitude relevance” rather than “compositional confidence”

Instead of:

> SPLADE score encodes compositional confidence

write:

> **SPLADE score magnitude may provide a useful separation signal in these compositional reranking settings.**

Then test correlations with:

* hop count;
* gold/distractor status;
* query-term coverage;
* document length;
* lexical overlap;
* expansion density.

If you can show:

```text
score margin
        ↓
strongly associated with
        ↓
multi-hop gold-vs-distractor separation
```

your argument becomes much stronger.

---

# 8. Your “Feature Invariance” section is now appropriately cautious — keep it that way

This section is much better than the original.

You explicitly state:

> the full SF pipeline adds UMAP, Gaussian smoothing and spreading activation, therefore the pipeline-level claim is still a hypothesis. 

Excellent.

Do **not** turn this back into a theorem.

However, one sentence remains dangerous:

> “SF scores are a deterministic function of term-co-occurrence overlap”

That is too compressed.

The emitted SF score is not simply:

```text
qᵀd
```

because your own pipeline includes:

* UMAP;
* spatial projection;
* Gaussian filtering;
* spreading activation;
* normalization.

So use:

> **“raw SDR overlap is determined by binary bit intersection; the final SF score is a deterministic transformation of the encoded spatial representation.”**

Then clearly distinguish:

```text
raw SDR overlap
        ↓
spatial transformation
        ↓
final SF score
```

This avoids a mathematical reviewer attacking the premise.

---

# 9. The scaling section is better—but the interpretation is still too aggressive

Good decision:

> You abandoned the O(√N) claim. 

Excellent.

But then you say:

> “This directly validates the score-concentration prediction” 

No.

The N-sweep:

```text
20 → 50 → 100 → 494
```

is **not enough to validate score concentration as the causal explanation**.

It demonstrates:

> operator performance changes as pool size changes.

It does not demonstrate:

> score concentration caused the change.

And the pattern itself is weird:

```text
CombSUM = 1.000 at every N
RRF = .667 → .883 → .783
```

That could have many causes.

### Rewrite as:

> “The pool-growth experiment shows that operator behavior remains stable for CombSUM while RRF varies across candidate regimes. This is consistent with, but does not establish, score-concentration as the underlying mechanism.”

That is reviewer-proof.

---

# 10. There is an internal contradiction around “full corpus”

This is important.

You say:

> “Full-corpus retrieval (Regime B)” 

but then:

> HotpotQA = 494 documents.

That's not necessarily a **full corpus in the IR sense**.

It's the full corpus **of your constructed HotpotQA evaluation collection**, perhaps.

A SIGIR reviewer will distinguish:

```text
full dataset corpus
```

from:

```text
large-scale first-stage retrieval
```

Your manuscript itself eventually admits this.

Therefore rename:

> **Full-dataset reranking**

rather than:

> **Full-corpus retrieval**

And say:

> “full-corpus with respect to the 494-document HotpotQA collection.”

This removes unnecessary ammunition.

---

# 11. Your 10-query / 50-query design remains the largest empirical weakness

This is probably the single most important remaining issue.

You have an enormous experimental matrix:

```text
8 datasets
× 7 operators
× 4 model pairs
```

but your core map uses:

> **10-query probes** 

That is extremely small.

You correctly acknowledge it, but a reviewer won't necessarily forgive it just because you say:

> “directional evidence.”

Your n=50 results are better, but only cover:

* HotpotQA
* NQ-REaR
* selected operators
* selected pairs.

And your own text admits that apparently huge effects disappear:

> BM25+SPLADE: 0.950 vs 0.850 → 0.940 vs 0.945. 

This is extremely important.

### My recommendation

Do **not** pretend the 8×7 matrix is statistically confirmatory.

Call it:

> **exploratory operator landscape**

Then designate:

### Confirmatory core

Use perhaps:

* HotpotQA
* MuSiQue
* NQ-REaR

and:

* SF+SPLADE
* SF+DPR

with:

* RRF
* CombSUM
* Linear

and **n≥100**, preferably the full available datasets.

Then use the remaining matrix as exploratory.

That would substantially improve the paper.

---

# 12. Your current statistical language is inconsistent

In §6.1:

> HotpotQA CombSUM = 1.000 vs RRF = 0.750.

This looks spectacular.

But the paper later tells the reader the n=10 map is exploratory.

The abstract nonetheless says:

> “we demonstrate…”

and the contribution says:

> “show…”

I would systematically separate:

### Strong verbs

For formally supported claims:

* establish
* demonstrate
* prove

### Medium

For empirical patterns:

* observe
* find
* consistently observe

### Weak

For n=10:

* suggest
* indicate
* provide exploratory evidence

This matters a lot for a journal reviewer.

---

# 13. Your synthetic experiment contains a conceptual inconsistency

This table:

| Condition |  A |  B | Margin |
| --------- | -: | -: | -----: |
| large     | 45 | 12 |    +33 |
| small     | 20 | 18 |     +2 |
| reversed  | 12 | 45 |    -33 |

cannot simultaneously be described as:

> “rank is held fixed”

if “rank” means the ranking induced by the same scores.

In the reversed condition:

```text
A = 12
B = 45
```

B is rank 1.

So you are holding **document identity / intended ordering** fixed, not score-induced rank.

That distinction is crucial.

### Fix the terminology

Say:

> “document identities are fixed, while score magnitudes are manipulated.”

If you want **rank truly fixed**, use two separate score vectors:

```text
ranking signal:
A > B
```

and manipulate a secondary magnitude channel.

Or apply monotonic transformations:

```text
[45,12] → [0.9,0.2] → [450,120]
```

which preserves rank.

---

# 14. Your “CombMNZ” explanation is too strong

You say:

> “CombMNZ explicitly rewards agreement across retrievers — a proxy for multi-hop evidence convergence.”

That's plausible, but not demonstrated.

CombMNZ rewards:

> number of systems returning the document.

That is not necessarily:

> complementary multi-hop evidence.

Agreement can mean redundancy.

I would write:

> “CombMNZ adds a multiplicity factor that rewards documents appearing in multiple component rankings. In our setting this may favor documents receiving support from both signals, although we do not equate multiplicity with genuine multi-hop evidence.”

Much safer.

---

# 15. The Kendall τ section has a serious methodological problem

You currently discuss:

> “Kendall's τ as a pre-fusion diagnostic.”

But the reported table is:

> τ between **operator rankings**. 

Earlier, however, your conceptual framework defines complementarity using:

> τ between the two **retrievers**.

These are different quantities.

You need explicitly distinguish:

### τ_signal

```text
τ(SF, SPLADE)
```

Measures component ranking agreement.

### τ_operator

```text
τ(RRF, CombSUM)
```

Measures whether two fusion operators produce similar final rankings.

Do not mix them.

This is likely to be caught by a reviewer.

---

# 16. Your statement “high τ → fusion adds little” is too simplistic

You say:

> “high τ ... signals redundancy; low τ ... signals complementarity.” 

This is common intuition, but not sufficient.

Two systems can have:

```text
τ = .85
```

and still differ at:

```text
rank 1 / rank 2
```

which matters enormously to MRR.

Your own later “bottleneck” discussion actually discovers this.

So your stronger conceptual model should be:

> **Global rank correlation is insufficient; fusion gain is often determined by local disagreement near the relevance boundary/top ranks.**

This could actually become a nice contribution.

I would add:

### Top-k disagreement

Measure:

* τ
* top-1 disagreement
* top-3 overlap
* gold rank disagreement
* pairwise gold-vs-best-negative margin.

That would strengthen the paper substantially.

---

# 17. The title is now good

I prefer:

> **What Does Fusion Preserve? Task- and Score-Geometry Dependent Information Loss in Hybrid Retrieval**

over the earlier titles.

It does three things well:

1. Doesn't claim a universal law.
2. Puts the information-preservation idea first.
3. Makes SF secondary.

Keep it.

---

# 18. The abstract needs one major correction

The abstract currently says:

> “we isolate magnitude information loss as the mechanism” 

That's still slightly too strong.

Change to:

> **“we identify magnitude information loss as a mechanism underlying several observed multi-hop reranking failures.”**

Also:

> “score magnitude can determine multi-hop reranking outcomes, consistent with a causal role”

is acceptable only if you carefully define **causal role of the operator**, not causal relevance of magnitude.

I would rewrite contribution #3 as:

> **“Through controlled magnitude perturbations, we demonstrate that magnitude-sensitive fusion operators respond to score magnitude even when ordinal information is held fixed, and connect this controlled behavior to observed retrieval traces.”**

That is excellent and defensible.

---

# 19. There is still a serious mismatch in the dataset description

The abstract says:

> “eight closed-domain question-answering datasets”

But the paper includes:

* PopQA
* PubMedQA
* NarrativeQA
* Belebele
* 2Wiki
* HotpotQA
* MuSiQue
* NQ-REaR

Correct.

However, NQ-REaR isn't simply a conventional “factoid dataset,” and the task topology table says:

> NQ-REaR → “Magnitude / separation.” 

That's an assumption.

I would classify it separately:

> **large-pool factoid retrieval / reranking**

and avoid claiming it requires magnitude in the same sense as multi-hop reasoning.

This matters because NQ-REaR is one of your strongest operator-difference datasets.

---

# 20. The “single-hop = rank sufficient” claim is too broad

You say:

> “For single-hop matching, rank is often sufficient.” 

Fine.

But:

> “Single-hop tasks show little operator sensitivity.”

is only true for your datasets, many of which hit a ceiling.

Your own PubMedQA = 0.800 is flat.

That's not necessarily evidence that:

> rank is sufficient.

It could mean:

> the candidate pool is too easy / hard / evaluation doesn't discriminate operators.

So phrase:

> “In our single-hop candidate-reranking conditions, operator sensitivity is largely masked by ceiling or floor effects.”

That's much more precise.

---

# 21. Your feature-invariance section should probably be shortened

Right now:

> §8.1 Feature Invariance
> §8.2 Non-Collinear Feature Tests — not implemented

The second subsection is literally:

> “[To be filled...]” 

This **cannot appear in a submitted paper**.

This is a hard stop.

You must either:

### Option A — run it

Best option.

Or:

### Option B — remove §8.2 entirely

and say:

> “A stronger adversarial test is left for future work.”

I strongly recommend **Option B unless you can run it properly**.

Never leave:

> `[To be filled]`

in the manuscript.

---

# 22. Same problem with the appendices

You say:

> Appendix C — **planned** statistical tables. 

This is another hard stop.

A submitted journal manuscript cannot say:

> “planned.”

Either produce it or remove the promise.

More importantly, you repeatedly refer to:

> Appendix D
> Appendix E
> Appendix G

without the actual material being part of the manuscript.

Before submission, every cited appendix must exist.

---

# 23. Your references are not submission-ready

This remains a major issue.

The manuscript literally says:

> “Citation verification ... was web-blocked ... pending a verification pass before submission.” 

That sentence **must not be in the submitted paper**.

Also, there are obvious suspicious/incomplete entries:

* NarrativeQA-adjacent citation;
* PubMedQA mixed into the NarrativeQA reference;
* 2Wiki citation listed as 2017;
* “NQ-REaR is derived from Natural Questions” needs the actual NQ-REaR source;
* some titles/authors need verification.

For a theory-oriented IR paper, citation quality is especially important.

**Do a complete bibliography audit before anything else.**

---

# 24. I would remove the deployment section

The:

> “No GPU; CPU-only query; ~512 B/doc...” 

is not helping your central contribution.

It's inherited from the SF paper.

The journal paper is no longer about:

> “SF is economically useful.”

It's about:

> **fusion information preservation.**

Move deployment considerations into a short paragraph in the discussion or appendix.

Don't spend scarce journal pages on it.

---

# 25. The paper should not call SF “fully characterized”

You repeatedly use:

> “fully-characterized heterogeneous probe”

But later acknowledge:

> feature invariance is unresolved;
> score concentration remains a hypothesis;
> magnitude semantics are inferred.

So use:

> **“controlled and comparatively transparent probe”**

instead of:

> “fully characterized.”

That's more defensible.

---

# 26. What I think the paper's real contribution is now

If I were rewriting the contribution paragraph as your chief reviewer, I'd make it:

### Contribution 1 — Information-preservation framework

We distinguish fusion operators according to whether they preserve:

* ordinal information;
* score magnitude;
* score scale;
* multiplicity.

### Contribution 2 — Controlled empirical map

Across:

```text
8 datasets
7 operators
4 retrieval pairs
```

we show that operator effectiveness is **conditional on both task and signal geometry**.

### Contribution 3 — Magnitude-blindness mechanism

We identify cases where rank-only fusion loses useful score-separation information and demonstrate operator sensitivity using controlled score interventions.

### Contribution 4 — Boundary analysis

We characterize two limitations of the SF probe:

* overlap-feature invariance;
* candidate-pool score concentration.

This is much stronger than claiming:

> “We discovered a law.”

---

# 27. The paper's strongest experimental result is not the 62.2% result anymore

This is important.

The old paper revolved around:

> MuSiQue: 0.482 → 0.782 = +62.2%

I would **not make that the journal headline anymore**.

Why?

Because the new experiments show:

* HotpotQA is much more discriminative;
* MuSiQue RRF and CombSUM can tie;
* n=50 weakens some dramatic differences;
* model-pair effects matter.

The strongest journal result is now:

> **operator behavior changes when the score geometry of the fused signals changes, even under the same task and same operator family.**

That is much more scientifically interesting.

---

# 28. Your new strongest experiment is the four-pair comparison

This is the centerpiece I would promote.

You have:

```text
              SPLADE       DPR
SF            SF+SPLADE    SF+DPR
BM25          BM25+SPLADE  BM25+DPR
```

This gives you a 2×2 design.

That is excellent.

But don't analyze it as:

> signal B determines operator.

Analyze it as:

### Factor A

Signal A:

```text
SF vs BM25
```

### Factor B

Signal B:

```text
SPLADE vs DPR
```

### Factor C

Task:

```text
Hotpot vs NQ
```

### Factor D

Operator:

```text
Linear / RRF / CombSUM
```

Now you're effectively performing a **factorial experiment**.

That is much more journal-worthy.

---

# 29. I would add one statistical model

If feasible, this would significantly strengthen the manuscript.

Instead of only pairwise MRR:

Use a mixed-effects model at the query level:

```text
MRR ~ Operator × RetrieverPair × Task
      + (1 | Query)
```

or an equivalent permutation/randomization framework.

Then you can ask:

> Is the Operator × RetrieverPair interaction significant?

That directly tests your central hypothesis.

You don't need huge theory.

Even:

> permutation test of operator × pair interaction

would be valuable.

This is more informative than dozens of pairwise comparisons.

---

# 30. Final decision if I were the reviewer

### Current version

**Recommendation: Weak Reject / Major Revision**

Reason:

> The manuscript presents an interesting and potentially significant empirical framework for understanding hybrid fusion through information preservation and score geometry. The revised scope is appropriately cautious, and the inclusion of multiple fusion operators and retriever pairings substantially improves the work. However, several central claims remain insufficiently supported. In particular, the causal interpretation of magnitude is stronger than the intervention actually establishes; the claim that signal-B geometry determines operator choice is contradicted by parts of the reported matrix; the n=10 exploratory evaluation is too small for several headline comparisons; and the candidate-growth experiment does not establish score concentration as the causal mechanism. Finally, the manuscript contains unfinished sections and unverified references that would preclude acceptance in its current form.

That's a **fixable review**, not a rejection of the research idea.

---

# 31. The exact final revision plan I recommend

Don't run 20 more experiments.

Do these **8 things**, in this order:

### P0 — Must fix before submission

**1. Remove causal overclaim**

Replace:

> causal magnitude contribution

with:

> controlled evidence for magnitude sensitivity.

---

**2. Fix the synthetic experiment**

Turn it into:

> real-score rank-preserving intervention + magnitude perturbation.

---

**3. Remove “signal B determines operator”**

Replace with:

> **operator effectiveness emerges from the interaction between task requirements and the score geometry of both component signals.**

---

**4. Resolve τ terminology**

Separate:

```text
τ_signal
τ_operator
```

and don't call operator-ranking τ a complementarity measure.

---

### P1 — Strongly recommended

**5. Make the 4-pair experiment the centerpiece**

Treat it as:

> 2 × 2 factorial signal-geometry experiment.

---

**6. Add one proper statistical interaction test**

Preferably:

```text
Operator × RetrieverPair × Task
```

with query-level resampling/permutation or mixed-effects analysis.

---

**7. Remove unfinished material**

Delete:

> `[To be filled]`

Delete:

> “planned appendix”

unless actually completed.

---

**8. Audit every reference and numerical claim**

Especially:

* NQ-REaR source;
* 2WikiMultihopQA citation;
* PubMedQA;
* NarrativeQA;
* Bruch et al.;
* all model/checkpoint references.

---

# 32. And one strategic recommendation

**Do not expand this paper further after these fixes.**

You are at the point where adding:

* another dataset,
* another operator,
* another theoretical principle,
* another architecture,

could actually make the paper **weaker**.

The paper now has a coherent story:

> **Fusion is an information bottleneck. Different operators preserve different score properties. Whether that information matters depends jointly on task requirements and signal geometry. SF gives us a controlled heterogeneous probe with which to study this interaction.**

That's a good journal story.

The remaining work is to make every experiment **support exactly that story—and nothing stronger.**

The most important revision is therefore not another benchmark. It is to make the paper **epistemically precise**: clearly distinguish what is **proved**, what is **experimentally established**, what is **strongly suggested**, and what remains **a hypothesis**. Your current draft is already moving in that direction; the remaining inconsistencies are concentrated and fixable. 
