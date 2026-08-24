If I were doing the research myself, with the full benchmark datasets, retrieval models, and compute available, I would **stop adding breadth** and build a tightly controlled experimental program around the weaknesses we identified.

The goal would be to turn the paper from:

> “We observe that some operators work better than others.”

into:

> **“We can experimentally isolate which information each fusion operator preserves, demonstrate when that information matters, and quantify how this interaction depends on task and retriever score geometry.”**

Below is the implementation plan I would actually execute.

---

# 1. First: freeze the research question

I would formally redefine the main hypothesis as:

> **H1 — Fusion Information Hypothesis:** Fusion operators differ in the information they preserve. The utility of that preserved information depends on the retrieval task and on the score geometry of the component retrievers.

Then split it into four testable hypotheses:

### H1 — Rank invariance

If two score sets induce the same ranking, a rank-only operator should produce the same result.

Test:

```text
s
↓ monotonic transformation
f(s)
↓
same ranking
```

RRF/Borda should be invariant.

---

### H2 — Magnitude sensitivity

Score-space operators should respond to magnitude changes even when ranking information is unchanged.

This is where your current synthetic experiment needs to become much stronger.

---

### H3 — Magnitude relevance

Magnitude-sensitive operators improve retrieval **only when magnitude contains useful relevance information**.

This is the difficult one.

We need to distinguish:

```text
magnitude sensitivity
```

from:

```text
useful magnitude information
```

---

### H4 — Interaction

Operator effectiveness depends on:

```text
Task × Signal Geometry × Operator
```

not simply:

```text
Task → Operator
```

This should become your strongest empirical contribution.

---

# 2. Build a proper experimental framework first

Before running experiments, I'd create one canonical data structure.

Something like:

```python
@dataclass
class RetrievalRun:
    query_id: str
    doc_id: str
    relevance: int
    score: float
    rank: int
    retriever: str
    dataset: str
    split: str
```

Every experiment should consume this same object.

For each:

```text
dataset
query
retriever
```

store:

```text
doc_id
relevance
raw_score
rank
```

Never recompute scores differently for different experiments.

---

# 3. Store the raw retrieval outputs permanently

For every model:

### Lexical

```text
BM25
```

### Sparse neural

```text
SPLADE
```

### Dense

```text
DPR
```

### Your probe

```text
SF
```

I'd generate:

```text
runs/
    hotpotqa/
        bm25.parquet
        splade.parquet
        dpr.parquet
        sf.parquet

    musique/
        ...
```

Each record:

```text
query_id
doc_id
score
rank
relevance
```

This is essential because all subsequent interventions should operate on **frozen retrieval traces**.

No model rerunning.

---

# 4. Experiment 1 — prove rank invariance properly

This is the easy experiment.

For every retrieval score vector `s`, generate:

### Linear

```python
s1 = s
```

### Scaling

```python
s2 = 10 * s
```

### Translation

```python
s3 = s + 100
```

### Exponential

```python
s4 = np.exp(s)
```

### Power

```python
s5 = np.sign(s) * np.abs(s)**3
```

All monotonic transformations.

They preserve:

```text
rank(s) == rank(si)
```

Then evaluate:

```text
RRF(s)
Borda(s)
CombSUM(s)
CombMNZ(s)
Linear(s)
```

Expected:

| Operator | Rank preserved? | Output invariant? |
| -------- | --------------: | ----------------: |
| RRF      |             Yes |           **Yes** |
| Borda    |             Yes |           **Yes** |
| CombSUM  |              No |                No |
| CombMNZ  |              No |                No |
| Linear   |              No |                No |

This gives you a clean empirical verification of the information taxonomy.

But this isn't the main contribution.

It's your **mechanistic validation experiment**.

---

# 5. Experiment 2 — real retrieval traces + magnitude intervention

This is the experiment I would make central.

Take a real query.

Suppose:

```text
Retriever A

Gold       score=0.82 rank=2
Negative   score=0.91 rank=1
```

and Retriever B:

```text
Gold       score=43.0 rank=1
Negative   score=12.0 rank=2
```

Now construct controlled counterfactuals.

---

## Condition A — Original

```text
A: [0.91, 0.82]
B: [43, 12]
```

---

## Condition B — Magnitude compressed

Normalize B:

```text
[43,12] → [1.0, 0.99]
```

while preserving ranking.

---

## Condition C — Magnitude amplified

```text
[43,12] → [1000, 1]
```

---

## Condition D — Magnitude equalized

```text
rank 1 → 1
rank 2 → 0.999
```

---

## Condition E — Magnitude randomized

Generate:

```text
same ranking
random magnitudes
```

The critical property:

> **Ranking stays unchanged.**

Now compare:

```text
RRF
Borda
CombSUM
CombMNZ
Linear
```

If RRF remains identical but CombSUM changes dramatically, you have demonstrated:

> **operator-level magnitude sensitivity on real retrieval traces.**

That is much stronger than the current two-document toy example.

---

# 6. Experiment 3 — distinguish magnitude sensitivity from magnitude usefulness

This is the experiment I consider most important.

For every query, calculate:

```text
gold score
best-negative score
```

Then:

```text
Δscore = score_gold - score_best_negative
```

For each retriever calculate:

```text
P(Δscore > 0)
mean Δscore
median Δscore
AUC(score)
```

Do this separately for:

```text
single-hop
multi-hop
```

and:

```text
SF
SPLADE
DPR
BM25
```

Now you can ask:

> Does score magnitude actually separate relevant from non-relevant documents?

If:

```text
multi-hop:
gold margin >> single-hop
```

then your magnitude hypothesis gains empirical support.

If it doesn't, **you must weaken the paper's claim**.

That's science, and it protects the paper.

---

# 7. Experiment 4 — directly test the “compositional confidence” hypothesis

I would absolutely do this.

For HotpotQA and MuSiQue, use the annotated reasoning structure.

For each candidate document calculate:

```text
hop relevance
```

For example:

```text
Doc A:
hop 1 = relevant
hop 2 = relevant

Doc B:
hop 1 = relevant
hop 2 = irrelevant
```

Then compare score magnitude.

You want to know:

```text
SPLADE score
      ↓
does it correlate with
      ↓
number of satisfied hops?
```

Calculate:

```text
Spearman(score, hop_coverage)
```

and perhaps:

```text
score ~ hop_coverage + lexical_overlap + doc_length
```

using a regression.

If you obtain:

```text
β_hop > 0
p < 0.05
```

after controlling for lexical overlap and document length, **your paper becomes much stronger**.

Now you can legitimately say magnitude contains information related to compositional evidence.

If it doesn't:

Don't call it compositional confidence.

Call it:

> **score-separation information.**

---

# 8. Experiment 5 — control for obvious SPLADE confounders

A reviewer will ask:

> Maybe SPLADE magnitude is just document length.

So calculate:

```text
score
document length
query length
query-document lexical overlap
number of activated terms
SPLADE non-zero dimensions
```

Then test:

```text
score ~ length + overlap + expansion_density + hop_coverage
```

This is critical.

You need to determine whether:

```text
SPLADE magnitude
```

contains information beyond trivial lexical/document-size effects.

---

# 9. Experiment 6 — your four-pair experiment should become a factorial experiment

You already have:

```text
                 SPLADE       DPR
SF               ✓            ✓
BM25             ✓            ✓
```

Don't just report four tables.

Treat it formally.

Factors:

```text
A = first retriever
    SF / BM25

B = second retriever
    SPLADE / DPR

C = operator
    RRF / Borda / CombSUM / CombMNZ / Linear

D = task
    single-hop / multi-hop
```

Then calculate:

```text
MRR ~ A * B * C * D
```

or, better, query-level mixed effects / permutation analysis.

Your key question becomes:

> Is there an interaction between operator and retriever pair?

That's exactly what the paper claims.

---

# 10. Experiment 7 — stop using only MRR

This is important.

MRR is very sensitive to rank 1.

Add:

### MRR

For consistency with previous paper.

### Recall@k

```text
R@1
R@5
R@10
```

### nDCG@k

Especially important if relevance isn't binary.

### Rank of gold

Report:

```text
mean reciprocal rank
median rank
P(gold in top-k)
```

### Pairwise margin

For compositional tasks:

```text
score_gold - score_best_negative
```

This directly connects to your hypothesis.

---

# 11. Experiment 8 — evaluate the fusion operators at multiple candidate-set sizes

Don't use only:

```text
20
50
100
494
```

where possible.

Create:

```text
N ∈ {5, 10, 20, 50, 100, 250, 500, 1000}
```

For every query.

But here's the critical part:

### Don't sample arbitrary negatives.

Construct:

#### Random negatives

and

#### Hard negatives

separately.

Because score concentration on random negatives tells you little about realistic retrieval.

You want:

```text
random negatives
vs.
BM25 hard negatives
vs.
dense hard negatives
```

Then plot:

```text
candidate pool size
       ↓
score distribution
       ↓
operator performance
```

---

# 12. Experiment 9 — actually test the scaling hypothesis

Don't claim:

> score concentration causes performance degradation

until you measure it.

For every candidate size N calculate:

### Mean

```text
μ
```

### Standard deviation

```text
σ
```

### Coefficient of variation

```text
CV = σ / |μ|
```

### Top-vs-random separation

```text
Δ = score_gold - mean(score_negatives)
```

### Standardized separation

```text
z = Δ / σ_negative
```

Then plot:

```text
N
↓
CV
↓
gold-vs-negative z-score
↓
MRR
```

Now you can test whether:

```text
score concentration
```

actually predicts:

```text
retrieval degradation
```

using correlation/regression.

---

# 13. Experiment 10 — replace “Scaling Wall” with a measurable phenomenon

I would rename it:

> **Score Concentration Under Candidate Growth**

Then define:

[
C(N)=\frac{\mu_{\text{gold}}-\mu_{\text{negative}}}{\sigma_{\text{negative}}}
]

This is much cleaner.

You can say:

> As candidate pool size increases, does normalized score separation decline?

If yes:

**excellent.**

You have a real empirical phenomenon.

No need for a questionable asymptotic theorem.

---

# 14. Experiment 11 — local disagreement is more important than global Kendall τ

I would add this.

For each query:

```text
τ(SF,SPLADE)
```

but also:

```text
top1 agreement
top3 overlap
top5 overlap
gold-rank difference
```

Then calculate:

```text
Fusion Gain =
MRR(fused) - max(MRR(A), MRR(B))
```

Test:

```text
Fusion Gain ~ τ
```

and separately:

```text
Fusion Gain ~ top-k disagreement
```

I suspect you'll find:

> global Kendall τ is a weak predictor.

That would be an interesting secondary result.

---

# 15. Experiment 12 — identify where complementarity actually occurs

For every query classify it:

```text
A correct, B wrong
A wrong, B correct
A correct, B correct
A wrong, B wrong
```

Then measure fusion performance.

This produces a very interpretable table:

| A | B | RRF | CombSUM | Linear |
| - | - | --: | ------: | -----: |
| ✓ | ✓ |     |         |        |
| ✓ | ✗ |     |         |        |
| ✗ | ✓ |     |         |        |
| ✗ | ✗ |     |         |        |

This is much more informative than overall MRR.

Especially:

> **How does each operator behave when the signals disagree?**

That's where fusion matters.

---

# 16. Experiment 13 — analyze rank-vs-score disagreement explicitly

For every query where operators disagree:

```text
RRF says A
CombSUM says B
```

calculate:

```text
rank_A
rank_B
score_A
score_B
score_margin
```

Then determine:

> **what information caused the decision difference?**

You can automatically generate cases.

This can replace your manually constructed case studies.

---

# 17. Make the qualitative examples data-driven

Instead of:

> “Here is a hypothetical query…”

extract:

### Top 10 disagreement cases

where:

```text
RRF ≠ CombSUM
```

and:

```text
CombSUM correct
RRF wrong
```

Then show:

```text
query
gold document
RRF ranks
CombSUM ranks
raw scores
score margins
```

Do the reverse too:

```text
RRF correct
CombSUM wrong
```

This gives you genuine evidence for the mechanism.

---

# 18. Experiment 14 — add calibration

This would be excellent.

For each retriever calculate whether score magnitude is calibrated to relevance.

For example:

```text
score bins:
0–10
10–20
20–30
30–40
40+
```

Then:

```text
P(relevant | score bin)
```

If higher score actually means higher relevance probability, magnitude has semantic value.

If:

```text
P(relevant | score)
```

is nearly flat, your magnitude hypothesis is weak.

This is an extremely clean test.

---

# 19. Experiment 15 — normalize scores before fusion

You need this because someone will say:

> “Linear fusion loses because you didn't normalize scores.”

Run:

### Raw Linear

```text
αA + (1-α)B
```

### Min-max

```text
α A' + (1-α) B'
```

### Z-score

```text
α Z(A) + (1-α) Z(B)
```

### Rank-normalized

```text
α rank(A) + (1-α) rank(B)
```

Then compare with RRF.

This lets you make a much more sophisticated argument:

> **The problem is not simply “linear fusion is bad”; rather, normalization determines which information survives.**

That would directly connect your work to classical fusion literature.

---

# 20. Experiment 16 — α sweep

Do not use:

```text
α = 0.3
```

as the only value.

Use:

```text
α ∈ {
0.0,
0.1,
0.2,
...
0.9,
1.0
}
```

For each:

```text
dataset
retriever pair
candidate size
```

Plot:

```text
MRR vs α
```

Then you can identify:

> whether multi-hop actually prefers magnitude-preserving mixtures.

And whether:

> the optimum α is stable across datasets.

If α varies wildly, that's itself an important result.

---

# 21. Experiment 17 — RRF k sweep

Similarly:

```text
k ∈ {1,5,10,20,30,60,100,200}
```

Don't simply say:

> k=60 is standard.

Show robustness.

If RRF remains poor across k:

**very strong evidence.**

If some k rescues it:

your claim needs to become more nuanced.

---

# 22. Experiment 18 — operator family rather than individual operators

Your paper will become conceptually cleaner if you classify:

### Rank-only

```text
RRF
Borda
```

### Magnitude-preserving

```text
CombSUM
Linear
```

### Multiplicity-aware

```text
CombMNZ
```

### Normalized score-space

```text
z-score
min-max
```

Then ask:

> Does the **information-preservation class** predict performance better than the operator identity?

That's a much more general contribution.

---

# 23. The final experimental matrix I would actually run

Don't run everything on every dataset.

I'd use:

### Full benchmark

```text
8 datasets
×
5 core operators
```

for broad mapping.

Core:

```text
RRF
Borda
CombSUM
CombMNZ
Linear
```

---

### Deep analysis

Use:

```text
HotpotQA
MuSiQue
NQ-REaR
PubMedQA
```

because they represent:

```text
multi-hop
multi-hop deeper
large candidate pool
domain-specific
```

---

### Retriever pairs

```text
SF + SPLADE
SF + DPR
BM25 + SPLADE
BM25 + DPR
```

---

### Deep intervention

Only:

```text
HotpotQA
MuSiQue
```

with real traces.

That is enough.

---

# 24. The final paper would then have this evidence hierarchy

### Level 1 — Mathematical

Prove:

```text
RRF(f(s)) = RRF(s)
```

for monotonic transformations.

Therefore:

> RRF discards score magnitude.

---

### Level 2 — Controlled experiment

Show:

```text
same ranking
different magnitude
        ↓
RRF unchanged
CombSUM changes
```

Therefore:

> operators differ in information sensitivity.

---

### Level 3 — Real retrieval

Show:

```text
magnitude perturbation
        ↓
different ranking outcomes
```

Therefore:

> magnitude can affect fusion decisions.

---

### Level 4 — Relevance analysis

Show:

```text
score magnitude
        ↓
relevance / hop coverage / separation
```

Therefore:

> magnitude is not merely arbitrary numerical scale.

---

### Level 5 — Task interaction

Show:

```text
task × signal geometry × operator
        ↓
performance
```

Therefore:

> the usefulness of preserved information is task-dependent.

---

# 25. Then your conclusion becomes extremely defensible

You would be able to say:

> **We do not claim that rank-based fusion is inherently unsuitable for multi-hop retrieval, nor that score magnitude universally represents reasoning depth. Instead, we show that fusion operators impose distinct information bottlenecks: rank-based operators discard score magnitude, whereas score-based operators retain it. Through controlled interventions and real retrieval traces, we demonstrate that this discarded information can affect ranking decisions, and that its utility varies with task requirements and the score geometry of the component retrievers.**

That is a **very strong SIGIR/TOIS conclusion**.

---

# 26. If I had to prioritize the work

I would spend the research time like this:

| Priority | Experiment                                    | Importance     |
| -------- | --------------------------------------------- | -------------- |
| 🔴 P0    | Real-trace magnitude intervention             | **Critical**   |
| 🔴 P0    | Full 4-pair factorial analysis                | **Critical**   |
| 🔴 P0    | Score magnitude vs relevance / hop coverage   | **Critical**   |
| 🔴 P0    | Remove “B determines operator” claim          | **Critical**   |
| 🔴 P0    | α + RRF-k sensitivity                         | **Critical**   |
| 🟠 P1    | Candidate-size × score-concentration analysis | Very high      |
| 🟠 P1    | Hard vs random negatives                      | Very high      |
| 🟠 P1    | Local disagreement vs Kendall τ               | High           |
| 🟠 P1    | Score normalization experiments               | High           |
| 🟡 P2    | Calibration                                   | Useful         |
| 🟡 P2    | Additional datasets                           | Not necessary  |
| 🟢 P3    | More fusion operators                         | Already enough |

---

## The one experiment I would not compromise on

If you only have time to implement **one** major experiment, make it this:

```text
REAL RETRIEVAL TRACE
        │
        ├── Original scores
        │
        ├── Rank-preserving compressed scores
        │
        ├── Rank-preserving amplified scores
        │
        ├── Rank-preserving randomized magnitudes
        │
        └── Magnitude-swapped scores
                 │
                 ▼
        ┌────────────────────┐
        │ RRF / Borda        │
        │ CombSUM / CombMNZ  │
        │ Linear / normalized│
        └────────────────────┘
                 │
                 ▼
       Ranking + MRR + nDCG
                 │
                 ▼
      Does discarded magnitude
      actually change decisions?
```

Run this on **real HotpotQA + MuSiQue traces**, then connect the resulting score margins to **gold-vs-negative separation and hop coverage**.

That experiment directly addresses the central weakness I identified in the current manuscript. If it produces the expected result, I would become substantially more positive about the paper; if it does not, you'll have a scientifically valuable negative result that tells us to narrow the central claim rather than forcing the “magnitude fallacy” narrative.
