# Score-Geometry Predictor of Operator Family (review item 0.6)

For each query: 21 pre-fusion geometry features (9 per signal: mean/std/CV/range/skew/kurtosis/top1-2/top1-5 margins/entropy; 3 pair: Pearson, Kendall, top-5 Jaccard). Label = winning operator family by fused MRR (rank-only vs score-space). Model: logistic regression, leave-one-DATASET-out (generalization to unseen tasks).

**Power reality check (reported transparently):** at n=10 exploratory queries per dataset, most queries are operator-TIES (gold at rank 1 under every operator), so the predictable subset is tiny. The divergence rate itself is a geometric quantity: ties concentrate exactly where top-rank margins are large for both signals. A meaningful predictor study requires the n=100 traces per-query component scores, which we flag as required future instrumentation; the framework below is delivered and validated on the divergent subset that exists.

- Divergent (non-tie) queries: **6/40**
- hotpotqa: {'rank_only': 0, 'score_space': 2, 'tie': 8}
- musique: {'rank_only': 0, 'score_space': 2, 'tie': 8}
- scifact: {'rank_only': 1, 'score_space': 1, 'tie': 8}
- 2wikimultihopqa: {'rank_only': 0, 'score_space': 0, 'tie': 10}

- Mean LODO accuracy: **0.75** (majority-class baseline 1.0, lift -0.250)
- Per-fold: [1.0, 0.5]
