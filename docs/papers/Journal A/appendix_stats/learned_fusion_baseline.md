# Learned Fusion Baseline vs Fixed Operators (Item 20)

Logistic regression over [s_A, s_B, s_A_norm, s_B_norm]; leave-one-query-out CV (train on other queries' documents, score the held-out query). Identical pools/golds as §7.5.

| Dataset | n | rrf | combsum | learned (LOQO-CV) |
|---------|--:|----:|--------:|------------------:|
| hotpotqa | 10 | 0.883 | 1.0 | 1.0 |
| musique | 10 | 0.861 | 0.914 | 0.933 |
| scifact | 10 | 0.821 | 0.82 | 0.823 |
