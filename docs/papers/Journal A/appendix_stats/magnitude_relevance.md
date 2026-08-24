# Magnitude Relevance Analysis (S3; H3 test)

Real component traces; delta = gold score - best negative score per retriever. Hop coverage = candidate is a dataset-annotated supporting document. Regression controls: doc word length, query-term lexical overlap.


## Margin statistics by retriever and task

| Dataset | Task | Retriever | P(delta>0) | mean delta | median delta | AUC(score) | n queries |
|---------|------|-----------|-----------:|-----------:|-------------:|-----------:|----------:|
| hotpotqa | multi-hop | SF | 0.20 | -0.066 | -0.030 | 0.966 | 10 |
| hotpotqa | multi-hop | SPLADE | 1.00 | 0.191 | 0.179 | 0.981 | 10 |
| musique | multi-hop | SF | 0.33 | -0.063 | -0.082 | 0.900 | 9 |
| musique | multi-hop | SPLADE | 1.00 | 0.301 | 0.365 | 0.866 | 10 |
| scifact | claim-verification | SF | 0.60 | 0.142 | 0.321 | 0.965 | 10 |
| scifact | claim-verification | SPLADE | 0.80 | 0.296 | 0.469 | 0.975 | 10 |

## Score ~ supporting-status analysis (title-matched supporting docs; controls: length, overlap)

| Dataset | Retriever | Spearman(score, supporting) | n docs | n supporting |
|---------|-----------|----------------------------:|-------:|-------------:|
| hotpotqa | SF | 0.240 | 940 | 21 |
| hotpotqa | SPLADE | 0.247 | 940 | 21 |
| musique | SF | 0.176 | 1000 | 16 |
| musique | SPLADE | 0.173 | 1000 | 19 |
| scifact | SF | 0.207 | 660 | 11 |
| scifact | SPLADE | 0.211 | 660 | 11 |

## Calibration: P(supporting/gold | score bin)

| Dataset | Retriever | bin | n | P(gold) |
|---------|-----------|-----|--:|--------:|
| hotpotqa | SPLADE | [0.0,0.2) | 756 | 0.000 |
| hotpotqa | SPLADE | [0.2,0.4) | 103 | 0.029 |
| hotpotqa | SPLADE | [0.4,0.6) | 30 | 0.033 |
| hotpotqa | SPLADE | [0.6,0.8) | 24 | 0.042 |
| hotpotqa | SPLADE | [0.8,1.0) | 27 | 0.556 |
| musique | SPLADE | [0.0,0.2) | 293 | 0.003 |
| musique | SPLADE | [0.2,0.4) | 510 | 0.008 |
| musique | SPLADE | [0.4,0.6) | 142 | 0.021 |
| musique | SPLADE | [0.6,0.8) | 34 | 0.000 |
| musique | SPLADE | [0.8,1.0) | 21 | 0.524 |
| scifact | SPLADE | [0.0,0.2) | 497 | 0.000 |
| scifact | SPLADE | [0.2,0.4) | 107 | 0.009 |
| scifact | SPLADE | [0.4,0.6) | 31 | 0.000 |
| scifact | SPLADE | [0.6,0.8) | 11 | 0.091 |
| scifact | SPLADE | [0.8,1.0) | 14 | 0.643 |
