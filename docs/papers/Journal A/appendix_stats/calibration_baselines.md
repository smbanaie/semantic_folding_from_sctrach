# Calibration Baselines: magnitude vs calibration (review item 0.8)

Per-signal normalization applied before fusion; fused with three score-space operators. n=10 exploratory queries per dataset (same traces as §7.4). If calibration alone explains CombSUM's advantage, some normalizer should match or beat raw.


## hotpotqa

| Normalization | CombSUM | CombMNZ | Linear(α=0.3) |
|---------------|--------:|--------:|--------:|
| raw | 1.000 | 1.000 | 1.000 |
| minmax | 1.000 | 1.000 | 1.000 |
| zscore | 1.000 | 1.000 | 1.000 |
| l2 | 1.000 | 1.000 | 1.000 |
| rank_gauss | 0.900 | 0.900 | 1.000 |
| sigmoid | 0.883 | 0.883 | 0.883 |
| quantile | 0.683 | 0.683 | 0.950 |
| softmax | 1.000 | 1.000 | 1.000 |

## musique

| Normalization | CombSUM | CombMNZ | Linear(α=0.3) |
|---------------|--------:|--------:|--------:|
| raw | 0.914 | 0.903 | 0.950 |
| minmax | 0.917 | 0.905 | 0.950 |
| zscore | 0.920 | 0.911 | 0.950 |
| l2 | 0.905 | 0.903 | 0.950 |
| rank_gauss | 0.950 | 0.911 | 0.920 |
| sigmoid | 0.853 | 0.853 | 0.820 |
| quantile | 0.855 | 0.853 | 0.911 |
| softmax | 0.920 | 0.914 | 0.950 |

## scifact

| Normalization | CombSUM | CombMNZ | Linear(α=0.3) |
|---------------|--------:|--------:|--------:|
| raw | 0.820 | 0.820 | 0.823 |
| minmax | 0.820 | 0.820 | 0.823 |
| zscore | 0.820 | 0.820 | 0.823 |
| l2 | 0.821 | 0.821 | 0.823 |
| rank_gauss | 0.824 | 0.824 | 0.820 |
| sigmoid | 0.828 | 0.828 | 0.826 |
| quantile | 0.821 | 0.821 | 0.820 |
| softmax | 0.820 | 0.820 | 0.823 |
