# Win/Tie/Loss and Rank-1 Change Analysis: CombSUM vs RRF (n=100)

Per-query paired outcomes from the n=100 confirmatory core. 'Rank-1 change' counts queries where switching RRF -> CombSUM changes which document is ranked first — the information-bottleneck statistic.

| Dataset | n | CombSUM wins | RRF wins | ties | win% | rank-1 changes | rank-1 % | mean Δ | dz | n needed (power .8) |
|---------|--:|-------------:|---------:|-----:|-----:|---------------:|---------:|-------:|---:|--------------------:|
| hotpotqa | 100 | 21 | 1 | 78 | 21.0% | 10 | 10.0% | +0.093 | 0.45 | 40 |
| musique | 100 | 8 | 0 | 92 | 8.0% | 4 | 4.0% | +0.043 | 0.29 | 93 |
| nq_rear | 100 | 18 | 11 | 71 | 18.0% | 18 | 18.0% | +0.028 | 0.13 | 488 |
