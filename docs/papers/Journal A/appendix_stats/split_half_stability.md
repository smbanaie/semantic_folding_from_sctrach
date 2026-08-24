# Split-Half Stability of the n=100 Confirmatory Core

200 random disjoint 50/50 query partitions (seed=42). Reported: per-operator mean MRR across halves, its split-to-split std, and the fraction of splits where CombSUM beats RRF on BOTH halves (sign stability of the paired difference).

| Dataset | Operator | mean MRR (split-half) | std across 400 halves |
|---------|----------|----------------------:|----------------------:|
| hotpotqa | combsum | 0.947 | 0.016 |
| hotpotqa | rrf | 0.854 | 0.024 |
| hotpotqa | zscore | 0.896 | 0.023 |
| hotpotqa | linear | 0.702 | 0.031 |
| **hotpotqa** | **CombSUM>RRF on both halves** | **100.0%** | |
| musique | combsum | 0.952 | 0.019 |
| musique | rrf | 0.908 | 0.026 |
| musique | zscore | 0.952 | 0.019 |
| musique | linear | 0.832 | 0.034 |
| **musique** | **CombSUM>RRF on both halves** | **99.5%** | |
| nq_rear | combsum | 0.746 | 0.035 |
| nq_rear | rrf | 0.718 | 0.035 |
| nq_rear | zscore | 0.733 | 0.034 |
| nq_rear | linear | 0.682 | 0.036 |
| **nq_rear** | **CombSUM>RRF on both halves** | **79.0%** | |
