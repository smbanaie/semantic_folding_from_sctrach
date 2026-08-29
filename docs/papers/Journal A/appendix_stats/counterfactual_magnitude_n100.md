# Counterfactual Relevance-Aligned Magnitude Intervention (Item 1, n=100)

Rank-preserving; RRF invariant by construction. CombSUM/CombMNZ changes are attributable to magnitude, not rank.


## hotpotqa (n=100)

RRF invariance minimum Kendall τ across worlds: **1.0000** (1.0 = ranks identical => RRF identical, as required).\n

### MRR by operator × world

| Operator | orig | compress | rpr | world+ (ρ=1.25) | world+ (ρ=1.5) | world+ (ρ=2.0) | world− |
|----------|-----:|--------:|----:|:|:|:|------:|
| rrf | 0.7973 | 0.7973 | 0.7973 | 0.7973 | 0.7973 | 0.7973 | 0.7973 |
| combsum | 0.9011 | 0.9011 | 0.7906 | 0.9266 | 0.9549 | 0.9624 | 0.8191 |
| combmnz | 0.9011 | 0.9011 | 0.7906 | 0.9266 | 0.9549 | 0.9624 | 0.8191 |
| linear | 0.9402 | 0.9402 | 0.8871 | 0.9618 | 0.9668 | 0.9668 | 0.8630 |
| borda | 0.7904 | 0.7904 | 0.7904 | 0.7904 | 0.7904 | 0.7904 | 0.7904 |
| zscore | 0.9368 | 0.9368 | 0.8871 | 0.9618 | 0.9668 | 0.9668 | 0.8697 |
| minmax | 0.9402 | 0.9402 | 0.8871 | 0.9618 | 0.9668 | 0.9668 | 0.8630 |

### Causal contrast (CombSUM, bootstrap 95% CI, B=10000)

- World+ vs orig: ΔMRR = +0.0538 (95% CI [+0.0254, +0.0864])\n
- orig vs World−: ΔMRR = +0.0820 (95% CI [+0.0447, +0.1254])\n
- RRF World+ vs orig ΔMRR = +0.0000 (must be ~0; invariance check)\n

### H3a — rank-conditioned relevance gap E[s|y=1,r]−E[s|y=0,r]\n
| rank bucket | gap |
|------------|----:|
| 1 | n/a |
| 2-3 | 0.0418 |
| 4-5 | 0.0182 |
| 6-10 | 0.0229 |

P(y=1 | large separation) = 0.000; P(y=1 | small separation) = 0.025; AUC(score) = 0.980\n

## musique (n=100)

RRF invariance minimum Kendall τ across worlds: **1.0000** (1.0 = ranks identical => RRF identical, as required).\n

### MRR by operator × world

| Operator | orig | compress | rpr | world+ (ρ=1.25) | world+ (ρ=1.5) | world+ (ρ=2.0) | world− |
|----------|-----:|--------:|----:|:|:|:|------:|
| rrf | 0.7478 | 0.7478 | 0.7478 | 0.7478 | 0.7478 | 0.7478 | 0.7478 |
| combsum | 0.8074 | 0.8074 | 0.7380 | 0.8477 | 0.8724 | 0.9097 | 0.7549 |
| combmnz | 0.8074 | 0.8074 | 0.7380 | 0.8477 | 0.8724 | 0.9097 | 0.7549 |
| linear | 0.9299 | 0.9299 | 0.7927 | 0.9499 | 0.9499 | 0.9499 | 0.8679 |
| borda | 0.7263 | 0.7263 | 0.7264 | 0.7263 | 0.7263 | 0.7263 | 0.7263 |
| zscore | 0.9380 | 0.9380 | 0.7919 | 0.9448 | 0.9448 | 0.9498 | 0.8506 |
| minmax | 0.9299 | 0.9299 | 0.7927 | 0.9499 | 0.9499 | 0.9499 | 0.8679 |

### Causal contrast (CombSUM, bootstrap 95% CI, B=10000)

- World+ vs orig: ΔMRR = +0.0650 (95% CI [+0.0322, +0.1035])\n
- orig vs World−: ΔMRR = +0.0525 (95% CI [+0.0278, +0.0814])\n
- RRF World+ vs orig ΔMRR = +0.0000 (must be ~0; invariance check)\n

### H3a — rank-conditioned relevance gap E[s|y=1,r]−E[s|y=0,r]\n
| rank bucket | gap |
|------------|----:|
| 1 | n/a |
| 2-3 | 0.0739 |
| 4-5 | 0.0291 |
| 6-10 | 0.0148 |

P(y=1 | large separation) = 0.002; P(y=1 | small separation) = 0.021; AUC(score) = 0.900\n

## nq_rear (n=100)

RRF invariance minimum Kendall τ across worlds: **1.0000** (1.0 = ranks identical => RRF identical, as required).\n

### MRR by operator × world

| Operator | orig | compress | rpr | world+ (ρ=1.25) | world+ (ρ=1.5) | world+ (ρ=2.0) | world− |
|----------|-----:|--------:|----:|:|:|:|------:|
| rrf | 0.6702 | 0.6702 | 0.6702 | 0.6702 | 0.6702 | 0.6702 | 0.6702 |
| combsum | 0.7359 | 0.7359 | 0.6744 | 0.7607 | 0.7885 | 0.7962 | 0.6569 |
| combmnz | 0.7359 | 0.7359 | 0.6744 | 0.7607 | 0.7885 | 0.7962 | 0.6569 |
| linear | 0.7120 | 0.7120 | 0.7157 | 0.7669 | 0.7933 | 0.8034 | 0.6597 |
| borda | 0.6744 | 0.6744 | 0.6744 | 0.6744 | 0.6744 | 0.6744 | 0.6744 |
| zscore | 0.7240 | 0.7240 | 0.7157 | 0.7539 | 0.7789 | 0.8050 | 0.6560 |
| minmax | 0.7120 | 0.7120 | 0.7157 | 0.7669 | 0.7933 | 0.8034 | 0.6597 |

### Causal contrast (CombSUM, bootstrap 95% CI, B=10000)

- World+ vs orig: ΔMRR = +0.0526 (95% CI [+0.0251, +0.0836])\n
- orig vs World−: ΔMRR = +0.0790 (95% CI [+0.0435, +0.1195])\n
- RRF World+ vs orig ΔMRR = +0.0000 (must be ~0; invariance check)\n

### H3a — rank-conditioned relevance gap E[s|y=1,r]−E[s|y=0,r]\n
| rank bucket | gap |
|------------|----:|
| 1 | n/a |
| 2-3 | 0.0135 |
| 4-5 | 0.0086 |
| 6-10 | 0.0377 |

P(y=1 | large separation) = 0.000; P(y=1 | small separation) = 0.044; AUC(score) = 0.986\n