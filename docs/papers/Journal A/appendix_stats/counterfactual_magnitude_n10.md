# Counterfactual Relevance-Aligned Magnitude Intervention (Item 1, n=10)

Rank-preserving; RRF invariant by construction. CombSUM/CombMNZ changes are attributable to magnitude, not rank.


## hotpotqa (n=10)

RRF invariance minimum Kendall τ across worlds: **1.0000** (1.0 = ranks identical => RRF identical, as required).\n

### MRR by operator × world

| Operator | orig | compress | rpr | world+ (ρ=1.25) | world+ (ρ=1.5) | world+ (ρ=2.0) | world− |
|----------|-----:|--------:|----:|:|:|:|------:|
| rrf | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| combsum | 1.0000 | 1.0000 | 0.8833 | 1.0000 | 1.0000 | 1.0000 | 0.6033 |
| combmnz | 1.0000 | 1.0000 | 0.8833 | 1.0000 | 1.0000 | 1.0000 | 0.6033 |
| linear | 1.0000 | 1.0000 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 0.7750 |
| borda | 0.8833 | 0.8833 | 0.8833 | 0.8833 | 0.8833 | 0.8833 | 0.8833 |
| zscore | 1.0000 | 1.0000 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 0.7750 |
| minmax | 1.0000 | 1.0000 | 0.9500 | 1.0000 | 1.0000 | 1.0000 | 0.7750 |

### Causal contrast (CombSUM, bootstrap 95% CI, B=10000)

- World+ vs orig: ΔMRR = +0.0000 (95% CI [+0.0000, +0.0000])\n
- orig vs World−: ΔMRR = +0.3967 (95% CI [+0.2167, +0.5600])\n
- RRF World+ vs orig ΔMRR = +0.0000 (must be ~0; invariance check)\n

### H3a — rank-conditioned relevance gap E[s|y=1,r]−E[s|y=0,r]\n
| rank bucket | gap |
|------------|----:|
| 1 | n/a |
| 2-3 | -0.0021 |
| 4-5 | -0.1184 |
| 6-10 | -0.0671 |

P(y=1 | large separation) = 0.000; P(y=1 | small separation) = 0.043; AUC(score) = 0.976\n

## musique (n=10)

RRF invariance minimum Kendall τ across worlds: **0.9980** (1.0 = ranks identical => RRF identical, as required).\n

### MRR by operator × world

| Operator | orig | compress | rpr | world+ (ρ=1.25) | world+ (ρ=1.5) | world+ (ρ=2.0) | world− |
|----------|-----:|--------:|----:|:|:|:|------:|
| rrf | 0.8111 | 0.8111 | 0.8111 | 0.8111 | 0.8111 | 0.8111 | 0.8111 |
| combsum | 0.9125 | 0.9125 | 0.8043 | 0.9143 | 0.9143 | 0.9200 | 0.8625 |
| combmnz | 0.9125 | 0.9125 | 0.8043 | 0.9143 | 0.9143 | 0.9200 | 0.8625 |
| linear | 0.9250 | 0.9250 | 0.9111 | 0.9500 | 0.9500 | 0.9500 | 0.8700 |
| borda | 0.8042 | 0.8042 | 0.8042 | 0.8042 | 0.8042 | 0.8042 | 0.8042 |
| zscore | 0.9167 | 0.9167 | 0.9111 | 0.9200 | 0.9250 | 0.9500 | 0.8667 |
| minmax | 0.9250 | 0.9250 | 0.9111 | 0.9500 | 0.9500 | 0.9500 | 0.8700 |

### Causal contrast (CombSUM, bootstrap 95% CI, B=10000)

- World+ vs orig: ΔMRR = +0.0018 (95% CI [+0.0000, +0.0054])\n
- orig vs World−: ΔMRR = +0.0500 (95% CI [+0.0000, +0.1500])\n
- RRF World+ vs orig ΔMRR = +0.0000 (must be ~0; invariance check)\n

### H3a — rank-conditioned relevance gap E[s|y=1,r]−E[s|y=0,r]\n
| rank bucket | gap |
|------------|----:|
| 1 | n/a |
| 2-3 | 0.0954 |
| 4-5 | 0.0112 |
| 6-10 | 0.0053 |

P(y=1 | large separation) = 0.002; P(y=1 | small separation) = 0.023; AUC(score) = 0.904\n

## scifact (n=10)

RRF invariance minimum Kendall τ across worlds: **1.0000** (1.0 = ranks identical => RRF identical, as required).\n

### MRR by operator × world

| Operator | orig | compress | rpr | world+ (ρ=1.25) | world+ (ρ=1.5) | world+ (ρ=2.0) | world− |
|----------|-----:|--------:|----:|:|:|:|------:|
| rrf | 0.8214 | 0.8214 | 0.8214 | 0.8214 | 0.8214 | 0.8214 | 0.8214 |
| combsum | 0.8204 | 0.8204 | 0.8205 | 0.8205 | 0.8205 | 0.8205 | 0.8198 |
| combmnz | 0.8204 | 0.8204 | 0.8205 | 0.8205 | 0.8205 | 0.8205 | 0.8198 |
| linear | 0.8229 | 0.8229 | 0.8200 | 0.8229 | 0.8230 | 0.8231 | 0.8229 |
| borda | 0.8205 | 0.8205 | 0.8205 | 0.8205 | 0.8205 | 0.8205 | 0.8205 |
| zscore | 0.8229 | 0.8229 | 0.8199 | 0.8231 | 0.8231 | 0.8232 | 0.8229 |
| minmax | 0.8229 | 0.8229 | 0.8200 | 0.8229 | 0.8230 | 0.8231 | 0.8229 |

### Causal contrast (CombSUM, bootstrap 95% CI, B=10000)

- World+ vs orig: ΔMRR = +0.0001 (95% CI [+0.0000, +0.0004])\n
- orig vs World−: ΔMRR = +0.0006 (95% CI [+0.0000, +0.0017])\n
- RRF World+ vs orig ΔMRR = +0.0000 (must be ~0; invariance check)\n

### H3a — rank-conditioned relevance gap E[s|y=1,r]−E[s|y=0,r]\n
| rank bucket | gap |
|------------|----:|
| 1 | n/a |
| 2-3 | 0.4876 |
| 4-5 | n/a |
| 6-10 | 0.0985 |

P(y=1 | large separation) = 0.000; P(y=1 | small separation) = 0.033; AUC(score) = 0.952\n

## 2wikimultihopqa (n=10)

RRF invariance minimum Kendall τ across worlds: **0.9928** (1.0 = ranks identical => RRF identical, as required).\n

### MRR by operator × world

| Operator | orig | compress | rpr | world+ (ρ=1.25) | world+ (ρ=1.5) | world+ (ρ=2.0) | world− |
|----------|-----:|--------:|----:|:|:|:|------:|
| rrf | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| combsum | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| combmnz | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| linear | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| borda | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| zscore | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| minmax | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Causal contrast (CombSUM, bootstrap 95% CI, B=10000)

- World+ vs orig: ΔMRR = +0.0000 (95% CI [+0.0000, +0.0000])\n
- orig vs World−: ΔMRR = +0.0000 (95% CI [+0.0000, +0.0000])\n
- RRF World+ vs orig ΔMRR = +0.0000 (must be ~0; invariance check)\n

### H3a — rank-conditioned relevance gap E[s|y=1,r]−E[s|y=0,r]\n
| rank bucket | gap |
|------------|----:|
| 1 | n/a |
| 2-3 | 0.1731 |
| 4-5 | 0.0849 |
| 6-10 | 0.0527 |

P(y=1 | large separation) = 0.000; P(y=1 | small separation) = 0.054; AUC(score) = 0.976\n