# Normalization ablation — nq_rear (n=100, SF+SPLADE)

| A \ B | scheme | MRR_cs | MRR_rrf | ΔMRR | top1_chg | τ(cs,rrf) | World− deg |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | raw | 0.7587 | 0.6909 | +0.0677 | 19 | 0.716 | +0.0814 |
| raw | minmax | 0.7408 | 0.6909 | +0.0499 | 23 | 0.765 | +0.0680 |
| raw | zscore | 0.7595 | 0.6890 | +0.0705 | 40 | 0.793 | +0.0607 |
| raw | ranknorm | 0.6897 | 0.6909 | -0.0012 | 27 | 0.711 | +0.0647 |
| minmax | raw | 0.7587 | 0.6909 | +0.0677 | 19 | 0.716 | +0.0814 |
| minmax | minmax | 0.7408 | 0.6909 | +0.0499 | 23 | 0.765 | +0.0680 |
| minmax | zscore | 0.7595 | 0.6890 | +0.0705 | 40 | 0.793 | +0.0607 |
| minmax | ranknorm | 0.6897 | 0.6909 | -0.0012 | 27 | 0.711 | +0.0647 |
| zscore | raw | 0.6908 | 0.6901 | +0.0007 | 34 | 0.766 | +0.0884 |
| zscore | minmax | 0.6987 | 0.6901 | +0.0085 | 32 | 0.747 | +0.0906 |
| zscore | zscore | 0.7387 | 0.6921 | +0.0466 | 25 | 0.588 | +0.0630 |
| zscore | ranknorm | 0.6316 | 0.6901 | -0.0586 | 41 | 0.806 | +0.0650 |
| ranknorm | raw | 0.7234 | 0.6909 | +0.0324 | 25 | 0.725 | +0.0585 |
| ranknorm | minmax | 0.7258 | 0.6909 | +0.0349 | 29 | 0.635 | +0.0514 |
| ranknorm | zscore | 0.7743 | 0.6890 | +0.0852 | 41 | 0.819 | +0.0762 |
| ranknorm | ranknorm | 0.6826 | 0.6909 | -0.0084 | 10 | 0.878 | +0.0308 |

World- deg > 0 => magnitude effect operative under that normalization. raw/raw is the paper's baseline.