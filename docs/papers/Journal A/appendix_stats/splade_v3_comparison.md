# Second Learned Sparse Model: SPLADE-v3 vs SPLADE-cocondenser (n=50)

Pair: SF + learned sparse. Identical indices, queries, pools; only the
SPLADE checkpoint changes (naver/splade-cocondenser-ensembledistil -> naver/splade-v3).

| Operator | HotpotQA v2 | HotpotQA v3 | MuSiQue v2 | MuSiQue v3 |
|----------|------------:|------------:|-----------:|-----------:|
| linear | 0.832 | 0.822 | 0.887 | 0.900 |
| rrf | 0.893 | 0.903 | 0.917 | 0.943 |
| combsum | 0.947 | 0.960 | 0.977 | 0.987 |
| combmnz | 0.893 | 0.882 | 0.919 | 0.917 |
| borda | 0.857 | 0.862 | 0.770 | 0.790 |
| zscore | 0.897 | 0.922 | 0.953 | 0.963 |
| minmax | 0.832 | 0.822 | 0.887 | 0.900 |

Best v3 operator: combsum on HotpotQA (0.960), combsum on MuSiQue (0.987).
