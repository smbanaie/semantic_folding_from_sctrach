# Tau Analysis: signal-level vs operator-level agreement (S4)

tau_signal  = Kendall(SF ranking, SPLADE ranking) per query — component complementarity diagnostic.
tau_operator= Kendall between two operators' fused rankings — operator agreement, NOT a complementarity measure.
Fusion Gain = MRR(fused) - max(MRR(A), MRR(B)) per query.


## hotpotqa (n=10 queries)

- mean tau_signal: **0.309**
- Fusion Gain(combsum) vs tau_signal: rho=None, p=None
- Fusion Gain(combsum) vs top-1 disagreement: rho=None, p=None

### Complementarity 4-cell table (A=top-1 correct? B=top-1 correct?)

| Cell (A,B) | n | linear | rrf | combsum | combmnz | borda | zscore | minmax | mean MRR over ops |
|-----|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| FT | 8 | 1.00 | 0.85 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 0.932 |

## musique (n=10 queries)

- mean tau_signal: **-0.062**
- Fusion Gain(combsum) vs tau_signal: rho=0.522, p=0.1215
- Fusion Gain(combsum) vs top-1 disagreement: rho=-0.167, p=0.6454

### Complementarity 4-cell table (A=top-1 correct? B=top-1 correct?)

| Cell (A,B) | n | linear | rrf | combsum | combmnz | borda | zscore | minmax | mean MRR over ops |
|-----|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| FT | 7 | 0.93 | 0.80 | 0.88 | 0.86 | 0.79 | 0.93 | 0.93 | 0.874 |

## scifact (n=10 queries)

- mean tau_signal: **0.318**
- Fusion Gain(combsum) vs tau_signal: rho=0.234, p=0.5161
- Fusion Gain(combsum) vs top-1 disagreement: rho=0.248, p=0.4888

### Complementarity 4-cell table (A=top-1 correct? B=top-1 correct?)

| Cell (A,B) | n | linear | rrf | combsum | combmnz | borda | zscore | minmax | mean MRR over ops |
|-----|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| FT | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| FF | 2 | 0.12 | 0.11 | 0.10 | 0.10 | 0.10 | 0.12 | 0.12 | 0.109 |
