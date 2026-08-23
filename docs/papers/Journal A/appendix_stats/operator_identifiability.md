# Operator Identifiability (Item 25)

Fraction of queries for which two operators produce **identical fused** rankings on the same candidate pool (SF+SPLADE, alpha=0.3, k=60; real component scores from the alpha-sweep endpoint runs).

| Dataset | Op pair | identical | n | gap |
|---------|---------|----------:|--:|----:|
| hotpotqa | borda vs minmax | 0.00 | 10 | 1.00 |
| hotpotqa | borda vs zscore | 0.00 | 10 | 1.00 |
| hotpotqa | combmnz vs borda | 0.00 | 10 | 1.00 |
| hotpotqa | combmnz vs minmax | 0.00 | 10 | 1.00 |
| hotpotqa | combmnz vs zscore | 0.00 | 10 | 1.00 |
| hotpotqa | combsum vs borda | 0.00 | 10 | 1.00 |
| hotpotqa | combsum vs combmnz | 1.00 | 10 | 0.00 |
| hotpotqa | combsum vs minmax | 0.00 | 10 | 1.00 |
| hotpotqa | combsum vs zscore | 0.00 | 10 | 1.00 |
| hotpotqa | linear vs borda | 0.00 | 10 | 1.00 |
| hotpotqa | linear vs combmnz | 0.00 | 10 | 1.00 |
| hotpotqa | linear vs combsum | 0.00 | 10 | 1.00 |
| hotpotqa | linear vs minmax | 1.00 | 10 | 0.00 |
| hotpotqa | linear vs rrf | 0.00 | 10 | 1.00 |
| hotpotqa | linear vs zscore | 0.10 | 10 | 0.90 |
| hotpotqa | rrf vs borda | 0.00 | 10 | 1.00 |
| hotpotqa | rrf vs combmnz | 0.00 | 10 | 1.00 |
| hotpotqa | rrf vs combsum | 0.00 | 10 | 1.00 |
| hotpotqa | rrf vs minmax | 0.00 | 10 | 1.00 |
| hotpotqa | rrf vs zscore | 0.00 | 10 | 1.00 |
| hotpotqa | zscore vs minmax | 0.10 | 10 | 0.90 |
| musique | borda vs minmax | 0.00 | 10 | 1.00 |
| musique | borda vs zscore | 0.00 | 10 | 1.00 |
| musique | combmnz vs borda | 0.00 | 10 | 1.00 |
| musique | combmnz vs minmax | 0.00 | 10 | 1.00 |
| musique | combmnz vs zscore | 0.00 | 10 | 1.00 |
| musique | combsum vs borda | 0.00 | 10 | 1.00 |
| musique | combsum vs combmnz | 0.00 | 10 | 1.00 |
| musique | combsum vs minmax | 0.00 | 10 | 1.00 |
| musique | combsum vs zscore | 0.00 | 10 | 1.00 |
| musique | linear vs borda | 0.00 | 10 | 1.00 |
| musique | linear vs combmnz | 0.00 | 10 | 1.00 |
| musique | linear vs combsum | 0.00 | 10 | 1.00 |
| musique | linear vs minmax | 1.00 | 10 | 0.00 |
| musique | linear vs rrf | 0.00 | 10 | 1.00 |
| musique | linear vs zscore | 0.00 | 10 | 1.00 |
| musique | rrf vs borda | 0.00 | 10 | 1.00 |
| musique | rrf vs combmnz | 0.00 | 10 | 1.00 |
| musique | rrf vs combsum | 0.00 | 10 | 1.00 |
| musique | rrf vs minmax | 0.00 | 10 | 1.00 |
| musique | rrf vs zscore | 0.00 | 10 | 1.00 |
| musique | zscore vs minmax | 0.00 | 10 | 1.00 |
| scifact | borda vs minmax | 0.00 | 10 | 1.00 |
| scifact | borda vs zscore | 0.00 | 10 | 1.00 |
| scifact | combmnz vs borda | 0.00 | 10 | 1.00 |
| scifact | combmnz vs minmax | 0.00 | 10 | 1.00 |
| scifact | combmnz vs zscore | 0.00 | 10 | 1.00 |
| scifact | combsum vs borda | 0.00 | 10 | 1.00 |
| scifact | combsum vs combmnz | 1.00 | 10 | 0.00 |
| scifact | combsum vs minmax | 0.00 | 10 | 1.00 |
| scifact | combsum vs zscore | 0.00 | 10 | 1.00 |
| scifact | linear vs borda | 0.00 | 10 | 1.00 |
| scifact | linear vs combmnz | 0.00 | 10 | 1.00 |
| scifact | linear vs combsum | 0.00 | 10 | 1.00 |
| scifact | linear vs minmax | 1.00 | 10 | 0.00 |
| scifact | linear vs rrf | 0.00 | 10 | 1.00 |
| scifact | linear vs zscore | 0.10 | 10 | 0.90 |
| scifact | rrf vs borda | 0.00 | 10 | 1.00 |
| scifact | rrf vs combmnz | 0.00 | 10 | 1.00 |
| scifact | rrf vs combsum | 0.00 | 10 | 1.00 |
| scifact | rrf vs minmax | 0.00 | 10 | 1.00 |
| scifact | rrf vs zscore | 0.00 | 10 | 1.00 |
| scifact | zscore vs minmax | 0.10 | 10 | 0.90 |

## Headline pairs

| Dataset | linear vs rrf | combsum vs rrf | minmax vs zscore |
|---------|--------------:|---------------:|-----------------:|
| hotpotqa | 0.00 | 0.00 | 0.10 |
| musique | 0.00 | 0.00 | 0.00 |
| scifact | 0.00 | 0.00 | 0.10 |
