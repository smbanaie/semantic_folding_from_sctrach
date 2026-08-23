# Deep-Pool BM25+SPLADE N-Sweep on HotpotQA (n=10 queries per N)

Candidate pool padded to N docs per query (seed=42); signal A = BM25,
signal B = SPLADE-cocondenser. All seven operators evaluated.

| N | linear MRR | rrf MRR | combsum MRR | combmnz MRR | borda MRR | zscore MRR | minmax MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.900 | 0.850 | 0.950 | 0.950 | 0.850 | 0.950 | 0.900 |
| 50 | 0.900 | 0.850 | 0.950 | 0.950 | 0.850 | 0.950 | 0.900 |
| 100 | 0.900 | 0.850 | 0.950 | 0.950 | 0.850 | 0.950 | 0.900 |
| 494 | 0.900 | 0.850 | 0.950 | 0.950 | 0.850 | 0.950 | 0.900 |

| N | linear P@1 | rrf P@1 | combsum P@1 | combmnz P@1 | borda P@1 | zscore P@1 | minmax P@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.800 | 0.800 | 0.900 | 0.900 | 0.800 | 0.900 | 0.800 |
| 50 | 0.800 | 0.800 | 0.900 | 0.900 | 0.800 | 0.900 | 0.800 |
| 100 | 0.800 | 0.800 | 0.900 | 0.900 | 0.800 | 0.900 | 0.800 |
| 494 | 0.800 | 0.800 | 0.900 | 0.900 | 0.800 | 0.900 | 0.800 |

Best operator per N: N=20: combmnz (0.950), N=50: combmnz (0.950), N=100: combmnz (0.950), N=494: combmnz (0.950).
