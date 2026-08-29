# Top-rank ΔRR decomposition — n=100

| dataset | n | mean ΔRR | top20% share of |ΔRR| | H6 (≥80% in 20%) | #zero | #pos | #neg | Type A/B/C/D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | 100 | +0.0889 | 0.980 | PASS | 75 | 21 | 4 | A=3/B=18/C=75/D=4 |
| musique | 100 | +0.0621 | 0.973 | PASS | 71 | 23 | 6 | A=5/B=18/C=71/D=6 |
| nq_rear | 100 | +0.0665 | 1.000 | PASS | 80 | 17 | 3 | A=3/B=14/C=80/D=3 |

Mean joint_margin by type (negative-margin regime = where magnitude fusion wins):
  hotpotqa: A=-0.29693989644323254, B=-0.10192555464356419, C=-0.09272239081808044, D=-0.1252567467162357
  musique: A=-0.08550958626612276, B=-0.14568586325723068, C=-0.008040392290210816, D=-0.2797503143967009
  nq_rear: A=-0.15847344384481607, B=-0.0871372014546705, C=-0.09513360266449442, D=-0.28019344069733226