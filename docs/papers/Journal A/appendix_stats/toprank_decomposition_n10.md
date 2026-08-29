# Top-rank ΔRR decomposition — n=10

| dataset | n | mean ΔRR | top20% share of |ΔRR| | H6 (≥80% in 20%) | #zero | #pos | #neg | Type A/B/C/D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | 10 | +0.1167 | 1.000 | PASS | 8 | 2 | 0 | A=1/B=1/C=8/D=0 |
| musique | 10 | +0.0514 | 1.000 | PASS | 8 | 2 | 0 | A=0/B=2/C=8/D=0 |
| scifact | 10 | -0.0011 | 1.000 | PASS | 8 | 1 | 1 | A=0/B=1/C=8/D=1 |
| 2wikimultihopqa | 10 | +0.0000 | 0.000 | fail | 10 | 0 | 0 | A=0/B=0/C=10/D=0 |

Mean joint_margin by type (negative-margin regime = where magnitude fusion wins):
  hotpotqa: A=0.013160182882863858, B=0.10492274193769097, C=-0.21651709362053856, D=None
  musique: A=None, B=-0.3397806362493356, C=-0.16658601053639452, D=None
  scifact: A=None, B=-0.5187843768512169, C=0.40756065178701917, D=-0.8320162980339797
  2wikimultihopqa: A=None, B=None, C=-0.1373429246641337, D=None