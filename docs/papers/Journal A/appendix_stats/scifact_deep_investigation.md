# SciFact 5,183-doc Collapse Investigation

Pool: 5183 candidate docs/query (constructed from BEIR corpus), 10 queries. All operators see identical candidates.


## Fused MRR by operator (confirms collapse)

| Operator | MRR |
|----------|----:|
| linear | 0.130 |
| rrf | 0.130 |
| combsum | 0.130 |
| combmnz | 0.130 |
| borda | 0.130 |
| zscore | 0.130 |
| minmax | 0.130 |

## Diagnosis

- Gold rank under CombSUM (per query): [10, 5, 1]
- Best gold rank achievable by ANY operator (oracle): [10, 5, 1]
- Mean score CV within pools: 0.05831
- Mean top-1 minus top-2 margin: 3.35e-03
- Mean top-10 intersection ratio across 7 ops: 1.0
- Queries whose gold is present in the pool: 3/10

**Verdict:** candidate/pool failure dominates: gold present in only 3/10 pools, so MRR is bounded at ~0.13 regardless of operator (7 queries have zero gold in pool); fusion saturation: operators produce near-identical rankings (top-10 intersection = 1.0 — all seven operators rank the pool identically at the top)
