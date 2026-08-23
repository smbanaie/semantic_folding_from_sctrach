# Score Margin vs Fusion Error (Item 14)

Per query, per signal: margin = (best gold score − best non-gold score)/max|score| (negative = gold scores *below* a distractor in that signal). Joint margin = mean of the two signals' margins. 'Rescue' = RRF top-1 misses gold while CombSUM top-1 hits it; 'both fail' = neither recovers gold.


### hotpotqa (n=10 queries)

| Joint-margin bin | #queries | #rescues | rescue rate | #both-fail |
|------------------|---------:|---------:|------------:|-----------:|
| neg[<-.10] | 0 | 0 | — | 0 |
| small[-.10,0) | 0 | 0 | — | 0 |
| pos[0,.10) | 8 | 1 | 0.12 | 0 |
| med[.10,.30) | 2 | 0 | 0.00 | 0 |
| large[.30+] | 0 | 0 | — | 0 |

Overall: 1/10 rescues. Negative-joint-margin queries: 0/0 rescued. Non-negative-margin queries: 1/10 rescued. RRF/CombSUM top-1 disagreement: 9/10 queries.


### musique (n=10 queries)

| Joint-margin bin | #queries | #rescues | rescue rate | #both-fail |
|------------------|---------:|---------:|------------:|-----------:|
| neg[<-.10] | 0 | 0 | — | 0 |
| small[-.10,0) | 1 | 1 | 1.00 | 0 |
| pos[0,.10) | 3 | 0 | 0.00 | 0 |
| med[.10,.30) | 5 | 1 | 0.20 | 0 |
| large[.30+] | 0 | 0 | — | 0 |

Overall: 2/10 rescues. Negative-joint-margin queries: 1/1 rescued. Non-negative-margin queries: 1/8 rescued. RRF/CombSUM top-1 disagreement: 8/10 queries.


### scifact (n=10 queries)

| Joint-margin bin | #queries | #rescues | rescue rate | #both-fail |
|------------------|---------:|---------:|------------:|-----------:|
| neg[<-.10] | 0 | 0 | — | 0 |
| small[-.10,0) | 0 | 0 | — | 0 |
| pos[0,.10) | 1 | 0 | 0.00 | 0 |
| med[.10,.30) | 1 | 0 | 0.00 | 0 |
| large[.30+] | 6 | 0 | 0.00 | 0 |

Overall: 0/10 rescues. Negative-joint-margin queries: 0/2 rescued. Non-negative-margin queries: 0/8 rescued. RRF/CombSUM top-1 disagreement: 10/10 queries.

