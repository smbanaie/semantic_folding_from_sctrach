# Score Margin vs Fusion Error (Item 9)


## Hotpotqa (n=100 queries)


| Joint-margin bin | #queries | #rescues | rescue rate | #both-fail |

|------------------|---------:|---------:|------------:|-----------:|

| neg[<-.10] |   0 |     — |           — |     0 |
| small[-.10,0) |   0 |     — |           — |     0 |
| pos[0,.10) |   0 |     — |           — |     0 |
| med[.10,.30) |   0 |     — |           — |     0 |
| large[.30+] | 100 |     — |           — |   100 |

Note: Single-signal margin analysis from n=100 traces; cross-signal rescue (RRF misses, CombSUM hits) requires both component signals simultaneously. Both-fail rate shown above.
