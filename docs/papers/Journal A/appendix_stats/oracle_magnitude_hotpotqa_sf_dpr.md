# Oracle / Controlled Magnitude (Item 11) -- Hotpotqa / sf_dpr

n=100 queries. Base MRR: CombSUM=0.5543, RRF=0.5543.

| rho_oracle | CombSUM MRR | RRF MRR | ΔCombSUM vs orig | RRF invariant |
|-----------:|------------:|--------:|-----------------:|:------------:|
|   1.5 | 0.5543 | 0.5543 | +0.0000 | yes |
|   3.0 | 0.5543 | 0.5543 | +0.0000 | yes |
|  10.0 | 0.5543 | 0.5543 | +0.0000 | yes |

> RRF invariance holds across all oracle worlds (ranks preserved by construction).
> If CombSUM MRR rises with rho_oracle while RRF is flat, the effect is relevance-aligned
> *separation* (magnitude utility), not the specific scale of either real retriever.
