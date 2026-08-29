# Oracle / Controlled Magnitude (Item 11) -- Hotpotqa / bm25_splade

n=100 queries. Base MRR: CombSUM=0.9343, RRF=0.9060.

| rho_oracle | CombSUM MRR | RRF MRR | ΔCombSUM vs orig | RRF invariant |
|-----------:|------------:|--------:|-----------------:|:------------:|
|   1.5 | 0.9543 | 0.9060 | +0.0200 | yes |
|   3.0 | 0.9593 | 0.9060 | +0.0250 | yes |
|  10.0 | 0.9593 | 0.9060 | +0.0250 | yes |

> RRF invariance holds across all oracle worlds (ranks preserved by construction).
> If CombSUM MRR rises with rho_oracle while RRF is flat, the effect is relevance-aligned
> *separation* (magnitude utility), not the specific scale of either real retriever.
