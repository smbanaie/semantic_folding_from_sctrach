# SIGIR_REVIEW.md Audit — item-by-item status against Journal_V3.md

| Item | Demand (short) | Status | Evidence / gap |
|------|----------------|--------|----------------|
| 0.1 | One coherent protocol; regimes named | None | — |
| 0.2 | n=10 framed exploratory; strong verbs reserved | APPLIED | — |
| 0.3 | n>=100 confirmatory + rank-1-change counts + power analysis | APPLIED | — |
| 0.4 | No causal overclaim | APPLIED | — |
| 0.5 | 'signal B determines' removed | APPLIED | — |
| 0.6 | Score-geometry predictor experiment | APPLIED | — |
| 0.7 | Magnitude vs calibration distinction | APPLIED | — |
| 0.8 | Calibration baselines battery | APPLIED | — |
| 0.9 | Feature-invariance experiment run or removed as contribution | APPLIED | — |
| 0.10 | 'full corpus' terminology fixed | APPLIED | — |
| 0.11 | SciFact 5183 investigated with components | APPLIED | — |
| 1 | Title kept fusion-centric | APPLIED | — |
| 2 | Abstract rewritten per review wording | APPLIED | — |
| 3.1 | SF usefulness strengthened | APPLIED | — |
| 3.2 | Hop sentence softened | APPLIED | — |
| 3.3 | RQ4 boundaries wording | None | — |
| 4 | Reference audit complete; no pending notes | APPLIED | — |
| 5 | Formal score-geometry definition G(s) | APPLIED | — |
| 6 | Prop1 as foundation not new theory | APPLIED | — |
| 7 | Dataset table rigor + query-selection explanation | APPLIED | — |
| 7.1 | Multi-seed / split stability | APPLIED | split_half_stability.py: 200 splits, CombSUM>RRF both-halves 100%/99.5%/79% |
| 8 | Regime A/B/C precise naming | APPLIED | — |
| 9 | Second dense retriever | SCOPED | reviewer marks lower priority than calibration; limitations names Contriever/E5/BGE as future work |
| 10 | Learned/calibrated fusion baseline | APPLIED | — |
| 11 | Alpha response-surface framing + CI | APPLIED | — |
| 12 | Stats: effect size, win rate, bootstrap, power in paper body | APPLIED | Table in §4.7 (wins/losses/ties, dz, power-n) + fig6 |
| 13 | SF architecture trimmed / appendix pointer | APPLIED | Appendix-A pointer sentence added at SF-as-probe section |
| 14 | Exploratory vs confirmatory visually separated | APPLIED | — |
| 15 | CombSUM/CombMNZ explanation evidence-backed | APPLIED | — |
| 16 | §6.5 joint-geometry rewrite done | APPLIED | — |
| 17 | SPLADE-v3 scoped to two checkpoints | APPLIED | — |
| 18 | Alpha CI stated | None | — |
| 19 | Kendall tau split + fusion-gain fit reported | APPLIED | — |
| 20 | Synthetic magnitude grid expanded + heatmap figure | APPLIED | fig4_phase_diagram (3 regime facets) referenced in §7.2 |
| 21 | Real perturbation battery incl. new conditions | APPLIED | — |
| 22 | Feature invariance decision stated, no placeholders | APPLIED | — |
| 23 | Scaling treatment; larger N point present | APPLIED | — |
| 24 | SciFact decomposition | APPLIED | — |
| 25 | Complete-collection naming | None | — |
| 26 | Task-operator compatibility conditional framing | None | — |
| 27 | Bruch positioning table | APPLIED | — |
| 28 | Tau descriptive diagnostic wording | None | — |
| 29 | Deployment one paragraph | APPLIED | — |
| 30 | Limitations list extended | APPLIED | — |
| 31 | References verified, no carry-over | APPLIED | — |
| 32 | Required figures generated+referenced | APPLIED | figs 1-6 via scripts/journal_figures.py (plotly, PNG+SVG), referenced inline |
| 33 | Central diagnostic table | APPLIED | Table 4 central diagnostic map after n=100 block |
| 34 | Three-level claims block | APPLIED | — |
| 35 | Four contribution claims aligned | APPLIED | — |
| 36 | Banned framings removed | APPLIED | — |
| 37 | Execution order respected | APPLIED | — |

**Counts:** APPLIED=44, SCOPED=1, PARTIAL=0, MISSING=0

## Baseline gates

- causal_occurrences: 2
- venue_verify: 0
- to_be_filled: 0
- chars: 105908
