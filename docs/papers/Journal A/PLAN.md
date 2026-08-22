1|# Journal Paper Expansion — PLAN
2|
3|## Version Control
4|
5|- **Branch:** `main` (merged from `feature/journal-a-expansion` at 2fc83a0)
6|- All work below is committed to `main`. No separate branch needed.
7|
8|## Overview
9|
10|This plan addresses all **5 major SIGIR reviewer requirements** and executes the **complete experimental matrix** for the journal paper. The plan breaks down into **6 phases** with explicit dependencies. Results are logged in `docs/EXPANSION-RESULT.md` and per-dataset reports in `docs/reports/`.
11|
12|---
13|
14|## Phase 0: Current State Audit (Week 0) — COMPLETED
15|
16|### 0.1 Environment Verification — DONE
17|- [x] Verify `.venv` works
18|- [x] All 9 datasets available in `data/<dataset>/converted/*.jsonl` (SciFact added)
19|- [x] Benchmark infrastructure: `semantic_folding/dataset_benchmark/generic_benchmark.py`
20|- [x] Sanity check passed on Belebele 10 queries
21|
22|### 0.2 Current State Audit — DONE
23|- [x] Extracted existing results from `docs/reports/BENCHMARK_RESULTS.md`
24|- [x] Identified gaps in master table (MuSiQue missing from fusion table, pool size = 1 errors for NarrativeQA/Belebele)
25|- [x] Operator coverage: 7 operators implemented (Linear, RRF, CombSUM, CombMNZ, Borda, z-score, min-max)
26|- [x] Model pair coverage: 4 pairs implemented (SF+SPLADE, SF+DPR, BM25+SPLADE, BM25+DPR)
27|- [x] N-sweep deep-pool harness implemented (`--deep-pool N`)
28|- [x] Full-corpus harness implemented (`--full-corpus`, scifact/hotpotqa sidecars created)
29|
30|**Deliverable:** `docs/EXPANSION-RESULT.md` initialized with Phase 0 audit results
31|
32|---
33|
34|## Phase 1: Complete 9-Dataset Master Fusion Table (Weeks 1-2)
35|
36|**CRITICAL REVIEWER #1 FIX:** The paper claims "9 datasets" but the main fusion table only has 8 (MuSiQue missing). Must run complete 7-operator × 9-dataset matrix for SF+SPLADE pair.
37|
37|### 1.1 Run Complete SF+SPLADE Matrix on All 9 Datasets
38|**Datasets (9):** PopQA, PubMedQA, NarrativeQA, Belebele, 2WikiMultihopQA, HotpotQA, MuSiQue, NQ-REaR, **SciFact (NEW)**
39|**Queries:** 50 per dataset (confirmatory n=50; n=10 probe already exists for 8 datasets)
40|**Operators (7):** Linear, RRF, CombSUM, CombMNZ, Borda, z-score, min-max
41|**Model Pair:** SF+SPLADE (baseline pair from §6.1)
42|
43|```bash
44|# For each dataset - run complete 7-operator matrix
45|for ds in popqa pubmedqa narrativeqa belebele 2wikimultihopqa hotpotqa musique nq_rear scifact; do
46|  .venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
47|    --dataset $ds --jsonl data/$ds/converted/$ds.jsonl \
48|    --max-queries 50 \
49|    --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax \
50|    --run-dir outputs/${ds}_benchmark/full_matrix_$(date +%Y%m%d_%H%M%S)
51|done
52|```
53|
54|### 1.2 Fix Pool Size = 1 Error (Reviewer #2)
38|**CRITICAL:** Table 1 says NarrativeQA/Belebele pool size = 1, but MRR < 1.0. Must audit and correct.
39|- Audit: What exactly is in `paragraphs` for NarrativeQA/Belebele? Gold + how many distractors?
40|- Fix: Update pool sizes in Table 1 to actual measured counts
41|- If pool size > 1, correct the table. If pool size = 1, explain MRR < 1 (e.g., multiple supporting docs).
42|
43|### 1.3 Statistical Analysis
44|- [ ] Implement paired bootstrap test in `benchmark_analyzer.py` (1000 resamples)
45|- [ ] Add 95% CI for all MRR values
46|- [ ] Add Holm correction for multiple comparisons
47|- [ ] Generate master table with all cells filled + CIs + significance markers
48|
49|**Deliverable:** Complete 9×7 master table in `docs/reports/cross-dataset/master_table_v1_<timestamp>.md` + Phase 1 section in EXPANSION-RESULT.md
50|
51|---
52|
53|## Phase 2: Second Model Pair Validation (Weeks 2-3)
54|
55|**REVIEWER #4 FIX:** "Isn't the multi-hop result just SPLADE-specific?" Must run full operator matrix on 3 additional pairs.
56|
57|### 2.1 DPR Integration — ALREADY IMPLEMENTED
58|- [x] `semantic_folding/dpr_scorer.py` exists with DPR encoding
59|- [x] Pre-encode corpus for each dataset (cache embeddings)
60|- [x] DPR scoring function compatible with fusion interface
61|- [ ] Verify DPR scores on all 9 datasets
62|
59|### 2.2 Run 4 Model Pairs on All 9 Datasets (Confirmatory n=50 on discriminating pairs)
60|| Signal A | Signal B | Purpose |
61||----------|----------|---------|
62|| BM25 | SPLADE | Baseline learned+lexical |
63|| BM25 | DPR | Baseline lexical+dense |
64|| SF | SPLADE | Current main pair (Phase 1) |
65|| **SF | DPR** | **Critical: SF with different score geometry** |
66|
67|**Full design:** 4 model pairs × 9 datasets × 7 operators = 252 configurations (exploratory n=10 for all; confirmatory n=50 for 4 discriminating pairs on HotpotQA/NQ-REaR)
68|
69|```bash
70|# Example: SF+DPR on all datasets
71|for ds in popqa pubmedqa narrativeqa belebele 2wikimultihopqa hotpotqa musique nq_rear scifact; do
72|  .venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
73|    --dataset $ds --jsonl data/$ds/converted/$ds.jsonl \
74|    --max-queries 50 \
75|    --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax \
76|    --signal-a sf --retriever-b dpr \
77|    --run-dir outputs/${ds}_benchmark/sf_dpr_$(date +%Y%m%d_%H%M%S)
78|done
79|```
72|
73|### 2.3 Kendall's τ Complementarity Analysis (Already done for HotpotQA/NQ-REaR — extend to all)
74|- [ ] Compute Kendall's τ between component signal rankings for all pairs/datasets
75|- [ ] Add to master table
76|
77|**Deliverable:** 4×9×7 results table + τ analysis in `docs/reports/cross-dataset/model_pairs_v1_<timestamp>.md` + Phase 2 section in EXPANSION-RESULT.md
78|
79|---
79|
80|## Phase 3: Synthetic Magnitude Experiment (Week 3)
81|
82|**REVIEWER #7 FIX:** "Your explanation of the Multi-Hop Magnitude Fallacy is interesting—but currently unproven." Must implement controlled magnitude perturbation.
83|
84|### 3.1 Implement Synthetic Score Generator
85|**Location:** `semantic_folding/synthetic_magnitude_experiment.py`
86|- [ ] Generate synthetic retrieval results with controlled rank/magnitude
87|- [ ] Conditions:
88|  - Condition 1: Large margin (A=45, B=12, rank_A=1, rank_B=2)
89|  - Condition 2: Small margin (A=20, B=18, rank_A=1, rank_B=2)
90|  - Condition 3: Reversed margin (A=12, B=45, rank_A=1, rank_B=2)
91|  - Additional: Vary margin systematically (45/12, 40/15, 35/20, 30/25, 25/30...)
92|- [ ] Apply all 7 fusion operators
93|- [ ] Measure: Which operator correctly ranks A > B?
94|
95|### 3.2 Rank-Preserving Transformation Test
96|- [ ] Apply monotonic transformations: log, sqrt, exp, sigmoid, min-max, z-score
97|- [ ] Verify RRF invariance: RRF(f(s)) == RRF(s) for all monotonic f
98|- [ ] Measure score-fusion sensitivity: F(f(s)) != F(s)
99|- [ ] Plot: Performance vs. magnitude separation for each operator
100|
100|### 3.3 Connect to Real Data
101|- [ ] Extract real SPLADE score distributions from multi-hop vs single-hop queries
102|- [ ] Show: multi-hop queries have larger score margins between relevant/irrelevant
103|- [ ] Demonstrate: synthetic margin manipulation mimics real multi-hop behavior
104|
105|**Deliverable:** Synthetic experiment results + figures in `results/synthetic_magnitude_<timestamp>.json` + Phase 3 section in EXPANSION-RESULT.md
106|
107|---
108|
109|## Phase 4: Full-Corpus Evaluation (Weeks 3-4)
110|
111|**REVIEWER #5 FIX:** "You are not actually doing retrieval." Must run ≥2 full-corpus evaluations.
112|
113|### 4.1 Full-Corpus Sidecars — PARTIAL (HotpotQA done, SciFact done)
114|- [x] HotpotQA: `hotpotqa_corpus.txt` (494 docs) + full-corpus benchmark completed
115|- [x] SciFact: `scifact_full_corpus.txt` (5183 docs) + full.jsonl created
116|- [ ] Run full-corpus benchmark on SciFact
117|- [ ] Run full-corpus benchmark on HotpotQA with 4 model pairs (not just SF+SPLADE)
118|
119|### 4.2 Full-Corpus Experiments
120|```bash
121|# SciFact full-corpus (SF+SPLADE)
122|.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
123|  --dataset scifact --jsonl data/scifact/converted/scifact_full.jsonl \
124|  --max-queries 50 --full-corpus \
125|  --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax
126|
127|# HotpotQA full-corpus (all 4 model pairs)
128|for pair in sf_splade sf_dpr bm25_splade bm25_dpr; do
129|  .venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
130|    --dataset hotpotqa --jsonl data/hotpotqa/converted/hotpotqa_full.jsonl \
131|    --max-queries 50 --full-corpus \
132|    --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax \
133|    --signal-a ${pair%_*} --retriever-b ${pair#*_}
134|done
135|```
136|
137|**Deliverable:** Full-corpus results on ≥2 datasets showing whether controlled-reranking findings generalize + Phase 4 section in EXPANSION-RESULT.md
138|
139|---
140|
141|## Phase 5: Feature Invariance & Score Concentration (Weeks 4-5)
142|
143|**REVIEWER #8 FIX:** "Your current case study is not enough" + "Scaling Wall O(√N) claim rejected."
144|
145|### 5.1 Feature Invariance — Adversarial Features (NEW HARNESS NEEDED)
146|**Location:** New `semantic_folding/feature_invariance.py`
147|- [ ] Implement non-collinear features as controlled perturbations:
148|  - Term rarity (IDF-based)
149|  - Document length normalization
150|  - Phrase coverage (% of query phrases in doc)
151|  - Query-term diversity (entropy of query terms)
152|  - Proximity (min span of query terms in doc)
153|  - Score margin (top-1 minus top-2 score)
154|  - Independent BM25 score
155|- [ ] Measure correlation with SF overlap (qᵀd)
156|- [ ] Ablation: Add each feature to SF ranking, measure ΔMRR
157|- [ ] Plot: corr(feature, overlap) vs ΔMRR scatter
158|
149|### 5.2 Score Concentration — Scaling Experiment (REPLACES O(√N) CLAIM)
150|**Location:** `semantic_folding/score_concentration_scaling.py`
151|- [ ] Candidate sizes: N ∈ {20, 50, 100, 250, 500, 1k, 5k, 10k} (already have 20,50,100,494 on HotpotQA)
152|- [ ] For each N: Sample N candidates (1 gold + N-1 BM25 negatives from full corpus)
153|- [ ] Measure: mean, std, CV, max score, gold score, gold rank, MRR, Recall@k
154|- [ ] Compare: SF, BM25, SPLADE, DPR
155|- [ ] Plot: CV vs N, MRR vs N, gold rank vs N, separation Δ/σ
156|- [ ] Derive expected overlap statistics for binary SDRs: E[qᵀd]=Kρ, Var=Kρ(1-ρ)
157|
158|### 5.3 Theoretical Analysis
159|- [ ] Connect empirical CV curves to binomial overlap model
160|- [ ] Report as "Candidate-Growth-Induced Score Concentration" (not "Scaling Wall")
161|
162|**Deliverable:** Feature invariance scatter + scaling experiment figures in `results/scaling_<timestamp>.json` + Phase 5 section in EXPANSION-RESULT.md
163|
164|---
165|
166|## Phase 6: Paper Rewrite & Restructuring (Weeks 5-7)
167|
168|**REVIEWER #6 FIX:** "Kill the word 'strictly'" + "Theorem 1 is not a theorem" + "Theorem → Hypothesis"
169|
169|### 6.1 Restructure Conference Paper to Journal Format
170|- [ ] Move SF architecture to Appendix A
171|- [ ] Write new Section 3: Conceptual Framework (score properties, rank vs magnitude, complementarity, task-operator-signal-geometry hypothesis)
172|- [ ] Expand Section 2: Related Work (add Bruch et al. positioning, fix citations)
173|- [ ] Rewrite Section 4: Experimental Methodology (two regimes, statistical protocol, pool size audit)
174|- [ ] Write Section 5: Zero-Shot Semantic Signal (tone down SF claims, honest AP caveats)
175|- [ ] Write Section 6: Fusion Operator Analysis (main empirical contribution, master table, model pairs, τ)
176|- [ ] Write Section 7: Magnitude Information Hypothesis (synthetic + real, causal language scoped)
177|- [ ] Write Section 8: Representation and Scaling Boundaries (feature invariance + score concentration, honest future work)
178|- [ ] Write Section 9: Discussion (reviewer test answers, practical guidelines, what we don't establish)
179|- [ ] Write Section 10: Limitations and Conclusion
180|- [ ] Update Abstract, Title, Contributions (4 contributions per SPEC)
181|- [ ] Change title to: **"What Does Fusion Preserve? Task-Dependent Information Loss in Hybrid Information Retrieval"** (Option 4, advisor preference)
182|
183|### 6.2 Generate All Figures/Tables
184|- [ ] Master table (9 datasets × 7 operators × 4 model pairs) with 95% CIs
185|- [ ] Synthetic magnitude experiment figure (margin vs operator performance)
186|- [ ] Scaling experiment figure (CV vs N, MRR vs N, Δ/σ vs N)
187|- [ ] Feature invariance figure (corr vs ΔMRR scatter)
188|- [ ] Operator topology decision matrix (score geometry → winning family)
189|- [ ] Full-corpus vs controlled comparison
190|- [ ] Kendall's τ heatmap
191|
192|### 6.3 Statistical Validation
193|- [ ] All MRR values with 95% bootstrap CIs
194|- [ ] All pairwise comparisons with p-values
195|- [ ] Holm correction applied to confirmatory tests
196|- [ ] Effect sizes reported (Cliff's delta or Cohen's d)
197|
198|### 6.4 Final Review Against Reviewer Test Questions
199|- [ ] Answer all 5 reviewer questions with experimental evidence
200|- [ ] Verify no "must remove" phrases remain (§9.4 of SPEC)
201|- [ ] Verify title is journal-appropriate
202|- [ ] Verify contributions match 4 contributions in SPEC
203|- [ ] Verify all "strictly"/"law"/"proves" language removed unless formal proof provided
204|
205|**Deliverable:** Complete journal paper draft at `docs/papers/Journal A/Beyond Vocabulary Mismatch..._journal.md`
206|
207|---
208|
209|## Dependency Graph
210|
211|```
212|Phase 0 (Setup) — COMPLETED
213|    │
214|    ├──→ Phase 1 (9×7 Master Table) ←──────────────────┐
215|    │                                                │
524|    ├──→ Phase 2 (4 Model Pairs) ←─────────────────────┤  (can run in parallel after Phase 0)
217|    │                                                │
218|    ├──→ Phase 3 (Synthetic Experiment) ←────────────┤  (needs Phase 1 operators)
219|    │                                                │
220|    ├──→ Phase 4 (Full Corpus) ←─────────────────────┤  (needs Phase 2 DPR)
221|    │                                                │
222|    └──→ Phase 5 (Feature Invariance + Scaling) ─────┘  (independent, needs Phase 0)
223|                              │
224|                              ▼
225|                       Phase 6 (Paper Rewrite)
226|                              │
227|                              ▼
228|                        SUBMISSION
229|```
230|
231|---
232|
233|## Resource Requirements
234|
235|| Resource | Needed For | Estimated Time |
236||----------|------------|----------------|
237|| GPU (1x) | DPR encoding, SPLADE encoding | Phase 2, 4 |
238|| CPU (8+ cores) | Benchmark runs, bootstrap (1000 resamples) | All phases |
239|| Disk (~50GB) | Cached embeddings, full corpus indices | Phase 2, 4 |
240|| Time | Full matrix: 252 configs × 50 queries × 9 datasets | ~60-80 hrs compute |
241|
242|---
243|
244|## Risk Mitigation
245|
246|| Risk | Mitigation |
247||------|------------|
248|| Full-corpus too slow | Use top-1000 BM25 candidates + rerank; sample 50 queries |
249|| DPR encoding OOM | Batch encoding, FP16, gradient checkpointing |
250|| Synthetic experiment not convincing | Add more margin conditions; connect to real score distributions |
251|| Reviewer rejects "task-operator" claim | Scope claim to tested operators/datasets; emphasize "hypothesis" not "law" |
252|| Multiple comparison burden | Pre-register primary comparisons; use Holm; report exploratory separately |
253|| Pool size audit reveals more errors | Be honest; report actual pool sizes per dataset |
254|
255|---
256|
257|## Milestone Tracking
258|
259|| Milestone | Target Date | Status |
260||-----------|-------------|--------|
261|| Phase 0 complete | Day 0 | ✅ |
262|| Phase 1 complete (master table) | Day 7-10 | 🔄 IN PROGRESS |
263|| Phase 2 complete (4 model pairs) | Day 14-17 | ⏳ PENDING |
264|| Phase 3 complete (synthetic) | Day 20 | ⏳ PENDING |
265|| Phase 4 complete (full corpus) | Day 25 | ⏳ PENDING |
266|| Phase 5 complete (invariance+scaling) | Day 30 | ⏳ PENDING |
267|| Phase 6 complete (paper draft) | Day 40-45 | ⏳ PENDING |
268|| Internal review | Day 45 | ⏳ PENDING |
269|| Submission ready | Day 50 | ⏳ PENDING |
270|
271|---
272|
273|## Commands Reference
274|
275|### Run Full Operator Matrix (Phase 1)
276|```bash
277|for ds in popqa pubmedqa narrativeqa belebele 2wikimultihopqa hotpotqa musique nq_rear scifact; do
278|  .venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
279|    --dataset $ds --jsonl data/$ds/converted/$ds.jsonl \
280|    --max-queries 50 \
281|    --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax \
282|    --run-dir outputs/${ds}_benchmark/full_matrix_$(date +%Y%m%d_%H%M%S)
283|done
284|```
285|
286|### Run 4 Model Pairs (Phase 2)
287|```bash
288|for ds in popqa pubmedqa narrativeqa belebele 2wikimultihopqa hotpotqa musique nq_rear scifact; do
289|  for pair in sf_splade sf_dpr bm25_splade bm25_dpr; do
290|    .venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
291|      --dataset $ds --jsonl data/$ds/converted/$ds.jsonl \
292|      --max-queries 50 \
293|      --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax \
294|      --signal-a ${pair%_*} --retriever-b ${pair#*_} \
295|      --run-dir outputs/${ds}_benchmark/${pair}_$(date +%Y%m%d_%H%M%S)
296|  done
297|done
298|```
299|
300|### Run Synthetic Experiment (Phase 3)
301|```bash
302|.venv\Scripts\python semantic_folding/synthetic_magnitude_experiment.py \
303|  --output results/synthetic_magnitude_$(date +%Y%m%d_%H%M%S).json
304|```
305|
306|### Run Scaling Experiment (Phase 5)
307|```bash
308|.venv\Scripts\python semantic_folding/score_concentration_scaling.py \
309|  --dataset musique --candidate-sizes 20,50,100,250,500,1000,5000,10000 \
310|  --output results/scaling_$(date +%Y%m%d_%H%M%S).json
311|```
312|
313|### Compute Statistics
314|```bash
315|.venv\Scripts\python semantic_folding/dataset_benchmark/benchmark_analyzer.py \
316|  --run-dir outputs/<dataset>_benchmark/full_matrix_<ts> \
317|  --paired-bootstrap --holm-correction \
318|  --output results/stats_<dataset>_<ts>.json
318|```
319|
320|---
321|
322|## Next Steps (Immediate)
323|
324|1. **Now:** Complete Phase 1 — run 9×7 SF+SPLADE matrix (SciFact currently running)
325|2. **Next:** Run Phase 2 — 4 model pairs on HotpotQA/NQ-REaR (confirmatory n=50)
326|3. **Parallel:** Start Phase 3 synthetic experiment implementation
327|4. **Then:** Phase 4 full-corpus on SciFact + HotpotQA (4 pairs)
328|5. **Then:** Phase 5 feature invariance harness + scaling experiment
329|6. **Finally:** Phase 6 paper rewrite
330|
331|Update `docs/EXPANSION-RESULT.md` after each completed step.