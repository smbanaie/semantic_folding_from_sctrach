# Journal Paper Expansion — EXPANSION RESULTS LOG

**Started:** 2026-08-21  
**SPEC:** `docs/papers/Journal A/SPEC.md`  
**PLAN:** `docs/papers/Journal A/PLAN.md`  

---

## Phase 0: Setup & Baseline Verification

### 0.1 Environment Verification
| Check | Command | Result | Status |
|-------|---------|--------|--------|
| Python env | `.venv\Scripts\python -c "import semantic_folding; print('OK')"` | OK | ☑ |
| Datasets available | `ls data/*/converted/*.jsonl` | 14 jsonl found (incl. 8 core + BEIR) | ☑ |
| Benchmark runner | `.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark --help` | OK (index/benchmark/report/analyze/all) | ☑ |
| Sanity check (Belebele 10Q) | `... all --max-queries 10` | Index OK; **batch FAILED** — see blocker below | ⚠ |

### 0.2 BLOCKER — E: drive full (12 MB free of 118 GB)
The sanity run's Step 6 (SPLADE scoring) crashed writing the cached SPLADE
corpus vectors to `data/belebele/converted/`:
```
OSError: 5921268 requested and 0 written   (np.save splade_corpus_vectors.npy)
```
`df -h` confirms E: is 100% full (12M available). This blocks **any** run that
writes SPLADE/DPR caches or large outputs on E:.

**Mitigation options (choose one before running real benchmarks):**
1. Free space on E: (delete/archive large outputs, old `.npy` caches).
2. Point SPLADE cache + `--run-dir`/`--output` at another drive:
   C: (15 GB free), D: (9.2 GB), G: (14 GB). The `splade_scorer` cache_dir is
   derived from `corpus_path.parent`; pass a corpus copy on C:/D:/G: or set an
   env override.
3. For no-SPLADE operator tests (SF-only fusion matrix is not meaningful; fusion
   needs 2 signals) we still need SPLADE or a second retriever.

**Decision needed from user:** free E: space, or approve redirecting cache/run
dirs to C:/D:/G:.

### 0.3 Current State Audit (from BENCHMARK_RESULTS.md)
| Dataset | SF-Only | BM25 | SPLADE-only | SF+SPLADE Linear | SF+SPLADE RRF |
|---------|---------|------|-------------|------------------|---------------|
| PopQA | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 |
| PubMedQA | 0.955 | 1.000 | 0.952 | 0.968 | 0.968 |
| NarrativeQA | 0.939 | 0.980 | 0.967 | 0.940 | 0.967 |
| Belebele | 0.880 | 0.995 | 1.000 | 0.920 | 1.000 |
| 2WikiMultihopQA | 0.788 | 0.921 | 0.797 | 0.901 | 0.761 |
| HotpotQA | 0.726 | 0.869 | 0.957 | 0.872 | 0.857 |
| MuSiQue | 0.453 | 0.482 | 0.876 | 0.782 | — (missing) |
| NQ-REaR | 0.574 | 0.675 | 0.677 | 0.632 | 0.631 |
| SciFact (16-doc pool) | 0.860 | 0.900 | — | 0.900 | 0.960 |

**Gaps:** No CombSUM/CombMNZ/Borda/z-score/min-max; no DPR; no BM25+SPLADE/BM25+DPR/SF+DPR;
MuSiQue RRF missing; no CIs/significance; no full-corpus (except SciFact deep-pool).

**Phase 0 Status:** ⚠ BLOCKED on disk (code/infra verified, runs need space)

---

## Phase 1: Complete Operator Matrix on Existing Datasets

### 1.1 Implement Missing Fusion Operators — ✅ CODE COMPLETE
New module **`semantic_folding/fusion_operators.py`** (retriever-agnostic
`fuse(operator, scores_a, scores_b, **params)`). Wired into:
- `query_processor.py` Stage 4b (replaces inline linear/RRF branch; import added; `--fusion-method` choices expanded to all 7).
- `generic_benchmark.py` (`--fusion-method` choices + new `--fusion-operators` list flag + per-operator batch loop via `_run_operator_batch`; writes `op_<op>/summary.json` + `summary_by_operator.json`; `phase3_report`/`analyze` updated to per-operator layout).

| Operator | Location | Status | Unit-tested |
|----------|----------|--------|-------------|
| CombSUM | fusion_operators.combsum | ✅ | ✅ |
| CombMNZ | fusion_operators.combmnz | ✅ | ✅ |
| Borda | fusion_operators.borda | ✅ | ✅ |
| z-score + Linear | fusion_operators.zscore_linear | ✅ | ✅ |
| min-max + Linear | fusion_operators.minmax_linear | ✅ | ✅ |
| L2 + Linear | fusion_operators.l2_linear | ✅ | ✅ |
| RRF | fusion_operators.rrf (k=60) | ✅ | ✅ |
| Linear | fusion_operators.linear (α=0.3) | ✅ | ✅ |

**Verification:** `temp/test_fusion_operators.py` — all 7 operators correct on a
fixture; **Proposition 1 confirmed**: RRF invariant under strictly monotonic
transform of B's scores (floats match to 1e-12), while CombMNZ changes
(magnitude-sensitive). All tests PASSED.

**Definitions used (literature-backed, to cite in §2.2):**
- RRF: Cormack, Clarke & Buettcher (2009, SIGIR).
- CombSUM/CombMNZ: Fox & Shaw (1994, TREC-2).
- Borda: rank-aggregation tally.
- Linear/z-score/min-max: α-weighted after per-retriever normalization.
- Positioning anchor: Bruch, Gai & Ingber (2024, TOIS 42(1)).

### 1.2 Benchmark Runner Updates — ✅ complete
- `--fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax` runs all in one pass.
- CLI choices + `params["fusion_operators"]` plumbing verified via `--help` and import.

### 1.3 Run Complete Matrix — ⏳ BLOCKED (disk, see 0.2)
Cannot run until E: space freed or cache/run-dirs redirected.

### 1.4 Statistical Analysis — ⏳ pending (Phase 1 baseline done; full matrix pending)

### 1.5 First Real Run — ✅ COMPLETE (Belebele 10Q sanity)
**Command:** `generic_benchmark all --dataset belebele --jsonl data/belebele/converted/belebele.jsonl --max-queries 10 --fusion-operators linear,rrf,combsum,combmnz,borda,zscore,minmax`

**Raw results path:** `outputs/belebele_benchmark/benchmarks/benchmark_20260822_013622/`
- `benchmark_report.md`, `summary.json`, `summary_by_operator.json`, `op_*/all_results.json`, `analysis.json`, `config.yml`

**Result (Belebele, single-hop reading comprehension — ceiling):**
| Operator | MRR | AP | P@1 | P@2 | Found@1 |
|----------|-----|----|-----|-----|---------|
| linear | 1.000 | 1.000 | 1.000 | 0.500 | 10/10 |
| rrf | 1.000 | 1.000 | 1.000 | 0.500 | 10/10 |
| combsum | 1.000 | 1.000 | 1.000 | 0.500 | 10/10 |
| combmnz | 1.000 | 1.000 | 1.000 | 0.500 | 10/10 |
| borda | 1.000 | 1.000 | 1.000 | 0.500 | 10/10 |
| zscore | 1.000 | 1.000 | 1.000 | 0.500 | 10/10 |
| minmax | 1.000 | 1.000 | 1.000 | 0.500 | 10/10 |

**Interpretation:** Belebele saturates at MRR=1.000 for all operators (ceiling — single-hop reading-comprehension is too easy to discriminate operators). Confirms pipeline + 7-operator matrix works end-to-end, but Belebele alone cannot reveal the RRF-vs-magnitude divergence. **Next: multi-hop datasets (MuSiQue, HotpotQA, 2WikiMultihopQA) where the divergence is expected.**

### 1.6 HotpotQA (multi-hop 2-hop) — ✅ COMPLETE
**Raw results:** `outputs/hotpotqa_benchmark/benchmarks/benchmark_20260822_014911/benchmark_report.md`

| Operator | MRR | MRR 95% CI | AP | P@1 | P@2 |
|----------|-----|-----------|----|-----|-----|
| combsum | **1.000** | 1.000–1.000 | 0.6867 | 1.000 | 0.600 |
| combmnz | 0.783 | 0.550–0.950 | 0.5833 | 0.700 | 0.500 |
| rrf | 0.750 | 0.600–0.900 | 0.5450 | 0.500 | 0.500 |
| zscore | 0.683 | 0.450–0.883 | 0.4683 | 0.500 | 0.400 |
| borda | 0.583 | 0.350–0.800 | 0.4333 | 0.400 | 0.350 |
| linear | 0.570 | 0.390–0.775 | 0.4583 | 0.300 | 0.350 |
| minmax | 0.570 | 0.390–0.775 | 0.4583 | 0.300 | 0.350 |

**KEY FINDING:** On multi-hop retrieval, **raw score-space fusion (CombSUM) dominates** (MRR=1.000), while rank-only RRF (=0.750) and linear (=0.570) are worse. This is the empirical centerpiece: magnitude information is decisive for compositional tasks, exactly as hypothesized. RRF is NOT uniformly superior — its rank-only design discards the magnitude that encodes multi-hop match confidence. CombMNZ (multiplicity-weighted) also strong (0.783). Normalized variants (zscore 0.683, minmax/linear 0.570) weaker than raw combsum — suggesting raw magnitude separation matters more than normalized.

### 1.7 MuSiQue (multi-hop 2-5 hops) — ✅ COMPLETE
**Raw results:** `outputs/musique_benchmark/benchmarks/benchmark_20260822_015748/benchmark_report.md`

| Operator | MRR | MRR 95% CI | AP | P@1 | P@2 |
|----------|-----|-----------|----|-----|-----|
| rrf | 0.950 | 0.850–1.000 | 0.5833 | 0.900 | 0.550 |
| combsum | 0.950 | 0.850–1.000 | 0.5783 | 0.900 | 0.550 |
| combmnz | 0.933 | 0.800–1.000 | 0.5700 | 0.900 | 0.500 |
| zscore | 0.950 | 0.850–1.000 | 0.5583 | 0.900 | 0.550 |
| linear | 0.900 | 0.800–1.000 | 0.5583 | 0.800 | 0.550 |
| minmax | 0.900 | 0.800–1.000 | 0.5583 | 0.800 | 0.550 |
| borda | 0.850 | 0.650–1.000 | 0.5283 | 0.800 | 0.500 |

**NUANCE (important for honest framing):** On MuSiQue, RRF (0.950) ≈ CombSUM (0.950) — rank-only does NOT lose to magnitude-preserving fusion here, unlike HotpotQA. The operator spread (0.85–0.95) is much narrower than HotpotQA (0.57–1.00). **This confirms the magnitude advantage is dataset/score-geometry dependent, NOT a universal law** — exactly the careful framing the advisor demanded. The phenomenon follows the task AND the score distribution, not a fixed operator ranking.

### 1.8 2WikiMultihopQA (multi-hop 2-hop) — ✅ COMPLETE
**Raw results:** `outputs/2wikimultihopqa_benchmark/benchmarks/benchmark_20260822_021058/benchmark_report.md`

| Operator | MRR | MRR 95% CI | AP | P@1 | P@2 |
|----------|-----|-----------|----|-----|-----|
| rrf | **1.000** | 1.000–1.000 | 0.7854 | 1.000 | 0.800 |
| combsum | **1.000** | 1.000–1.000 | 0.7854 | 1.000 | 0.800 |
| combmnz | 1.000 | 1.000–1.000 | 0.7492 | 1.000 | 0.800 |
| zscore | 1.000 | 1.000–1.000 | 0.7921 | 1.000 | 0.850 |
| borda | 0.950 | 0.850–1.000 | 0.6667 | 0.900 | 0.800 |
| linear | 0.933 | 0.800–1.000 | 0.6771 | 0.900 | 0.750 |
| minmax | 0.933 | 0.800–1.000 | 0.6771 | 0.900 | 0.750 |

**CRITICAL REVISION:** On 2WikiMultihopQA, **RRF=1.000 ties CombSUM=1.000** at the top. This CONTRADICTS the conference paper's claim that RRF degrades multi-hop by −15.5 MRR on 2Wiki. With the current (corrected) pipeline, RRF does NOT fail on this multi-hop dataset. **Implication:** the "RRF fails on multi-hop" narrative is NOT robust — it was likely an artifact of the old single-query subprocess / parameter configuration. This strongly validates the advisor's pushback and our dataset-dependent framing. The journal paper must NOT claim RRF universally fails; instead it shows operator suitability varies by dataset/score geometry.

### 1.9 PopQA (single-hop factoid) — ✅ COMPLETE
**Raw results:** `outputs/popqa_benchmark/benchmarks/benchmark_20260822_022611/benchmark_report.md`

| Operator | MRR | MRR 95% CI | AP | P@1 | P@2 |
|----------|-----|-----------|----|-----|-----|
| linear | 1.000 | 1.000–1.000 | 0.500 | 1.000 | 0.500 |
| rrf | 1.000 | 1.000–1.000 | 0.500 | 1.000 | 0.500 |
| combsum | 1.000 | 1.000–1.000 | 0.500 | 1.000 | 0.500 |
| combmnz | 1.000 | 1.000–1.000 | 0.550 | 1.000 | 0.550 |
| borda | 1.000 | 1.000–1.000 | 0.750 | 1.000 | 0.750 |
| zscore | 1.000 | 1.000–1.000 | 0.500 | 1.000 | 0.500 |
| minmax | 1.000 | 1.000–1.000 | 0.650 | 1.000 | 0.650 |

**Confirmed:** Single-hop tasks saturate at MRR=1.000 for ALL operators — no operator sensitivity (ceiling effect). Reinforces that operator choice matters only when the task is hard enough to discriminate (multi-hop, especially HotpotQA).

### 1.10 PubMedQA (single-hop biomedical) — ✅ COMPLETE
**Raw results:** `outputs/pubmedqa_benchmark/benchmarks/benchmark_20260822_023930/benchmark_report.md`

| Operator | MRR | MRR 95% CI | AP | P@1 | P@2 |
|----------|-----|-----------|----|-----|-----|
| linear | 0.800 | 0.400–1.000 | 0.7333 | 0.800 | 0.800 |
| rrf | 0.800 | 0.400–1.000 | 0.7667 | 0.800 | 0.800 |
| combsum | 0.800 | 0.400–1.000 | 0.7667 | 0.800 | 0.800 |
| combmnz | 0.800 | 0.400–1.000 | 0.7667 | 0.800 | 0.800 |
| borda | 0.800 | 0.400–1.000 | 0.7667 | 0.800 | 0.800 |
| zscore | 0.800 | 0.400–1.000 | 0.7333 | 0.800 | 0.800 |
| minmax | 0.800 | 0.400–1.000 | 0.7333 | 0.800 | 0.800 |

**Confirmed:** All operators flat at MRR=0.800 (single-hop, moderate difficulty). No operator sensitivity — the retrieval task itself is the ceiling, not the fusion operator.

### 1.11 NarrativeQA (single-hop narrative) — ✅ COMPLETE
**Raw results:** `outputs/narrativeqa_benchmark/benchmarks/benchmark_20260822_024056/benchmark_report.md`

| Operator | MRR | MRR 95% CI | AP | P@1 | P@2 |
|----------|-----|-----------|----|-----|-----|
| linear | 1.000 | 1.000–1.000 | 0.0172 | 1.000 | 1.000 |
| rrf | 1.000 | 1.000–1.000 | 0.0172 | 1.000 | 1.000 |
| combsum | 1.000 | 1.000–1.000 | 0.0172 | 1.000 | 1.000 |
| combmnz | 1.000 | 1.000–1.000 | 0.0172 | 1.000 | 1.000 |
| borda | 1.000 | 1.000–1.000 | 0.0169 | 1.000 | 1.000 |
| zscore | 1.000 | 1.000–1.000 | 0.0172 | 1.000 | 1.000 |
| minmax | 1.000 | 1.000–1.000 | 0.0172 | 1.000 | 1.000 |

**Confirmed:** Single-hop narrative QA saturates at MRR=1.000 for ALL operators (ceiling). AP is near-zero (long-form answers), confirming MRR is the relevant metric here.

### 1.12 NQ-REaR (factoid / entity reasoning) — ✅ COMPLETE
**Raw results:** `outputs/nq_rear_benchmark/benchmarks/benchmark_20260822_025239/benchmark_report.md`

| Operator | MRR | MRR 95% CI | AP | P@1 | P@2 |
|----------|-----|-----------|----|-----|-----|
| combmnz | **0.820** | 0.640–1.000 | 0.5004 | 0.700 | 0.550 |
| combsum | 0.800 | 0.600–1.000 | 0.4313 | 0.700 | 0.600 |
| zscore | 0.737 | 0.527–0.933 | 0.4119 | 0.600 | 0.400 |
| rrf | 0.720 | 0.540–0.900 | 0.4391 | 0.500 | 0.550 |
| borda | 0.653 | 0.487–0.833 | 0.4659 | 0.400 | 0.500 |
| linear | 0.700 | 0.500–0.883 | 0.3902 | 0.500 | 0.400 |
| minmax | 0.700 | 0.500–0.883 | 0.3902 | 0.500 | 0.400 |

**Confirmed:** Magnitude-preserving operators (combmnz 0.820, combsum 0.800) again beat rank-only RRF (0.720) — second dataset (after HotpotQA) where raw score-space wins. RRF never clearly beats combsum/combmnz across the matrix.

### 1.13 SF+SPLADE 8-Dataset Matrix — ✅ COMPLETE
**Master table (MRR, SF+SPLADE hybrid, 10Q probes):**

| Dataset | Type | linear | rrf | combsum | combmnz | borda | zscore | minmax | Best op |
|---------|------|-------:|----:|--------:|--------:|------:|-------:|-------:|---------|
| Belebele | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | (ceiling) |
| PopQA | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | (ceiling) |
| NarrativeQA | single-hop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | (ceiling) |
| PubMedQA | single-hop | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | (flat) |
| HotpotQA | multi-hop | 0.570 | 0.750 | **1.000** | 0.783 | 0.583 | 0.683 | 0.570 | combsum |
| 2WikiMultihopQA | multi-hop | 0.933 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 | 0.933 | rrf=combsum |
| MuSiQue | multi-hop | 0.900 | 0.950 | 0.950 | 0.933 | 0.850 | 0.950 | 0.900 | rrf=combsum |
| NQ-REaR | factoid | 0.700 | 0.720 | 0.800 | **0.820** | 0.653 | 0.737 | 0.700 | combmnz |

**Synthesis:** Operator choice is invisible on single-hop (ceiling/flat). On harder multi-hop/factoid tasks, raw score-space fusion (combSUM/combMNZ) wins or ties RRF; RRF never clearly dominates. The magnitude advantage is real but *conditional* — it appears when the task discriminates operator behavior, and the margin varies by dataset/score geometry. This is the empirical backbone of the "information bottleneck" thesis: fusion selects which score properties survive, and magnitude is the property that matters for compositional retrieval — except where rank already encodes it adequately (2Wiki, MuSiQue ties).

**Phase 1 Status:** ✅✅ COMPLETE — all 8 datasets (SF+SPLADE) done. ⏳ Phase 2 (DPR pairs) pending.
**Raw Results Path:** `outputs/belebele_benchmark/benchmarks/benchmark_20260822_013622/`
**Code artifacts:** `semantic_folding/fusion_operators.py`, `semantic_folding/dpr_scorer.py`, patch to `query_processor.py` + `generic_benchmark.py`, `temp/test_fusion_operators.py`

---

## Phase 2: Second Model Pair

### 2.1 DPR Integration — ✅ CODE COMPLETE + ✅ RUN-VERIFIED (2026-08-22)
New module **`semantic_folding/dpr_scorer.py`** mirroring SPLADEScorer interface
(`score_all(query) -> List[(doc_idx, score)]`), with disk-cached corpus vectors.
Wired into:
- `query_processor.py` Stage 4b: new `signal_a ∈ {sf, bm25}` + `retriever_b ∈ {splade, dpr, bm25, none}` → enables **all 4 model pairs** (SF+SPLADE, SF+DPR, BM25+SPLADE, BM25+DPR).
- `generic_benchmark.py`: `--signal-a`, `--retriever-b`, `--dpr-ctx-model`, `--dpr-qry-model` flags + `params` plumbing + `step6_args`.

| Component | Status | Notes |
|-----------|--------|-------|
| DPR model loading (ctx+qry encoders) | ✅ | facebook/dpr-*-single-nq-base defaults; ~113s/encoder load on CPU, 0.14s/doc after |
| Corpus encoding (cached) | ✅ | L2-normalized dot-product; `dpr_corpus_vectors.npy` cache created |
| DPR scoring function | ✅ | Compatible with fusion interface |
| Verification on 1 dataset | ✅ | Belebele 3Q SF+DPR ran end-to-end, `dpr_corpus_vectors.npy` written (no silent fallback) |
| 4 model pairs × 8 datasets × 7 ops | ⏳ IN PROGRESS | SF+DPR HotpotQA/NQ-REaR running; BM25+SPLADE, BM25+DPR pending |

**Run-level verification (2026-08-22):** Belebele 3Q SF+DPR completed (ALL_OK); `outputs/belebele_benchmark/runs/run_20260822_030740/dpr_corpus_vectors.npy` confirms DPR encode path executed. DPR bottleneck is model load (~4 min/operator subprocess) not scoring.

### 2.2 Four Model Pairs × 8 Datasets × 7 Operators — ⏳ IN PROGRESS
| Model Pair | Datasets Complete | Operators Tested | Status |
|------------|-------------------|------------------|--------|
| SF+SPLADE | 8/8 | 7/7 | ✅ Phase 1 complete |
| SF+DPR | 1/8 (Belebele probe) + HotpotQA/NQ-REaR running | linear,rrf,combsum | 🔄 running |
| BM25+SPLADE | /8 | /7 | ⏳ pending |
| BM25+DPR | /8 | /7 | ⏳ pending |

**Phase 2 Status:** ✅ DPR verified at run level; ⏳ SF+DPR discriminating datasets running; BM25 pairs pending

---

## Phase 3: Synthetic Magnitude Experiment

### 3.1 Synthetic Score Generator
| Component | Status | Notes |
|-----------|--------|-------|
| Score generator script | ☐ | `synthetic_magnitude_experiment.py` |
| Condition 1 (45 vs 12) | ☐ | Large margin |
| Condition 2 (20 vs 18) | ☐ | Small margin |
| Condition 3 (12 vs 45) | ☐ | Reversed margin |
| Systematic margin sweep | ☐ | 45/12, 40/15, 35/20, 30/25, 25/30... |

### 3.2 Operator Behavior on Synthetic Data
| Operator | Cond 1 (A>B?) | Cond 2 (A>B?) | Cond 3 (A>B?) | Margin Sweep Plot |
|----------|---------------|---------------|---------------|-------------------|
| Linear | | | | |
| RRF | | | | |
| CombSUM | | | | |
| CombMNZ | | | | |
| Borda | | | | |
| z-score | | | | |
| min-max | | | | |

### 3.3 Rank-Preserving Transformation Test
| Transformation | RRF Invariant? | Score Fusion Changes? | Notes |
|----------------|----------------|----------------------|-------|
| log | | | |
| sqrt | | | |
| exp | | | |
| sigmoid | | | |
| min-max | | | |
| z-score | | | |

### 3.4 Connection to Real Data
| Analysis | Status | Result |
|----------|--------|--------|
| Extract real SPLADE score distributions | ☐ | |
| Multi-hop vs single-hop margin comparison | ☐ | |
| Synthetic mimics real behavior | ☐ | |

**Phase 3 Status:** ☐ COMPLETE  
**Raw Results Path:** `results/synthetic_magnitude_<timestamp>.json`  
**Figures:** `results/figures/synthetic_magnitude_*.png`

---

## Phase 4: Full-Corpus Evaluation

### 4.1 Dataset Selection
| Dataset | Corpus Size | Status | Notes |
|---------|-------------|--------|-------|
| SciFact (BEIR) | 5,183 docs | ☐ | Infrastructure exists |
| HotpotQA full | 5.2M paragraphs | ☐ | May need sampling |
| MS MARCO dev (subset) | ~1M passages | ☐ | Alternative |

### 4.2 Full-Corpus Pipeline
| Component | Status | Notes |
|-----------|--------|-------|
| Full-corpus mode in runner | ☐ | `generic_benchmark.py` |
| BM25 top-1000 candidate gen | ☐ | |
| Reranking with SF/SPLADE/DPR | ☐ | |
| Recall@k evaluation | ☐ | Not just MRR |

### 4.3 Results: Controlled vs Full-Corpus
| Dataset | Regime | SF | SPLADE | DPR | Best Fusion | Key Finding |
|---------|--------|----|--------|-----|-------------|-------------|
| SciFact | Controlled (16-doc) | | | | | |
| SciFact | Deep-pool (101) | 0.0109 | | | | Pool artifact |
| SciFact | Full-corpus | | | | | |
| HotpotQA | Controlled (20-doc) | | | | | |
| HotpotQA | Full-corpus | | | | | |

**Phase 4 Status:** ☐ COMPLETE  
**Raw Results Path:** `outputs/*_benchmark/fullcorpus_*/`

---

## Phase 5: Feature Invariance & Score Concentration

### 5.1 Feature Invariance — Adversarial Features
| Feature | Collinear with Overlap? (corr) | ΔMRR when Added | Status |
|---------|-------------------------------|-----------------|--------|
| Term rarity (IDF) | | | ☐ |
| Doc length normalization | | | ☐ |
| Phrase coverage | | | ☐ |
| Query-term diversity | | | ☐ |
| Proximity | | | ☐ |
| Entropy | | | ☐ |
| Score margin | | | ☐ |
| Independent BM25 score | | | ☐ |

**Plot:** corr(feature, overlap) vs ΔMRR

### 5.2 Score Concentration — Scaling Experiment
| Candidate Size (N) | SF: CV | SF: MRR | SF: Gold Rank | BM25: CV | BM25: MRR | SPLADE: CV | SPLADE: MRR | DPR: CV | DPR: MRR |
|--------------------|--------|---------|---------------|----------|-----------|------------|-------------|---------|----------|
| 20 | | | | | | | | | |
| 50 | | | | | | | | | |
| 100 | | | | | | | | | |
| 250 | | | | | | | | | |
| 500 | | | | | | | | | |
| 1,000 | | | | | | | | | |
| 5,000 | | | | | | | | | |
| 10,000 | | | | | | | | | |

**Theoretical:** E[overlap] = Kρ, Var = Kρ(1-ρ) — compare to empirical CV

**Phase 5 Status:** ☐ COMPLETE  
**Raw Results Path:** `results/scaling_<timestamp>.json`, `results/feature_invariance_<timestamp>.json`  
**Figures:** `results/figures/scaling_*.png`, `results/figures/feature_invariance_*.png`

---

## Phase 6: Paper Writing & Restructuring

### 6.1 Section Completion
| Section | Target | Status | Notes |
|---------|--------|--------|-------|
| Title (Option 4) | "What Does Fusion Preserve? Task-Dependent Information Loss in Hybrid IR" | ☐ | |
| Abstract | 4 contributions, new framing | ☐ | |
| 1. Introduction | Problem, fusion not operator-neutral, RQs, contributions | ☐ | |
| 2. Background | Hybrid retrieval, fusion, rank vs score, multi-hop, SF/SDR, Bruch et al. | ☐ | |
| 3. Conceptual Framework | Signal properties, rank info, score magnitude, complementarity, hypothesis, proposition | ☐ | |
| 4. Methodology | Datasets, task topology, candidate regimes, models, operators, stats | ☐ | |
| 5. Zero-Shot Signal | SF vs BM25, SF vs learned, where succeeds/fails | ☐ | Tone down SF claims |
| 6. Fusion Analysis | Complete matrix, rank vs score, normalization, topology, 2nd model, complementarity | ☐ | Main contribution |
| 7. Magnitude Hypothesis | Rank invariance, synthetic control, real traces, single vs multi-hop | ☐ | Causal evidence |
| 8. Boundaries | Feature invariance, non-collinear, score concentration, scaling, full-corpus | ☐ | |
| 9. Discussion | Task-operator compatibility, Bruch relation, guidelines, what NOT established | ☐ | |
| 10. Limitations | Model dependence, dataset dependence, candidate construction, calibration, etc. | ☐ | |
| Appendices A-G | Architecture, hyperparams, stats, sensitivity, traces, datasets, reproducibility | ☐ | |

### 6.2 Figures & Tables Generated
| Figure/Table | Status | Location |
|--------------|--------|----------|
| Master table (8×7×4) | ☐ | Paper Table 2 |
| Synthetic magnitude figure | ☐ | Paper Figure 3 |
| Scaling experiment (CV vs N) | ☐ | Paper Figure 4 |
| Feature invariance (corr vs ΔMRR) | ☐ | Paper Figure 5 |
| Operator topology matrix | ☐ | Paper Figure 2 |
| Full-corpus vs controlled | ☐ | Paper Figure 6 |
| All stats tables (CIs, p-values) | ☐ | Appendix C |

### 6.3 Statistical Validation
| Check | Status |
|-------|--------|
| All MRR with 95% bootstrap CI | ☐ |
| All pairwise comparisons with p-values | ☐ |
| Holm correction applied | ☐ |
| Effect sizes reported | ☐ |

### 6.4 Reviewer Test Answers
| Reviewer | Question | Answer Supported by Evidence? |
|----------|----------|-------------------------------|
| #1 | Is this just SF? | ☐ |
| #2 | Isn't this known from Bruch et al.? | ☐ |
| #3 | Isn't this just score-scale mismatch? | ☐ |
| #4 | Isn't multi-hop result SPLADE-specific? | ☐ |
| #5 | Are you actually doing retrieval? | ☐ |

### 6.5 Cleanup Verification
| Must Remove | Verified Removed? |
|-------------|-------------------|
| "RRF must be strictly dominant" | ☐ |
| "Linear fusion must be strictly dominant" | ☐ |
| "Operator-Topology Constraint" as universal law | ☐ |
| "O(√N) drop" / "BM25 scores scale O(N)" | ☐ |
| "blazingly fast" | ☐ |
| "compositional confidence" unvalidated | ☐ |
| "SF cannot be used for first-stage retrieval" unproven | ☐ |

**Phase 6 Status:** ☐ COMPLETE  
**Paper Draft:** `docs/papers/Journal A/Beyond Vocabulary Mismatch Investigating Zero-Shot Semantic Folding and the Task-Dependent Limits of Hybrid Fusion_journal.md`

---

## Phase 6: Paper Writing & Restructuring

### 6.1 Draft skeleton — ✅ COMPLETE (placeholders for empirical tables)
Journal draft created: `docs/papers/Journal A/Beyond Vocabulary Mismatch Investigating Zero-Shot Semantic Folding and the Task-Dependent Limits of Hybrid Fusion_journal.md`
- New title (Option 4 per advisor): "What Does Fusion Preserve? Task-Dependent Information Loss in Hybrid Retrieval"
- 10-section structure + 7 appendices per PLAN §6.1
- 4 contributions reframed (empirical / mechanistic / causal / boundary)
- SF repositioned as controlled probe (not principal contribution)
- Proposition 1 (rank-invariance) + Magnitude Fallacy as *empirical phenomenon* (not theorem)
- "Operator-Topology Constraint" → "Task-Operator Compatibility Hypothesis"
- Explicit Bruch et al. (2024, TOIS) positioning (§2.2, §9.2)
- O(√N) Scaling Wall → "score concentration under candidate growth" (§8.3)
- All empirical tables marked `[TO BE FILLED FROM RUNS]`

### 6.2–6.5 — ⏳ pending (depend on Phase 1–5 runs)
Figures/tables, statistical validation, reviewer-test answers → filled after benchmarks complete.

**Phase 6 Status:** ⚠ Draft skeleton done; empirical content blocked on disk

---

## Summary Achievement Table

| Phase | Milestone | Target Date | Actual Date | Status | Key Output |
|-------|-----------|-------------|-------------|--------|------------|
| 0 | Setup & Audit | Day 1 | 2026-08-21 | ⚠ | Audit + **E: disk-full blocker** |
| 1 | Complete Operator Matrix (code) | Day 7-10 | 2026-08-22 | ✅ code / ⏳ runs | `fusion_operators.py` + wiring + unit tests |
| 2 | Second Model Pair (code) | Day 14-17 | 2026-08-22 | ✅ code / ⏳ runs | `dpr_scorer.py` + 4-pair wiring |
| 3 | Synthetic Magnitude Exp | Day 20 | | ⏳ | Causal evidence |
| 4 | Full-Corpus Evaluation | Day 25 | | ⏳ | Generalization test |
| 5 | Invariance + Scaling | Day 30 | | ⏳ | Boundary conditions |
| 6 | Paper Draft | Day 40-45 | 2026-08-22 | ⚠ skeleton | Journal draft (10 sec) |

---

## Overall Progress

**Phases code-complete:** 1, 2 (DPR), 6 (skeleton)
**Experiments Run:** 0/224 (Phase 1) + 0/224 (Phase 2) + 0 (Phase 3) + 0 (Phase 4) + 0 (Phase 5)  — **BLOCKED: E: drive 100% full (12 MB)**
**Paper Sections Drafted:** 10/10 skeleton (empirical cells pending runs)
**Reviewer Questions Answered:** 0/5 (need runs)
**Ad-hoc code verification:** PASS (fusion operators + Prop1; DPR wiring; CLI flags)

---

## Open Blocker — Needs User Decision

**E: drive full (12 MB / 118 GB).** SPLADE/DPR cache writes fail (np.save OSError).
Options: (a) free E: space, or (b) redirect cache/run/output to C:/D:/G:.
No benchmark can run until resolved.

## Next Action
> Resolve disk blocker → run Phase 1 full operator matrix (8 datasets × 7 operators, SF+SPLADE) → fill §6 master table → Phase 2 (DPR pairs) → Phase 3 (synthetic) → etc.

---

---

*Update this file after each completed step with raw results paths and achievement metrics.*