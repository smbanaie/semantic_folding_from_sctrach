# BioASQ Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download BioASQ dataset, run Semantic Folding + BM25 benchmarks, and update the benchmark report with results.

**Architecture:** All infrastructure already exists — adapter (`bioasq_adapter.py`), benchmark runner (`generic_benchmark.py`), BM25 baseline (`bm25_benchmark.py`). We only need to execute the pipeline and update documentation.

**Tech Stack:** Python, BigBIO HuggingFace datasets, existing SF pipeline

---

### Task 1: Download BioASQ Data

**Covers:** Data acquisition for BioASQ benchmark

**Files:**
- Create: `data/bioasq/raw/bioasq_cache.jsonl` (auto-created by adapter)
- Create: `data/bioasq/converted/bioasq.jsonl` (auto-created by adapter)

- [ ] **Step 1: Create data directories**

```bash
mkdir -p data/bioasq/raw data/bioasq/converted
```

- [ ] **Step 2: Download BioASQ via BigBIO**

```bash
.venv\Scripts\python -c "
from semantic_folding.dataset_benchmark.adapters import get_adapter
from pathlib import Path

adapter = get_adapter('bioasq')
raw_dir = Path('data/bioasq/raw')
adapter.download(raw_dir)
print('Download complete')
"
```

Expected: BigBIO config `bioasq_task_b_source` loads, ~4000+ rows cached to `data/bioasq/raw/bioasq_cache.jsonl`.

- [ ] **Step 3: Convert to MuSiQue format**

```bash
.venv\Scripts\python -c "
from semantic_folding.dataset_benchmark.adapters import get_adapter
from pathlib import Path

adapter = get_adapter('bioasq')
raw_dir = Path('data/bioasq/raw')
converted_dir = Path('data/bioasq/converted')
jsonl_path = adapter.convert_to_musique_format(raw_dir, converted_dir, max_queries=500)
print(f'Converted to: {jsonl_path}')
"
```

Expected: `data/bioasq/converted/bioasq.jsonl` created with MuSiQue-compatible entries.

- [ ] **Step 4: Verify JSONL structure**

```bash
.venv\Scripts\python -c "
import json
with open('data/bioasq/converted/bioasq.jsonl') as f:
    first = json.loads(f.readline())
print(f'Fields: {list(first.keys())}')
print(f'Question: {first[\"question\"][:100]}...')
print(f'Paragraphs: {len(first[\"paragraphs\"])}')
print(f'Gold passages: {sum(1 for p in first[\"paragraphs\"] if p[\"is_supporting\"])}')
"
```

Expected: Fields match MuSiQue format (`id`, `question`, `answer`, `paragraphs` with `is_supporting`).

---

### Task 2: Run Semantic Folding Benchmark

**Covers:** SF pipeline evaluation on BioASQ

**Files:**
- Create: `outputs/bioasq_benchmark/benchmarks/benchmark_<ts>/benchmark_report.md`

- [ ] **Step 1: Run full SF benchmark (50 queries)**

```bash
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset bioasq \
  --jsonl data/bioasq/converted/bioasq.jsonl \
  --max-queries 50
```

Expected: Three-phase pipeline completes:
- Phase 1: Index corpus (Steps 1-5)
- Phase 2: Query benchmark (Step 6 per query)
- Phase 3: Report generation with MRR, AP, P@K, R@K, NDCG@K

- [ ] **Step 2: Verify benchmark output**

Check `outputs/bioasq_benchmark/benchmarks/benchmark_<ts>/benchmark_report.md` exists and contains metrics.

Expected: Report shows MRR, AP, P@1, P@2, P@5, NDCG scores with per-query breakdown.

---

### Task 3: Run BM25 Baseline

**Covers:** BM25 baseline comparison

**Files:**
- Create: `outputs/bioasq_benchmark/runs/run_<ts>/bm25/bm25_report.md`

- [ ] **Step 1: Run BM25 baseline**

```bash
.venv\Scripts\python -m semantic_folding.dataset_benchmark.bm25_benchmark \
  --dataset bioasq \
  --jsonl data/bioasq/converted/bioasq.jsonl \
  --query-end 50
```

Expected: BM25 baseline metrics computed for comparison with SF.

- [ ] **Step 2: Verify BM25 output**

Check BM25 report exists with MRR, AP scores.

---

### Task 4: Generate Comparison Report

**Covers:** Cross-method comparison analysis

**Files:**
- Create: `docs/reports/bioasq/v1_<timestamp>.md`
- Modify: `docs/reports/REPORTS.md` (add entry)
- Modify: `docs/reports/BENCHMARK_RESULTS.md` (add BioASQ row)

- [ ] **Step 1: Create report directory**

```bash
mkdir -p docs/reports/bioasq
```

- [ ] **Step 2: Copy and version benchmark report**

```bash
# Find the latest benchmark
$benchDir = Get-ChildItem outputs/bioasq_benchmark/benchmarks/benchmark_* | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "$($benchDir.FullName)/benchmark_report.md" "docs/reports/bioasq/v1_${timestamp}.md"
```

- [ ] **Step 3: Update REPORTS.md**

Add row to `docs/reports/REPORTS.md`:
```
| BioASQ | v1 | <date> | `docs/reports/bioasq/v1_<ts>.md` | 50 queries | SF vs BM25 |
```

- [ ] **Step 4: Update BENCHMARK_RESULTS.md**

Add BioASQ row to the ranking table in `docs/reports/BENCHMARK_RESULTS.md`:

```markdown
| <rank> | **BioASQ** | Biomedical | <N> | <SF_MRR> | <BM25_MRR> | <ratio>% | Biomedical QA |
```

Add detailed metrics section:
```markdown
#### BioASQ (Biomedical QA)
| Metric | SF | BM25 | Notes |
|--------|-----|------|-------|
| MRR | <value> | <value> | |
| AP | <value> | — | |

**Finding**: <analysis based on results>
```

---

### Task 5: Verify and Finalize

**Covers:** Quality assurance and documentation

- [ ] **Step 1: Run verification**

```bash
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark analyze \
  --dataset bioasq \
  --jsonl data/bioasq/converted/bioasq.jsonl \
  --max-queries 50
```

Expected: Deep analysis with per-query breakdown, failure patterns.

- [ ] **Step 2: Check BENCHMARK_RESULTS.md consistency**

Verify BioASQ row is correctly positioned in ranking table (by SF/BM25 ratio).

- [ ] **Step 3: Update thesis foundation if needed**

If BioASQ results are significant, update `semantic_folding/benchmarks.md` with new dataset entry.

---

## Notes

- **Data source**: BigBIO HuggingFace (`bigbio/bioasq_task_b`) — no registration required
- **Expected dataset size**: ~4000+ questions from BioASQ Task B
- **Query limit**: Start with 50 queries for initial test, expand to full dataset later
- **BioASQ domain**: Biomedical QA — expect strong SF performance based on PubMedQA results (95.5% SF/BM25 ratio)
- **Key comparison**: BioASQ is more complex than PubMedQA (multi-document, multi-snippet) — will test SF's ability to handle biomedical multi-passage retrieval
