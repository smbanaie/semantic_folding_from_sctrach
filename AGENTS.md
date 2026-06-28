# Semantic Folding Pipeline — Project Rules

## Project Structure

- Pipeline scripts: `semantic_folding/*.py`
- Entry point: `semantic_folding/semantic_folder.py` (interactive TUI)
- Individual steps: `semantic_folding/phrase_extractor.py` → `term_context.py` → `semantic_space.py` → `phrase_fingerprints.py` → `doc_fingerprints.py` → `query_processor.py`
- Utilities: `semantic_folding/lib.py`
- Visualizers: `semantic_folding/{phrase,doc,query}_visualizer.py`
- Notebooks: `semantic_folding/notebooks/`
- Outputs: `outputs/<dataset>_benchmark/`
- Config: `config/semantic_folding.yml`, `config/exec_state.yml`

### Documentation Structure

```
docs/
├── recommendations.md                          # Future work & improvement roadmap
├── reports/
│   ├── BENCHMARK_RESULTS.md                    # SINGLE SOURCE OF TRUTH (all metrics, all datasets)
│   ├── REPORTS.md                              # Version history index + file log
│   ├── <dataset>/
│   │   ├── v2_*_comprehensive_analysis.md      # Per-dataset deep dive (keep)
│   │   ├── v1_*.md                             # Per-dataset detailed report (keep)
│   │   └── failure_analysis_*.md               # Root cause analysis (keep)
│   └── (hipporag2_full_benchmark.md)           # Archived — content in BENCHMARK_RESULTS.md
├── research/                                   # Literature reviews, method comparisons
└── thesis/                                     # Thesis drafts
```

| File | Role | Update Frequency |
|------|------|-----------------|
| `docs/reports/BENCHMARK_RESULTS.md` | **Single source of truth** — all metrics, all datasets, all experiments | After every benchmark run |
| `docs/reports/REPORTS.md` | **Index** — version history table + report file references | After every benchmark run |
| `docs/recommendations.md` | **Roadmap** — future work, improvement priorities, tested experiments | When new improvements are tested |
| `docs/reports/<dataset>/` | **Deep dives** — per-dataset detailed analysis with failure patterns | Per-dataset |
| `docs/thesis/benchmarks.md` | **Thesis foundation** — methodology, parameter justification, academic framing | After significant changes |

## PhD Thesis Markdown Files (Foundation Documents)

These files are the foundation of the PhD thesis and MUST be updated after each successful improvement:

| File | Topic | Status |
|------|-------|--------|
| `docs/thesis/benchmarks.md` | Benchmarking methodology, multi-dataset results | ✅ Updated |
| `docs/thesis/archive/metrics.md` | Retrieval metrics & evaluation framework | ✅ Current |
| `docs/thesis/archive/parameters_tuning.md` | Parameter tuning experiments | ✅ Current |
| `docs/thesis/archive/fingerprints.md` | Phrase/doc fingerprint encoding | ✅ Current |
| `docs/thesis/archive/query_processing.md` | Query processing architecture | ✅ Current |
| `docs/thesis/archive/semantic_space.md` | Semantic space construction | ✅ Current |
| `docs/thesis/archive/phrase_extractor.md` | Phrase extraction pipeline | ✅ Current |
| `docs/thesis/archive/term_context.md` | Term-context matrix | ✅ Current |
| `docs/thesis/archive/lib.md` | Library utilities | ✅ Current |

**Rule:** After any benchmark improvement, update `benchmarks.md` with new results, metrics, and academic evidence.

## Reports Convention

Every benchmark run MUST be saved in `docs/reports/` with this structure:

```
docs/reports/
├── REPORTS.md                          # Master index of all reports
├── <dataset>/
│   ├── <version>_<timestamp>.md        # Full report with params, metrics, analysis
│   └── ...
└── cross-dataset/
    └── comparison_<timestamp>.md       # Cross-dataset comparison tables
```

### Report Filename Format
`<version>_<YYYYMMDD>_<HHMMSS>.md`

Examples:
- `v1_20260613_003000.md` — first run on Belebele
- `v2_20260613_143000.md` — improved parameters
- `cross_20260613_150000.md` — cross-dataset comparison

### Report Content Requirements

Each report MUST include:
1. **Header**: Dataset name, version, timestamp, pipeline parameters
2. **Configuration**: Full parameter table (grid_size, spread, top_percent, etc.)
3. **Metrics**: MRR, AP, P@K, R@K, NDCG@K (mean, min, max, std)
4. **Found-at distribution**: Rank histogram
5. **Per-query analysis**: Top performers and failures with query text
6. **Comparison**: vs BM25 baseline and previous runs
7. **Analysis**: Why results occurred, failure patterns
8. **Recommendations**: Next steps, parameter changes to try
9. **Reproduction**: Commands to reproduce the exact run

### Saving a Benchmark Result

```bash
# After running benchmark, save report:
# 1. Create directory if needed
mkdir -p docs/reports/<dataset>

# 2. Copy benchmark_report.md with version prefix
cp outputs/<dataset>_benchmark/benchmarks/benchmark_<ts>/benchmark_report.md \
   docs/reports/<dataset>/<version>_<timestamp>.md

# 3. Update master index
# Append entry to docs/reports/REPORTS.md
```

## Python Environment

- Virtual env: `.venv\Scripts\python` (Windows)
- spaCy model: `en_core_web_sm`
- Key deps: `numpy`, `scipy`, `spacy`, `plotly`, `scikit-learn`, `pyyaml`

## Pipeline Parameters

These are the **verified optimal defaults** — use for all datasets unless evidence shows otherwise:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid size | **64** | 128×128 tested on PubMedQA → MRR −5.3%. Do not change. |
| Encoding | Morton Z-order (`use_morton: true`) | |
| Smoothing | Gaussian blur, **sigma=1.5** | sigma=0 tested → MRR −31.2%. **Critical for performance.** |
| Dim reduction | **t-SNE** (default) | t-SNE MRR=0.88 vs UMAP MRR=0.80 on Belebele 50 queries. |
| Spreading | **radius=1, decay=0.5** | spread=2 tested → MRR −7.1% on short queries. Keep at 1. |
| top_percent | **0.10** | 0.05 tested → MRR −5.3%. Keep at 0.10. |
| Query weighting | **IDF** | uniform tested → MRR −0.86%. Keep IDF. |
| Normalization | L2 for query, **L2 for documents** (`--doc-norm l2`) | sqrt_nnz tested → MRR −4.0% on Belebele |
| `--geometric` | **Do not use** | Crashes with access violation at query 144. Buggy. |
| `--dynamic-spreading` | **Do not use** | Makes short-query MRR worse. |
| keep_verbs | true | Not worth testing — other param changes all failed. |
| min_freq | 1 | Not worth testing. |

**Key finding:** t-SNE outperforms UMAP on Belebele (MRR 0.88 vs 0.80). UMAP is faster (10-100x) and preserves global structure better, but t-SNE's local focus is better for phrase-level semantic matching. Use UMAP when dataset > 10k contexts or out-of-sample projection is needed.

## Benchmarking (MuSiQue)

- Script: `semantic_folding/dataset_benchmark/musique/run_benchmark.py`
- Three-phase design: index (Steps 1-5) → benchmark (Step 6) → report
- Run registry: `semantic_folding/dataset_benchmark/musique/runs/registry.yml`
- Analysis: `semantic_folding/dataset_benchmark/musique/benchmark_analyzer.py`
- Metrics: MRR, AP, P@K, R@K, NDCG@K

## Multi-Dataset Benchmarking

- Script: `semantic_folding/dataset_benchmark/run_all_benchmarks.py`
- Generic runner: `semantic_folding/dataset_benchmark/generic_benchmark.py`
- BM25 baseline: `semantic_folding/dataset_benchmark/bm25_benchmark.py`
- Adapters: `semantic_folding/dataset_benchmark/adapters/`
- Datasets: PubMedQA, Belebele, BioASQ, DROP, DocFinQA, CUAD

### Reporting Rule (MANDATORY)

**Every benchmark run MUST be saved to `docs/reports/` with this workflow:**

```bash
# 1. Run benchmark
.venv\Scripts\python -m semantic_folding.dataset_benchmark.generic_benchmark all \
  --dataset <name> --jsonl data/<name>/converted/<name>.jsonl --max-queries 50

# 2. Create report directory if needed
New-Item -ItemType Directory -Force -Path "docs/reports/<name>" | Out-Null

# 3. Copy report with versioned filename
Copy-Item "outputs/<name>_benchmark/benchmarks/benchmark_<ts>/benchmark_report.md" \
  "docs/reports/<name>/v<N>_<YYYYMMDD>_<HHMMSS>.md"

# 4. Update docs/reports/REPORTS.md (add row to version history + report list)

# 5. Update docs/reports/BENCHMARK_RESULTS.md (add/update dataset metrics table)
```

**Report filename format:** `v<N>_<YYYYMMDD>_<HHMMSS>.md`

**Reports index:** `docs/reports/REPORTS.md` — MUST be updated after every benchmark.

**Single source of truth:** `docs/reports/BENCHMARK_RESULTS.md` — MUST be updated after every benchmark with new metrics.

### Adapter Pattern

Each dataset has an adapter that:
1. `download(output_dir)` — fetch raw data
2. `convert_to_musique_format(raw_path, output_dir, max_queries)` — convert to JSONL
3. `get_recommended_params()` — dataset-specific parameter overrides

### Running Benchmarks

```bash
# Single dataset
.venv\Scripts\python semantic_folding\dataset_benchmark\generic_benchmark.py all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl --max-queries 100

# All datasets
.venv\Scripts\python semantic_folding\dataset_benchmark\run_all_benchmarks.py \
  --datasets belebele --max-queries 100

# BM25 baseline only
.venv\Scripts\python semantic_folding\dataset_benchmark\bm25_benchmark.py \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl \
  --run-dir outputs/belebele_benchmark/runs/run_<ts> --query-end 100

# Run with UMAP (default is now UMAP)
.venv\Scripts\python semantic_folding\dataset_benchmark\generic_benchmark.py all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl \
  --method umap --umap-n-neighbors 15 --umap-min-dist 0.0 --umap-metric cosine

# Run with t-SNE (for comparison)
.venv\Scripts\python semantic_folding\dataset_benchmark\generic_benchmark.py all \
  --dataset belebele --jsonl data/belebele/converted/belebele.jsonl \
  --method tsne --perplexity 30 --tsne-iter 1000
```

### Running Benchmarks with Log Capture

Use `Start-Process` to capture stdout/stderr to log files for later inspection:

```powershell
# Generic pattern
Start-Process -NoNewWindow -FilePath ".venv\Scripts\python" `
  -ArgumentList "-m semantic_folding.dataset_benchmark.generic_benchmark all --dataset <name> --jsonl data/<name>/converted/<name>.jsonl --max-queries 50 --query-end 50" `
  -RedirectStandardOutput "temp/<name>_benchmark.log" `
  -RedirectStandardError "temp/<name>_benchmark_err.log"

# Example: BioASQ benchmark
Start-Process -NoNewWindow -FilePath ".venv\Scripts\python" `
  -ArgumentList "-m semantic_folding.dataset_benchmark.generic_benchmark all --dataset bioasq --jsonl data/bioasq/converted/bioasq.jsonl --max-queries 50 --query-end 50 --no-splade" `
  -RedirectStandardOutput "temp/bioasq_A1.log" `
  -RedirectStandardError "temp/bioasq_A1_err.log"
```

Use `--no-splade` to skip SPLADE embedding (faster runs). Log files go to `temp/` which is gitignored.

### Batch Query Processing (March 2025+)

`phase2_benchmark` in `generic_benchmark.py` uses **batch query processing** via `--query-file` internally. Instead of calling `query_processor.py` once per query (reloading fingerprints, IDF, spaCy, and models each time), all gold-bearing queries are written to a `queries.txt` file and processed in a single subprocess call. Results are saved to `all_results.json` and split back per-query for metrics.

**Performance gain**: For 10 queries, batch processing saves ~N× fingerprint/IDF loading overhead (1-2 min per query → 1-2 min total). For SPLADE configs, the model loads once and caches per query, saving ~80s/query startup.

**What changed**:
- Index phase is unchanged (Steps 1-5 run sequentially)
- Benchmark phase now writes `queries.txt` + single `--query-file` call + `--output all_results.json`
- CSV and per-query files retain the same format
- `elapsed_s` in CSV is averaged across all queries in the batch
- `summary.json` includes `batch_elapsed_s` for total batch wall time

## Naming Conventions

- Python: snake_case for functions/variables
- CLI flags: kebab-case (e.g., `--grid-size`)
- Config keys: snake_case
- Output dirs: `snake_case`

## Common Workflows

### Run full pipeline (TUI)
```bash
.venv\Scripts\python semantic_folding\semantic_folder.py
```

### Run individual steps
1. `phrase_extractor.py --corpus data\corpus.txt`
2. `term_context.py --vocab <vocab.csv> --mapping <mapping.json>`
3. `semantic_space.py --matrix <npz> --metadata <json> --method tsne --grid-size 64`
4. `phrase_fingerprints.py --coordinates <coords.json> --grid-size 64 --morton`
5. `doc_fingerprints.py --corpus data\corpus.txt --fingerprints <dir> --grid-size 64`
6. `query_processor.py --query "<text>" --grid-size 64 --spreading-steps 1 --top-k 5`

### Run MuSiQue benchmark
```bash
.venv\Scripts\python semantic_folding\dataset_benchmark\musique\run_benchmark.py --mode index --split dev --max-queries 100 --grid-size 64 --spreading-steps 1 --top-percent 0.10 --weighting idf --benchmark
```

### Run experiments
```bash
.venv\Scripts\python run_experiments.py
.venv\Scripts\python run_geometric_experiments.py
```

### Compute IR metrics
```bash
.venv\Scripts\python tools\compute_ir_metrics.py
```

## Verification (No Test Suite)

This project has **no automated test suite**. To verify changes:
1. Run the full pipeline TUI end-to-end
2. Verify the output corpus matches expected structure
3. Run the MuSiQue benchmark (baseline: MRR 1.000, AP 0.869)
4. Run the 5-query QA sample and check the evaluation report
5. Manually inspect visualizations in the run's output directory

## Temporary Scripts Convention

- All ad-hoc scripts (experiments, inspections, runners) must be placed in `temp/`
- Run them from `temp/` via `.venv\Scripts\python temp\script_name.py`
- When resuming a paused task, check `temp/` first for existing scripts and their log/output state
- Each script in `temp/` has a README entry explaining its purpose and safe deletion criteria
- The `temp/` directory is gitignored — never commit its contents

## Agent Behavior Rules

- **Never commit changes unless explicitly asked**
- **Never merge branches into main without user confirmation**
- **Prefer editing existing files over creating new ones** unless task requires new files
- **No explanatory comments in code** unless task explicitly requests them
- **Keep responses concise** (under 4 lines for informational answers)

## Git Conventions

- Tags follow `v<major>.<minor>` pattern
- Commit messages: lowercase, descriptive
- Every new feature or bug fix on a **dedicated branch** (e.g., `feature/<name>`)
- Push and present changes for review before merging
- **Only merge into `main` after explicit user confirmation**
- Delete feature/fix branch after merge

### Thesis Chapter Edits

- Thesis chapter edits (`docs/thesis/`) are committed directly to `main`
- No separate thesis branch required

---

## Citation Management & Academic Writing

### Paper & Thesis Location

- **Paper draft**: `docs/papers/paper1/semantic_folding_paper.md`
- **Thesis index**: `docs/thesis/THESIS.md`
- **Thesis chapters** (in order):

| # | File | Title |
|---|------|-------|
| 1 | `chapter1_introduction.md` | Introduction |
| 2 | `chapter2_literature_review.md` | Literature Review |
| 3 | `chapter3_sf_pipeline.md` | The Semantic Folding Pipeline |
| 4 | `chapter4_parameter_tuning.md` | Parameter Tuning |
| 5 | `chapter5_sparse_vs_dense.md` | Sparse vs Dense Retrieval |
| 6 | `chapter6_similarity_metrics.md` | Similarity Metrics |
| 7 | `chapter7_experiments.md` | Experiments and Benchmark Results |
| 8 | `chapter8_discussion.md` | Discussion |
| 9 | `chapter9_conclusions.md` | Conclusions and Future Work |

### Citation Format

All citations use numbered references `[N]` in the paper, and author-year `(Author, Year)` in the thesis chapters.

**Paper format:**
```
[1] Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
```

**Thesis format:**
```
Kanerva (1988) proposed Sparse Distributed Memory...
```

### Citation Sources

| Source | Search Pattern | Purpose |
|--------|---------------|---------|
| Google Scholar | `scholar.google.com/scholar?q=...` | General academic search |
| ScienceDirect | `site:sciencedirect.com` via Google Scholar | Elsevier journals |
| arXiv | `arxiv.org/abs/...` | Preprints and working papers |

### ScienceDirect Search Pattern

```
https://scholar.google.com/scholar?q=<topic>+site:sciencedirect.com
```

**Example searches:**
- `semantic embedding sparse vectors site:sciencedirect.com`
- `domain specific question answering medical site:sciencedirect.com`
- `sparse distributed memory information retrieval site:sciencedirect.com`
- `glossary integration terminology retrieval site:sciencedirect.com`
- `multi hop question answering reasoning site:sciencedirect.com`

### Citation Verification Checklist

When adding citations to paper or thesis:

1. **Search Google Scholar** for each claim requiring citation
2. **Search ScienceDirect** using `site:sciencedirect.com` pattern for domain-specific papers
3. **Verify the citation** by checking the actual paper title, authors, and year
4. **Format consistently** using the established format for paper/thesis
5. **Update in-text references** to use numbered `[N]` (paper) or author-year (thesis)
6. **Add to references section** at the end of the document
7. **Update `docs/recommendations.md`** with new citations found

### Citation Categories

| Category | Papers | Topics |
|----------|--------|--------|
| Semantic Folding & SDR | [1-5] | Kanerva, Hawkins, Webber, HTM |
| Dense Retrieval | [6-9] | DPR, ColBERT, SPLADE |
| Classical IR | [10-12] | Vector space, BM25 |
| Distributional Semantics | [13-15] | Harris, Firth, vocabulary mismatch |
| Dimensionality Reduction | [16-17] | t-SNE, UMAP |
| Morton Encoding | [18] | Z-order curve |
| Orthogonality Constraint | [19] | Semantic interference |
| Closed-Domain QA | [20-23] | Domain-specific QA, ontology |
| Domain-Specific Retrieval | [24-25] | Medical, biomedical IR |
| Benchmark Datasets | [26-32] | PubMedQA, SciFact, etc. |
| Hyperdimensional Computing | [33-34] | VSA, HDC |
| IR Surveys | [35-36] | BERT for IR, LLMs for IR |
| ScienceDirect Papers | [37-69] | Domain QA, sparse vectors, hybrid retrieval |

### Applying Recommendations

After citation search, check `docs/recommendations.md` for:

1. **New citations found** — Section "New Citations Added"
2. **Recommendations for paper enhancement** — Specific sections to update
3. **Key insights from search** — Patterns identified across papers

**Apply recommendations by:**
1. Reading the specific section mentioned in the recommendation
2. Adding new citations to in-text references
3. Expanding discussion based on new findings
4. Updating the recommendations.md file with completion status

### Thesis Chapter Update Process

1. **Read the current chapter** to understand existing content
2. **Identify claims needing citations** (especially domain-specific claims)
3. **Search Google Scholar and ScienceDirect** for supporting papers
4. **Add citations** in author-year format for thesis
5. **Expand discussion** based on new findings
6. **Update references section** at end of chapter
7. **Verify consistency** across all chapters
