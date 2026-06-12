# Semantic Folding Pipeline — Project Rules

## Project Structure

- Pipeline scripts: `semantic_folding/*.py`
- Entry point: `semantic_folding/semantic_folder.py` (interactive TUI)
- Individual steps: `semantic_folding/phrase_extractor.py` → `term_context.py` → `semantic_space.py` → `phrase_fingerprints.py` → `doc_fingerprints.py` → `query_processor.py`
- Utilities: `semantic_folding/lib.py`
- Visualizers: `semantic_folding/{phrase,doc,query}_visualizer.py`
- Notebooks: `semantic_folding/notebooks/`
- Outputs: `outputs/run_<timestamp>/`
- Config: `config/semantic_folding.yml`, `config/exec_state.yml`

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
| Dim reduction | t-SNE (default) | |
| Spreading | **radius=1, decay=0.5** | spread=2 tested → MRR −7.1% on short queries. Keep at 1. |
| top_percent | **0.10** | 0.05 tested → MRR −5.3%. Keep at 0.10. |
| Query weighting | **IDF** | uniform tested → MRR −0.86%. Keep IDF. |
| Normalization | L2 for query, `sqrt(nnz)` for document fingerprints | |
| `--geometric` | **Do not use** | Crashes with access violation at query 144. Buggy. |
| `--dynamic-spreading` | **Do not use** | Makes short-query MRR worse. |
| keep_verbs | true | Not worth testing — other param changes all failed. |
| min_freq | 1 | Not worth testing. |

**Key finding:** The default pipeline parameters (grid=64, spread=1, top%=0.10, smoothing=1.5, weighting=idf) are optimal for PubMedQA. Do not deviate without re-testing.

## Benchmarking (MuSiQue)

- Script: `semantic_folding/dataset_benchmark/musique/run_benchmark.py`
- Three-phase design: index (Steps 1-5) → benchmark (Step 6) → report
- Run registry: `semantic_folding/dataset_benchmark/musique/runs/registry.yml`
- Analysis: `semantic_folding/dataset_benchmark/musique/benchmark_analyzer.py`
- Metrics: MRR, AP, P@K, R@K, NDCG@K

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
