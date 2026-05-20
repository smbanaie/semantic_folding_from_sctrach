"""
Run parameter sweep experiments for the Semantic Folding pipeline.
Each experiment varies one parameter from baseline to isolate its impact.
"""
import subprocess, sys, json, time, shutil
from pathlib import Path

BASE = Path("outputs/experiments")
VENV = Path(".venv/Scripts/python")
CORPUS = Path("data/corpus.txt")
QA_SAMPLE = Path("data/qa-sample.md")
SHARED = BASE / "shared"
SHARED.mkdir(parents=True, exist_ok=True)

# ── Read the 5 queries from qa-sample.md ──
def extract_queries(path):
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("### Question"):
            q = line.split('"')[1] if '"' in line else line.split(": ", 1)[1]
            queries.append(q.strip('"'))
    return queries

QUERIES = extract_queries(QA_SAMPLE)

# Write queries file once
QUERIES_FILE = SHARED / "queries.txt"
QUERIES_FILE.write_text("\n".join(QUERIES), encoding="utf-8")
print(f"Wrote {len(QUERIES)} queries to {QUERIES_FILE}")

def run_step(script, args, label):
    cmd = [str(VENV), str(script)] + args
    print(f"\n{'='*70}\n[{label}] {' '.join(str(a) for a in cmd)}\n{'='*70}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    # Print only key lines (not full detail)
    for line in r.stderr.splitlines():
        if any(kw in line for kw in ["SUCCESS", "ERROR", "WARNING", "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6", "Pipeline", "Done", "complete", "skipped", "sparsity", "Processing"]):
            print(f"  {line.strip()}")
    if r.returncode != 0:
        print(f"  ERROR (code {r.returncode}): {r.stderr[-500:]}")
        sys.exit(1)
    print(f"  [{label}] finished in {elapsed:.1f}s")
    return elapsed

# ═══════════════════════════════════════════════════════════════
# Step 1: Phrase Extraction (shared across all experiments)
# ═══════════════════════════════════════════════════════════════
STEP1_OUT = SHARED / "extracted_phrases"
if not (STEP1_OUT / "vocabulary.csv").exists():
    run_step("semantic_folding/phrase_extractor.py", [
        "--corpus", CORPUS,
        "--output", STEP1_OUT,
        "--min-freq", "1",
        "--min-word-length", "3",
        "--keep-verbs",
    ], "Step 1")
else:
    print(f"[Step 1] Using cached output: {STEP1_OUT}")

# ═══════════════════════════════════════════════════════════════
# Step 2: Term-Context Matrix (shared)
# ═══════════════════════════════════════════════════════════════
STEP2_OUT = SHARED / "term_context_matrix"
if not (STEP2_OUT / "term_context_matrix.npz").exists():
    run_step("semantic_folding/term_context.py", [
        "--corpus", CORPUS,
        "--vocab", STEP1_OUT / "vocabulary.csv",
        "--mapping", STEP1_OUT / "phrase_to_contexts.json",
        "--output", STEP2_OUT,
    ], "Step 2")
else:
    print(f"[Step 2] Using cached output: {STEP2_OUT}")

IDF_WEIGHTS = STEP2_OUT / "idf_weights.json"
MATRIX_NPZ = STEP2_OUT / "term_context_matrix.npz"
METADATA_JSON = STEP2_OUT / "term_context_matrix.json"

# ═══════════════════════════════════════════════════════════════
# Step 3-4 for grid=128 (shared across experiments A-H)
# ═══════════════════════════════════════════════════════════════
STEP3_128 = SHARED / "semantic_space_128"
STEP4_128 = SHARED / "phrase_fingerprints_128"

if not (STEP3_128 / "context_coordinates.json").exists():
    run_step("semantic_folding/semantic_space.py", [
        "--matrix", MATRIX_NPZ,
        "--metadata", METADATA_JSON,
        "--output", STEP3_128,
        "--method", "tsne",
        "--grid-size", "128",
    ], "Step 3 (grid=128)")
else:
    print(f"[Step 3 (g128)] Using cached output: {STEP3_128}")

if not (STEP4_128 / "phrase_fingerprints.npz").exists():
    run_step("semantic_folding/phrase_fingerprints.py", [
        "--coordinates", STEP3_128 / "context_coordinates.json",
        "--metadata", METADATA_JSON,
        "--output", STEP4_128,
        "--grid-size", "128",
        "--smoothing-sigma", "1.5",
    ], "Step 4 (grid=128)")
else:
    print(f"[Step 4 (g128)] Using cached output: {STEP4_128}")

# ═══════════════════════════════════════════════════════════════
# Step 3-4 for grid=64 (experiment I)
# ═══════════════════════════════════════════════════════════════
STEP3_64 = SHARED / "semantic_space_64"
STEP4_64 = SHARED / "phrase_fingerprints_64"

if not (STEP3_64 / "context_coordinates.json").exists():
    run_step("semantic_folding/semantic_space.py", [
        "--matrix", MATRIX_NPZ,
        "--metadata", METADATA_JSON,
        "--output", STEP3_64,
        "--method", "tsne",
        "--grid-size", "64",
    ], "Step 3 (grid=64)")
else:
    print(f"[Step 3 (g64)] Using cached output: {STEP3_64}")

if not (STEP4_64 / "phrase_fingerprints.npz").exists():
    run_step("semantic_folding/phrase_fingerprints.py", [
        "--coordinates", STEP3_64 / "context_coordinates.json",
        "--metadata", METADATA_JSON,
        "--output", STEP4_64,
        "--grid-size", "64",
        "--smoothing-sigma", "1.5",
    ], "Step 4 (grid=64)")
else:
    print(f"[Step 4 (g64)] Using cached output: {STEP4_64}")

# ═══════════════════════════════════════════════════════════════
# DEFINE EXPERIMENTS
# ═══════════════════════════════════════════════════════════════
# Each experiment: (name, step5_overrides, step6_overrides, phrase_fp_dir)
# step5_overrides override default: top_percent=0.10, smoothing_sigma=1.5,
#   min_peak_distance=2, no_normalize
# step6_overrides override default: spreading_steps=1, weighting=idf,
#   spreading_decay=0.5

EXPERIMENTS = [
    # ── Baseline ──
    dict(name="A_baseline",
         step5=dict(top_percent=0.10, smoothing_sigma=1.5),
         step6=dict(spreading_steps=1, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── No spreading (exact match only) ──
    dict(name="B_no_spreading",
         step5=dict(top_percent=0.10, smoothing_sigma=1.5),
         step6=dict(spreading_steps=0, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── More spreading ──
    dict(name="C_more_spreading",
         step5=dict(top_percent=0.10, smoothing_sigma=1.5),
         step6=dict(spreading_steps=2, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── Sparser fingerprints ──
    dict(name="D_sparser_fp",
         step5=dict(top_percent=0.05, smoothing_sigma=1.5),
         step6=dict(spreading_steps=1, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── Denser fingerprints ──
    dict(name="E_denser_fp",
         step5=dict(top_percent=0.15, smoothing_sigma=1.5),
         step6=dict(spreading_steps=1, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── Uniform weighting ──
    dict(name="F_uniform_weighting",
         step5=dict(top_percent=0.10, smoothing_sigma=1.5),
         step6=dict(spreading_steps=1, weighting="uniform", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── Weaker smoothing ──
    dict(name="G_weak_smoothing",
         step5=dict(top_percent=0.10, smoothing_sigma=1.0),
         step6=dict(spreading_steps=1, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── Stronger smoothing ──
    dict(name="H_strong_smoothing",
         step5=dict(top_percent=0.10, smoothing_sigma=2.0),
         step6=dict(spreading_steps=1, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_128),
    # ── Smaller grid ──
    dict(name="I_small_grid_64",
         step5=dict(top_percent=0.10, smoothing_sigma=1.5),
         step6=dict(spreading_steps=1, weighting="idf", spreading_decay=0.5),
         fp_dir=STEP4_64),
]

# ═══════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════
results_summary = {}

for exp in EXPERIMENTS:
    name = exp["name"]
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {name}")
    print(f"{'#'*70}")

    exp_dir = BASE / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    grid = 64 if "64" in name else 128

    # ── Step 5: Document Fingerprints ──
    step5_out = exp_dir / "doc_fingerprints"
    step5_args = [
        "--corpus", CORPUS,
        "--fingerprints", exp["fp_dir"],
        "--idf-weights", IDF_WEIGHTS,
        "--output", step5_out,
        "--grid-size", str(grid),
        "--top-percent", str(exp["step5"]["top_percent"]),
        "--no-normalize",
        "--normalize-method", "l2",
        "--min-word-length", "3",
        "--keep-verbs",
        "--min-peak-distance", "2",
        "--smoothing-sigma", str(exp["step5"]["smoothing_sigma"]),
    ]
    run_step("semantic_folding/doc_fingerprints.py", step5_args, f"{name} Step 5")

    # ── Step 6: Query Processing ──
    step6_out = exp_dir / "query_results.json"
    step6_args = [
        "--query-file", QUERIES_FILE,
        "--fingerprints", exp["fp_dir"],
        "--doc-fingerprints", step5_out,
        "--output", step6_out,
        "--grid-size", str(grid),
        "--top-k", "5",
        "--weighting", exp["step6"]["weighting"],
        "--spreading-steps", str(exp["step6"]["spreading_steps"]),
        "--spreading-decay", str(exp["step6"]["spreading_decay"]),
    ]
    if exp["step6"]["weighting"] == "idf":
        step6_args += ["--idf-weights", IDF_WEIGHTS]

    run_step("semantic_folding/query_processor.py", step6_args, f"{name} Step 6")

    # ── Parse results ──
    with open(step6_out) as f:
        data = json.load(f)
    results_summary[name] = []
    for qr in data:
        docs = [int(r[0]) for r in qr["results"][:5]]
        scores = [round(r[1], 4) for r in qr["results"][:5]]
        results_summary[name].append({"docs": docs, "scores": scores})

# ═══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
# Ground truth (updated with latest corrections from qa-sample.md)
GT = [
    # Q1: C02 primary, C01 secondary
    {0: 2, 1: 1},
    # Q2: C16 primary, C17 primary
    {0: 16, 1: 17},
    # Q3: C07, C06, C14
    {0: 7, 1: 6, 2: 14},
    # Q4: C08, C10, C09
    {0: 8, 1: 10, 2: 9},
    # Q5: C00, C03, C09
    {0: 0, 1: 3, 2: 9},
]

FMT = "  {:>28s} | {:s} | {:s} | {:s} | {:s} | {:s}"
SEP = "  " + "-" * 28 + "-+-" + "-" * 45 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 5 + "-+-" + "-" * 30

report = []
report.append("# Parameter Sweep — Experiment Results\n")
report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"**Grid size:** 128×128 (exp A–H), 64×64 (exp I)")
report.append(f"**Ground truth:** {len(GT)} queries (updated)\n")

report.append("## Per-Experiment Top-5 Results\n")

for exp in EXPERIMENTS:
    name = exp["name"]
    r = results_summary[name]
    report.append(f"### {name}\n")
    report.append("| Q | Params Changed | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Relevant Found |")
    report.append("|---|---------------|--------|--------|--------|--------|--------|---------------|")

    for qi in range(len(GT)):
        docs = r[qi]["docs"]
        scores = r[qi]["scores"]
        gt_set = set(GT[qi].values())
        found = gt_set.intersection(docs)
        # Describe params changed
        changed = []
        if exp["step5"]["top_percent"] != 0.10: changed.append(f"top={exp['step5']['top_percent']}")
        if exp["step5"]["smoothing_sigma"] != 1.5: changed.append(f"σ={exp['step5']['smoothing_sigma']}")
        if exp["step6"]["spreading_steps"] != 1: changed.append(f"spread={exp['step6']['spreading_steps']}")
        if exp["step6"]["weighting"] != "idf": changed.append(f"w={exp['step6']['weighting']}")
        if "64" in name: changed.append("grid=64")
        param_str = ", ".join(changed) if changed else "baseline"

        doc_labels = [f"C{d:02d}({s:.2f})" for d, s in zip(docs, scores)]
        report.append(f"| Q{qi+1} | {param_str} | {doc_labels[0]} | {doc_labels[1]} | {doc_labels[2]} | {doc_labels[3]} | {doc_labels[4]} | {', '.join(f'C{d:02d}' for d in sorted(found))} |")

    report.append("")

# ── Aggregate metrics table ──
report.append("## Aggregate Metrics Comparison\n")

def compute_ap(ranked_docs, gt_set):
    """Average precision for ranked docs (top-5) vs binary ground truth."""
    hits = 0
    ap = 0.0
    for i, d in enumerate(ranked_docs[:5]):
        if d in gt_set:
            hits += 1
            ap += hits / (i + 1)
    return ap / min(len(gt_set), 5)

def compute_mrr(ranked_docs, gt_set):
    for i, d in enumerate(ranked_docs[:5]):
        if d in gt_set:
            return 1.0 / (i + 1)
    return 0.0

def compute_ndcg(ranked_docs, gt_set):
    """NDCG@5 with binary relevance (1 if in gt_set)."""
    dcg = 0.0
    for i, d in enumerate(ranked_docs[:5]):
        rel = 1.0 if d in gt_set else 0.0
        dcg += rel / (i + 2)  # log2(i+2)
    # IDCG: all relevant at top
    ideal = sorted(gt_set, key=lambda d: (d in ranked_docs[:5], -ranked_docs.index(d) if d in ranked_docs else 0), reverse=True)
    idcg = sum(1.0 / (i + 2) for i in range(min(len(gt_set), 5)))
    return dcg / idcg if idcg > 0 else 0.0

header = "| Metric | " + " | ".join(f"{e['name']:>28s}" for e in EXPERIMENTS) + " |"
report.append(header)
report.append("|---|" + "|".join("---:" for _ in EXPERIMENTS) + "|")

for metric_name in ["P@5", "R@5", "MRR", "NDCG@5", "AP"]:
    row = [f"**{metric_name}**"]
    for exp in EXPERIMENTS:
        vals = []
        for qi in range(len(GT)):
            docs = results_summary[exp["name"]][qi]["docs"]
            gt_set = set(GT[qi].values())
            if metric_name == "P@5":
                v = len(gt_set.intersection(docs[:5])) / 5
            elif metric_name == "R@5":
                v = len(gt_set.intersection(docs[:5])) / len(gt_set)
            elif metric_name == "MRR":
                v = compute_mrr(docs, gt_set)
            elif metric_name == "NDCG@5":
                v = compute_ndcg(docs, gt_set)
            elif metric_name == "AP":
                v = compute_ap(docs, gt_set)
            vals.append(v)
        mean = sum(vals) / len(vals)
        row.append(f"{mean:.4f}")
    report.append(" | ".join(row) + " |")

# Write report
REPORT_FILE = BASE / "experiment_comparison.md"
REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
print(f"\n{'='*70}")
print(f"Report written to {REPORT_FILE}")
print(f"{'='*70}")

# Also print quick comparison to stdout
print("\n═══ QUICK COMPARISON ═══")
for exp in EXPERIMENTS:
    name = exp["name"]
    p5_vals = []
    for qi in range(len(GT)):
        docs = results_summary[name][qi]["docs"]
        gt_set = set(GT[qi].values())
        p5_vals.append(len(gt_set.intersection(docs[:5])) / 5)
    top5_str = " | ".join(
        ",".join(str(d) for d in results_summary[name][qi]["docs"][:3])
        for qi in range(5)
    )
    print(f"  {name:>28s} → P@5={sum(p5_vals)/len(p5_vals):.3f} | top-5: {top5_str}")
