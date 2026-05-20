"""
Run geometric scoring experiments (J, K) against the winner (I).
Uses shared artifacts from the main experiment sweep.
"""
import subprocess, sys, json, time
from pathlib import Path

BASE = Path("outputs/experiments")
VENV = Path(".venv/Scripts/python")
CORPUS = Path("data/corpus.txt")
SHARED = BASE / "shared"
QUERIES_FILE = SHARED / "queries.txt"
IDF_WEIGHTS = SHARED / "term_context_matrix" / "idf_weights.json"

GRID_64_FP = SHARED / "phrase_fingerprints_64"

def run_step(script, args, label):
    cmd = [str(VENV), str(script)] + args
    print(f"\n{'='*70}\n[{label}] {' '.join(str(a) for a in cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    for line in r.stderr.splitlines():
        if any(kw in line for kw in ["SUCCESS", "ERROR", "WARNING", "Step 5", "Step 6", "Pipeline", "complete", "sparsity", "Processing", "GEOMETRIC"]):
            print(f"  {line.strip()}")
    if r.returncode != 0:
        print(f"  ERROR (code {r.returncode}): {r.stderr[-500:]}")
        sys.exit(1)
    print(f"  [{label}] finished in {elapsed:.1f}s")
    return elapsed

EXPERIMENTS = [
    # J: geometric scoring replacing spreading on grid=64 (spread=0, geometric)
    dict(name="J_geometric_no_spread",
         step5=dict(top_percent=0.10, smoothing_sigma=1.5),
         step6=dict(spreading_steps=0, weighting="idf", geometric=True)),
    # K: geometric scoring WITH spreading (spread=1, geometric)
    dict(name="K_geometric_with_spread",
         step5=dict(top_percent=0.10, smoothing_sigma=1.5),
         step6=dict(spreading_steps=1, weighting="idf", geometric=True)),
]

for exp in EXPERIMENTS:
    name = exp["name"]
    print(f"\n{'#'*70}\n# EXPERIMENT: {name}\n{'#'*70}")
    exp_dir = BASE / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    grid = 64

    # Step 5
    step5_out = exp_dir / "doc_fingerprints"
    run_step("semantic_folding/doc_fingerprints.py", [
        "--corpus", CORPUS,
        "--fingerprints", GRID_64_FP,
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
    ], f"{name} Step 5")

    # Step 6
    step6_out = exp_dir / "query_results.json"
    step6_args = [
        "--query-file", QUERIES_FILE,
        "--fingerprints", GRID_64_FP,
        "--doc-fingerprints", step5_out,
        "--output", step6_out,
        "--grid-size", str(grid),
        "--top-k", "5",
        "--weighting", exp["step6"]["weighting"],
        "--spreading-steps", str(exp["step6"]["spreading_steps"]),
    ]
    if exp["step6"]["weighting"] == "idf":
        step6_args += ["--idf-weights", IDF_WEIGHTS]
    if exp["step6"].get("geometric"):
        step6_args += ["--geometric"]

    run_step("semantic_folding/query_processor.py", step6_args, f"{name} Step 6")

# Parse results
GT = [{0: 2, 1: 1}, {0: 16, 1: 17}, {0: 7, 1: 6, 2: 14}, {0: 8, 1: 10, 2: 9}, {0: 0, 1: 3, 2: 9}]

def compute_ap(ranked, gt_set):
    hits = 0; ap = 0.0
    for i, d in enumerate(ranked[:5]):
        if d in gt_set:
            hits += 1; ap += hits / (i + 1)
    return ap / min(len(gt_set), 5)

def compute_mrr(ranked, gt_set):
    for i, d in enumerate(ranked[:5]):
        if d in gt_set: return 1.0 / (i + 1)
    return 0.0

def compute_ndcg(ranked, gt_set):
    dcg = sum(1.0/(i+2) for i,d in enumerate(ranked[:5]) if d in gt_set)
    idcg = sum(1.0/(i+2) for i in range(min(len(gt_set),5)))
    return dcg / idcg if idcg else 0.0

for exp in EXPERIMENTS:
    name = exp["name"]
    with open(BASE / name / "query_results.json") as f:
        data = json.load(f)
    print(f"\n{'='*70}\n# RESULTS: {name}")
    all_metrics = {"P@5":[],"R@5":[],"MRR":[],"NDCG@5":[],"AP":[]}
    for qi in range(5):
        docs = [int(r[0]) for r in data[qi]["results"]]
        scores = [round(r[1],4) for r in data[qi]["results"]]
        gt_set = set(GT[qi].values())
        found = sorted(gt_set & set(docs))
        labels = [f"C{d:02d}({s:.2f})" for d,s in zip(docs,scores)]
        print(f"  Q{qi+1}: {' | '.join(labels)}  → {found}")
        all_metrics["P@5"].append(len(found)/5)
        all_metrics["R@5"].append(len(found)/len(gt_set))
        all_metrics["MRR"].append(compute_mrr(docs, gt_set))
        all_metrics["NDCG@5"].append(compute_ndcg(docs, gt_set))
        all_metrics["AP"].append(compute_ap(docs, gt_set))

    print(f"  Metrics: P@5={sum(all_metrics['P@5'])/5:.3f}  "
          f"R@5={sum(all_metrics['R@5'])/5:.3f}  "
          f"MRR={sum(all_metrics['MRR'])/5:.3f}  "
          f"NDCG={sum(all_metrics['NDCG@5'])/5:.3f}  "
          f"AP={sum(all_metrics['AP'])/5:.3f}")
