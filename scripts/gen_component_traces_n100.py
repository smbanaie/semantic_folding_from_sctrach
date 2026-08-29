"""Regenerate n=100 component-score traces (comp_1.0 / comp_0.0) for the
counterfactual-magnitude experiment (SIGIR-Final-Tasks Item 1).

Faithful reproduction of how docs/papers/Journal A/appendix_alpha/<ds>_comp_*.json
were originally produced (see temp/alpha_sweep_offline.py): run query_processor.py
with --signal-a sf --retriever-b splade --hybrid-alpha 1.0 (pure SF) and
--hybrid-alpha 0.0 (pure SPLADE), --top-k 100, over the dataset's candidate
corpus + fingerprints. The fused "linear(1.0)" is exactly maxnorm(SF); "linear(0.0)"
is maxnorm(SPLADE) — identical to the n=10 comp files, just at n=100.

Outputs:
  docs/papers/Journal A/appendix_alpha/<ds>_comp_1.0_n100.json
  docs/papers/Journal A/appendix_alpha/<ds>_comp_0.0_n100.json
"""
import json
import subprocess
import sys
from pathlib import Path

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys.path.insert(0, str(PROJ))
import semantic_folding.dataset_benchmark.generic_benchmark as gb

QP = PROJ / "semantic_folding/query_processor.py"
VENV = PROJ / ".venv/Scripts/python.exe"
ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
ALPHA_DIR.mkdir(parents=True, exist_ok=True)

NQ = 100
TOPK = 100
PER_RUN_TIMEOUT = 1800  # 30 min hard cap per component run

TARGETS = {
    "hotpotqa": (
        PROJ / "outputs/hotpotqa_benchmark/runs/run_20260824_032535",
        PROJ / "data/hotpotqa/converted/hotpotqa.jsonl",
    ),
    "musique": (
        PROJ / "outputs/musique_benchmark/runs/run_20260824_033236",
        PROJ / "data/musique/converted/musique.jsonl",
    ),
    "nq_rear": (
        PROJ / "outputs/nq_rear_benchmark/runs/run_20260824_033353",
        PROJ / "data/nq_rear/converted/nq_rear.jsonl",
    ),
}


def run_component(dataset, run_dir, jsonl, endpoint):
    entries = gb.load_entries(jsonl)[:NQ]
    qf = ALPHA_DIR / f"{dataset}_q_n100_{endpoint}.txt"
    qf.write_text("\n".join(e["question"] for e in entries), encoding="utf-8")
    out_json = ALPHA_DIR / f"{dataset}_comp_{endpoint}_n100.json"
    if out_json.exists():
        print(f"  [cached] {dataset} endpoint {endpoint}")
        return True
    args = [
        str(VENV), str(QP), "--query-file", str(qf),
        "--fingerprints", str(run_dir / "phrase_fingerprints"),
        "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
        "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
        "--grid-size", "64", "--top-k", str(TOPK), "--weighting", "idf",
        "--spreading-steps", "1", "--output", str(out_json),
        "--keep-verbs", "--min-word-length", "3",
        "--splade", "--splade-model", "naver/splade-cocondenser-ensembledistil",
        "--hybrid-alpha", str(endpoint), "--signal-a", "sf", "--retriever-b", "splade",
        "--doc-norm", "l2", "--corpus", str(run_dir / "corpus.txt"),
        "--run-dir", str(run_dir),
    ]
    try:
        r = subprocess.run(args, cwd=str(PROJ), capture_output=True, text=True,
                           timeout=PER_RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {dataset} endpoint {endpoint}")
        return False
    if not out_json.exists():
        print(f"  [FAIL] {dataset} ep {endpoint}: {r.stderr[-300:]}")
        return False
    print(f"  [ok] {dataset} endpoint {endpoint} -> {out_json.name}")
    return True


def main():
    for dataset, (run_dir, jsonl) in TARGETS.items():
        print(f"=== {dataset} (n={NQ}) ===")
        ok1 = run_component(dataset, run_dir, jsonl, 1.0)
        ok0 = run_component(dataset, run_dir, jsonl, 0.0)
        print(f"  SF ok={ok1} SPLADE ok={ok0}")


if __name__ == "__main__":
    main()
