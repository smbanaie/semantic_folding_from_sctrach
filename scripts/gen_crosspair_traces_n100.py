"""Generate cross-pair component-score traces for Item 3 operator identifiability.

The SF+SPLADE traces already exist (gen_component_traces_n100.py). This script generates
the OTHER three pairs needed for the §9 cross-pair identifiability table:
  SF+DPR, BM25+SPLADE, BM25+DPR
for hotpotqa / musique / nq_rear at n=100.

For each pair we emit two component files:
  <ds>_<pair>_comp_1.0_n100.json   (pure signal A)
  <ds>_<pair>_comp_0.0_n100.json   (pure signal B)
using the SAME run dirs / fingerprints as the SF+SPLADE generation, only changing
--signal-a / --retriever-b.

SF+DPR and BM25+DPR do NOT need SPLADE (safe). BM25+SPLADE needs SPLADE (may hit the
flaky model-download; the script skips-and-reports on failure rather than aborting).

Outputs go to docs/papers/Journal A/appendix_alpha/.
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
PER_RUN_TIMEOUT = 1800

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

# (pair_label, signal_a, retriever_b, needs_splade)
PAIRS = [
    ("sf_dpr", "sf", "dpr", False),
    ("bm25_splade", "bm25", "splade", True),
    ("bm25_dpr", "bm25", "dpr", False),
]


def run_component(dataset, run_dir, jsonl, pair, signal_a, retriever_b, endpoint):
    entries = gb.load_entries(jsonl)[:NQ]
    qf = ALPHA_DIR / f"{dataset}_{pair}_q_n100_{endpoint}.txt"
    qf.write_text("\n".join(e["question"] for e in entries), encoding="utf-8")
    out_json = ALPHA_DIR / f"{dataset}_{pair}_comp_{endpoint}_n100.json"
    if out_json.exists():
        print(f"  [cached] {dataset} {pair} endpoint {endpoint}")
        return True
    args = [
        str(VENV), str(QP), "--query-file", str(qf),
        "--fingerprints", str(run_dir / "phrase_fingerprints"),
        "--doc-fingerprints", str(run_dir / "doc_fingerprints"),
        "--idf-weights", str(run_dir / "term_context_matrix" / "idf_weights.json"),
        "--grid-size", "64", "--top-k", str(TOPK), "--weighting", "idf",
        "--spreading-steps", "1", "--output", str(out_json),
        "--keep-verbs", "--min-word-length", "3",
        "--signal-a", signal_a, "--retriever-b", retriever_b,
        "--hybrid-alpha", str(endpoint), "--corpus", str(run_dir / "corpus.txt"),
        "--run-dir", str(run_dir),
    ]
    if retriever_b == "splade":
        args += ["--splade", "--splade-model", "naver/splade-cocondenser-ensembledistil"]
    try:
        r = subprocess.run(args, cwd=str(PROJ), capture_output=True, text=True,
                           timeout=PER_RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {dataset} {pair} endpoint {endpoint}")
        return False
    if not out_json.exists():
        print(f"  [FAIL] {dataset} {pair} ep {endpoint}: {r.stderr[-200:]}")
        return False
    print(f"  [ok] {dataset} {pair} endpoint {endpoint} -> {out_json.name}")
    return True


def main():
    for dataset, (run_dir, jsonl) in TARGETS.items():
        print(f"=== {dataset} (n={NQ}) ===")
        for pair, sa, rb, needs_splade in PAIRS:
            ok1 = run_component(dataset, run_dir, jsonl, pair, sa, rb, 1.0)
            ok0 = run_component(dataset, run_dir, jsonl, pair, sa, rb, 0.0)
            print(f"  {pair}: A={ok1} B={ok0}")


if __name__ == "__main__":
    main()
