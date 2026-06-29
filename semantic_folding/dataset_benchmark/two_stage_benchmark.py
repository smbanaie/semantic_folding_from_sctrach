"""
Two-Stage Neuro-Lexical Pipeline Benchmark

BM25 retrieves top-K candidates → SF re-ranks those candidates.

Usage:
    python -m semantic_folding.dataset_benchmark.two_stage_benchmark \
        --dataset bioasq \
        --jsonl data/bioasq/converted/bioasq.jsonl \
        --run-dir outputs/bioasq_benchmark/runs/run_XXX \
        --pool-size 100 \
        --alpha 0.5
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from semantic_folding.lib import get_logger

logger = get_logger("two_stage")

from .generic_benchmark import (
    load_entries, compute_metrics, filter_results_to_candidates,
    register_run, update_run_status, OUTPUTS_DIR, PIPELINE_DEFAULTS,
)
from .bm25_benchmark import BM25Scorer
from .sf_reranker_simple import (
    build_query_fingerprint_simple,
    rerank_candidates_with_sf_simple,
    load_phrase_fingerprints_from_npz,
)


def run_two_stage_benchmark(
    dataset: str,
    jsonl_path: Path,
    run_dir: Path,
    pool_size: int = 100,
    max_queries: int = None,
    top_k: int = 5,
    alpha: float = 0.5,
) -> Optional[Path]:
    """
    Run two-stage retrieval: BM25 top-K + SF re-ranking.
    
    Parameters
    ----------
    alpha : float
        Interpolation weight: final = alpha * bm25_norm + (1-alpha) * sf
    """
    # === LOAD DATA ===
    entries = load_entries(jsonl_path)
    if max_queries:
        entries = entries[:max_queries]
    
    # Load gold labels from query_gold.json (created during indexing)
    # Try multiple possible locations
    query_gold = {}
    for qg_path in [
        run_dir / "query_gold.json",
        run_dir.parent / "query_gold.json",  # Sometimes in parent
        Path("outputs") / dataset / "query_gold.json",
    ]:
        if qg_path.exists():
            with open(qg_path) as f:
                query_gold = json.load(f)
            logger.info(f"Loaded gold labels from {qg_path}")
            break
    
    if len(query_gold) == 0:
        logger.warning("No gold labels found, will skip metrics")
    
    # Load corpus
    corpus_file = run_dir / "corpus.txt"
    if not corpus_file.exists():
        logger.error(f"Corpus not found: {corpus_file}")
        return None
    
    doc_ids, texts = [], []
    with open(corpus_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma_idx = line.find(',')
            if comma_idx > 0 and line[:comma_idx].startswith('doc_'):
                doc_ids.append(line[:comma_idx])
                texts.append(line[comma_idx+1:])
    
    logger.info(f"Loaded corpus: {len(doc_ids)} documents")
    
    # === BM25 INDEX ===
    bm25 = BM25Scorer(texts)
    logger.info("BM25 index built")
    
    # === LOAD SF FINGERPRINTS ===
    # Document fingerprints
    doc_fp_dir = run_dir / "doc_fingerprints"
    doc_fingerprints = {}
    if doc_fp_dir.exists():
        npz_file = doc_fp_dir / "doc_fingerprints.npz"
        meta_file = doc_fp_dir / "doc_fingerprints_meta.json"
        
        if npz_file.exists() and meta_file.exists():
            doc_data = np.load(npz_file)
            doc_fps_mat = doc_data['fingerprints']
            with open(meta_file) as f:
                doc_meta = json.load(f)
            
            doc_to_row = doc_meta.get('doc_to_row', {})
            for doc_id, idx in doc_to_row.items():
                doc_fingerprints[doc_id] = sparse.csr_matrix(doc_fps_mat[idx])
            
            logger.info(f"Loaded {len(doc_fingerprints)} document fingerprints from .npz")
        else:
            logger.warning("Document fingerprints .npz not found, skipping SF re-ranking")
    else:
        logger.warning("Document fingerprints not found, skipping SF re-ranking")
    
    # Phrase fingerprints
    phrase_fp_dir = run_dir / "phrase_fingerprints"
    phrase_fingerprints = {}
    if phrase_fp_dir.exists():
        npz_file = phrase_fp_dir / "phrase_fingerprints.npz"
        meta_file = phrase_fp_dir / "phrase_fingerprints_meta.json"
        
        if npz_file.exists() and meta_file.exists():
            phrase_fingerprints = load_phrase_fingerprints_from_npz(npz_file, meta_file, grid_size=64)
            logger.info(f"Loaded {len(phrase_fingerprints)} phrase fingerprints from .npz")
        else:
            logger.warning("Phrase fingerprints .npz not found")
    else:
        logger.warning("Phrase fingerprints not found")
    
    # === RUN BENCHMARK ===
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = OUTPUTS_DIR / f"{dataset}_benchmark" / "benchmarks" / f"benchmark_{ts}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    failed = 0
    
    for idx, entry in enumerate(entries):
        # Handle different JSONL formats
        query = entry.get("question") or entry.get("query") or ""
        
        # Load gold from query_gold.json
        gold = query_gold.get(str(idx), [])
        
        if not query:
            logger.debug(f"[{idx:04d}] No query, skipping")
            failed += 1
            continue
        
        if not gold:
            logger.debug(f"[{idx:04d}] No gold, skipping")
            failed += 1
            continue
        
        # === STAGE 1: BM25 TOP-K ===
        bm25_results = bm25.score(query)
        candidates = [(did, score) for did, score in bm25_results[:pool_size]]
        
        # === STAGE 2: SF RE-RANKING ===
        if len(doc_fingerprints) > 0 and len(phrase_fingerprints) > 0:
            reranked = rerank_candidates_with_sf_simple(
                query, candidates, doc_fingerprints, phrase_fingerprints,
                grid_size=64, top_k=top_k, alpha=alpha,
            )
        else:
            # Fall back to BM25 order
            reranked = candidates[:top_k]
        
        # === SAVE RESULTS ===
        ranked_ids = [did for did, _ in reranked]
        ranked_with_scores = [(did, score) for did, score in reranked]
        
        per_query_dir = bench_dir / "per_query" / f"{idx:04d}"
        per_query_dir.mkdir(parents=True, exist_ok=True)
        
        with open(per_query_dir / "filtered_results.json", "w") as f:
            json.dump({
                "query_idx": idx,
                "query": query,
                "gold": gold,
                "candidates": [did for did, _ in candidates],
                "filtered_ranked": reranked,
                "alpha": alpha,
            }, f, indent=2)
        
        # Compute metrics
        metrics = compute_metrics(ranked_with_scores, gold, top_k_list=[top_k])
        results.append(metrics)
        
        if (idx + 1) % 10 == 0 or idx == len(entries) - 1:
            n = idx + 1
            mean_mrr = np.mean([r["mrr"] for r in results]) if results else 0.0
            logger.info(f"  [{idx:04d}/{n-1}] MRR={mean_mrr:.4f}  ({n}/{len(entries)})")
    
    # === SUMMARY ===
    if len(results) == 0:
        logger.error("No valid queries")
        update_run_status(bench_dir, dataset, "failed")
        return None
    
    # Compute mean metrics (handle both p@k and p@k formats)
    mean_mrr = float(np.mean([r["mrr"] for r in results]))
    mean_ap = float(np.mean([r["ap"] for r in results]))
    
    # Extract p@k values (try both formats)
    p_at_1 = [r.get("p@1", r.get("p@1", 0.0)) for r in results]
    p_at_2 = [r.get("p@2", r.get("p@2", 0.0)) for r in results]
    p_at_5 = [r.get("p@5", r.get("p@5", 0.0)) for r in results]
    
    summary = {
        "mean_mrr": mean_mrr,
        "mean_ap": mean_ap,
        "mean_p@1": float(np.mean(p_at_1)),
        "mean_p@2": float(np.mean(p_at_2)),
        "mean_p@5": float(np.mean(p_at_5)),
        "n_queries": len(results),
        "failed": failed,
    }
    
    with open(bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    update_run_status(bench_dir, dataset, "completed")
    logger.success(f"Two-stage benchmark complete: {bench_dir}")
    logger.info(f"  MRR={summary['mean_mrr']:.4f}  AP={summary['mean_ap']:.4f}  "
                f"P@1={summary['mean_p@1']:.4f}  P@2={summary['mean_p@2']:.4f}")
    
    return bench_dir


def cli_main():
    parser = argparse.ArgumentParser(
        description="Two-stage neuro-lexical pipeline benchmark (BM25 + SF re-ranking)",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Existing Phase 1 run directory (needs corpus.txt, fingerprints)")
    parser.add_argument("--pool-size", type=int, default=100,
                        help="BM25 candidate pool size (default: 100)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="BM25+SF interpolation weight (default: 0.5)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Final top-k results (default: 5)")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()
    
    bench_dir = run_two_stage_benchmark(
        dataset=args.dataset,
        jsonl_path=args.jsonl,
        run_dir=args.run_dir,
        pool_size=args.pool_size,
        max_queries=args.max_queries,
        top_k=args.top_k,
        alpha=args.alpha,
    )
    if bench_dir is None:
        sys.exit(1)
    print(f"\nTWO_STAGE_OK:{bench_dir}")


if __name__ == "__main__":
    cli_main()
