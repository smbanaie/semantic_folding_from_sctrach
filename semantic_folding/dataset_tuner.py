"""Dataset parameter tuner for Semantic Folding benchmarks.

Runs a 4-way grid (UMAP/t-SNE x max_doc_freq=0 / max_doc_freq_pct=5%) for each
requested profile (sf_only, sf_splade), picks the MRR winner per profile, then
compares the two profile winners and writes the BEST one as the active flat
top-level entry for the dataset in the target registry YAML.

Mirrors the process captured in temp/tune_*_splade.log (the original tuning runs).

Usage:
    python -m semantic_folding.dataset_tuner \
        --dataset scifact --jsonl data/scifact/converted/scifact.jsonl \
        --registry config/dataset_registry.yml --max-queries 50

The registry is updated in place. Non-dataset keys (defaults, protocols, other
datasets) are preserved. The dataset entry is written as a flat top-level block
(the schema actually consumed by load_dataset_registry); both profiles' tuned
results are also recorded under a `tuning:` subkey for documentation.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml

from semantic_folding.lib import get_logger

logger = get_logger("tune")

# --- paths -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from .dataset_benchmark.adapters import get_adapter
from .dataset_benchmark.generic_benchmark import (
    GenericBenchmarkRunner,
    load_dataset_registry,
)

# 4-way grid: (label, method, max_doc_freq, max_doc_freq_pct)
GRID = [
    ("umap_mdf0", "umap", 0, None),
    ("umap_mdf05pct", "umap", 20, 0.05),
    ("tsne_mdf0", "tsne", 0, None),
    ("tsne_mdf05pct", "tsne", 20, 0.05),
]

PROFILES = {
    "sf_only": {"splade": False},
    "sf_splade": {"splade": True, "splade_alpha": 0.3},
}


def build_params(registry_path: Path, dataset: str, profile: str,
                 method: str, max_doc_freq: int, max_doc_freq_pct) -> Dict[str, Any]:
    base = load_dataset_registry(registry_path=registry_path, dataset=dataset)
    base.update(PROFILES[profile])
    base["method"] = method
    base["max_doc_freq"] = max_doc_freq
    base["max_doc_freq_pct"] = max_doc_freq_pct
    base["use_morton"] = True
    base["doc_norm"] = base.get("doc_norm", "l2")
    base["weighting"] = base.get("weighting", "idf")
    base["smoothing_sigma"] = base.get("smoothing_sigma", 1.5)
    base["grid_size"] = base.get("grid_size", 64)
    base["top_percent"] = base.get("top_percent", 0.10)
    base["spreading_steps"] = base.get("spreading_steps", 1)
    base["oov_expansion"] = False
    base["fusion_method"] = base.get("fusion_method", "rrf")
    base["rrf_k"] = base.get("rrf_k", 60)
    # strip keys the runner does not expect / are structural
    for k in ("description", "max_queries", "config_profile", "profiles", "tuning"):
        base.pop(k, None)
    return base


def run_grid(adapter, registry_path: Path, dataset: str, jsonl_path: Path,
             profile: str, max_queries: int) -> Tuple[Optional[str], Dict[str, float], Dict[str, float]]:
    logger.info("")
    logger.info(f"  Profile: {profile} (splade={PROFILES[profile]['splade']})")
    logger.info(f"  {'─'*50}")
    results = {}
    timings = {}
    for label, method, mdf, mdf_pct in GRID:
        params = build_params(registry_path, dataset, profile, method, mdf, mdf_pct)
        runner = GenericBenchmarkRunner(adapter, params=params)
        t0 = time.time()
        run_dir = runner.phase1_index(jsonl_path, max_queries=max_queries)
        if run_dir is None:
            logger.error(f"    [{label}] index FAILED")
            results[label] = -1.0
            timings[label] = 0.0
            continue
        bench_dir = runner.phase2_benchmark(run_dir, jsonl_path, query_start=0, query_end=max_queries)
        elapsed = time.time() - t0
        if bench_dir is None:
            logger.error(f"    [{label}] benchmark FAILED")
            results[label] = -1.0
            timings[label] = elapsed
            continue
        summary_path = bench_dir / "summary.json"
        mrr = -1.0
        if summary_path.exists():
            with open(summary_path) as f:
                mrr = float(json.load(f).get("mean_mrr", -1.0))
        ap = 0.0
        if summary_path.exists():
            with open(summary_path) as f:
                ap = float(json.load(f).get("mean_ap", 0.0))
        results[label] = mrr
        timings[label] = elapsed
        logger.info(f"    [{label}] method={method}, MRR={mrr:.4f}, AP={ap:.4f}, time={elapsed:.0f}s")
    # pick winner: highest MRR, tiebreak by speed
    valid = {k: v for k, v in results.items() if v > 0}
    if not valid:
        return None, results, timings
    winner = sorted(valid.items(), key=lambda kv: (-kv[1], timings[kv[0]]))[0][0]
    logger.info(f"  WINNER ({profile}): {winner}, MRR={results[winner]:.4f}, time={timings[winner]:.0f}s")
    return winner, results, timings


def profile_winner_params(registry_path: Path, dataset: str, profile: str, winner_label: str) -> Dict[str, Any]:
    method = "umap" if winner_label.startswith("umap") else "tsne"
    if winner_label.endswith("mdf0"):
        mdf, mdf_pct = 0, None
    else:
        mdf, mdf_pct = 20, 0.05
    p = build_params(registry_path, dataset, profile, method, mdf, mdf_pct)
    p["splade"] = PROFILES[profile]["splade"]
    if profile == "sf_splade":
        p["splade_alpha"] = 0.3
    return p


def update_registry(registry_path: Path, dataset: str, active_params: Dict[str, Any],
                    best_profile: str, all_results: Dict[str, Any]):
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    original = registry.get(dataset, {})
    block = {
        "description": original.get("description", f"{dataset} (tuned)"),
        "max_queries": original.get("max_queries", 50),
        "splade": active_params.get("splade", False),
        "method": active_params.get("method", "umap"),
        "max_doc_freq": active_params.get("max_doc_freq", 0),
        "max_doc_freq_pct": active_params.get("max_doc_freq_pct"),
        "perplexity": active_params.get("perplexity", 50),
        "doc_norm": active_params.get("doc_norm", "l2"),
        "fusion_method": active_params.get("fusion_method", "rrf"),
        "rrf_k": active_params.get("rrf_k", 60),
    }
    if active_params.get("splade"):
        block["splade_alpha"] = active_params.get("splade_alpha", 0.3)
    # drop None values cleanly
    block = {k: v for k, v in block.items() if v is not None}
    block["tuning"] = {
        "best_profile": best_profile,
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "grid": all_results,
    }
    registry[dataset] = block
    # preserve header comment manually
    header = "# Dataset Parameter Registry (auto-tuned entries appended by dataset_tuner.py)\n"
    with open(registry_path, "w", encoding="utf-8") as f:
        f.write("# ============================================================\n")
        f.write("# Dataset Parameter Registry\n")
        f.write("# ============================================================\n")
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  Registry updated: {dataset} -> {block}")


def main():
    ap = argparse.ArgumentParser(description="Tune SF benchmark params for a dataset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--registry", type=Path,
                    default=PROJECT_ROOT / "config" / "dataset_registry.yml")
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--profiles", nargs="+", choices=list(PROFILES.keys()),
                    default=list(PROFILES.keys()))
    args = ap.parse_args()

    registry_path = args.registry.resolve()
    if not registry_path.exists():
        logger.error(f"Registry not found: {registry_path}")
        return
    adapter = get_adapter(args.dataset)

    logger.info("=" * 60)
    logger.info(f"  Tuning {args.dataset} ({adapter.display_name})")
    logger.info(f"  jsonl: {args.jsonl}")
    logger.info(f"  max_queries: {args.max_queries}")
    logger.info(f"  profiles: {args.profiles}")
    logger.info("=" * 60)

    profile_winners = {}
    all_results = {}
    for profile in args.profiles:
        winner, results, timings = run_grid(
            adapter, registry_path, args.dataset, args.jsonl, profile, args.max_queries)
        all_results[profile] = {
            "winner": winner,
            "mrr": results,
            "time": {k: round(v, 1) for k, v in timings.items()},
        }
        if winner:
            profile_winners[profile] = (winner, results[winner])

    if not profile_winners:
        logger.error("  No valid profile results — aborting registry update.")
        return

    # best profile = highest MRR winner across profiles
    best_profile = sorted(profile_winners.items(), key=lambda kv: -kv[1][1])[0][0]
    winner_label = profile_winners[best_profile][0]
    active_params = profile_winner_params(registry_path, args.dataset, best_profile, winner_label)
    # carry dataset-level description/max_queries from existing registry if present
    existing = load_dataset_registry(registry_path=registry_path, dataset=args.dataset)
    active_params["description"] = existing.get("description", f"{args.dataset} (tuned)")
    active_params["max_queries"] = existing.get("max_queries", args.max_queries)

    update_registry(registry_path, args.dataset, active_params, best_profile, all_results)

    logger.info("=" * 60)
    logger.info(f"  Tuning Complete — {args.dataset}")
    logger.info(f"  best_profile: {best_profile} ({winner_label}, MRR={profile_winners[best_profile][1]:.4f})")
    for prof, (w, mrr) in profile_winners.items():
        logger.info(f"    {prof:12s} -> {w}, MRR={mrr:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
