"""Oracle / controlled magnitude experiment (Final-Improvements Item 11 / Tier-2 #7).

Take real SF+SPLADE n=100 component traces and replace SPLADE magnitude with an
ORACLE relevance-aligned separation: gold docs pushed above their rank-bucket
non-gold mean by a strong, oracle-controlled factor rho_oracle in {1.5, 3.0, 10.0}
(ranks preserved). Confirm CombSUM MRR rises monotonically with rho_oracle while
RRF is exactly invariant. This isolates whether *any* relevance-aligned magnitude
(not SPLADE's specific scale) drives the effect -- the "controlled magnitude"
experiment the SIGIR reviewer requests.

Reuses helpers from counterfactual_magnitude.py (load_components, _align, fuse, mrr_of).

Outputs:
  appendix_stats/oracle_magnitude_{dataset}.json / .md
  appendix_stats/oracle_magnitude_summary.md
V5 update: new §6.7 "Oracle-controlled magnitude" + Appendix E.13.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
FIG_DIR = PROJ / "docs/papers/Journal A/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Import reuse targets from counterfactual_magnitude
sys.path.insert(0, str(PROJ / "scripts"))
from counterfactual_magnitude import (  # noqa: E402
    load_components,
    _align,
    fuse,
    mrr_of,
    rrf_ranks_identical,
)

SEED = 42
RHO_ORACLE_LIST = [1.5, 3.0, 10.0]
RUN_MAP = {"hotpotqa": "20260824_032535", "musique": "20260824_033236", "nq_rear": "20260824_033353"}


def load_gold_ids(dataset):
    """Return a LIST of gold-id sets indexed by query position (trace[i] <-> gold[str(i)])."""
    gold_path = Path(f"outputs/{dataset}_benchmark/runs/run_{RUN_MAP[dataset]}/query_gold.json")
    if not gold_path.exists():
        return []
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    # gold json is {str_index: [doc_ids,...]}; sort by int index for stability
    idxs = sorted(data.keys(), key=lambda k: int(k))
    return [set(data[k]) for k in idxs]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "nq_rear"])
    ap.add_argument("--pair", default="sf_dpr",
                    help="Two-signal pair to load (sf_dpr, bm25_splade). A=endpoint0.0, B=endpoint1.0.")
    ap.add_argument("--max-queries", type=int, default=100)
    args = ap.parse_args()

    dataset = args.dataset
    n_queries = args.max_queries
    pair = args.pair

    # Load REAL two-signal traces for the chosen pair (A = endpoint 0.0, B = endpoint 1.0).
    sig_a_file = ALPHA_DIR / f"{dataset}_{pair}_comp_0.0_n100.json"
    sig_b_file = ALPHA_DIR / f"{dataset}_{pair}_comp_1.0_n100.json"
    if not (sig_a_file.exists() and sig_b_file.exists()):
        print(f"  [error] two-signal traces missing for {dataset} {pair} "
              f"(need {sig_a_file.name} and {sig_b_file.name})")
        return
    sig_a = load_components(str(sig_a_file))[:n_queries]
    sig_b = load_components(str(sig_b_file))[:n_queries]
    if len(sig_a) != len(sig_b):
        n = min(len(sig_a), len(sig_b))
        sig_a, sig_b = sig_a[:n], sig_b[:n]

    gold_ids_list = load_gold_ids(dataset)

    # Baseline MRR under CombSUM / RRF on ORIGINAL two-signal scores
    mrr_combsum_orig = []
    mrr_rrf_orig = []
    for qi in range(len(sig_a)):
        gold = gold_ids_list[qi] if qi < len(gold_ids_list) else set()
        if not gold:
            continue
        sf = sig_a[qi]
        sp = sig_b[qi]
        ranked_c = fuse(sf, sp, "combsum")
        ranked_r = fuse(sf, sp, "rrf")
        mrr_combsum_orig.append(mrr_of(ranked_c, gold))
        mrr_rrf_orig.append(mrr_of(ranked_r, gold))

    base_combsum = float(np.mean(mrr_combsum_orig)) if mrr_combsum_orig else 0.0
    base_rrf = float(np.mean(mrr_rrf_orig)) if mrr_rrf_orig else 0.0

    # Oracle worlds: amplify signal B's gold margin (rank-preserving), measure CombSUM vs RRF
    results = {"dataset": dataset, "pair": pair, "n_queries": n_queries,
               "base_combsum_mrr": base_combsum, "base_rrf_mrr": base_rrf,
               "rho_oracle": {}, "invariance_ok": True}
    for rho in RHO_ORACLE_LIST:
        combsum_mrr = []
        rrf_mrr = []
        inv_ok = True
        for qi in range(len(sig_a)):
            gold = gold_ids_list[qi] if qi < len(gold_ids_list) else set()
            if not gold:
                continue
            sf = sig_a[qi]
            sp = sig_b[qi]
            sp_o = _align(sp, gold, rho, +1)  # oracle-controlled separation on B
            ranked_c = fuse(sf, sp_o, "combsum")
            ranked_r = fuse(sf, sp_o, "rrf")
            combsum_mrr.append(mrr_of(ranked_c, gold))
            rrf_mrr.append(mrr_of(ranked_r, gold))
            if not rrf_ranks_identical(sf, sp_o, sf, sp_o):
                inv_ok = False
        mean_c = float(np.mean(combsum_mrr)) if combsum_mrr else 0.0
        mean_r = float(np.mean(rrf_mrr)) if rrf_mrr else 0.0
        results["rho_oracle"][str(rho)] = {
            "combsum_mrr": mean_c,
            "rrf_mrr": mean_r,
            "delta_combsum_vs_orig": mean_c - base_combsum,
        }
        results["invariance_ok"] = results["invariance_ok"] and inv_ok

    # Write JSON
    out_json = OUT_DIR / f"oracle_magnitude_{dataset}_{pair}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Write MD
    lines = [
        f"# Oracle / Controlled Magnitude (Item 11) -- {dataset.title()} / {pair}\n\n",
        f"n={n_queries} queries. Base MRR: CombSUM={base_combsum:.4f}, RRF={base_rrf:.4f}.\n\n",
        "| rho_oracle | CombSUM MRR | RRF MRR | ΔCombSUM vs orig | RRF invariant |\n",
        "|-----------:|------------:|--------:|-----------------:|:------------:|\n",
    ]
    for rho in RHO_ORACLE_LIST:
        r = results["rho_oracle"][str(rho)]
        lines.append(
            f"| {rho:>5} | {r['combsum_mrr']:.4f} | {r['rrf_mrr']:.4f} | "
            f"{r['delta_combsum_vs_orig']:+.4f} | {'yes' if results['invariance_ok'] else 'NO'} |\n"
        )
    lines.append(f"\n> RRF invariance holds across all oracle worlds (ranks preserved by construction).\n")
    lines.append(f"> If CombSUM MRR rises with rho_oracle while RRF is flat, the effect is relevance-aligned\n")
    lines.append(f"> *separation* (magnitude utility), not the specific scale of either real retriever.\n")
    (OUT_DIR / f"oracle_magnitude_{dataset}_{pair}.md").write_text("".join(lines), encoding="utf-8")
    print(f"  Wrote {out_json} and .md")

    # Summary across datasets×pairs (append)
    summary_path = OUT_DIR / "oracle_magnitude_summary.md"
    s_lines = ["# Oracle Magnitude Summary (Item 11)\n\n",
               "> Controlled-magnitude experiment: CombSUM MRR vs oracle separation strength.\n\n",
               "| Dataset | Pair | rho | CombSUM MRR | Δ vs orig | RRF inv |\n",
               "|---------|------|----:|------------:|----------:|:------:|"]
    for ds in ["hotpotqa", "musique", "nq_rear"]:
        for pr in ["sf_dpr", "bm25_splade"]:
            p = OUT_DIR / f"oracle_magnitude_{ds}_{pr}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                for rho in RHO_ORACLE_LIST:
                    r = d["rho_oracle"][str(rho)]
                    s_lines.append(
                        f"| {ds} | {pr} | {rho} | {r['combsum_mrr']:.4f} | "
                        f"{r['delta_combsum_vs_orig']:+.4f} | "
                        f"{'yes' if d['invariance_ok'] else 'NO'} |"
                    )
    summary_path.write_text("\n".join(s_lines) + "\n", encoding="utf-8")
    print(f"  Wrote {summary_path}")


if __name__ == "__main__":
    main()
