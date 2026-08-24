"""Promote the ad-hoc win/loss/rank-1 analysis to a tracked script
(Final V3, review items 0.3 + 12). Deterministic; reads only committed runs."""
import json
import math
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

RUNS = {
    "hotpotqa": ("benchmark_20260824_034107", "run_20260824_032535"),
    "musique": ("benchmark_20260824_034226", "run_20260824_033236"),
    "nq_rear": ("benchmark_20260824_034248", "run_20260824_033353"),
}


def load_op(bench, op, gold):
    out = {}
    pq = bench / f"op_{op}" / "per_query"
    for qdir in sorted(pq.iterdir()):
        if not qdir.is_dir():
            continue
        fr = json.loads((qdir / "filtered_results.json").read_text(encoding="utf-8"))
        qi = int(fr["query_idx"])
        g = set(gold.get(str(qi), []))
        ranked = [d for d, _ in (fr.get("filtered_ranked") or [])]
        mrr = 0.0
        r1 = None
        for i, d in enumerate(ranked, 1):
            if d in g:
                mrr = 1.0 / i
                r1 = d
                break
        out[qi] = {"mrr": mrr, "rank1": r1}
    return out


def main():
    summary = {}
    lines = [
        "# Win/Tie/Loss and Rank-1 Change Analysis: CombSUM vs RRF (n=100)\n",
        "Per-query paired outcomes from the n=100 confirmatory core. "
        "'Rank-1 change' counts queries where switching RRF -> CombSUM changes "
        "which document is ranked first — the information-bottleneck statistic.\n",
        "| Dataset | n | CombSUM wins | RRF wins | ties | win% | rank-1 changes | rank-1 % | mean \u0394 | dz | n needed (power .8) |",
        "|---------|--:|-------------:|---------:|-----:|-----:|---------------:|---------:|-------:|---:|--------------------:|",
    ]
    for ds, (bench_name, run_name) in RUNS.items():
        bench = PROJ / f"outputs/{ds}_benchmark/benchmarks/{bench_name}"
        gold = json.loads((PROJ / f"outputs/{ds}_benchmark/runs/{run_name}/query_gold.json").read_text(encoding="utf-8"))
        cs = load_op(bench, "combsum", gold)
        rf = load_op(bench, "rrf", gold)
        common = sorted(set(cs) & set(rf))
        wins = losses = r1_changes = 0
        deltas = []
        for qi in common:
            a, b = cs[qi]["mrr"], rf[qi]["mrr"]
            deltas.append(a - b)
            if a > b + 1e-12:
                wins += 1
            elif b > a + 1e-12:
                losses += 1
            if cs[qi]["rank1"] != rf[qi]["rank1"]:
                r1_changes += 1
        ties = len(common) - wins - losses
        n = len(common)
        mean_d = sum(deltas) / n
        sd = (sum((d - mean_d) ** 2 for d in deltas) / (n - 1)) ** 0.5 if n > 1 else 0.0
        dz = abs(mean_d) / sd if sd > 0 else float("inf")
        n_needed = int(math.ceil(((1.96 + 0.84) / dz) ** 2)) if sd > 0 and dz > 0 else None
        summary[ds] = {
            "n": n, "combsum_wins": wins, "rrf_wins": losses, "ties": ties,
            "win_pct": round(100 * wins / n, 1),
            "rank1_changed_queries": r1_changes,
            "rank1_change_pct": round(100 * r1_changes / n, 1),
            "mean_delta": round(mean_d, 4), "dz": round(dz, 3),
            "n_needed_for_power80": n_needed,
        }
        lines.append(
            f"| {ds} | {n} | {wins} | {losses} | {ties} | {summary[ds]['win_pct']}% "
            f"| {r1_changes} | {summary[ds]['rank1_change_pct']}% "
            f"| {mean_d:+.3f} | {dz:.2f} | {n_needed} |")

    out_md = OUT_DIR / "win_loss_rank1_n100.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "win_loss_rank1_n100.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
