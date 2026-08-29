"""Operator identifiability table (SIGIR-Final-Tasks Item 3).

Computes, per retriever pair and per operator pair, the identifiability metrics from
Reviews §9:
  I_global = #{q: full ranking F_A != F_B} / N
  I_k      = #{q: Top_k(F_A) != Top_k(F_B)} / N   (k in {1,5,10})
  I_1      = #{q: argmax(F_A) != argmax(F_B)} / N
plus per-query Kendall tau(F_A, F_B) and P(tau == 1) (exact rank equality).

Reuses SF+SPLADE component traces (same as Items 1-2). Other pairs (SF+DPR,
BM25+SPLADE, BM25+DPR) are accepted via --pair once their component traces exist
(see gen_component_traces_n100.py extension); the script is pair-agnostic.

Output: docs/papers/Journal A/appendix_stats/operator_identifiability_{pair}_n{N}.{json,md}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
sys.path.insert(0, str(PROJ))
from semantic_folding import fusion_operators as fo

ALPHA_DIR = PROJ / "docs/papers/Journal A/appendix_alpha"
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# pair -> (signal_A_comp_tag, signal_B_comp_tag, gold_run_dir)
PAIRS = {
    "sf_splade": (
        PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/query_gold.json",
    ),
    "sf_dpr": None,       # traces not yet generated
    "bm25_splade": None,  # traces not yet generated
    "bm25_dpr": None,     # traces not yet generated
}

OPERATORS = ["rrf", "combsum", "combmnz", "linear", "borda", "zscore", "minmax"]


def comp_path(ds, n, pair="sf_splade"):
    if pair == "sf_splade":
        # legacy naming (no pair prefix)
        if n == 100:
            return ALPHA_DIR / f"{ds}_comp_1.0_n100.json", ALPHA_DIR / f"{ds}_comp_0.0_n100.json"
        return ALPHA_DIR / f"{ds}_comp_1.0.json", ALPHA_DIR / f"{ds}_comp_0.0.json"
    if n == 100:
        return (ALPHA_DIR / f"{ds}_{pair}_comp_1.0_n100.json",
                ALPHA_DIR / f"{ds}_{pair}_comp_0.0_n100.json")
    return (ALPHA_DIR / f"{ds}_{pair}_comp_1.0.json",
            ALPHA_DIR / f"{ds}_{pair}_comp_0.0.json")


def load_components(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(q["results"]) for q in raw]


def load_gold(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = [raw[str(i)] if str(i) in raw else raw[i] for i in range(len(raw))]
        return [set(v) for v in items]
    return [set((q.get("gold") or q.get("gold_ids") or q.get("answer_docs")) or []) for q in raw]


def fuse(sf, sp, operator):
    docs = sorted(set(sf) | set(sp))
    sfv = {d: sf.get(d, 0.0) for d in docs}
    spv = {d: sp.get(d, 0.0) for d in docs}
    if operator == "borda":
        fused = fo.fuse(operator, sfv, spv, n_docs=len(docs))
    else:
        fused = fo.fuse(operator, sfv, spv, k=60)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def kendall_tau_ranked(a, b):
    pos_a = {d: i for i, d in enumerate(a)}
    pos_b = {d: i for i, d in enumerate(b)}
    docs = list(set(pos_a) & set(pos_b))
    n = len(docs)
    if n < 2:
        return 1.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            x = pos_a[docs[i]] - pos_a[docs[j]]
            y = pos_b[docs[i]] - pos_b[docs[j]]
            if x * y > 0:
                c += 1
            elif x * y < 0:
                d += 1
    den = c + d
    return 1.0 if den == 0 else (c - d) / den


def topk_set(ranked, k):
    return set(ranked[:k])


def identifiability(sf_list, sp_list, op_a, op_b):
    n = min(len(sf_list), len(sp_list))
    i_global = i1 = i5 = i10 = 0
    taus = []
    exact = 0
    for qi in range(n):
        ra = fuse(sf_list[qi], sp_list[qi], op_a)
        rb = fuse(sf_list[qi], sp_list[qi], op_b)
        if ra != rb:
            i_global += 1
        if topk_set(ra, 1) != topk_set(rb, 1):
            i1 += 1
        if topk_set(ra, 5) != topk_set(rb, 5):
            i5 += 1
        if topk_set(ra, 10) != topk_set(rb, 10):
            i10 += 1
        t = kendall_tau_ranked(ra, rb)
        taus.append(t)
        if t == 1.0:
            exact += 1
    return {
        "n": n,
        "I_global": i_global / n,
        "I_1": i1 / n,
        "I_5": i5 / n,
        "I_10": i10 / n,
        "mean_kendall_tau": float(np.mean(taus)),
        "P_exact_rank_equal": exact / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, choices=[10, 100])
    ap.add_argument("--ds", default="hotpotqa", help="dataset key for component traces")
    ap.add_argument("--pair", default="sf_splade",
                    help="pair label (used only for output naming when traces supplied)")
    args = ap.parse_args()
    N = args.n
    c1, c0 = comp_path(args.ds, N, args.pair)
    if not (c1.exists() and c0.exists()):
        print(f"SKIP {args.pair}: missing {c1.name}/{c0.name}")
        return
    sf_list = load_components(c1)
    sp_list = load_components(c0)
    # gold not needed for identifiability (it compares operator rankings)
    op_pairs = [("rrf", "combsum"), ("rrf", "combmnz"), ("combsum", "combmnz"),
                ("rrf", "linear"), ("combsum", "linear")]
    table = {}
    for oa, ob in op_pairs:
        key = f"{oa}_vs_{ob}"
        table[key] = identifiability(sf_list, sp_list, oa, ob)
        r = table[key]
        print(f"  {key:18s} I_global={r['I_global']:.3f} I_1={r['I_1']:.3f} "
              f"I_5={r['I_5']:.3f} I_10={r['I_10']:.3f} P(exact)={r['P_exact_rank_equal']:.3f}")
    out = {
        "pair": args.pair, "dataset": args.ds, "sample_size": N, "table": table,
    }
    name = f"operator_identifiability_{args.pair}_{args.ds}_n{N}"
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # markdown
    lines = [f"# Operator identifiability — {args.pair} / {args.ds} / n={N}", ""]
    lines.append("| operator pair | I_global | I_1 | I_5 | I_10 | mean τ | P(exact) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, r in table.items():
        lines.append(f"| {k} | {r['I_global']:.3f} | {r['I_1']:.3f} | {r['I_5']:.3f} | "
                     f"{r['I_10']:.3f} | {r['mean_kendall_tau']:.3f} | {r['P_exact_rank_equal']:.3f} |")
    (OUT_DIR / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {name}.json/.md")


if __name__ == "__main__":
    main()
