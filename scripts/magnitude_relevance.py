"""Magnitude-relevance analysis (Final-Improvements-V2 S3; reviewer #7,
resolver Experiments 3/4/5/14).

Questions answered on REAL retrieval traces:
  Q1 (margin stats)   Does score magnitude separate gold from distractors?
                      Per retriever: P(delta>0), mean/median delta, AUC(score)
                      where delta = score_gold - best_negative.
  Q2 (task split)     Are margins larger on multi-hop than single-hop/factoid?
  Q3 (hop coverage)   Does SPLADE score correlate with supporting-status
                      (hop coverage) after controlling for doc length +
                      lexical overlap? Spearman + OLS regression.
  Q4 (calibration)    P(gold | score-bin) per retriever - is magnitude
                      semantically meaningful or flat?

Data: real component scores captured during alpha-sweep endpoint runs
(maxnorm SF / maxnorm SPLADE) for hotpotqa/musique/scifact + supporting-doc
annotations from the converted datasets.

Outputs: appendix_stats/magnitude_relevance.md (+ .json)
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
OUT_DIR = PROJ / "docs/papers/Journal A/appendix_stats"

ALPHA = PROJ / "docs/papers/Journal A/appendix_alpha"
TRACES = {
    "hotpotqa": {
        "sf": ALPHA / "hotpotqa_comp_1.0.json",
        "splade": ALPHA / "hotpotqa_comp_0.0.json",
        "gold": PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/query_gold.json",
        "jsonl": PROJ / "data/hotpotqa/converted/hotpotqa.jsonl",
        "task": "multi-hop",
    },
    "musique": {
        "sf": ALPHA / "musique_comp_1.0.json",
        "splade": ALPHA / "musique_comp_0.0.json",
        "gold": PROJ / "outputs/musique_benchmark/runs/run_20260822_191925/query_gold.json",
        "jsonl": PROJ / "data/musique/converted/musique.jsonl",
        "task": "multi-hop",
    },
    "scifact": {
        "sf": ALPHA / "scifact_comp_1.0.json",
        "splade": ALPHA / "scifact_comp_0.0.json",
        "gold": PROJ / "outputs/scifact_benchmark/runs/run_20260822_191507/query_gold.json",
        "jsonl": PROJ / "data/scifact/converted/scifact.jsonl",
        "task": "claim-verification",
    },
}


def load_scores(path):
    return [dict(q["results"]) for q in json.loads(path.read_text(encoding="utf-8"))]


def load_dataset_meta(path):
    """Return per-query list of {doc_id: {'supporting': bool, 'length': int, 'overlap': float}}."""
    meta = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            paras = e.get("paragraphs", [])
            info = {}
            for p in paras:
                title = p.get("title", "")
                text = p.get("paragraph_text", "")
                info[title] = {
                    "title": title,
                    "supporting": bool(p.get("is_supporting", False)),
                    "length": len(text.split()),
                    "text": text.lower(),
                }
            meta.append({
                "paras": info,
                "question": e.get("question", "").lower(),
            })
    return meta


def build_supporting_map(jsonl_path, corpus_path):
    """doc_id -> supporting-title set per query.

    Corpus lines are 'gid, {title} {paragraph_text}'. For each query we know
    its candidate gids (query_doc_map) and gold titles (is_supporting flags in
    the converted JSONL). A doc supports query i iff the doc's title matches
    one of query i's supporting titles. Returns: {qid: {gid: True}}.
    """
    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    # doc_id -> (title, text) from corpus lines
    docs = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            gid, rest = line.split(",", 1)
            docs[gid.strip()] = rest.strip()

    maps = {}
    for qi, e in enumerate(entries):
        sup_titles = {p["title"] for p in e.get("paragraphs", [])
                      if p.get("is_supporting")}
        flags = {}
        if not sup_titles:
            maps[qi] = flags
            continue
        for gid, body in docs.items():
            # title match: body starts with "<title> " (build_combined_corpus
            # writes 'gid, title text'); require the full title token prefix.
            is_sup = any(body == t or body.startswith(t + " ")
                         for t in sup_titles)
            if is_sup:
                flags[gid] = True
        maps[qi] = flags
    return maps


def title_of(gid_body):
    """Best-effort title extraction: leading tokens before sentence text."""
    return gid_body


def load_corpus_map(run_dir):
    cmap = {}
    with open(Path(run_dir) / "corpus.txt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            gid, rest = line.split(",", 1)
            cmap[gid.strip()] = rest.strip()
    return cmap


def lexical_overlap(question_lower, text_lower):
    qw = set(w for w in question_lower.split() if len(w) > 2)
    if not qw:
        return 0.0
    tw = set(text_lower.split())
    return len(qw & tw) / len(qw)


def auc_score(pos, neg):
    """Rank-based AUC of score separating pos from neg samples."""
    if not pos or not neg:
        return float("nan")
    diffs = [p - n for p in pos for n in neg]
    return sum(1 for d in diffs if d > 0) / len(diffs)


def spearman(x, y):
    """Spearman rho with average ranks for ties (scipy-compatible)."""
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    rho = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).statistic
    return float(rho) if not np.isnan(rho) else float("nan")


def ols_beta(X, y):
    """OLS betas with plain numpy; returns coefficients + t-ish z via SE."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xc = np.column_stack([np.ones(len(X))] + list(X.T))
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ y
    resid = y - Xc @ beta
    dof = max(1, len(y) - Xc.shape[1])
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    return beta, beta / np.where(se == 0, 1e-12, se)


def main():
    corpus_maps = {
        "hotpotqa": load_corpus_map(PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656"),
        "musique": load_corpus_map(PROJ / "outputs/musique_benchmark/runs/run_20260822_191925"),
        "scifact": load_corpus_map(PROJ / "outputs/scifact_benchmark/runs/run_20260822_191507"),
    }
    ds_meta = {ds: load_dataset_meta(cfg["jsonl"]) for ds, cfg in TRACES.items()}

    results = {"margin_stats": {}, "hop_regression": {}, "calibration": {}}
    lines = [
        "# Magnitude Relevance Analysis (S3; H3 test)\n",
        "Real component traces; delta = gold score - best negative score per "
        "retriever. Hop coverage = candidate is a dataset-annotated supporting "
        "document. Regression controls: doc word length, query-term lexical overlap.\n",
    ]

    # ---------- Q1/Q2 margin stats ----------
    lines += ["\n## Margin statistics by retriever and task\n",
              "| Dataset | Task | Retriever | P(delta>0) | mean delta | median delta | AUC(score) | n queries |",
              "|---------|------|-----------|-----------:|-----------:|-------------:|-----------:|----------:|"]
    for ds, cfg in TRACES.items():
        sf = load_scores(cfg["sf"])
        sp = load_scores(cfg["splade"])
        gold = json.loads(cfg["gold"].read_text(encoding="utf-8"))
        cmap = corpus_maps[ds]
        meta = ds_meta[ds]
        stats = {}
        for retr_name, scores in (("SF", sf), ("SPLADE", sp)):
            deltas, pos_list, neg_all = [], [], []
            for qi in range(len(scores)):
                gset = set(gold.get(str(qi), []))
                sc = scores[qi]
                g_vals = [v for d, v in sc.items() if d in gset]
                n_vals = [v for d, v in sc.items() if d not in gset]
                if not g_vals or not n_vals:
                    continue
                delta = max(g_vals) - max(n_vals)
                deltas.append(delta)
                pos_list.extend(g_vals)
                neg_all.extend(n_vals)
            deltas = np.array(deltas)
            row = {
                "p_delta_pos": float((deltas > 0).mean()) if len(deltas) else None,
                "mean_delta": float(deltas.mean()) if len(deltas) else None,
                "median_delta": float(np.median(deltas)) if len(deltas) else None,
                "auc": auc_score(pos_list, neg_all),
                "n_queries": len(deltas),
            }
            stats[retr_name] = row
            lines.append(
                f"| {ds} | {cfg['task']} | {retr_name} | "
                f"{row['p_delta_pos']:.2f} | {row['mean_delta']:.3f} | "
                f"{row['median_delta']:.3f} | {row['auc']:.3f} | {row['n_queries']} |")
        results["margin_stats"][ds] = stats

    # ---------- Q3 hop coverage regression (score vs supporting status, title-matched) ----------
    lines += ["\n## Score ~ supporting-status analysis (title-matched supporting docs; controls: length, overlap)\n",
              "| Dataset | Retriever | Spearman(score, supporting) | n docs | n supporting |",
              "|---------|-----------|----------------------------:|-------:|-------------:|"]
    sup_maps = {
        "hotpotqa": build_supporting_map(
            PROJ / "data/hotpotqa/converted_n50_backup/hotpotqa.jsonl",
            PROJ / "outputs/hotpotqa_benchmark/runs/run_20260822_163656/corpus.txt"),
        "musique": build_supporting_map(
            PROJ / "data/musique/converted/musique.jsonl",
            PROJ / "outputs/musique_benchmark/runs/run_20260822_191925/corpus.txt"),
        "scifact": build_supporting_map(
            PROJ / "data/scifact/converted/scifact.jsonl",
            PROJ / "outputs/scifact_benchmark/runs/run_20260822_191507/corpus.txt"),
    }
    for ds, cfg in TRACES.items():
        sf = load_scores(cfg["sf"])
        sp = load_scores(cfg["splade"])
        gold = json.loads(cfg["gold"].read_text(encoding="utf-8"))
        cmap = corpus_maps[ds]
        smap = sup_maps.get(ds)
        if smap is None:
            continue
        for retr_name, scores in (("SF", sf), ("SPLADE", sp)):
            X_rows, y_rows = [], []
            r_pairs = []
            for qi in range(len(scores)):
                gset = set(gold.get(str(qi), []))
                flags = smap.get(qi, {})
                question = ""
                if qi < len(ds_meta[ds]):
                    question = ds_meta[ds][qi]["question"]
                for d, s in scores[qi].items():
                    body = cmap.get(d, "")
                    sup = bool(flags.get(d)) or (d in gset)
                    length = len(body.split())
                    overlap = lexical_overlap(question, body)
                    y_rows.append(1.0 if sup else 0.0)
                    X_rows.append([length, overlap])
                    r_pairs.append((s, 1.0 if sup else 0.0))
            if len(set(y_rows)) < 2:
                continue
            rho = spearman([p[0] for p in r_pairs], [p[1] for p in r_pairs])
            n_sup = int(sum(y_rows))
            results["hop_regression"][f"{ds}/{retr_name}"] = {
                "spearman": round(rho, 3), "n_docs": len(y_rows), "n_supporting": n_sup}
            lines.append(f"| {ds} | {retr_name} | {rho:.3f} | {len(y_rows)} | {n_sup} |")

    # ---------- Q4 calibration ----------
    lines += ["\n## Calibration: P(supporting/gold | score bin)\n",
              "| Dataset | Retriever | bin | n | P(gold) |",
              "|---------|-----------|-----|--:|--------:|"]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.001]
    for ds, cfg in TRACES.items():
        sp = load_scores(cfg["splade"])
        gold = json.loads(cfg["gold"].read_text(encoding="utf-8"))
        for lo, hi in zip(bins[:-1], bins[1:]):
            tot = hit = 0
            for qi in range(len(sp)):
                gset = set(gold.get(str(qi), []))
                for d, s in sp[qi].items():
                    if lo <= s < hi:
                        tot += 1
                        hit += int(d in gset)
            if tot:
                results["calibration"][f"{ds}/{lo:.1f}-{hi:.1f}"] = {
                    "n": tot, "p_gold": round(hit / tot, 4)}
                lines.append(f"| {ds} | SPLADE | [{lo:.1f},{hi:.1f}) | {tot} | {hit/tot:.3f} |")

    out_md = OUT_DIR / "magnitude_relevance.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "magnitude_relevance.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
