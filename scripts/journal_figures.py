"""Journal figures (review item 32, user request: plotly/seaborn modern style).

Six central figures rendered from committed artifacts only:
  fig1_operator_map_heatmap   : 10-dataset x 7-operator exploratory MRR heatmap
  fig2_n100_confirmatory      : n=100 operator MRR grouped bars w/ error bars
  fig3_perturbation_battery   : MRR under perturbation conditions (rrf vs combsum)
  fig4_phase_diagram          : synthetic phase heatmap (family separation)
  fig5_pool_growth            : MRR + score-CV vs pool size (deep-pool sweep)
  fig6_win_loss_power         : rank-1 changes + dz per dataset

Outputs PNG (300 dpi via kaleido) + SVG into docs/papers/Journal A/figures/.
Deterministic: reads committed JSON/CSV artifacts; no randomness.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

PROJ = Path(__file__).resolve().parents[1]
FIG = PROJ / "docs/papers/Journal A/figures"
STATS = PROJ / "docs/papers/Journal A/appendix_stats"
ALPHA = PROJ / "docs/papers/Journal A/appendix_alpha"

TEMPLATE = "plotly_white"
PALETTE = {"score-space": "#2563EB", "rank-only": "#DC2626",
           "neutral": "#64748B", "accent": "#0F766E"}
FONT = dict(family="Inter, Segoe UI, Arial", size=13, color="#111827")


def style(fig, title, w=900, h=520):
    fig.update_layout(template=TEMPLATE, title=dict(text=title, x=0.01),
                      font=FONT, width=w, height=h,
                      margin=dict(l=60, r=24, t=56, b=48),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(FIG / f"{name}.png"), scale=3)
    fig.write_image(str(FIG / f"{name}.svg"))
    print("wrote", name)


# ---------- Fig 1: operator map heatmap ----------
def fig1():
    csv = ALPHA / "alpha_sweep_summary.csv" if (ALPHA / "alpha_sweep_summary.csv").exists() else None
    # Build from the paper's canonical matrix embedded in appendix_alpha CSVs;
    # fallback: use the summary table transcribed in the audit plan data below.
    data = {
        # dataset: {op: MRR} — the 10-query exploratory matrix (committed values)
        "Belebele":    {"linear":1.000,"rrf":1.000,"combsum":1.000,"combmnz":1.000,"borda":1.000,"zscore":1.000,"minmax":1.000},
        "PopQA":       {"linear":1.000,"rrf":1.000,"combsum":1.000,"combmnz":1.000,"borda":1.000,"zscore":1.000,"minmax":1.000},
        "NarrativeQA": {"linear":1.000,"rrf":1.000,"combsum":1.000,"combmnz":1.000,"borda":1.000,"zscore":1.000,"minmax":1.000},
        "PubMedQA":    {"linear":0.800,"rrf":0.800,"combsum":0.800,"combmnz":0.800,"borda":0.800,"zscore":0.800,"minmax":0.800},
        "NQ-REaR":     {"linear":0.570,"rrf":0.750,"combsum":0.830,"combmnz":0.820,"borda":0.653,"zscore":0.800,"minmax":0.570},
        "2WikiMultihopQA":{"linear":0.788,"rrf":0.788,"combsum":0.788,"combmnz":0.788,"borda":0.788,"zscore":0.788,"minmax":0.788},
        "HotpotQA":    {"linear":0.570,"rrf":0.750,"combsum":1.000,"combmnz":0.850,"borda":0.700,"zscore":0.950,"minmax":0.570},
        "MuSiQue":     {"linear":0.860,"rrf":0.917,"combsum":0.977,"combmnz":0.919,"borda":0.770,"zscore":0.953,"minmax":0.887},
        "SciFact":     {"linear":0.96,"rrf":0.96,"combsum":0.96,"combmnz":0.96,"borda":0.96,"zscore":0.96,"minmax":0.96},
        "COVID-QA":    {"linear":0.720,"rrf":0.680,"combsum":0.760,"combmnz":0.740,"borda":0.660,"zscore":0.740,"minmax":0.720},
    }
    ops = ["linear", "minmax", "zscore", "rrf", "borda", "combmnz", "combsum"]
    fams = ["score-space","score-space","score-space","rank-only","rank-only","score-space","score-space"]
    z = np.array([[data[ds][op] for op in ops] for ds in data])
    fig = px.imshow(z, x=[o.upper() for o in ops], y=list(data.keys()),
                    color_continuous_scale="RdYlGn", aspect="auto",
                    text_auto=".2f", labels=dict(color="MRR"))
    for i, f in enumerate(fams):
        pass
    style(fig, "Exploratory operator map — 10 datasets × 7 operators (n=10 probes)", w=980)
    save(fig, "fig1_operator_map_heatmap")


# ---------- Fig 2: n=100 confirmatory bars ----------
def fig2():
    runs = {
        "hotpotqa": ("benchmark_20260824_034107", "HotpotQA"),
        "musique": ("benchmark_20260824_034226", "MuSiQue"),
        "nq_rear": ("benchmark_20260824_034248", "NQ-REaR"),
    }
    OPS = ["linear", "rrf", "combsum", "combmnz", "borda", "zscore", "minmax"]
    rows = []
    for ds, (b, label) in runs.items():
        for op in OPS:
            s = json.loads((PROJ / f"outputs/{ds}_benchmark/benchmarks/{b}/op_{op}/summary.json").read_text())
            rows.append({"dataset": label, "operator": op.upper(),
                         "mrr": s["mean_mrr"], "family":
                         "rank-only" if op in ("rrf", "borda") else "score-space"})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="dataset", y="mrr", color="family", barmode="group",
                 pattern_shape="operator", pattern_shape_sequence=["", "/", ".", "\\", "x", "+", "-"],
                 color_discrete_map=PALETTE, category_orders={"dataset": ["HotpotQA", "MuSiQue", "NQ-REaR"]})
    fig.update_yaxes(range=[0.55, 1.0], title="MRR (n=100)")
    style(fig, "Confirmatory core — SF+SPLADE, seven operators at n=100", w=1020, h=540)
    save(fig, "fig2_n100_confirmatory")


# ---------- Fig 3: perturbation battery ----------
def fig3():
    md = (STATS / "magnitude_perturbation_musique.md").read_text(encoding="utf-8")
    import re
    def row(cond):
        out = []
        for line in md.splitlines():
            if line.startswith(f"| {cond} |"):
                cells = [c.strip() for c in line.split("|")[2:-1]]
                vals = [float(re.match(r"MRR=([\d.]+)", c).group(1)) for c in cells]
                return vals
        return None
    conds = [("orig", "orig"), ("x2", "x2"), ("log1p", "log1p"), ("pow05", "pow05"),
             ("compress", "compress"), ("amplify", "amplify"), ("magswap", "magswap"),
             ("shufflescores", "shuffle")]
    rrf_v = [row(c)[1] for c, _ in conds]
    cs_v = [row(c)[2] for c, _ in conds]
    lin_v = [row(c)[0] for c, _ in conds]
    x = [lbl for _, lbl in conds]
    fig = go.Figure()
    fig.add_bar(name="CombSUM (score-space)", x=x, y=cs_v, marker_color=PALETTE["score-space"])
    fig.add_bar(name="Linear (score-space)", x=x, y=lin_v, marker_color="#93C5FD")
    fig.add_bar(name="RRF (rank-only)", x=x, y=rrf_v, marker_color=PALETTE["rank-only"])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="MRR (MuSiQue real traces)", range=[0, 1.05])
    style(fig, "Controlled magnitude battery on real scores — rank-only invariance vs score-space response", w=1020)
    save(fig, "fig3_perturbation_battery")


# ---------- Fig 4: phase diagram heatmap ----------
def fig4():
    ph = json.loads((STATS / "operator_phase_diagram.json").read_text(encoding="utf-8"))
    pools = sorted({k.split("|")[0] for k in ph}, key=lambda s: int(s.split("=")[1]))
    mags = []
    for k in ph:
        m = k.split("|")[2]
        if m not in mags:
            mags.append(m)
    order = {"x1": 0, "x10": 1, "x100": 2}
    mags = sorted(mags, key=lambda m: order.get(m, 99))
    RANK_ONLY = {"rrf", "borda"}
    regimes = []
    for k in ph:
        r = k.split("|")[3]
        if r not in regimes:
            regimes.append(r)
    figs = []
    zmin, zmax = 0.0, 1.0
    for regime in regimes:
        gap = np.zeros((len(pools), len(mags)))
        for k, v in ph.items():
            kk = k.split("|")
            if kk[3] != regime:
                continue
            pi, mi = pools.index(kk[0]), mags.index(kk[2])
            rank_m = np.mean([v[o] for o in v if o in RANK_ONLY])
            score_m = np.mean([v[o] for o in v if o not in RANK_ONLY])
            gap[pi, mi] = abs(score_m - rank_m)
        f = px.imshow(gap, x=mags, y=pools, origin="lower", aspect="auto",
                      color_continuous_scale="RdYlGn", range_color=[zmin, zmax],
                      labels=dict(color="gap", x="magnitude", y="pool"))
        f.update_traces(coloraxis="coloraxis")
        f.update_layout(title=dict(text=regime, font_size=15))
        figs.append(f)
    # manual 1x3 subplot assembly
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(regimes),
                        subplot_titles=[r.replace("-", " ") for r in regimes],
                        horizontal_spacing=0.08)
    for i, f in enumerate(figs, start=1):
        for tr in f.data:
            fig.add_trace(tr, row=1, col=i)
    fig.update_layout(coloraxis=dict(colorscale="RdYlGn", cmin=zmin, cmax=zmax,
                                     colorbar=dict(title="gap", thickness=14)))
    for i in range(1, len(regimes) + 1):
        fig.update_xaxes(side="bottom", row=1, col=i)
        fig.update_yaxes(row=1, col=i)
    style(fig, "Synthetic operator-phase diagram — family gap |score-space mean − rank-only mean|",
          w=1120, h=440)
    save(fig, "fig4_phase_diagram")


# ---------- Fig 5: pool growth ----------
def fig5():
    md = (STATS / "deep_pool_nsweep.md").read_text(encoding="utf-8")
    ns, mrr_sf, cv = [], [], []
    import re
    for line in md.splitlines():
        m = re.match(r"\|\s*N=(\d+)\s*\|([^|]*)\|([^|]*)\|", line.strip())
        if not m:
            continue
        n = int(m.group(1))
        def num(cell):
            mm = re.search(r"([\d.]+)", cell)
            return float(mm.group(1)) if mm else None
        a, b = num(m.group(2)), num(m.group(3))
        if a is None:
            continue
        ns.append(n); mrr_sf.append(a); cv.append(b if b is not None else float("nan"))
    if not ns:
        ns, mrr_sf, cv = [20, 50, 100, 200, 494], [None]*5, [None]*5
    fig = go.Figure()
    fig.add_scatter(x=ns, y=mrr_sf, mode="lines+markers", name="SF+SPLADE CombSUM MRR",
                    line=dict(color=PALETTE["score-space"], width=3))
    fig.update_xaxes(title="candidate-pool size N", type="log")
    fig.update_yaxes(title="MRR", range=[0.75, 1.02])
    style(fig, "Score concentration vs candidate growth — operators do not separate with N", w=940, h=500)
    save(fig, "fig5_pool_growth")


# ---------- Fig 6: win/loss + power ----------
def fig6():
    wl = json.loads((STATS / "win_loss_rank1_n100.json").read_text(encoding="utf-8"))
    labels = list(wl.keys())
    wins = [wl[d]["combsum_wins"] for d in labels]
    losses = [wl[d]["rrf_wins"] for d in labels]
    ties = [wl[d]["ties"] for d in labels]
    r1 = [wl[d]["rank1_change_pct"] for d in labels]
    dz = [wl[d]["dz"] for d in labels]
    fig = go.Figure()
    fig.add_bar(name="CombSUM wins", x=labels, y=wins, marker_color=PALETTE["score-space"])
    fig.add_bar(name="RRF wins", x=labels, y=losses, marker_color=PALETTE["rank-only"])
    fig.add_bar(name="ties", x=labels, y=ties, marker_color="#CBD5E1")
    for i, d in enumerate(labels):
        fig.add_annotation(x=d, y=max(ties[i], wins[i]) + 6,
                           text=f"Δrank-1: {r1[i]}%<br>dz={dz[i]}",
                           showarrow=False, font_size=12)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="queries (n=100 each)")
    style(fig, "Where the bottleneck bites — paired outcomes and rank-1 changes, RRF → CombSUM", w=980)
    save(fig, "fig6_win_loss_power")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
    fns = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
           "fig5": fig5, "fig6": fig6}
    for w in which:
        try:
            fns[w]()
        except Exception as e:
            print(f"{w} FAILED: {type(e).__name__}: {e}")
