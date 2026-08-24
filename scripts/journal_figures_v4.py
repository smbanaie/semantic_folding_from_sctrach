"""Signature figures for Journal_V4 (review items 14-15).
fig7_conceptual_phase_map: 2D conceptual operator-selection map with datasets
placed by measured geometry diagnostics. Labeled conceptual, not validated.
fig8_causal_centerpiece: 3-panel Original / Rank-preserving / Rank-destroying
with RRF vs CombSUM vs Borda MRR (MuSiQue real traces).
Deterministic; reads committed artifacts."""
import json, re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PROJ = Path(r"E:/PHD/GraphRag-Implementations/YaALI/knowledge-graph-builder")
STATS = PROJ / "docs/papers/Journal A/appendix_stats"
FIG = PROJ / "docs/papers/Journal A/figures"
FONT = dict(family="Inter, Segoe UI, Arial", size=13, color="#111827")

def save(fig, name, w=1000, h=560):
    fig.update_layout(template="plotly_white", font=FONT, width=w, height=h,
                      margin=dict(l=64, r=28, t=56, b=52))
    fig.write_image(str(FIG / f"{name}.png"), scale=3)
    fig.write_image(str(FIG / f"{name}.svg"))
    print("wrote", name)

# ---------- fig7 conceptual phase map ----------
magrel = json.loads((STATS / "magnitude_relevance.json").read_text(encoding="utf-8"))
tau = json.loads((STATS / "tau_analysis.json").read_text(encoding="utf-8"))
wl = json.loads((STATS / "win_loss_rank1_n100.json").read_text(encoding="utf-8"))

pts = []
for ds in ("hotpotqa", "musique", "scifact"):
    p_pos = magrel["margin_stats"][ds]["SPLADE"]["p_delta_pos"]
    if ds in wl:
        dis = wl[ds]["rank1_change_pct"] / 100.0
    else:
        dis = 0.10
    name = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "scifact": "SciFact"}[ds]
    pts.append((name, p_pos, dis))

fig = go.Figure()
quads = [("#F1F5F9", 0, .5, 0, .5), ("#FEF9C3", .5, 1, 0, .5),
         ("#DBEAFE", 0, .5, .5, 1), ("#DCFCE7", .5, 1, .5, 1)]
for col, x0, x1, y0, y1 in quads:
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                  fillcolor=col, opacity=0.55, line_width=0)
labels = {(0.25, 0.22): "operators similar",
          (0.75, 0.22): "score-space potentially useful",
          (0.25, 0.80): "rank-based fusion sufficient",
          (0.75, 0.85): "<b>score-space fusion most useful</b>"}
for xy, lab in labels.items():
    fig.add_annotation(x=xy[0], y=xy[1], text=lab, showarrow=False,
                       font=dict(size=12, color="#475569"))
xs = [min(p[1], 0.93) for p in pts]
ys = [min(max(p[2], 0.06), 0.92) for p in pts]
names = [p[0] for p in pts]
fig.add_scatter(x=xs, y=ys, mode="markers+text", text=names,
                textposition="middle left", marker=dict(size=16, color="#2563EB"),
                name="datasets (measured)")
fig.update_xaxes(title="Magnitude informativeness — SPLADE P(gold-margin > 0)", range=[0, 1])
fig.update_yaxes(title="Rank disagreement — operator rank-1 change rate", range=[0, 1])
fig.update_layout(title="Conceptual operator-selection map derived from the empirical findings "
                        "(not a validated predictor)")
save(fig, "fig7_conceptual_phase_map", w=900, h=640)

# ---------- fig8 causal centerpiece ----------
md = (STATS / "magnitude_perturbation_musique.md").read_text(encoding="utf-8")
def row(cond):
    for line in md.splitlines():
        if line.startswith(f"| {cond} |"):
            cells = [c.strip() for c in line.split("|")[2:-1]]
            return [float(re.match(r"MRR=([\d.]+)", c).group(1)) for c in cells]

panels = [("Original", ["orig"]),
          ("Rank-preserving magnitude changes", ["x2","log1p","pow05","compress","amplify","magswap"]),
          ("Rank destroyed", ["shufflescores"])]
group_names, rrf_v, cs_v, borda_v = [], [], [], []
for pname, conds in panels:
    group_names.append(pname)
    rrf_v.append(float(np.mean([row(c)[1] for c in conds])))
    cs_v.append(float(np.mean([row(c)[2] for c in conds])))
    borda_v.append(float(np.mean([row(c)[4] for c in conds])))

fig = go.Figure()
for vals, nm, col in ((cs_v, "CombSUM", "#2563EB"),
                      (borda_v, "Borda", "#F59E0B"),
                      (rrf_v, "RRF", "#DC2626")):
    fig.add_bar(x=[f"<b>{g}</b>" for g in group_names], y=vals, name=nm,
                marker_color=col, text=[f"{v:.2f}" for v in vals],
                textposition="outside")
fig.add_annotation(x=0.0, y=1.10, xref="paper", text="baseline",
                   showarrow=False, font_size=12, font_color="#64748B")
fig.add_annotation(x=0.5, y=1.10, xref="paper",
                   text="<b>RRF unchanged (τ = 1.000)</b> — CombSUM responds",
                   showarrow=False, font_size=13)
fig.add_annotation(x=1.0, y=1.10, xref="paper", text="<b>RRF collapses</b>",
                   showarrow=False, font_size=13, font_color="#DC2626")
fig.update_layout(barmode="group",
                  title="Rank-preserving magnitude interventions separate fusion operators "
                        "(MuSiQue real traces)",
                  legend=dict(orientation="h", y=-0.14))
fig.update_yaxes(title="MRR", range=[0, 1.22])
save(fig, "fig8_causal_centerpiece", w=1060, h=620)
