"""
Query Visualizer
PhD Thesis: Semantic Folding for Closed-Domain QA
Step 6: Query + Top-1 Doc Fingerprint Dashboard

Layout: 2 rows (query, top doc), each = fingerprint heatmap | metadata table.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from lib import get_logger, load_document_fingerprints

logger = get_logger("query_visualizer")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def morton_to_xy(index: int, grid_size: int) -> Tuple[int, int]:
    def _compact_bits(v: int) -> int:
        v &= 0x55555555
        v = (v ^ (v >> 1)) & 0x33333333
        v = (v ^ (v >> 2)) & 0x0F0F0F0F
        v = (v ^ (v >> 4)) & 0x00FF00FF
        v = (v ^ (v >> 8)) & 0x0000FFFF
        return v
    x = _compact_bits(index)
    y = _compact_bits(index >> 1)
    return x, y


def inverse_flatten(flat_vector: np.ndarray, grid_size: int, use_morton: bool = False) -> np.ndarray:
    if not use_morton:
        return flat_vector.reshape(grid_size, grid_size)
    grid_2d = np.zeros((grid_size, grid_size), dtype=flat_vector.dtype)
    for idx, val in enumerate(flat_vector):
        if val != 0:
            x, y = morton_to_xy(idx, grid_size)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid_2d[y, x] = val
    return grid_2d


def normalize_for_display(fp_2d: np.ndarray) -> np.ndarray:
    result = np.zeros_like(fp_2d)
    active_mask = fp_2d > 0
    if active_mask.any():
        active_vals = fp_2d[active_mask]
        min_val = active_vals.min()
        max_val = active_vals.max()
        result[active_mask] = (active_vals - min_val) / (max_val - min_val + 1e-10)
    return result


def _fingerprint_heatmap(fp_2d: np.ndarray, showscale: bool = True) -> go.Heatmap:
    fp_display = normalize_for_display(fp_2d)
    fp_masked = np.where(fp_2d > 0, fp_display, np.nan)
    return go.Heatmap(
        z=fp_masked,
        colorscale=[
            [0, 'white'],
            [0.001, 'lightblue'],
            [0.2, 'blue'],
            [0.5, 'purple'],
            [0.8, 'red'],
            [1, 'darkred'],
        ],
        zmin=0, zmax=1,
        showscale=showscale,
        hovertemplate='Cell: (%{x}, %{y})<br>Activation: %{z:.4f}<extra></extra>',
        xgap=1, ygap=1,
    )


# ---------------------------------------------------------------------------
# Main visualisation function
# ---------------------------------------------------------------------------


def create_query_visualization(
    query_text: str,
    query_fp_dense: np.ndarray,
    query_metadata: Dict,
    results: List[Tuple[str, float]],
    doc_fingerprints_dense: Dict[str, np.ndarray] = None,
    grid_size: int = 64,
    use_morton: bool = True,
    top_n: int = 3,
    output_html: Optional[Path] = None,
    generate_png: bool = False,
) -> go.Figure:
    """
    Multi-row dashboard: query fingerprint + matched document fingerprints.
    Row 1 = query; rows 2+ = top docs (rank 1, 2, 4 by default).
    Each row = heatmap | table.
    """
    logger.info(f"Creating query visualization: {query_text!r}")

    query_grid = inverse_flatten(query_fp_dense, grid_size, use_morton)

    # Collect doc rows — default: results at indices 0, 1, 3 (ranks 1, 2, 4)
    doc_indices = [0, 1, 3]
    doc_infos: List[Tuple[int, str, float, np.ndarray]] = []
    if results and doc_fingerprints_dense is not None:
        for i in doc_indices:
            if i < len(results):
                doc_id, score = results[i]
                fp = doc_fingerprints_dense.get(doc_id)
                if fp is not None:
                    doc_infos.append((i + 1, doc_id, score,
                                      inverse_flatten(fp, grid_size, use_morton)))

    n_doc_rows = len(doc_infos)
    n_rows = 1 + n_doc_rows

    titles = ['<b>Query Fingerprint</b>', '<b>Query Metadata</b>']
    for rank, doc_id, score, _ in doc_infos:
        titles.append(f'<b>#{rank}: Doc@{doc_id}</b>')
        titles.append(f'<b>#{rank}</b> Qry–Doc Overlap')
        titles.append(f'<b>#{rank}</b> Details')

    specs = [[{'type': 'heatmap'}, {'type': 'table'}, None]]
    for _ in doc_infos:
        specs.append([{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'table'}])

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=titles,
        specs=specs,
        column_widths=[0.38, 0.32, 0.30],
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )

    # ── Row 1: Query ──────────────────────────────────────────────────────────
    fig.add_trace(_fingerprint_heatmap(query_grid), row=1, col=1)
    _draw_grid_borders(fig, grid_size, xref='x', yref='y')
    fig.update_xaxes(row=1, col=1, constrain='domain', showticklabels=False)
    fig.update_yaxes(row=1, col=1, scaleanchor='x', scaleratio=1, showticklabels=False)

    meta_rows = _build_query_metadata_rows(query_text, query_fp_dense, query_metadata, results)
    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Property</b>', '<b>Value</b>'],
                fill_color='lightblue', align='left', font=dict(size=12),
            ),
            cells=dict(
                values=list(zip(*meta_rows)),
                fill_color='white', align='left', font=dict(size=11), height=24,
            ),
        ),
        row=1, col=2,
    )

    # ── Rows 2…(n_rows): Documents ──────────────────────────────────────────
    for offset, (rank, doc_id, score, dgrid) in enumerate(doc_infos):
        row = offset + 2  # rows 2, 3, 4, …

        # Col 1: Document fingerprint heatmap
        fig.add_trace(_fingerprint_heatmap(dgrid, showscale=(offset == len(doc_infos) - 1)),
                      row=row, col=1)

        fp_ax = 2 * row - 2
        x_axis_fp = f'x{fp_ax}'
        y_axis_fp = f'y{fp_ax}'
        _draw_grid_borders(fig, grid_size, xref=x_axis_fp, yref=y_axis_fp)
        fig.update_xaxes(row=row, col=1, constrain='domain', showticklabels=False)
        fig.update_yaxes(row=row, col=1, scaleanchor=x_axis_fp, scaleratio=1,
                         showticklabels=False)

        # Col 2: Query–doc overlap heatmap
        overlap = np.minimum(query_grid, dgrid)
        fig.add_trace(
            go.Heatmap(
                z=overlap,
                colorscale=[
                    [0, 'white'],
                    [0.001, 'lavender'],
                    [0.2, 'mediumpurple'],
                    [0.5, 'purple'],
                    [0.8, 'indigo'],
                    [1, 'darkviolet'],
                ],
                zmin=0, zmax=1,
                showscale=False,
                hovertemplate='Cell: (%{x}, %{y})<br>Overlap: %{z:.4f}<extra></extra>',
                xgap=1, ygap=1,
            ),
            row=row, col=2,
        )

        overlap_ax = 2 * row - 1
        x_axis_ov = f'x{overlap_ax}'
        y_axis_ov = f'y{overlap_ax}'
        _draw_grid_borders(fig, grid_size, xref=x_axis_ov, yref=y_axis_ov)
        fig.update_xaxes(row=row, col=2, constrain='domain', showticklabels=False)
        fig.update_yaxes(row=row, col=2, scaleanchor=x_axis_ov, scaleratio=1,
                         showticklabels=False)

        # Col 3: Metadata table with overlap metrics
        active = int(np.count_nonzero(dgrid))
        query_active = int(np.count_nonzero(query_grid))
        total = grid_size * grid_size
        sparsity = 1 - (active / total) if total > 0 else 1.0
        overlap_cells = int(np.count_nonzero(overlap))
        overlap_ratio = overlap_cells / active if active > 0 else 0.0
        query_coverage = overlap_cells / query_active if query_active > 0 else 0.0
        q_flat = query_grid.ravel()
        d_flat = dgrid.ravel()
        cos_sim = float(np.dot(q_flat, d_flat) / (np.linalg.norm(q_flat) * np.linalg.norm(d_flat) + 1e-10))

        doc_rows = [
            ('Rank', f'#{rank}'),
            ('Document ID', doc_id),
            ('Score', f'{score:.4f}'),
            ('Active Bits', str(active)),
            ('Sparsity', f'{sparsity:.2%}'),
            ('Vector Size', str(total)),
            ('Overlap Cells', str(overlap_cells)),
            ('Overlap / Doc', f'{overlap_ratio:.2%}'),
            ('Query Coverage', f'{query_coverage:.2%}'),
            ('Cosine Similarity', f'{cos_sim:.4f}'),
        ]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Property</b>', '<b>Value</b>'],
                    fill_color='lightgreen', align='left', font=dict(size=12),
                ),
                cells=dict(
                    values=list(zip(*doc_rows)),
                    fill_color='white', align='left', font=dict(size=11), height=24,
                ),
            ),
            row=row, col=3,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    row_h = 320 if n_rows > 2 else 350
    height = 100 + n_rows * row_h
    fig.update_layout(
        title=dict(
            text=f'<b>Query Analysis: "{query_text}"</b>',
            x=0.5, xanchor='center', font=dict(size=16),
        ),
        height=height,
        width=1500,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=60, r=60, t=100, b=60),
    )

    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html))
        logger.info(f"Saved HTML: {output_html}")
        if generate_png:
            try:
                fig.write_image(str(output_html.with_suffix('.png')), width=1500, height=height, scale=2)
            except Exception as e:
                logger.warning(f"PNG export failed (install kaleido): {e}")

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_grid_borders(
    fig: go.Figure, grid_size: int,
    xref: str = 'x', yref: str = 'y',
    block_size: int = 4,
    border_color: str = 'lightgray', border_width: float = 1.0,
    max_shapes: int = 5000,
) -> None:
    num_blocks = grid_size // block_size
    shape_count = 0
    for i in range(num_blocks + 1):
        if shape_count >= max_shapes:
            break
        x_pos = i * block_size - 0.5
        fig.add_shape(
            type='line', layer='above',
            x0=x_pos, y0=-0.5, x1=x_pos, y1=grid_size - 0.5,
            line=dict(color=border_color, width=border_width),
            xref=xref, yref=yref,
        )
        shape_count += 1
    for i in range(num_blocks + 1):
        if shape_count >= max_shapes:
            break
        y_pos = i * block_size - 0.5
        fig.add_shape(
            type='line', layer='above',
            x0=-0.5, y0=y_pos, x1=grid_size - 0.5, y1=y_pos,
            line=dict(color=border_color, width=border_width),
            xref=xref, yref=yref,
        )
        shape_count += 1


def _build_query_metadata_rows(
    query_text: str,
    query_fp_dense: np.ndarray,
    query_metadata: Dict,
    results: List[Tuple[str, float]],
) -> List[Tuple[str, str]]:
    active_bits = int(np.count_nonzero(query_fp_dense))
    total_bits = len(query_fp_dense)
    sparsity = 1 - (active_bits / total_bits) if total_bits > 0 else 1.0

    rows = [
        ('Query', query_text[:100] + ('…' if len(query_text) > 100 else '')),
        ('Active Bits', str(active_bits)),
        ('Sparsity', f'{sparsity:.2%}'),
        ('Vector Size', str(total_bits)),
        ('Total Phrases', str(query_metadata.get('num_phrases', 'N/A'))),
        ('Matched Phrases', str(query_metadata.get('num_matched', 'N/A'))),
        ('Missing Phrases', str(query_metadata.get('num_missing', 0))),
    ]

    missing = query_metadata.get('missing_phrases', [])
    if missing:
        rows.append(('OOV Terms', ', '.join(missing[:5])
                     + ('…' if len(missing) > 5 else '')))

    rows.append(('Weighting', str(query_metadata.get('weighting', 'N/A'))))
    rows.append(('Normalization', str(query_metadata.get('normalization', 'N/A'))))

    spreading = query_metadata.get('_spreading_info', {})
    if spreading:
        rows.append(('Spreading Steps', str(spreading.get('steps', 0))))
        rows.append(('Spread Decay', str(spreading.get('decay', 'N/A'))))

    if results:
        rows.append(('Top Result', f'{results[0][0]} ({results[0][1]:.4f})'))
        if len(results) > 1:
            rows.append(('Runner-up', f'{results[1][0]} ({results[1][1]:.4f})'))
        if len(results) > 2:
            rows.append(('3rd Result', f'{results[2][0]} ({results[2][1]:.4f})'))

    rows.append(('Total Results', str(len(results))))

    return rows


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualize a query fingerprint with top document matches.',
    )
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--query-fp', type=Path, required=True)
    parser.add_argument('--doc-fp-dir', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--query-meta', type=Path, default=None)
    parser.add_argument('--grid-size', type=int, default=64)
    parser.add_argument('--no-morton', dest='use_morton', action='store_false', default=True)
    parser.add_argument('--top-n', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('viz.html'))
    parser.add_argument('--png', action='store_true')

    args = parser.parse_args()

    query_fp_dense = np.load(str(args.query_fp))

    query_metadata: Dict = {}
    if args.query_meta and args.query_meta.exists():
        with open(args.query_meta, 'r', encoding='utf-8') as f:
            query_metadata = json.load(f)

    with open(args.results, 'r', encoding='utf-8') as f:
        results_raw = json.load(f)
    results: List[Tuple[str, float]] = [(r[0], r[1]) for r in results_raw]

    doc_fp_sparse, doc_meta = load_document_fingerprints(args.doc_fp_dir)
    doc_fingerprints_dense = {d: fp.toarray().ravel() for d, fp in doc_fp_sparse.items()}
    grid_size = doc_meta.get('grid_size', args.grid_size)
    use_morton = doc_meta.get('use_morton', args.use_morton)

    create_query_visualization(
        query_text=args.query,
        query_fp_dense=query_fp_dense,
        query_metadata=query_metadata,
        results=results,
        doc_fingerprints_dense=doc_fingerprints_dense,
        grid_size=grid_size,
        use_morton=use_morton,
        top_n=args.top_n,
        output_html=args.output,
        generate_png=args.png,
    )

    logger.success("Query visualization complete.")


if __name__ == '__main__':
    main()
