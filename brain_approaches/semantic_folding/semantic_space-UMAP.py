#!/usr/bin/env python3
"""
Modernized Semantic Space Construction for Semantic Folding Pipeline
Creates semantic space by UMAP layout of term-context relationships,
generating a configurable grid (16×16 default) with comprehensive visualizations.
"""
import argparse
import csv
import json, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from lib import load_contexts_dict
import loguru
from loguru import logger
from tqdm import tqdm
import umap
# Try to import required dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    logger.warning("networkx not available. Install with: pip install networkx")
    NETWORKX_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("numpy not available. Install with: pip install numpy")
    NUMPY_AVAILABLE = False
try:
    import scipy.sparse
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available. Install with: pip install scipy")
    SCIPY_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
    MATPLOTLIB_AVAILABLE = False
try:
    import plotly.graph_objs as go
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

def load_sparse_matrix(matrix_path: Path) -> Tuple[Optional[scipy.sparse.csr_matrix], Dict[str, Any]]:
    """Load sparse term-context matrix from NPZ format"""
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy not available for sparse matrix loading")
    logger.info(f"Loading sparse matrix from: {matrix_path}")
    # Load the NPZ file
    npz_data = np.load(matrix_path)
    matrix = scipy.sparse.csr_matrix(
        (npz_data['data'], npz_data['indices'], npz_data['indptr']),
        shape=npz_data['shape']
    )
    # Load metadata
    metadata_path = matrix_path.with_suffix('.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    logger.success(f"Loaded matrix: {matrix.shape}, {matrix.nnz} non-zero entries")
    return matrix, metadata

def load_dense_matrix(matrix_path: Path) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Load dense term-context matrix from CSV format (fallback)"""
    logger.info(f"Loading dense matrix from: {matrix_path}")
    matrix = {}
    phrases = []
    with open(matrix_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        phrases = header[1:]  # Skip "Context ID" column
        for row in reader:
            context_id = row[0]
            values = [int(x) for x in row[1:]]
            matrix[context_id] = values
    metadata = {
        'num_contexts': len(matrix),
        'num_phrases': len(phrases),
        'phrases': phrases
    }
    logger.success(f"Loaded dense matrix: {len(matrix)} contexts × {len(phrases)} phrases")
    return matrix, metadata
def build_semantic_graph(matrix: scipy.sparse.csr_matrix,
                        metadata: Dict[str, Any],
                        contexts :  Dict[str, str],
                        edge_threshold: float = 0.1,
                        max_edges: int = 50000) -> nx.Graph:
    """Build semantic graph from term-context matrix"""
    if not NETWORKX_AVAILABLE:
        raise RuntimeError("networkx not available for graph construction")
    logger.info("Building semantic graph from term-context matrix")
    num_contexts = metadata['num_contexts']
    # For sparse matrix, we need to compute context-context similarities
    # This is expensive for large matrices, so we'll sample or use approximation
    if num_contexts > 1000:
        logger.warning(f"Large matrix ({num_contexts} contexts). Using sampled graph construction.")
        # Sample a subset of contexts for graph building
        sample_size = min(1000, num_contexts)
        indices = np.random.choice(num_contexts, sample_size, replace=False)
        submatrix = matrix[indices]
        context_ids = [f"context_{i}" for i in indices]
    else:
        submatrix = matrix
        context_ids = [f"context_{i}" for i in range(num_contexts)]
    logger.info(f"Computing similarities for {len(context_ids)} contexts")
    G = nx.Graph()
    # Add nodes
    for i, context_id in enumerate(context_ids):
        G.add_node(context_id)
    # Compute pairwise similarities (dot product of context vectors)
    logger.info("Computing context-context similarities...")
    # For efficiency, only compute upper triangle and threshold
    edge_count = 0
    for i in tqdm(range(len(context_ids)), desc="Building graph"):
        for j in range(i + 1, len(context_ids)):
            vector_i = matrix[i].toarray().flatten()
            vector_j = matrix[j].toarray().flatten()
            dot_product = np.dot(vector_i, vector_j)
            magnitude_i = np.linalg.norm(vector_i)
            magnitude_j = np.linalg.norm(vector_j)
            similarity = dot_product / (magnitude_i * magnitude_j)
            if similarity > edge_threshold:
                logger.info(f"Similarity of Context_{i} and Context_{j} : {similarity}")
                # time.sleep(0.5)
                G.add_edge(context_ids[i], context_ids[j], weight=similarity)
                edge_count += 1
                # Limit edges to prevent memory issues
                if edge_count >= max_edges:
                    logger.warning(f"Reached maximum edges limit ({max_edges}). Stopping graph construction.")
                    break
        if edge_count >= max_edges:
            break
    logger.success(f"Created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def compute_umap_layout(document_vectors, grid_size: int) -> Dict[str, Tuple[float, float]]:
    """Compute UMAP layout for semantic positioning"""
    logger.info(f"Computing UMAP layout with grid size {grid_size}")
    umap_model = umap.UMAP(n_components=2, random_state=42)
    document_vectors_umap = umap_model.fit_transform(document_vectors)
    positions = {f"context_{i}": (document_vectors_umap[i, 0], document_vectors_umap[i, 1]) for i in range(len(document_vectors))}
    logger.success("UMAP layout computed")
    return positions

def map_to_grid(positions: Dict[str, Tuple[float, float]],
                grid_size: int) -> Dict[str, Tuple[int, int]]:
    """Map continuous positions to discrete grid coordinates"""
    logger.info(f"Mapping positions to {grid_size}×{grid_size} grid")
    grid_coords = {}
    for node_id, (x, y) in positions.items():
        # Normalize positions from UMAP layout (-1 to 1 range) to grid coordinates
        # Add 1 to shift from (-1,1) to (0,2), then scale to grid
        grid_x = int(((x + 1) / 2) * grid_size)
        grid_y = int(((y + 1) / 2) * grid_size)
        # Clamp to grid boundaries
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        grid_coords[node_id] = (grid_x, grid_y)
    logger.success(f"Mapped {len(grid_coords)} nodes to grid coordinates")
    return grid_coords

def save_coordinates(coordinates: Dict[str, Tuple[int, int]], output_path: Path) -> None:
    """Save grid coordinates to CSV file"""
    logger.info(f"Saving coordinates to: {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Context ID', 'Grid Coordinates'])
        for context_id, (x, y) in coordinates.items():
            writer.writerow([context_id, f"{x},{y}"])
    logger.success(f"Saved coordinates for {len(coordinates)} contexts")

def create_visualizations(G: nx.Graph,
                         positions: Dict[str, Tuple[float, float]],
                         grid_coords: Dict[str, Tuple[int, int]],
                         context_texts: Dict[str, str],
                         grid_size: int,
                         output_dir: Path) -> None:
    """Create comprehensive visualizations of the semantic space"""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping matplotlib visualizations (not available)")
        return
    logger.info("Creating semantic space visualizations")
    # Set style
    sns.set_style("whitegrid")
    # 1. Network graph visualization
    if len(G.nodes) <= 500:  # Only visualize small graphs
        logger.info("Creating network graph visualization")
        plt.figure(figsize=(12, 10))
        # Draw the graph
        nx.draw(G, positions,
                node_color='skyblue',
                node_size=100,
                edge_color='gray',
                alpha=0.7,
                with_labels=False)
        plt.title(f'Semantic Space Network Graph\n{len(G.nodes)} nodes, {len(G.edges)} edges')
        plt.axis('equal')
        network_path = output_dir / "semantic_network.png"
        plt.savefig(network_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.success(f"Created network visualization: {network_path}")
    # 2. Grid heatmap visualization
    logger.info("Creating grid heatmap visualization")
    grid_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for coords in grid_coords.values():
        x, y = coords
        grid_matrix[y, x] += 1  # Note: matrix indexing
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid_matrix,
                annot=True if grid_size <= 20 else False,
                fmt='d',
                cmap='YlGnBu',
                cbar=True)
    plt.title(f'Semantic Space Grid Distribution\n{grid_size}×{grid_size} grid')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    grid_path = output_dir / "semantic_grid_heatmap.png"
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Created grid heatmap: {grid_path}")
    # 3. Interactive Plotly visualization (if available)
    if PLOTLY_AVAILABLE and len(grid_coords) <= 1000:
        logger.info("Creating interactive visualization")
        # Prepare data for Plotly
        x_coords = [coord[0] for coord in grid_coords.values()]
        y_coords = [coord[1] for coord in grid_coords.values()]
        context_ids = list(grid_coords.keys())
        # Create hover text with context previews
        hover_texts = []
        for context_id in context_ids:
            text = context_texts.get(context_id, "")
            # Truncate long texts for hover
            preview = text[:200] + "..." if len(text) > 200 else text
            hover_texts.append(f"ID: {context_id}<br>Text: {preview}")
        trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=8,
                color='skyblue',
                opacity=0.7
            ),
            text=hover_texts,
            hoverinfo='text'
        )
        layout = go.Layout(
            title=f'Semantic Space Grid Visualization ({grid_size}×{grid_size})',
            xaxis=dict(title='Grid X', range=[-0.5, grid_size - 0.5]),
            yaxis=dict(title='Grid Y', range=[-0.5, grid_size - 0.5]),
            hovermode='closest'
        )
        fig = go.Figure(data=[trace], layout=layout)
        plotly_path = output_dir / "semantic_grid_interactive.html"
        fig.write_html(str(plotly_path))
        logger.success(f"Created interactive visualization: {plotly_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Build semantic space from term-context matrix")
    parser.add_argument("--matrix_path", required=True, help="Path to term_context_matrix.npz or .csv file")
    parser.add_argument("--corpus_path", required=True, help="Path to corpus.txt file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--grid_size", type=int, default=16, help="Grid size (default: 16)")
    parser.add_argument("--edge_threshold", type=float, default=0.1, help="Minimum edge weight threshold")
    parser.add_argument("--max_edges", type=int, default=50000, help="Maximum edges in graph")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()
    logger.info("Starting semantic space construction...")
    logger.info(f"Matrix: {args.matrix_path}")
    logger.info(f"Corpus: {args.corpus_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Grid size: {args.grid_size}x{args.grid_size}")
    # Load term-context matrix
    matrix_path = Path(args.matrix_path)
    if matrix_path.suffix == '.npz':
        matrix, metadata = load_sparse_matrix(matrix_path)
    elif matrix_path.suffix == '.csv':
        matrix, metadata = load_dense_matrix(matrix_path)
    else:
        raise ValueError("Matrix file must be .npz (sparse) or .csv (dense)")
    # Load context texts
    corpus_path = Path(args.corpus_path)
    context_texts = load_contexts_dict(corpus_path)
    # from pprint import pformat
    # logger.info(pformat(context_texts))
    # time.sleep(30)
    # Build semantic graph
    G = build_semantic_graph(matrix, metadata, context_texts, args.edge_threshold, args.max_edges)

    # Get document vectors from the graph
    document_vectors = []
    for node in G.nodes:
        vector = matrix[int(node.split("_")[1])].toarray().flatten()
        document_vectors.append(vector)

    # Compute UMAP layout
    positions = compute_umap_layout(document_vectors, args.grid_size)

    # Map to grid coordinates
    grid_coords = map_to_grid(positions, args.grid_size)

    # Save coordinates
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_path = output_dir / "context_coordinates.csv"
    save_coordinates(grid_coords, coords_path)
    # Create visualizations
    if not args.no_visualization:
        create_visualizations(G, positions, grid_coords, context_texts,
                            args.grid_size, output_dir)
    # Log final statistics
    logger.info("Semantic Space Construction Summary:")
    logger.info(f"  Grid size: {args.grid_size}×{args.grid_size}")
    logger.info(f"  Contexts mapped: {len(grid_coords)}")
    logger.info(f"  Graph nodes: {len(G.nodes)}")
    logger.info(f"  Graph edges: {len(G.edges)}")
    # Compute grid utilization
    total_cells = args.grid_size * args.grid_size
    occupied_cells = len(set(grid_coords.values()))
    utilization = occupied_cells / total_cells * 100
    logger.info(f"  Grid utilization: {occupied_cells}/{total_cells} cells ({utilization:.1f}%)")
    logger.success("Semantic space construction completed")

if __name__ == "__main__":
    main()
