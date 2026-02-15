#!/usr/bin/env python3
"""
Modernized Phrase Fingerprint Generator for Semantic Folding Pipeline

Generates fingerprint matrices for each phrase based on semantic space coordinates,
creating 16×16 (or configurable) binary matrices representing phrase distributions.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import loguru
from loguru import logger
from tqdm import tqdm

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("numpy not available. Install with: pip install numpy")
    NUMPY_AVAILABLE = False

# Try to import scipy for sparse matrix support
try:
    import scipy.sparse
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available. Install with: pip install scipy")
    SCIPY_AVAILABLE = False


def load_context_coordinates(coordinates_path: Path) -> Dict[str, Tuple[int, int]]:
    """Load context coordinates from CSV file"""
    logger.info(f"Loading context coordinates from: {coordinates_path}")

    coordinates = {}
    with open(coordinates_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) >= 2:
                context_id, coords_str = row[0], row[1]
                try:
                    x, y = map(int, coords_str.split(','))
                    coordinates[context_id] = (x, y)
                except ValueError as e:
                    logger.warning(f"Invalid coordinates for context {context_id}: {coords_str}")

    logger.success(f"Loaded coordinates for {len(coordinates)} contexts")
    return coordinates


def load_phrases(phrases_path: Path) -> List[str]:
    """Load phrases from file"""
    logger.info(f"Loading phrases from: {phrases_path}")

    phrases = []
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                phrase = line.split(':', 1)[0].strip()
                if phrase:
                    phrases.append(phrase)

    logger.success(f"Loaded {len(phrases)} phrases")
    return phrases


def load_sparse_matrix(matrix_path: Path) -> Optional[scipy.sparse.csr_matrix]:
    """Load sparse term-context matrix"""
    if not SCIPY_AVAILABLE:
        logger.error("scipy not available for sparse matrix loading")
        return None

    logger.info(f"Loading sparse matrix from: {matrix_path}")

    try:
        # Load the NPZ file
        npz_data = np.load(matrix_path)
        matrix = scipy.sparse.csr_matrix(
            (npz_data['data'], npz_data['indices'], npz_data['indptr']),
            shape=npz_data['shape']
        )
        logger.success(f"Loaded sparse matrix: {matrix.shape}")
        return matrix
    except Exception as e:
        logger.error(f"Failed to load sparse matrix: {e}")
        return None


def load_dense_matrix(matrix_path: Path) -> Optional[Dict[str, List[int]]]:
    """Load dense term-context matrix (fallback)"""
    logger.info(f"Loading dense matrix from: {matrix_path}")

    try:
        matrix = {}
        with open(matrix_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            for row in reader:
                if row:
                    context_id = row[0]
                    values = [int(x) for x in row[1:]]
                    matrix[context_id] = values

        logger.success(f"Loaded dense matrix for {len(matrix)} contexts")
        return matrix
    except Exception as e:
        logger.error(f"Failed to load dense matrix: {e}")
        return None


def create_phrase_fingerprint(phrase: str,
                            phrase_idx: int,
                            matrix,
                            coordinates: Dict[str, Tuple[int, int]],
                            grid_size: int) -> np.ndarray:
    """Create fingerprint matrix for a single phrase"""
    fingerprint = np.zeros((grid_size, grid_size), dtype=np.int32)

    if SCIPY_AVAILABLE and hasattr(matrix, 'shape'):
        # Sparse matrix case
        num_contexts, num_phrases = matrix.shape
        if phrase_idx >= num_phrases:
            logger.warning(f"Phrase index {phrase_idx} out of range for matrix with {num_phrases} phrases")
            return fingerprint

        # Get contexts where this phrase appears
        phrase_vector = matrix[:, phrase_idx].toarray().flatten()
        context_indices = np.where(phrase_vector > 0)[0]

        for context_idx in context_indices:
            context_id = f"context_{context_idx}"
            if context_id in coordinates:
                x, y = coordinates[context_id]
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    fingerprint[y, x] += 1  # Note: matrix indexing

    else:
        # Dense matrix case (fallback)
        for context_id, context_vector in matrix.items():
            if context_id in coordinates and phrase_idx < len(context_vector):
                if context_vector[phrase_idx] > 0:
                    x, y = coordinates[context_id]
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        fingerprint[y, x] += 1

    return fingerprint


def save_fingerprint(fingerprint: np.ndarray,
                    phrase: str,
                    output_dir: Path) -> None:
    """Save fingerprint matrix to file"""
    # Create safe filename
    safe_name = "".join(c for c in phrase if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')[:50]  # Limit length

    filename = f"{safe_name}_fingerprint.txt"
    filepath = output_dir / filename

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in fingerprint:
                f.write('\t'.join(map(str, row)) + '\n')
    except Exception as e:
        logger.error(f"Failed to save fingerprint for phrase '{phrase}': {e}")


def create_fingerprint_visualization(fingerprint: np.ndarray,
                                   phrase: str,
                                   output_dir: Path) -> None:
    """Create visualization of fingerprint (optional)"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(fingerprint,
                   annot=False,
                   cmap='Blues',
                   cbar=True,
                   square=True)

        plt.title(f'Phrase Fingerprint: {phrase[:30]}...')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')

        viz_path = output_dir / "visualizations" / f"{phrase[:30].replace(' ', '_')}_fingerprint.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        pass  # Skip visualization if matplotlib not available
    except Exception as e:
        logger.warning(f"Failed to create visualization for phrase '{phrase}': {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate phrase fingerprints")
    parser.add_argument("--matrix_path", required=True, help="Path to term_context_matrix.npz or .csv")
    parser.add_argument("--coordinates_path", required=True, help="Path to context_coordinates.csv")
    parser.add_argument("--phrases_path", required=True, help="Path to phrases.txt")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--grid_size", type=int, default=16, help="Grid size (default: 16)")
    parser.add_argument("--max_phrases", type=int, help="Limit number of phrases to process (for testing)")
    parser.add_argument("--visualize", action="store_true", help="Create fingerprint visualizations")

    args = parser.parse_args()

    logger.info("Starting phrase fingerprint generation...")
    logger.info(f"Matrix: {args.matrix_path}")
    logger.info(f"Coordinates: {args.coordinates_path}")
    logger.info(f"Phrases: {args.phrases_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Grid size: {args.grid_size}x{args.grid_size}")

    # Load inputs
    coordinates = load_context_coordinates(Path(args.coordinates_path))
    phrases = load_phrases(Path(args.phrases_path))

    # Load term-context matrix
    matrix_path = Path(args.matrix_path)
    if matrix_path.suffix == '.npz':
        matrix = load_sparse_matrix(matrix_path)
    elif matrix_path.suffix == '.csv':
        matrix = load_dense_matrix(matrix_path)
    else:
        raise ValueError("Matrix file must be .npz (sparse) or .csv (dense)")

    if matrix is None:
        raise RuntimeError("Failed to load term-context matrix")

    # Limit phrases for testing
    if args.max_phrases:
        phrases = phrases[:args.max_phrases]
        logger.info(f"Limited processing to {len(phrases)} phrases for testing")

    # Create output directory
    output_dir = Path(args.output_dir)
    fingerprints_dir = output_dir / "fingerprints"
    fingerprints_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating fingerprints for {len(phrases)} phrases")
    logger.info(f"Grid size: {args.grid_size}×{args.grid_size}")

    # Process each phrase
    successful_fingerprints = 0
    with tqdm(total=len(phrases), desc="Generating fingerprints") as pbar:
        for phrase_idx, phrase in enumerate(phrases):
            try:
                fingerprint = create_phrase_fingerprint(
                    phrase, phrase_idx, matrix, coordinates, args.grid_size
                )

                # Save fingerprint
                save_fingerprint(fingerprint, phrase, fingerprints_dir)

                # Optional visualization
                if args.visualize:
                    create_fingerprint_visualization(fingerprint, phrase, output_dir)

                successful_fingerprints += 1

                # Log progress periodically
                if (phrase_idx + 1) % 1000 == 0:
                    occupied_cells = np.count_nonzero(fingerprint)
                    logger.info(f"Processed {phrase_idx + 1}/{len(phrases)} phrases. "
                              f"Last fingerprint has {occupied_cells} occupied cells.")

            except Exception as e:
                logger.error(f"Failed to process phrase '{phrase}': {e}")

            pbar.update(1)

    # Final statistics
    logger.info("Phrase fingerprint generation completed:")
    logger.info(f"  Total phrases: {len(phrases)}")
    logger.info(f"  Successful fingerprints: {successful_fingerprints}")
    logger.info(f"  Output directory: {fingerprints_dir}")
    logger.info(f"  Grid size: {args.grid_size}×{args.grid_size}")

    # Sample statistics
    if successful_fingerprints > 0:
        sample_files = list(fingerprints_dir.glob("*.txt"))[:5]
        if sample_files:
            total_size = sum(f.stat().st_size for f in sample_files)
            avg_size = total_size / len(sample_files)
            logger.info(f"  Average fingerprint file size: {avg_size:.0f} bytes")


if __name__ == "__main__":
    main()