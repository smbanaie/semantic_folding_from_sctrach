#!/usr/bin/env python3
"""
cross_attention_scorer.py — Block-level cross-attention scoring for Semantic Folding.

Implements ColBERT-style late interaction over grid blocks. Divides the 64x64 grid
into 8x8 blocks, computes attention between query and document blocks, and produces
a weighted similarity score.

Usage:
    python -m semantic_folding.cross_attention_scorer \\
        --query-fp query_fingerprint.npy \\
        --doc-fp doc_fingerprint.npy \\
        --grid-size 64 \\
        --block-size 8
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


def block_cross_attention_score(
    query_fp: np.ndarray,
    doc_fp: np.ndarray,
    grid_size: int = 64,
    block_size: int = 8,
    temperature: float = 1.0,
) -> float:
    """
    Compute block-level cross-attention score between query and document fingerprints.
    
    Args:
        query_fp: Query fingerprint (1D array of length grid_size^2)
        doc_fp: Document fingerprint (1D array of length grid_size^2)
        grid_size: Grid side length
        block_size: Block side length for partitioning
        temperature: Temperature for softmax (lower = sharper attention)
    
    Returns:
        Cross-attention score (float)
    """
    # Reshape to 2D grid
    query_grid = query_fp.reshape(grid_size, grid_size)
    doc_grid = doc_fp.reshape(grid_size, grid_size)
    
    # Number of blocks per dimension
    n_blocks = grid_size // block_size
    
    # Extract blocks and flatten
    query_blocks = []
    doc_blocks = []
    
    for i in range(n_blocks):
        for j in range(n_blocks):
            q_block = query_grid[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            d_block = doc_grid[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            query_blocks.append(q_block.flatten())
            doc_blocks.append(d_block.flatten())
    
    query_blocks = np.array(query_blocks)  # (n_blocks^2, block_size^2)
    doc_blocks = np.array(doc_blocks)      # (n_blocks^2, block_size^2)
    
    # Compute pairwise attention scores
    # query_blocks: (N, D), doc_blocks: (N, D)
    # attention[i,j] = softmax(query_blocks[i] @ doc_blocks[j]^T / sqrt(D))
    D = block_size * block_size
    attention_scores = np.dot(query_blocks, doc_blocks.T) / np.sqrt(D)
    
    # Apply temperature and softmax
    attention_scores = attention_scores / temperature
    # Softmax over doc blocks for each query block
    exp_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Compute cosine similarity for each block pair
    q_norms = np.linalg.norm(query_blocks, axis=1, keepdims=True) + 1e-8
    d_norms = np.linalg.norm(doc_blocks, axis=1, keepdims=True) + 1e-8
    block_cosine = np.dot(query_blocks / q_norms, (doc_blocks / d_norms).T)
    
    # Weighted sum: attention_weights * block_cosine
    # For each query block, weight its contribution by attention to doc blocks
    weighted_scores = attention_weights * block_cosine
    
    # Average over query blocks
    score = np.mean(weighted_scores)
    
    return float(score)


def block_cross_attention_score_batch(
    query_fp: np.ndarray,
    doc_fps: np.ndarray,
    grid_size: int = 64,
    block_size: int = 8,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute cross-attention scores for multiple documents at once.
    
    Args:
        query_fp: Query fingerprint (1D array of length grid_size^2)
        doc_fps: Document fingerprints (N x grid_size^2)
        grid_size: Grid side length
        block_size: Block side length
        temperature: Temperature for softmax
    
    Returns:
        Array of scores (N,)
    """
    n_docs = doc_fps.shape[0]
    scores = np.zeros(n_docs)
    
    for i in range(n_docs):
        scores[i] = block_cross_attention_score(
            query_fp, doc_fps[i], grid_size, block_size, temperature
        )
    
    return scores


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Block cross-attention scoring")
    parser.add_argument("--query-fp", required=True, help="Query fingerprint path (.npy)")
    parser.add_argument("--doc-fp", required=True, help="Document fingerprint path (.npy)")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size")
    parser.add_argument("--block-size", type=int, default=8, help="Block size")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    
    args = parser.parse_args()
    
    query_fp = np.load(args.query_fp)
    doc_fp = np.load(args.doc_fp)
    
    score = block_cross_attention_score(
        query_fp, doc_fp, args.grid_size, args.block_size, args.temperature
    )
    
    print(f"Cross-attention score: {score:.4f}")
    print(f"Grid size: {args.grid_size}")
    print(f"Block size: {args.block_size}")
    print(f"Number of blocks: {(args.grid_size // args.block_size) ** 2}")
