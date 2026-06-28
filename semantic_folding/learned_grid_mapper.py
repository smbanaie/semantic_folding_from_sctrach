#!/usr/bin/env python3
"""
learned_grid_mapper.py — Supervised 2D grid mapping using contrastive learning.

Trains a small MLP to map term-context vectors to 2D grid coordinates,
using contrastive loss to place co-occurring terms closer on the grid.

Usage:
    python -m semantic_folding.learned_grid_mapper \\
        --matrix term_context_matrix.npz \\
        --metadata term_context_matrix.json \\
        --corpus corpus.txt \\
        --grid-size 64 \\
        --output coordinates.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available. Install with: uv add torch")


class TermContextDataset(Dataset):
    """Dataset of (term_vector, positive_neighbor, negative_neighbor) triples."""

    def __init__(self, vectors: np.ndarray, cooccurrence_pairs: List[Tuple[int, int]],
                 all_indices: np.ndarray, num_negatives: int = 5):
        """
        Args:
            vectors: (N, D) term-context vectors
            cooccurrence_pairs: List of (i, j) indices that co-occur in gold documents
            all_indices: All valid indices for negative sampling
            num_negatives: Number of negative samples per positive pair
        """
        self.vectors = vectors
        self.pairs = cooccurrence_pairs
        self.all_indices = all_indices
        self.num_negatives = num_negatives
        self.pair_array = np.array(cooccurrence_pairs)

    def __len__(self):
        return len(self.pairs) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        pair_idx = idx // (1 + self.num_negatives)
        is_negative = (idx % (1 + self.num_negatives)) > 0

        i, j = self.pairs[pair_idx]
        anchor = self.vectors[i]

        if is_negative:
            # Random negative sample
            neg_idx = np.random.choice(self.all_indices)
            while neg_idx == i or neg_idx == j:
                neg_idx = np.random.choice(self.all_indices)
            target = self.vectors[neg_idx]
            label = 0.0  # negative pair
        else:
            target = self.vectors[j]
            label = 1.0  # positive pair

        return (
            torch.FloatTensor(anchor),
            torch.FloatTensor(target),
            torch.FloatTensor([label])
        )


class GridMapperMLP(nn.Module):
    """Small MLP to map high-dim vectors to 2D grid coordinates."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, x):
        coords = self.net(x)
        # Scale to grid coordinates [0, grid_size-1]
        return coords * (self.grid_size - 1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning similarity."""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_dist, positive_dist, label):
        # label: 1 for positive pairs, 0 for negative pairs
        loss = (1 - label) * anchor_dist + label * torch.clamp(self.margin - anchor_dist, min=0)
        return loss.mean()


def extract_cooccurrence_pairs(corpus_path: str, min_cooccurrence: int = 2) -> List[Tuple[int, int]]:
    """
    Extract term co-occurrence pairs from corpus.
    
    Two terms co-occur if they appear in the same document/paragraph.
    """
    from collections import Counter
    import re
    
    # Simple tokenization
    def tokenize(text):
        return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    
    doc_tokens = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split on comma (CSV format: id,text)
                comma_idx = line.find(',')
                if comma_idx > 0:
                    text = line[comma_idx+1:]
                else:
                    text = line
                tokens = tokenize(text)
                if tokens:
                    doc_tokens.append(tokens)
    
    # Count co-occurrences
    cooccurrence = Counter()
    for tokens in doc_tokens:
        token_list = sorted(tokens)
        for i in range(len(token_list)):
            for j in range(i+1, min(i+10, len(token_list))):  # Limit to nearby tokens
                cooccurrence[(token_list[i], token_list[j])] += 1
    
    # Filter by minimum co-occurrence
    pairs = [(a, b) for (a, b), count in cooccurrence.items() if count >= min_cooccurrence]
    logger.info(f"Extracted {len(pairs)} co-occurrence pairs from {len(doc_tokens)} documents")
    return pairs


def train_grid_mapper(
    vectors: np.ndarray,
    term_names: List[str],
    cooccurrence_pairs: List[Tuple[int, int]],
    grid_size: int = 64,
    hidden_dim: int = 128,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tuple[GridMapperMLP, Dict]:
    """
    Train the grid mapper using contrastive loss.
    
    Args:
        vectors: (N, D) term-context vectors
        term_names: List of term names
        cooccurrence_pairs: List of (i, j) index pairs that co-occur
        grid_size: Output grid size
        hidden_dim: Hidden layer dimension
        num_epochs: Training epochs
        learning_rate: Adam learning rate
        batch_size: Batch size
        device: 'cpu' or 'cuda'
    
    Returns:
        Tuple of (trained_model, training_info)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required. Install with: uv add torch")
    
    input_dim = vectors.shape[1]
    model = GridMapperMLP(input_dim, hidden_dim, grid_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(margin=1.0)
    
    # Filter pairs to valid indices
    valid_pairs = [(i, j) for i, j in cooccurrence_pairs if i < len(vectors) and j < len(vectors)]
    logger.info(f"Training with {len(valid_pairs)} valid co-occurrence pairs")
    
    if len(valid_pairs) == 0:
        logger.warning("No valid co-occurrence pairs found. Using random pairs.")
        # Generate random pairs as fallback
        n = len(vectors)
        valid_pairs = [(np.random.randint(n), np.random.randint(n)) for _ in range(min(1000, n))]
    
    # Create dataset and dataloader
    all_indices = np.arange(len(vectors))
    dataset = TermContextDataset(vectors, valid_pairs, all_indices, num_negatives=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for anchor, target, label in dataloader:
            anchor = anchor.to(device)
            target = target.to(device)
            label = label.to(device)
            
            # Forward pass
            anchor_coords = model(anchor)
            target_coords = model(target)
            
            # Compute distance
            anchor_dist = torch.sqrt(((anchor_coords - target_coords) ** 2).sum(dim=1) + 1e-8)
            
            # Compute loss
            loss = criterion(anchor_dist, None, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    training_info = {
        "epochs": num_epochs,
        "final_loss": losses[-1] if losses else 0,
        "num_pairs": len(valid_pairs),
        "grid_size": grid_size,
    }
    
    return model, training_info


def apply_grid_mapping(
    model: GridMapperMLP,
    vectors: np.ndarray,
    term_names: List[str],
    grid_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """
    Apply trained grid mapper to term-context vectors.
    
    Args:
        model: Trained GridMapperMLP
        vectors: (N, D) term-context vectors
        term_names: List of term names
        grid_size: Grid size
        device: 'cpu' or 'cuda'
    
    Returns:
        (N, 2) integer grid coordinates
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(vectors).to(device)
        coords = model(x).cpu().numpy()
    
    # Round to integer grid coordinates
    grid_coords = np.clip(np.round(coords).astype(int), 0, grid_size - 1)
    
    return grid_coords


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Learned grid mapping for Semantic Folding")
    parser.add_argument("--matrix", required=True, help="Term-context matrix path (.npz)")
    parser.add_argument("--metadata", required=True, help="Metadata JSON path")
    parser.add_argument("--corpus", required=True, help="Corpus file path")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", required=True, help="Output coordinates JSON path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Load matrix and metadata
    from scipy.sparse import load_npz
    matrix = load_npz(args.matrix).toarray()
    
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    term_names = metadata.get("phrases", list(range(matrix.shape[0])))
    
    logger.info(f"Loaded matrix: {matrix.shape}")
    logger.info(f"Grid size: {args.grid_size}")
    logger.info(f"Device: {args.device}")
    
    # Extract co-occurrence pairs
    cooccurrence_pairs = extract_cooccurrence_pairs(args.corpus)
    
    # Train grid mapper
    model, info = train_grid_mapper(
        matrix, term_names, cooccurrence_pairs,
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )
    
    # Apply grid mapping
    coords = apply_grid_mapping(model, matrix, term_names, args.grid_size, args.device)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "method": "learned",
        "grid_size": args.grid_size,
        "num_terms": len(term_names),
        "coordinates": coords.tolist(),
        "training_info": info,
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.success(f"Saved coordinates to {output_path}")
    logger.info(f"Training info: {info}")


if __name__ == "__main__":
    main()
