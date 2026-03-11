#!/usr/bin/env python3
"""
Modernized Document Fingerprint Generator for Semantic Folding Pipeline

Aggregates phrase fingerprints to create document-level semantic fingerprints,
with efficient phrase matching and comprehensive metadata storage.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
from lib import process_lemmatize, load_contexts_dict, load_phrases, load_fingerprint_cache
from phrase_extractor import expand_phrases, extract_phrases_spacy, extract_phrases_fallback
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

# Try to import spaCy for phrase extraction (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_MODEL_AVAILABLE = True
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        SPACY_MODEL_AVAILABLE = False
        nlp = None
except ImportError:
    logger.warning("spaCy not available. Install with: pip install spacy")
    SPACY_AVAILABLE = False
    SPACY_MODEL_AVAILABLE = False
    nlp = None


def apply_top_percent_threshold(fingerprint: np.ndarray, top_percent: float = 0.05) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply threshold to keep only top N% of cells by value"""
    total_cells = fingerprint.size
    top_k = max(1, int(total_cells * top_percent))  # At least 1 cell

    # Flatten and find top values
    flat_fingerprint = fingerprint.flatten()

    # Get indices of top values
    if np.sum(flat_fingerprint) == 0:
        # All zeros, nothing to threshold
        return fingerprint, {
            'threshold_applied': False,
            'threshold_percent': top_percent,
            'cells_kept': total_cells,
            'cells_zeroed': 0
        }

    # Find threshold value (top_k-th largest value)
    sorted_values = np.sort(flat_fingerprint[flat_fingerprint > 0])
    if len(sorted_values) >= top_k:
        threshold_value = sorted_values[-top_k]  # Keep values >= this threshold
    else:
        threshold_value = 0  # Keep all non-zero values

    # Apply threshold
    thresholded_fingerprint = fingerprint.copy()
    cells_kept = np.sum(thresholded_fingerprint >= threshold_value)
    cells_zeroed = total_cells - cells_kept

    # Zero out cells below threshold
    thresholded_fingerprint[thresholded_fingerprint < threshold_value] = 0

    threshold_info = {
        'threshold_applied': True,
        'threshold_percent': top_percent,
        'threshold_value': int(threshold_value),
        'cells_kept': int(cells_kept),
        'cells_zeroed': int(cells_zeroed),
        'sparsity_after_threshold': cells_kept / total_cells
    }

    return thresholded_fingerprint, threshold_info


def generate_document_fingerprint(context_text: str,
                                fingerprint_cache: Dict[str, Optional[np.ndarray]],
                                phrases: List[str],
                                grid_size: int,
                                no_threshold : bool, 
                                top_percent: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate document fingerprint by aggregating phrase fingerprints"""
    doc_fingerprint = np.zeros((grid_size, grid_size), dtype=np.int32)

    # Extract phrases from document
    logger.info("\n"+"==="*40)
    doc_phrases = phrases
    logger.info(f"Context: {context_text}")
    logger.info("\n"+"---"*40)
    logger.info(f"Extracted Phrases: {doc_phrases}")
    # Match document phrases to our vocabulary
    matched_phrases = []
    unmatched_phrases = []

    metadata = {}

    for doc_phrase in doc_phrases:
        logger.info("\n"+"-+-"*10)
        logger.info(f"Processing Phrase : {doc_phrase}")
        
        # Try exact match first
        if doc_phrase in fingerprint_cache and fingerprint_cache[doc_phrase] is not None:
            logger.info(f"Phrase : {doc_phrase} Added to DOC Fingerprints")
            doc_fingerprint += fingerprint_cache[doc_phrase]
            matched_phrases.append(doc_phrase)
        else:
            unmatched_phrases.append(doc_phrase)
    logger.info("\n"+"-"*20)
    logger.info(f"Unmatched Phrases: {unmatched_phrases}")
    logger.info("\n"+"-"*20)
    logger.info(f"Matched Phrases: {matched_phrases}")
    # Apply threshold cutoff to retain only top cells
    if not no_threshold and np.sum(doc_fingerprint) > 0:
        doc_fingerprint, threshold_info = apply_top_percent_threshold(doc_fingerprint, top_percent)
        metadata.update(threshold_info)
    else:
        metadata.update({
            'threshold_applied': False,
            'threshold_percent': 0.0,
            'cells_kept': int(np.sum(doc_fingerprint > 0)),
            'cells_zeroed': 0
        })

    # Metadata
    metadata.update({
        'total_doc_phrases': len(doc_phrases),
        'matched_phrases': len(matched_phrases),
        'unmatched_phrases': len(unmatched_phrases),
        'matched_phrase_list': matched_phrases[:10],  # Limit for storage
        'coverage': len(matched_phrases) / max(1, len(doc_phrases)),
        'total_fingerprint_sum': int(np.sum(doc_fingerprint))
    })
    logger.info("\n"+"---"*40)
    logger.info(metadata)
    logger.info("\n"+"==="*40)
    return doc_fingerprint, metadata


def save_document_fingerprint(fingerprint: np.ndarray,
                            metadata: Dict[str, Any],
                            context_id: str,
                            context_text: str,
                            output_dir: Path) -> None:
    """Save document fingerprint and metadata"""
    # Create safe filename
    safe_id = "".join(c for c in context_id if c.isalnum() or c in ('-', '_')).rstrip()
    safe_id = safe_id[:50]  # Limit length

    # Save fingerprint matrix
    fingerprint_file = output_dir / f"{safe_id}_fingerprint.txt"
    try:
        with open(fingerprint_file, 'w', encoding='utf-8') as f:
            for row in fingerprint:
                f.write('\t'.join(map(str, row)) + '\n')
    except Exception as e:
        logger.error(f"Failed to save fingerprint for context {context_id}: {e}")
        return

    # Save metadata
    metadata_file = output_dir / f"{safe_id}_metadata.json"
    try:
        full_metadata = {
            **metadata,
            'context_id': context_id,
            'context_text_preview': context_text[:200] + ('...' if len(context_text) > 200 else ''),
            'fingerprint_shape': fingerprint.shape,
            'fingerprint_file': str(fingerprint_file.name),
            'metadata_file': str(metadata_file.name)
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Failed to save metadata for context {context_id}: {e}")


def create_document_visualization(fingerprint: np.ndarray,
                                context_id: str,
                                output_dir: Path) -> None:
    """Create visualization of document fingerprint (optional)"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(fingerprint,
                   annot=False,
                   cmap='Reds',
                   cbar=True,
                   square=True)

        plt.title(f'Document Fingerprint: {context_id[:30]}...')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')

        viz_path = output_dir / "visualizations" / f"{context_id[:30].replace('/', '_')}_doc_fingerprint.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        pass  # Skip visualization if matplotlib not available
    except Exception as e:
        logger.warning(f"Failed to create visualization for document '{context_id}': {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate document fingerprints")
    parser.add_argument("--corpus_path", required=True, help="Path to corpus.txt file")
    parser.add_argument("--phrases_path", required=True, help="Path to phrases.txt file")
    parser.add_argument("--fingerprints_dir", required=True, help="Directory containing phrase fingerprints")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--grid_size", type=int, default=16, help="Grid size (default: 16)")
    parser.add_argument("--max_docs", type=int, help="Limit number of documents to process (for testing)")
    parser.add_argument("--top_percent", type=float, default=0.05, help="Keep only top N% of cells (default: 0.05)")
    parser.add_argument("--no_threshold", action="store_true", help="Disable thresholding (keep all cells)")
    parser.add_argument("--visualize", action="store_true", help="Create document fingerprint visualizations")
    parser.add_argument("--use_spacy", action="store_true", help="Force use of spaCy for phrase extraction")

    args = parser.parse_args()

    logger.info("Starting document fingerprint generation...")
    logger.info(f"Corpus: {args.corpus_path}")
    logger.info(f"Phrases: {args.phrases_path}")
    logger.info(f"Fingerprints: {args.fingerprints_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Grid size: {args.grid_size}x{args.grid_size}")

    # Override spaCy availability if requested
    global SPACY_AVAILABLE
    if args.use_spacy and not SPACY_AVAILABLE:
        logger.warning("spaCy requested but not available")
    elif not args.use_spacy:
        SPACY_AVAILABLE = False
    # Load inputs
    all_phrases = load_phrases(Path(args.phrases_path))
    contexts = load_contexts_dict(Path(args.corpus_path))

    # Limit documents for testing
    if args.max_docs:
        context_items = list(contexts.items())[:args.max_docs]
        contexts = dict(context_items)
        logger.info(f"Limited processing to {len(contexts)} documents for testing")

    # Load fingerprint cache
    fingerprint_cache = load_fingerprint_cache(
        Path(args.fingerprints_dir), all_phrases, args.grid_size
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    doc_fingerprints_dir = output_dir / "doc_fingerprints"
    doc_fingerprints_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating document fingerprints for {len(contexts)} documents")
    logger.info(f"Available phrase fingerprints: {sum(1 for v in fingerprint_cache.values() if v is not None)}")

    # Process each document
    successful_docs = 0
    total_matched_phrases = 0
    total_doc_phrases = 0

    with tqdm(total=len(contexts), desc="Generating document fingerprints") as pbar:
        for context_id, context_text in contexts.items():
            try:
                if args.use_spacy and SPACY_AVAILABLE and SPACY_MODEL_AVAILABLE:
                    context_phrases = extract_phrases_spacy(context_text, 1000)                   
                else:
                    context_phrases = extract_phrases_fallback(context_text)
                    # Generate document fingerprint
                context_phrases = expand_phrases(context_phrases)
                final_phrases = []
                for p in context_phrases : 
                    normalized_phrase = process_lemmatize(p)
                    if len(normalized_phrase) > 1 : 
                        final_phrases.append(normalized_phrase)
                final_phrases = [ item for item in set(final_phrases)]
                doc_fingerprint, metadata = generate_document_fingerprint(
                    context_text, fingerprint_cache, final_phrases, args.grid_size, args.no_threshold, args.top_percent
                )

                # Save fingerprint and metadata
                save_document_fingerprint(
                    doc_fingerprint, metadata, context_id, context_text, doc_fingerprints_dir
                )

                # Optional visualization
                if args.visualize:
                    create_document_visualization(doc_fingerprint, context_id, output_dir)

                successful_docs += 1
                total_matched_phrases += metadata['matched_phrases']
                total_doc_phrases += metadata['total_doc_phrases']

                # Log progress periodically
                if successful_docs % 500 == 0:
                    avg_coverage = total_matched_phrases / max(1, total_doc_phrases)
                    logger.info(f"Processed {successful_docs}/{len(contexts)} documents. "
                              f"Average phrase coverage: {avg_coverage:.3f}")

            except Exception as e:
                logger.error(f"Failed to process document '{context_id}': {e}")

            pbar.update(1)

    # Final statistics
    if successful_docs > 0:
        avg_coverage = total_matched_phrases / max(1, total_doc_phrases)
        avg_fingerprint_sum = total_matched_phrases / successful_docs

        logger.info("Document fingerprint generation completed:")
        logger.info(f"  Total documents: {len(contexts)}")
        logger.info(f"  Successful fingerprints: {successful_docs}")
        logger.info(f"  Average phrase coverage: {avg_coverage:.3f}")
        logger.info(f"  Average matched phrases per doc: {avg_fingerprint_sum:.1f}")
        logger.info(f"  Output directory: {doc_fingerprints_dir}")

        # Sample file statistics
        sample_files = list(doc_fingerprints_dir.glob("*_fingerprint.txt"))[:5]
        if sample_files:
            total_size = sum(f.stat().st_size for f in sample_files)
            avg_size = total_size / len(sample_files)
            logger.info(f"  Average fingerprint file size: {avg_size:.0f} bytes")


if __name__ == "__main__":
    main()