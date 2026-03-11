import numpy as np
import scipy.sparse
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

def load_sparse_matrix(matrix_path: Path) -> Tuple[scipy.sparse.csr_matrix, Dict[str, Any]]:
    """Load sparse term-context matrix from NPZ format"""
    npz_data = np.load(matrix_path)
    matrix = scipy.sparse.csr_matrix(
        (npz_data['data'], npz_data['indices'], npz_data['indptr']),
        shape=npz_data['shape']
    )
    # Load metadata
    metadata_path = matrix_path.with_suffix('.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return matrix, metadata

def load_phrases(phrases_path: Path) -> List[str]:
    """Load phrases from file"""
    phrases = []
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                phrase = line.split(':', 1)[0].strip()
                if phrase:
                    phrases.append(phrase)
    return phrases

def get_context_similarity(matrix: scipy.sparse.csr_matrix, context1: int, context2: int) -> float:
    """Get similarity between two contexts"""
    vector1 = matrix[context1].toarray().flatten()
    vector2 = matrix[context2].toarray().flatten()
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity

def get_shared_phrases(matrix: scipy.sparse.csr_matrix, phrases: List[str], context1: int, context2: int) -> List[str]:
    """Get shared phrases between two contexts"""
    vector1 = matrix[context1].toarray().flatten()
    vector2 = matrix[context2].toarray().flatten()
    shared_phrases = []
    for i in range(len(phrases)):
        if vector1[i] > 0 and vector2[i] > 0:
            shared_phrases.append(phrases[i])
    return shared_phrases

def main():
    parser = argparse.ArgumentParser(description="Get similarity and shared phrases between two contexts")
    parser.add_argument("--matrix_path", required=True, help="Path to term_context_matrix.npz")
    parser.add_argument("--phrases_path", required=True, help="Path to phrases.txt")
    args = parser.parse_args()

    matrix_path = Path(args.matrix_path)
    phrases_path = Path(args.phrases_path)

    matrix, metadata = load_sparse_matrix(matrix_path)
    phrases = load_phrases(phrases_path)

    num_contexts = metadata['num_contexts']
    print(f"Number of contexts: {num_contexts}")

    while True:
        try:
            context1 = int(input("Enter the index of the first context: "))
            context2 = int(input("Enter the index of the second context: "))
            if context1 < 0 or context1 >= num_contexts or context2 < 0 or context2 >= num_contexts:
                print("Invalid context index. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    similarity = get_context_similarity(matrix, context1, context2)
    print(f"Similarity between context {context1} and context {context2}: {similarity:.4f}")

    shared_phrases = get_shared_phrases(matrix, phrases, context1, context2)
    print("Shared phrases:")
    for phrase in shared_phrases:
        print(phrase)

if __name__ == "__main__":
    main()