#!/usr/bin/env python3
"""
splade_scorer.py — SPLADE integration for Semantic Folding hybrid ranking

Provides a SPLADEScorer class with the same interface as BM25Scorer,
enabling hybrid SF+SPLADE scoring via the existing --hybrid infrastructure.

Optimization: corpus vectors are saved to disk after first encoding,
so subsequent subprocess calls load pre-computed vectors instead of
re-encoding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton cache for loaded SPLADE models.
# Keyed by model_name, stores (model, tokenizer, torch_module).
# Multiple SPLADEScorer instances with the same model share one copy.
_MODEL_CACHE: dict = {}


class SPLADEScorer:
    """SPLADE-based scorer for hybrid SF+SPLADE ranking.

    Interface matches BM25Scorer: score_all(query_text) -> List[(doc_idx, score)]
    """

    def __init__(self, corpus_texts: List[str], model_name: str = "naver/splade-cocondenser-ensembledistil",
                 cache_dir: str = None):
        self.corpus_texts = corpus_texts
        self.doc_count = len(corpus_texts)
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._corpus_vectors = None
        self._cache_dir = cache_dir

        if cache_dir and self._load_cached_vectors():
            logger.info(f"  [SPLADE] Loaded cached corpus vectors from {cache_dir}")
            self._load_model()
        else:
            self._load_model()
            self._encode_corpus()
            if cache_dir:
                self._save_cached_vectors()

    def _get_cache_path(self) -> Path:
        return Path(self._cache_dir) / "splade_corpus_vectors.npy"

    def _load_cached_vectors(self) -> bool:
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                self._corpus_vectors = np.load(str(cache_path))
                if self._corpus_vectors.shape[0] == self.doc_count:
                    return True
                logger.warning(f"  [SPLADE] Cache shape mismatch: {self._corpus_vectors.shape[0]} vs {self.doc_count}")
            except Exception as e:
                logger.warning(f"  [SPLADE] Failed to load cache: {e}")
        return False

    def _save_cached_vectors(self):
        if self._cache_dir:
            cache_path = self._get_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), self._corpus_vectors)
            logger.info(f"  [SPLADE] Saved corpus vectors to {cache_path}")

    def _load_model(self):
        global _MODEL_CACHE
        if self.model_name in _MODEL_CACHE:
            self._model, self._tokenizer, self._torch = _MODEL_CACHE[self.model_name]
            logger.info(f"  [SPLADE] Reusing cached model: {self.model_name}")
            return
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch
            logger.info(f"  [SPLADE] Loading model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=False)
            model = AutoModelForMaskedLM.from_pretrained(self.model_name, token=False)
            model.eval()
            _MODEL_CACHE[self.model_name] = (model, tokenizer, torch)
            self._model = model
            self._tokenizer = tokenizer
            self._torch = torch
            logger.info(f"  [SPLADE] Model loaded and cached")
        except ImportError:
            raise ImportError("SPLADE requires 'transformers' and 'torch'.")
        except Exception as e:
            raise RuntimeError(f"Failed to load SPLADE model: {e}")

    def _encode(self, texts: List[str]) -> np.ndarray:
        import torch, gc
        all_vectors = []
        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._tokenizer(batch, padding=True, truncation=True,
                                       max_length=512, return_tensors="pt")
            with torch.no_grad():
                output = self._model(**encoded)
            logits = output.logits
            relu_logits = torch.nn.functional.relu(logits)
            log_relu = torch.log1p(relu_logits)
            pooled = log_relu.max(dim=1).values
            all_vectors.append(pooled.cpu().numpy())
            del output, logits, relu_logits, log_relu, pooled, encoded
            if i % 32 == 0:
                gc.collect()
        return np.vstack(all_vectors)

    def _encode_corpus(self):
        logger.info(f"  [SPLADE] Encoding {self.doc_count} corpus documents...")
        self._corpus_vectors = self._encode(self.corpus_texts)
        logger.info(f"  [SPLADE] Corpus encoded: shape={self._corpus_vectors.shape}")

    def score_all(self, query_text: str) -> List[Tuple[int, float]]:
        query_vec = self._encode([query_text])[0]
        scores = self._corpus_vectors.dot(query_vec)
        return [(i, float(scores[i])) for i in range(self.doc_count)]


def get_splade_scorer(corpus_texts: List[str], model_name: str = "naver/splade-cocondenser-ensembledistil",
                      cache_dir: str = None) -> SPLADEScorer:
    """Get a SPLADEScorer, using disk-cached corpus vectors when available."""
    return SPLADEScorer(corpus_texts, model_name=model_name, cache_dir=cache_dir)
