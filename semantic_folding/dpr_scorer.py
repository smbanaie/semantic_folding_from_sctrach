#!/usr/bin/env python3
"""
dpr_scorer.py — DPR (Dense Passage Retrieval) integration for hybrid ranking.

Provides a DPRScorer with the same interface as BM25Scorer / SPLADEScorer:
    score_all(query_text) -> List[(doc_idx, score)]

This is the second-model pair required by the journal expansion (reviewer #4:
"isn't the multi-hop result just SPLADE-specific?"). DPR has a *different*
score geometry than SPLADE (bi-encoder dot product of CLS embeddings vs.
sparse log1p-pooled expansions), so SF+DPR vs SF+SPLADE lets us ask whether
the magnitude phenomenon follows the task, the score geometry, or the model pair.

Optimization: corpus vectors are cached to disk after first encoding, so
subsequent subprocess calls load pre-computed vectors instead of re-encoding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton cache for loaded DPR models. Keyed by (ctx_model, qry_model).
_MODEL_CACHE: dict = {}

# Default DPR checkpoints (single-passage NQ-base; widely used, CPU-friendly).
DEFAULT_CTX_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
DEFAULT_QRY_MODEL = "facebook/dpr-question_encoder-single-nq-base"


class DPRScorer:
    """DPR-based scorer for hybrid SF+DPR / BM25+DPR ranking.

    Interface matches BM25Scorer / SPLADEScorer: score_all(query) -> [(idx, score)].
    """

    def __init__(self, corpus_texts: List[str],
                 ctx_model: str = DEFAULT_CTX_MODEL,
                 qry_model: str = DEFAULT_QRY_MODEL,
                 cache_dir: str = None):
        self.corpus_texts = corpus_texts
        self.doc_count = len(corpus_texts)
        self.ctx_model = ctx_model
        self.qry_model = qry_model
        self._ctx_encoder = None
        self._qry_encoder = None
        self._ctx_tokenizer = None
        self._qry_tokenizer = None
        self._corpus_vectors = None
        self._cache_dir = cache_dir

        key = (ctx_model, qry_model)
        if key in _MODEL_CACHE:
            self._ctx_encoder, self._ctx_tokenizer, self._qry_encoder, self._qry_tokenizer, self._torch = \
                _MODEL_CACHE[key]

        if cache_dir and self._load_cached_vectors():
            logger.info(f"  [DPR] Loaded cached corpus vectors from {cache_dir}")
        else:
            if self._ctx_encoder is None:
                self._load_model()
            self._encode_corpus()
            if cache_dir:
                self._save_cached_vectors()

    # ── caching ──────────────────────────────────────────────────────────
    def _get_cache_path(self) -> Path:
        return Path(self._cache_dir) / "dpr_corpus_vectors.npy"

    def _load_cached_vectors(self) -> bool:
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                self._corpus_vectors = np.load(str(cache_path))
                if self._corpus_vectors.shape[0] == self.doc_count:
                    return True
                logger.warning(f"  [DPR] Cache shape mismatch: {self._corpus_vectors.shape[0]} vs {self.doc_count}")
            except Exception as e:
                logger.warning(f"  [DPR] Failed to load cache: {e}")
        return False

    def _save_cached_vectors(self):
        if self._cache_dir:
            cache_path = self._get_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), self._corpus_vectors)
            logger.info(f"  [DPR] Saved corpus vectors to {cache_path}")

    # ── model loading ──────────────────────────────────────────────────────
    def _load_model(self):
        global _MODEL_CACHE
        key = (self.ctx_model, self.qry_model)
        if key in _MODEL_CACHE:
            self._ctx_encoder, self._ctx_tokenizer, self._qry_encoder, self._qry_tokenizer, self._torch = \
                _MODEL_CACHE[key]
            return
        try:
            from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
            import torch
            logger.info(f"  [DPR] Loading encoders: {self.ctx_model} / {self.qry_model}")
            ctx_tok = DPRContextEncoderTokenizer.from_pretrained(self.ctx_model, token=False)
            ctx_enc = DPRContextEncoder.from_pretrained(self.ctx_model, token=False)
            qry_tok = DPRQuestionEncoderTokenizer.from_pretrained(self.qry_model, token=False)
            qry_enc = DPRQuestionEncoder.from_pretrained(self.qry_model, token=False)
            ctx_enc.eval(); qry_enc.eval()
            _MODEL_CACHE[key] = (ctx_enc, ctx_tok, qry_enc, qry_tok, torch)
            self._ctx_encoder, self._ctx_tokenizer = ctx_enc, ctx_tok
            self._qry_encoder, self._qry_tokenizer = qry_enc, qry_tok
            self._torch = torch
            logger.info("  [DPR] Encoders loaded and cached")
        except ImportError:
            raise ImportError("DPR requires 'transformers' and 'torch'.")
        except Exception as e:
            raise RuntimeError(f"Failed to load DPR model: {e}")

    # ── encoding ──────────────────────────────────────────────────────────
    def _encode_texts(self, tokenizer, encoder, texts: List[str], is_query: bool) -> np.ndarray:
        torch = self._torch
        all_vecs = []
        batch_size = 16
        input_kw = "question_encoder_input" if is_query else "input_ids"
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            # DPR tokenizers return specific input-key names
            with torch.no_grad():
                out = encoder(**enc)
            # last_hidden_state pooled via CLS ([CLS] is index 0 for DPR)
            pooled = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
            all_vecs.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(all_vecs)

    def _encode_corpus(self):
        logger.info(f"  [DPR] Encoding {self.doc_count} corpus documents...")
        self._corpus_vectors = self._encode_texts(
            self._ctx_tokenizer, self._ctx_encoder, self.corpus_texts, is_query=False
        )
        # L2-normalize corpus vectors for stable dot-product similarity
        norms = np.linalg.norm(self._corpus_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._corpus_vectors = self._corpus_vectors / norms
        logger.info(f"  [DPR] Corpus encoded: shape={self._corpus_vectors.shape}")

    def score_all(self, query_text: str) -> List[Tuple[int, float]]:
        q_vec = self._encode_texts(self._qry_tokenizer, self._qry_encoder, [query_text], is_query=True)[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        scores = self._corpus_vectors.dot(q_vec)
        return [(i, float(scores[i])) for i in range(self.doc_count)]


def get_dpr_scorer(corpus_texts: List[str],
                   ctx_model: str = DEFAULT_CTX_MODEL,
                   qry_model: str = DEFAULT_QRY_MODEL,
                   cache_dir: str = None) -> DPRScorer:
    """Get a DPRScorer, using disk-cached corpus vectors when available."""
    return DPRScorer(corpus_texts, ctx_model=ctx_model, qry_model=qry_model, cache_dir=cache_dir)
