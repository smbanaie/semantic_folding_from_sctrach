from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional


class EmbeddingCache:
    """Simple on-disk cache for paragraph embeddings.

    Keys are SHA1 of paragraph text. Embeddings stored as lists of floats in a JSON file.
    This is intentionally small and dependency-free.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: Dict[str, List[float]] = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        return self._data.get(self._key(text))

    def set(self, text: str, emb: List[float]) -> None:
        self._data[self._key(text)] = list(emb)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)

    def clear(self) -> None:
        self._data.clear()
        try:
            self.path.unlink()
        except Exception:
            pass
