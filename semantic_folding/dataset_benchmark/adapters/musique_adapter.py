"""
MuSiQue Adapter

Source: HippoRAG2 datasets (originally from MuSiQue, Trivedi et al., 2022).
Multi-hop (2-5 hops) questions over Wikipedia passages.

The converted MuSiQue-like JSONL (id, question, answer, paragraphs with
is_supporting) is expected to already exist under <data>/musique/converted/.
This adapter locates that file and validates/copies it so the generic
benchmark runner can consume it identically to the other datasets.
"""

import json
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any

from .base_adapter import BaseDatasetAdapter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class MuSiQueAdapter(BaseDatasetAdapter):
    dataset_name = "musique"
    display_name = "MuSiQue"
    default_subset = "2hop"

    def _find_converted(self, raw_path: Path) -> Path:
        raw_path = Path(raw_path)
        # Accept either the directly-provided converted file or a sibling
        # converted/ directory produced by prior dataset preparation.
        candidates = [
            raw_path if raw_path.suffix == ".jsonl" else None,
            raw_path / "musique.jsonl",
            raw_path.parent / "converted" / "musique.jsonl",
            raw_path / "converted" / "musique.jsonl",
        ]
        for c in candidates:
            if c and c.exists():
                return c
        raise FileNotFoundError(
            f"MuSiQue converted JSONL not found. Looked in: "
            f"{[str(c) for c in candidates if c]}. "
            f"Place musique.jsonl under data/musique/converted/."
        )

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # The converted file is the source of truth; download is a no-op check.
        try:
            self._find_converted(output_dir)
            print(f"  MuSiQue data already available (converted jsonl present).")
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e) + f"\nSearched under {output_dir}")
        return output_dir

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        src = self._find_converted(raw_path)

        out_path = output_dir / "musique.jsonl"
        n_written = 0
        with open(src, "r", encoding="utf-8") as fin, \
                open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if not self.validate_entry(e):
                    continue
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written >= max_queries:
                    break

        stats = {"num_queries": n_written, "source": str(src)}
        with open(out_path.with_suffix(".stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  MuSiQue: wrote {n_written} queries -> {out_path}")
        return out_path

    def get_recommended_params(self) -> Dict[str, Any]:
        # Multi-hop: preserve magnitude — favour score-space fusion defaults.
        return {
            "grid_size": 64,
            "spreading_steps": 1,
            "top_percent": 0.10,
            "weighting": "idf",
            "smoothing_sigma": 1.5,
            "morton": True,
            "min_word_length": 3,
            "min_freq": 1,
            "keep_verbs": True,
            "top_k": 5,
            "tsne_perplexity": 50,
            "tsne_iter": 1000,
        }
