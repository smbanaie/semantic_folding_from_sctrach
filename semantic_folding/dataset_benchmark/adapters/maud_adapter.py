"""
MAUD Adapter — Merger Agreement Understanding Dataset

Source: data/maud/raw/extracted/ (CSV + contract txt files)
Paper:  Haber et al., 2023 (Atticus Project)

MAUD CSV format:
  Columns: data_type, contract_name, text, answer, label, question,
           subquestion, text_type, id, category

  - text: the relevant contract passage (gold)
  - question: the query about the passage
  - answer: ground truth answer
  - contract_name: which contract (e.g., contract_13)

The retrieval task: given a question about a merger clause, find the relevant
contract passage. We use the `text` field as the gold passage and add
distractors from other contracts' passages.

Output entry (MuSiQue-like):
  {
    "id": "maud_{id}",
    "question": question,
    "answer": answer,
    "paragraphs": [
      { "idx": 0, "title": "relevant_passage", "paragraph_text": text, "is_supporting": True },
      { "idx": 1..19, "title": "distractor", "paragraph_text": ..., "is_supporting": False }
    ]
  }
"""

import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Any

from .base_adapter import BaseDatasetAdapter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class MaudAdapter(BaseDatasetAdapter):
    dataset_name = "maud"
    display_name = "MAUD"
    default_subset = "dev"

    SPLIT_FILES = {
        "train": "MAUD_train.csv",
        "dev": "MAUD_dev.csv",
        "test": "MAUD_test.csv",
    }

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted = output_dir / "extracted"
        if extracted.exists():
            csv_files = list(extracted.glob("MAUD_*.csv"))
            if csv_files:
                print(f"  MAUD data found at {extracted}")
                return extracted

        try:
            from datasets import load_dataset
            print("  Downloading atticusproject/maud from HuggingFace...")
            ds = load_dataset("atticusproject/maud", split="train")
            print(f"  Downloaded {len(ds)} rows")
            print(f"  Columns: {ds.column_names}")
            sample = ds[0]
            for k in ds.column_names:
                v = str(sample[k])[:200]
                print(f"    {k}: {v}")

            extracted.mkdir(parents=True, exist_ok=True)
            csv_path = extracted / "MAUD_train.csv"
            import pandas as pd
            df = ds.to_pandas()
            df.to_csv(csv_path, index=False)
            print(f"  Saved {len(df)} rows -> {csv_path}")
            return extracted
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download MAUD: {e}\n"
                f"Download from https://huggingface.co/datasets/atticusproject/maud "
                f"and place CSVs in {output_dir}/raw/extracted/"
            )

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "maud.jsonl"

        import pandas as pd

        extracted = raw_path / "extracted" if (raw_path / "extracted").exists() else raw_path
        csv_file = extracted / self.SPLIT_FILES.get(self.default_subset, "MAUD_dev.csv")
        if not csv_file.exists():
            csv_files = list(extracted.glob("MAUD_*.csv"))
            if csv_files:
                csv_file = csv_files[0]
            else:
                raise FileNotFoundError(
                    f"MAUD CSV not found. Run download first. Looked in {extracted}"
                )

        print(f"  Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")

        random.seed(42)

        all_passages = df["text"].dropna().unique().tolist()
        all_passages = [str(p).strip() for p in all_passages if str(p).strip()]

        entries = []
        n_written = 0
        n_skipped = 0

        for _, row in df.iterrows():
            if n_written >= max_queries:
                break

            question = str(row.get("question", "")).strip()
            text = str(row.get("text", "")).strip()
            answer = str(row.get("answer", "")).strip()
            q_id = row.get("id", n_written)

            if not question or not text:
                n_skipped += 1
                continue

            distractor_pool = [p for p in all_passages if p != text]
            n_distractors = min(19, len(distractor_pool))
            distractors = random.sample(distractor_pool, n_distractors) if distractor_pool else []

            paragraphs = [{
                "idx": 0,
                "title": "relevant_passage",
                "paragraph_text": text,
                "is_supporting": True,
            }]
            for i, d in enumerate(distractors):
                paragraphs.append({
                    "idx": i + 1,
                    "title": f"distractor_{i:04d}",
                    "paragraph_text": d,
                    "is_supporting": False,
                })

            entries.append({
                "id": f"maud_{q_id}",
                "question": question,
                "answer": answer,
                "contract_name": str(row.get("contract_name", "")),
                "category": str(row.get("category", "")),
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        stats = {
            "num_queries": n_written,
            "num_skipped": n_skipped,
            "total_rows": len(df),
            "csv_file": csv_file.name,
        }
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  MAUD: wrote {n_written} queries -> {out_path} (skipped {n_skipped})")
        return out_path
