"""
COVID-QA Adapter (castorini / deepset COVID-QA)

Source: https://github.com/castorini/COVID-QA  (deepset-ai/COVID-QA)
Paper : Möller, Reina, Jayakumar, Pietsch (2020), "COVID-QA: A Question
        Answering Dataset for COVID-19", Proc. 1st Workshop on NLP for
        COVID-19 at ACL 2020.  https://aclanthology.org/2020.nlpcovid19-acl.18/

Dataset characteristics:
  - SQuAD 2.0-format JSON (data[].paragraphs[].{context, qas[]})
  - Each paragraph = one CORD-19 research-paper abstract (147 abstracts total)
  - Each QA: {question, id, answers:[{text, answer_start}], is_impossible}
  - COVID-QA.json = 2,019 question/answer pairs (the primary release)
  - All QAs answerable (is_impossible=False in this release)

Suitability for the fusion benchmark:
  - Same SQuAD structure as the existing QA datasets: a query + a gold context
    (the abstract the answer was extracted from) + distractor contexts.
  - Topology: single-context EXTRACTIVE / SCIENTIFIC FACTOID QA (not multi-hop).
    Adds a biomedical-scientific extractive topology to the 9-dataset matrix,
    useful for testing whether the operator findings generalize beyond the
    narrative/multi-hop/reading-comprehension topologies already covered.
  - Candidate pool = gold abstract + N-1 distractor abstracts (default 9, →10-doc
    pool, matching HotpotQA). Gold marked is_supporting=true.

Conversion to MuSiQue-like JSONL:
  {id, question, answer, paragraphs:[{idx,title,paragraph_text,is_supporting}]}
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base_adapter import BaseDatasetAdapter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Default source location (user-provided copy of the COVID-QA repo).
DEFAULT_SOURCE = Path(r"E:/Counseling/Done/COVID-QA-master/data/question-answering")

PRIMARY_FILE = "COVID-QA.json"        # 2,019 QA pairs (main release)
SUPPLEMENTARY_FILES = ["200421_covidQA.json", "200423_covidQA.json"]


def _norm(s: str) -> str:
    """Collapse internal newlines/tabs to spaces and strip (abstracts have
    embedded line breaks that break the line-based sidecar format)."""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.split())


class COVIDQAAdapter(BaseDatasetAdapter):
    dataset_name = "covidqa"
    display_name = "COVID-QA"
    default_subset = "default"

    def __init__(self, source_dir: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.source_dir = Path(source_dir) if source_dir else DEFAULT_SOURCE

    # ------------------------------------------------------------------
    def download(self, output_dir: Path) -> Path:
        """Copy the SQuAD JSON files from the local COVID-QA checkout."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        needed = [PRIMARY_FILE] + SUPPLEMENTARY_FILES
        missing = [f for f in needed if not (self.source_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"COVID-QA source files missing in {self.source_dir}: {missing}\n"
                f"Expected the castorini/COVID-QA data/question-answering/ files."
            )
        for f in needed:
            dst = output_dir / f
            if not dst.exists():
                shutil.copy(self.source_dir / f, dst)
                print(f"  copied {f} -> {dst}")
        print(f"  COVID-QA raw data ready at {output_dir}")
        return output_dir

    # ------------------------------------------------------------------
    def _load_all_qas(self, raw_path: Path, primary_only: bool = False):
        """Yield (context, qa) tuples from the SQuAD JSON files."""
        files = [PRIMARY_FILE] if primary_only else (needed_files(raw_path))
        qas = []
        for fname in files:
            fp = raw_path / fname
            if not fp.exists():
                continue
            d = json.load(open(fp, encoding="utf-8"))
            for art in d.get("data", []):
                for para in art.get("paragraphs", []):
                    ctx = _norm(para.get("context", ""))
                    if not ctx:
                        continue
                    title = _norm(art.get("title", "")) or _norm(ctx[:60])
                    for qa in para.get("qas", []):
                        qas.append((title, ctx, qa))
        return qas

    # ------------------------------------------------------------------
    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500,
        n_distractors: int = 9, seed: int = 42,
    ) -> Path:
        """
        Build a 10-doc candidate pool per query: gold abstract + n_distractors
        other abstracts. Pool size is dataset-specific and measured (not fixed).
        """
        import random
        rng = random.Random(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "covidqa.jsonl"

        qas = self._load_all_qas(raw_path, primary_only=True)
        # Build the global pool of unique contexts (for distractors).
        all_contexts = []
        seen_ctx = set()
        for title, ctx, _ in qas:
            title = _norm(title)
            ctx = _norm(ctx)
            key = (title, ctx)
            if key not in seen_ctx:
                seen_ctx.add(key)
                all_contexts.append((title, ctx))
        # de-dup qas by (question, context) to avoid repeats
        seen_q = set()
        entries = []
        n_written = 0
        n_skipped = 0

        for title, ctx, qa in qas:
            if n_written >= max_queries:
                break
            question = _norm(qa.get("question", "")).strip()
            answers = qa.get("answers", [])
            answer_texts = [_norm(a.get("text", "")).strip() for a in answers if _norm(a.get("text", "")).strip()]
            if not question or not answer_texts:
                n_skipped += 1
                continue
            qkey = (question, ctx)
            if qkey in seen_q:
                n_skipped += 1
                continue
            seen_q.add(qkey)

            # Gold paragraph = this QA's own context.
            paragraphs = [{
                "idx": 0,
                "title": title,
                "paragraph_text": ctx,
                "is_supporting": True,
            }]
            # Distractors: other contexts (exclude gold).
            others = [(t, c) for (t, c) in all_contexts if c != ctx]
            rng.shuffle(others)
            for i, (dt, dc) in enumerate(others[:n_distractors], start=1):
                paragraphs.append({
                    "idx": i,
                    "title": dt,
                    "paragraph_text": dc,
                    "is_supporting": False,
                })
            entries.append({
                "id": f"covidqa_{n_written:04d}",
                "question": question,
                "answer": answer_texts,
                "paragraphs": paragraphs,
            })
            n_written += 1

        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        stats = {"num_queries": n_written, "num_skipped": n_skipped,
                 "pool_size": n_distractors + 1, "total_contexts": len(all_contexts)}
        with open(out_path.with_suffix(".stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  COVID-QA: wrote {n_written} queries (pool={n_distractors+1}) -> {out_path} "
              f"(skipped {n_skipped})")
        return out_path

    # ------------------------------------------------------------------
    def convert_to_full_corpus_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500,
    ) -> Path:
        """Write every unique abstract to <name>_full_corpus.txt for Regime B.

        Newlines inside abstracts are collapsed to spaces so the line-based
        `doc_<id>, title text` sidecar format (one document per line) is valid.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        full_path = output_dir / "covidqa_full_corpus.txt"

        qas = self._load_all_qas(raw_path, primary_only=True)
        seen = set()
        lines = []
        next_id = 0
        for title, ctx, _ in qas:
            title = _norm(title)
            ctx = _norm(ctx)
            key = (title, ctx)
            if key in seen:
                continue
            seen.add(key)
            gid = f"doc_{next_id:06d}"
            lines.append(f"{gid}, {title} {ctx}")
            next_id += 1
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  COVID-QA full corpus: {len(lines)} documents -> {full_path}")
        return full_path

    # ------------------------------------------------------------------
    def get_recommended_params(self) -> Dict[str, Any]:
        return {
            "grid_size": 64, "spreading_steps": 1, "top_percent": 0.10,
            "weighting": "idf", "smoothing_sigma": 1.5, "morton": True,
            "min_word_length": 3, "min_freq": 1, "keep_verbs": True, "top_k": 10,
            "tsne_perplexity": 50, "tsne_iter": 1000, "method": "tsne",
            "rrf_k": 60,
        }


def needed_files(raw_path: Path) -> List[str]:
    files = [PRIMARY_FILE] + SUPPLEMENTARY_FILES
    return [f for f in files if (raw_path / f).exists()]
