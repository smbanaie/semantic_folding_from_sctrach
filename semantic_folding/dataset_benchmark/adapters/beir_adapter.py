"""
Generic BEIR Adapter

Handles all BEIR benchmark datasets (nfcorpus, scifact, quora, trec-covid, dbpedia-entity, etc.)

Source: BEIR benchmark (Thakur et al., 2021, NeurIPS)
Format: https://github.com/beir-cellar/beir

BEIR format:
  corpus.jsonl: {_id, title, text, metadata}
  queries.jsonl: {_id, text, metadata}
  qrels/: relevance judgments per split

Converts to MuSiQue-like JSONL:
  {
    "id": "<query_id>",
    "question": "<query text>",
    "answer": "<first relevant passage title>",
    "paragraphs": [
      { "idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true|false },
      ...
    ]
  }
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from .base_adapter import BaseDatasetAdapter

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("beir_adapter")


class BEIRAdapter(BaseDatasetAdapter):
    """Generic adapter for BEIR benchmark datasets."""

    def __init__(self, dataset_name: str, display_name: str, beir_subdir: str = None, **kwargs):
        self._dataset_name = dataset_name
        self._display_name = display_name
        self._beir_subdir = beir_subdir or dataset_name
        super().__init__(**kwargs)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def default_subset(self) -> str:
        return "test"

    def download(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        beir_path = Path("data") / "beir" / self._beir_subdir

        # Check for extracted BEIR data
        extracted = beir_path / self._beir_subdir
        if extracted.exists() and (extracted / "corpus.jsonl").exists():
            logger.info(f"  Found BEIR data at {extracted}")
            return extracted

        # Check for zip
        zip_path = beir_path / f"{self._beir_subdir}.zip"
        if zip_path.exists():
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(beir_path)
            extracted = beir_path / self._beir_subdir
            if extracted.exists():
                return extracted

        raise FileNotFoundError(
            f"BEIR dataset '{self._beir_subdir}' not found.\n"
            f"Expected: data/beir/{self._beir_subdir}/{self._beir_subdir}/corpus.jsonl\n"
            f"Download from: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self._beir_subdir}.zip"
        )

    def convert_to_musique_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        """
        Convert BEIR format to MuSiQue-like JSONL.

        For each query, we select the top-K relevant passages from qrels
        and add distractor passages from the corpus.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = raw_path / "corpus.jsonl"
        queries_path = raw_path / "queries.jsonl"
        qrels_dir = raw_path / "qrels"

        if not corpus_path.exists():
            raise FileNotFoundError(f"corpus.jsonl not found at {corpus_path}")

        # Load corpus
        corpus = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                doc_id = entry["_id"]
                corpus[doc_id] = {
                    "title": entry.get("title", ""),
                    "text": entry.get("text", ""),
                }
        logger.info(f"  Loaded {len(corpus)} corpus documents")

        # Load queries
        queries = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                queries[entry["_id"]] = entry.get("text", "")

        # Load qrels (test split)
        qrels_path = qrels_dir / "test.tsv"
        if not qrels_path.exists():
            # Try other splits
            for split in ["dev", "train"]:
                candidate = qrels_dir / f"{split}.tsv"
                if candidate.exists():
                    qrels_path = candidate
                    break

        if not qrels_path.exists():
            raise FileNotFoundError(f"No qrels found at {qrels_dir}")

        qrels = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                    if score > 0:  # Only relevant documents
                        if query_id not in qrels:
                            qrels[query_id] = []
                        qrels[query_id].append(doc_id)

        logger.info(f"  Loaded {len(queries)} queries, {len(qrels)} with relevance labels")

        # Convert to MuSiQue format
        output_path = output_dir / f"{self._dataset_name}.jsonl"
        n_written = 0
        all_doc_ids = list(corpus.keys())

        with open(output_path, "w", encoding="utf-8") as f:
            for q_idx, (query_id, query_text) in enumerate(queries.items()):
                if n_written >= max_queries:
                    break

                relevant_docs = qrels.get(query_id, [])
                if not relevant_docs:
                    continue

                # Build paragraphs: relevant docs + distractors
                paragraphs = []
                for i, doc_id in enumerate(relevant_docs[:5]):  # Max 5 relevant
                    if doc_id in corpus:
                        paragraphs.append({
                            "idx": len(paragraphs),
                            "title": corpus[doc_id]["title"],
                            "paragraph_text": corpus[doc_id]["text"],
                            "is_supporting": True,
                        })

                # Add distractors from corpus
                distractor_count = 0
                for doc_id in all_doc_ids:
                    if distractor_count >= 15:  # Max 15 distractors
                        break
                    if doc_id not in relevant_docs and doc_id in corpus:
                        paragraphs.append({
                            "idx": len(paragraphs),
                            "title": corpus[doc_id]["title"],
                            "paragraph_text": corpus[doc_id]["text"],
                            "is_supporting": False,
                        })
                        distractor_count += 1

                if not paragraphs:
                    continue

                entry = {
                    "id": f"{self._dataset_name}_{query_id}",
                    "question": query_text,
                    "answer": paragraphs[0]["title"] if paragraphs else "",
                    "paragraphs": paragraphs,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written += 1

        logger.info(f"  Written {n_written} entries -> {output_path}")
        return output_path

    def convert_to_full_corpus_format(
        self, raw_path: Path, output_dir: Path, max_queries: int = 500
    ) -> Path:
        """
        Full-corpus variant: writes two files so the benchmark can rank over the
        ENTIRE corpus (not a gold+15 distractor pool):
          - <name>_full.jsonl          : queries with gold passages (is_supporting=True)
          - <name>_full_corpus.txt     : every corpus doc, one line: "doc_<id>, title text"
        phase1_index (--full-corpus) loads the corpus file to build a global candidate
        set and maps every query to ALL corpus doc ids.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = raw_path / "corpus.jsonl"
        queries_path = raw_path / "queries.jsonl"
        qrels_dir = raw_path / "qrels"
        if not corpus_path.exists():
            raise FileNotFoundError(f"corpus.jsonl not found at {corpus_path}")

        corpus = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                corpus[e["_id"]] = {"title": e.get("title", ""), "text": e.get("text", "")}

        queries = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                queries[e["_id"]] = e.get("text", "")

        qrels_path = qrels_dir / "test.tsv"
        if not qrels_path.exists():
            for split in ["dev", "train"]:
                c = qrels_dir / f"{split}.tsv"
                if c.exists():
                    qrels_path = c
                    break
        if not qrels_path.exists():
            raise FileNotFoundError(f"No qrels found at {qrels_dir}")
        qrels = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3 and int(parts[2]) > 0:
                    qrels.setdefault(parts[0], []).append(parts[1])

        # Global corpus file (all docs, stable ids doc_<idx>)
        corpus_file = output_dir / f"{self._dataset_name}_full_corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            for idx, (_id, doc) in enumerate(corpus.items()):
                gid = f"doc_{idx:06d}"
                f.write(f"{gid}, {doc['title']} {doc['text']}\n")

        # Query file: gold passages only (supporting); candidate set = full corpus
        query_file = output_dir / f"{self._dataset_name}_full.jsonl"
        n_written = 0
        with open(query_file, "w", encoding="utf-8") as f:
            for q_idx, (qid, qtext) in enumerate(queries.items()):
                if n_written >= max_queries:
                    break
                rel = qrels.get(qid, [])
                if not rel:
                    continue
                paragraphs = []
                for did in rel[:5]:
                    if did in corpus:
                        paragraphs.append({
                            "idx": len(paragraphs),
                            "title": corpus[did]["title"],
                            "paragraph_text": corpus[did]["text"],
                            "is_supporting": True,
                        })
                if not paragraphs:
                    continue
                entry = {
                    "id": f"{self._dataset_name}_{qid}",
                    "question": qtext,
                    "answer": paragraphs[0]["title"],
                    "paragraphs": paragraphs,
                    "use_full_corpus": True,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written += 1
        logger.info(f"  Written {n_written} full-corpus queries -> {query_file}")
        logger.info(f"  Global corpus ({len(corpus)} docs) -> {corpus_file}")
        return query_file

    def get_recommended_params(self) -> Dict[str, Any]:
        params = super().get_recommended_params()
        # BEIR datasets are larger, use larger top_k
        params["top_k"] = 10
        return params


# Pre-configured instances for common BEIR datasets
class NFCorpusAdapter(BEIRAdapter):
    def __init__(self, **kwargs):
        super().__init__("nfcorpus", "NFCorpus", "nfcorpus", **kwargs)

class SciFactAdapter(BEIRAdapter):
    def __init__(self, **kwargs):
        super().__init__("scifact", "SciFact", "scifact", **kwargs)

class QuoraAdapter(BEIRAdapter):
    def __init__(self, **kwargs):
        super().__init__("quora", "Quora", "quora", **kwargs)

class TRECCOVIDAdapter(BEIRAdapter):
    def __init__(self, **kwargs):
        super().__init__("trec-covid", "TREC-COVID", "trec-covid", **kwargs)

class DBPediaAdapter(BEIRAdapter):
    def __init__(self, **kwargs):
        super().__init__("dbpedia-entity", "DBPedia", "dbpedia-entity", **kwargs)

class SciDocsAdapter(BEIRAdapter):
    def __init__(self, **kwargs):
        super().__init__("scidocs", "SciDocs", "scidocs", **kwargs)
