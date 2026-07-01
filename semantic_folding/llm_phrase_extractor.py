"""
llm_phrase_extractor.py — LLM-based phrase/concept extraction for Semantic Folding.

Uses a chat-completion API (OpenAI-compatible) to extract domain-specific
biomedical concepts and multi-word phrases from each corpus line.  The output
vocabulary can be merged with the spaCy-based phrase_extractor output to
improve recall on datasets where SF currently underperforms BM25.

Usage:
    from llm_phrase_extractor import extract_phrases_from_corpus
    phrases, mapping = extract_phrases_from_corpus(corpus_lines, ...)
"""

import json, os, re, time, csv, sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
import http.client
import urllib.parse

# Ensure semantic_folding/ is on sys.path for lib import
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from lib import get_logger

logger = get_logger("llm_phrase_extractor")

# ── Defaults from .env ────────────────────────────────────────────────────────

def _load_env(path: str = ".env") -> dict:
    """Load key=value pairs from a .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return {}
    env = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

_env = _load_env()
LLM_API_KEY = os.environ.get("LLM_API_KEY") or _env.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE") or _env.get("LLM_API_BASE", "https://opencode.ai/zen/v1")
LLM_MODEL = os.environ.get("LLM_PHRASE_MODEL") or _env.get("LLM_PHRASE_MODEL", "mimo-v2.5-free")


# ── LLM call helper ───────────────────────────────────────────────────────────

def _call_llm(messages: list, max_tokens: int = 4096, temperature: float = 0.1,
              retries: int = 3) -> Optional[str]:
    """Call the chat-completion API and return the assistant content string."""
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    parsed = urllib.parse.urlparse(url)
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    for attempt in range(1, retries + 1):
        try:
            conn = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=120)
            conn.request("POST", parsed.path, payload, {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            })
            resp = conn.getresponse()
            body = resp.read().decode()
            if resp.status != 200:
                logger.warning(f"  [LLM] attempt {attempt}/{retries} failed: {resp.status} {body[:200]}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                continue
            result = json.loads(body)
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"  [LLM] attempt {attempt}/{retries} error: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a biomedical NLP assistant. Extract domain-specific terms from biomedical text that a standard NLP parser would miss or under-represent.

For each input line, return a JSON array of strings. Each string is a specific term.

Rules:
- Output AT MOST 5 terms per item. Fewer is better — only high-value terms.
- Prefer proper nouns: drug names, gene/protein symbols, rare disease names, assay names
- Prefer multi-word terms (2-4 words) over single generic words
- Include abbreviations and their expansions as separate entries when both appear
- NEVER output generic terms: drug, therapy, treatment, gene, protein, disease, disorder, mutation, effect, role, mechanism, cell, tissue, process, pathway
- If the text contains only generic concepts, return []

OUTPUT FORMAT: A JSON list of strings. Example:
["myocardial infarction", "acute mi", "receptor tyrosine kinase RET"]

Return ONLY the JSON array, no explanation."""


# ── Batch phrase extraction ──────────────────────────────────────────────────

def _parse_llm_response(content: str) -> List[str]:
    """Parse a JSON array from the LLM response, handling common quirks."""
    if not content:
        return []
    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        # Remove ```json ... ``` or ```
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    content = content.strip()
    try:
        phrases = json.loads(content)
        if isinstance(phrases, list):
            return [str(p).strip() for p in phrases if p and str(p).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: try to find [...] in the response
    m = re.search(r"\[.*?\]", content, re.DOTALL)
    if m:
        try:
            phrases = json.loads(m.group())
            if isinstance(phrases, list):
                return [str(p).strip() for p in phrases if p and str(p).strip()]
        except json.JSONDecodeError:
            pass
    logger.warning(f"  [LLM] unparseable response: {content[:200]}")
    return []


def extract_phrases_batch(
    texts: List[str],
    doc_ids: List[str],
    system_prompt: str = None,
    batch_size: int = 8,
    max_tokens: int = 8192,
    temperature: float = 0.05,
    max_phrases_per_doc: int = 5,
) -> Dict[str, Set[str]]:
    """Extract phrases from a batch of corpus lines via a single LLM call.

    Each input line is passed as a numbered item.  The LLM returns a
    JSON object mapping item numbers to phrase arrays.

    Returns {phrase: set_of_doc_ids}.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    # Build numbered list prompt
    items_str = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
    user_prompt = f"""Extract key biomedical concepts from each item below.

Return a JSON object where keys are item numbers (0, 1, ...) and values are arrays of phrases.

Example:
{{
  "0": ["myocardial infarction", "heart attack"],
  "1": ["gene mutation", "RET", "coding sequence"]
}}

Items:
{items_str}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = _call_llm(messages, max_tokens=max_tokens, temperature=temperature)
    if not content:
        return {}

    # Parse JSON object
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"  [LLM] batch response not parseable as JSON")
        return {}

    if not isinstance(parsed, dict):
        logger.warning(f"  [LLM] batch response is not a dict")
        return {}

    result: Dict[str, Set[str]] = {}
    doc_phrase_count: Dict[str, int] = {}  # track phrases per doc_id
    for key, phrases in parsed.items():
        try:
            idx = int(key)
        except (ValueError, TypeError):
            continue
        if idx < 0 or idx >= len(doc_ids):
            continue
        doc_id = doc_ids[idx]
        if isinstance(phrases, list):
            for phrase in phrases:
                p = str(phrase).strip().lower() if phrase else ""
                if len(p) >= 3:
                    count = doc_phrase_count.get(doc_id, 0)
                    if count >= max_phrases_per_doc:
                        break
                    result.setdefault(p, set()).add(doc_id)
                    doc_phrase_count[doc_id] = count + 1

    logger.info(f"  [LLM] batch extracted {sum(len(v) for v in result.values())} phrase-ctx mappings for {len(texts)} lines")
    return result


# ── Full corpus extraction ────────────────────────────────────────────────────

def extract_phrases_from_corpus(
    corpus_lines: List[str],
    min_freq: int = 1,
    max_doc_freq: int = 0,
    batch_size: int = 8,
    max_batches: Optional[int] = None,
    sample: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[Counter, Dict[str, List[str]]]:
    """Run LLM phrase extraction over an entire corpus.

    Args:
        corpus_lines: Lines in ``doc_id, text`` format (same as corpus.txt).
        min_freq: Minimum document frequency to keep a phrase.
        max_doc_freq: Maximum doc frequency (0 = no limit).
        batch_size: Lines per LLM call.
        max_batches: Cap on number of batches (for testing).
        sample: If set, randomly sample this many lines first.
        cache_dir: If set, save/load raw LLM output to/from this directory.

    Returns:
        (phrase_counts, phrase_to_contexts) — same format as
        ``phrase_extractor.process_corpus_with_expansion`` output.
    """
    import random

    # Parse corpus
    doc_ids = []
    texts = []
    for line in corpus_lines:
        line = line.strip()
        if not line or "," not in line:
            continue
        doc_id, text = line.split(",", 1)
        doc_ids.append(doc_id.strip())
        texts.append(text.strip())

    if sample and sample < len(texts):
        indices = list(range(len(texts)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:sample]
        doc_ids = [doc_ids[i] for i in indices]
        texts = [texts[i] for i in indices]
        logger.info(f"  [LLM] sampled {sample} lines from {len(corpus_lines)}")

    n_total = len(texts)
    logger.info(f"  [LLM] extracting phrases from {n_total} lines in batches of {batch_size}")

    # Check for cached LLM output
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_vocab = cache_dir / "vocabulary.csv"
        cache_mapping = cache_dir / "phrase_to_contexts.json"
        if cache_vocab.exists() and cache_mapping.exists():
            logger.info(f"  [LLM] loading cached LLM output from {cache_dir}")
            cached_counts: Counter = Counter()
            with open(cache_vocab, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        cached_counts[row[0]] = int(row[1])
            with open(cache_mapping, "r", encoding="utf-8") as f:
                cached_mapping = json.load(f)
            logger.info(f"  [LLM] loaded {len(cached_counts)} cached phrases")
            return cached_counts, cached_mapping

    all_mapping: Dict[str, Set[str]] = {}

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_texts = texts[start:end]
        batch_ids = doc_ids[start:end]

        logger.info(f"  [LLM] batch {start // batch_size + 1}/{(n_total + batch_size - 1) // batch_size} ({start}-{end})")

        batch_result = extract_phrases_batch(batch_texts, batch_ids, batch_size=len(batch_texts))

        # Merge into all_mapping
        for phrase, ctx_set in batch_result.items():
            all_mapping.setdefault(phrase, set()).update(ctx_set)

        # Short delay to avoid rate limits
        if end < n_total:
            time.sleep(0.5)

        if max_batches and (start // batch_size + 1) >= max_batches:
            logger.info(f"  [LLM] stopped after {max_batches} batches (max_batches cap)")
            break

    # Frequency filter
    phrase_counts: Counter = Counter()
    filtered_mapping: Dict[str, List[str]] = {}
    dropped = 0
    for phrase, ctx_set in all_mapping.items():
        freq = len(ctx_set)
        if freq >= min_freq and (max_doc_freq == 0 or freq <= max_doc_freq):
            phrase_counts[phrase] = freq
            filtered_mapping[phrase] = sorted(ctx_set)
        else:
            dropped += 1

    logger.info(
        f"  [LLM] extracted {len(phrase_counts)} phrases "
        f"(kept), {dropped} dropped (min_freq={min_freq})"
    )

    # Save to cache if requested
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_llm_phrases(phrase_counts, filtered_mapping, cache_dir)
        logger.info(f"  [LLM] cached LLM output to {cache_dir}")

    return phrase_counts, filtered_mapping


QUERY_SYSTEM_PROMPT = """You are a biomedical query analyzer. Your ONLY job is to find 1-2 specific terms that a standard NLP parser would MISS from the question.

What spaCy already catches: disease names, body parts, common medical nouns, verbs, adjectives. DO NOT duplicate these.

What spaCy MISKS (your job):
- Drug/compound names (e.g., "metformin", "lacosamide")
- Gene/protein symbols (e.g., "RET", "BRCA1", "EGFR")
- Specific assay/test names (e.g., "ELISA", "Western blot")
- Rare disease subtypes not in general vocabulary

Rules:
- Output AT MOST 2 terms. Less is better — only add what spaCy truly misses.
- Each term must be a specific proper noun or technical term, NOT a concept or relationship.
- NEVER output generic terms: drug, therapy, treatment, gene, protein, disease, disorder, mutation, effect, role, interaction, mechanism.
- NEVER output multi-word relationship phrases. Only atomic terms.
- If spaCy would already catch everything, return []

Input: "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
Output: []  (spaCy catches: Hirschsprung, disease, mendelian, multifactorial, disorder)

Input: "Does metformin interfere thyroxine absorption?"
Output: ["metformin", "thyroxine"]  (drug names spaCy might miss)

Input: "What is the role of RET mutations in thyroid cancer?"
Output: ["RET"]  (gene symbol spaCy misses)

Return ONLY the JSON array, no explanation."""  # noqa: E501


def extract_query_phrases_batch(
    query_texts: List[str],
    system_prompt: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.01,
    retries: int = 2,
    max_phrases: int = 2,
) -> List[List[str]]:
    """Extract key concepts from query texts using LLM.

    Each query is sent individually (short text, high precision needed).
    Returns a list of phrase lists, one per query.
    """
    if system_prompt is None:
        system_prompt = QUERY_SYSTEM_PROMPT

    results: List[List[str]] = []
    for i, query_text in enumerate(query_texts):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text},
        ]
        content = _call_llm(messages, max_tokens=max_tokens, temperature=temperature, retries=retries)
        phrases = _parse_llm_response(content)[:max_phrases]
        results.append(phrases)
        if (i + 1) % 5 == 0:
            logger.info(f"  [LLM] query phrases: {i+1}/{len(query_texts)}")

    return results


# ── Save in phrase_extractor format ───────────────────────────────────────────

def save_llm_phrases(
    phrase_counts: Counter,
    phrase_to_contexts: Dict[str, List[str]],
    output_dir: Path,
) -> None:
    """Save LLM-extracted phrases in the same format as phrase_extractor.

    Creates ``<output_dir>/vocabulary.csv`` and
    ``<output_dir>/phrase_to_contexts.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocabulary.csv"
    with open(vocab_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for phrase, count in phrase_counts.most_common():
            writer.writerow([phrase, count])

    mapping_path = output_dir / "phrase_to_contexts.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(phrase_to_contexts, f, ensure_ascii=False, indent=2)

    logger.success(
        f"  [LLM] saved {len(phrase_counts)} phrases to {output_dir}"
    )


# ── Merge with spaCy phrases ──────────────────────────────────────────────────

def merge_with_spacy(
    llm_phrase_counts: Counter,
    llm_phrase_to_contexts: Dict[str, List[str]],
    spacy_vocab_path: Path,
    spacy_mapping_path: Path,
    output_dir: Path,
) -> Tuple[Counter, Dict[str, List[str]]]:
    """Merge LLM-extracted phrases with existing spaCy phrase output.

    Reads spaCy vocabulary + mapping, adds LLM phrases (union of context
    sets for overlap), and writes merged files to output_dir.
    """
    # Load spaCy output
    spacy_phrase_counts: Counter = Counter()
    with open(spacy_vocab_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                spacy_phrase_counts[row[0]] = int(row[1])

    with open(spacy_mapping_path, "r", encoding="utf-8") as f:
        spacy_mapping = json.load(f)

    # Merge: union of phrases, union of context sets for overlapping phrases
    merged_counts: Counter = Counter(spacy_phrase_counts)
    merged_mapping: Dict[str, List[str]] = dict(spacy_mapping)

    added = 0
    overlapped = 0
    for phrase, freq in llm_phrase_counts.items():
        ctx_set = set(llm_phrase_to_contexts.get(phrase, []))
        if phrase in merged_mapping:
            # Merge context sets: union
            existing_ctx = set(merged_mapping[phrase])
            new_ctx = ctx_set - existing_ctx
            if new_ctx:
                merged_mapping[phrase] = sorted(existing_ctx | ctx_set)
                merged_counts[phrase] = len(merged_mapping[phrase])
                overlapped += len(new_ctx)
        else:
            # New phrase
            merged_counts[phrase] = freq
            merged_mapping[phrase] = sorted(ctx_set)
            added += 1

    logger.info(
        f"  [LLM] merge: {added} new phrases added, "
        f"{overlapped} additional context mappings on existing phrases"
    )

    # Save merged output
    save_dir = output_dir / "extracted_phrases"
    save_dir.mkdir(parents=True, exist_ok=True)

    merged_vocab_path = save_dir / "vocabulary.csv"
    with open(merged_vocab_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for phrase, count in merged_counts.most_common():
            writer.writerow([phrase, count])

    merged_mapping_path = save_dir / "phrase_to_contexts.json"
    with open(merged_mapping_path, "w", encoding="utf-8") as f:
        json.dump(merged_mapping, f, ensure_ascii=False, indent=2)

    logger.success(
        f"  [LLM] merged vocabulary ({len(merged_counts)} total) saved to {save_dir}"
    )
    return merged_counts, merged_mapping


# ── CLI entry point for standalone testing ────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM phrase extraction for SF pipeline")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to corpus.txt")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample", type=int, default=None, help="Sample N lines for testing")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_lines = [line.strip() for line in f if line.strip()]

    phrase_counts, mapping = extract_phrases_from_corpus(
        corpus_lines,
        min_freq=args.min_freq,
        batch_size=args.batch_size,
        sample=args.sample,
        max_batches=args.max_batches,
    )
    save_llm_phrases(phrase_counts, mapping, args.output)
    print(f"Done: {len(phrase_counts)} phrases -> {args.output}")
