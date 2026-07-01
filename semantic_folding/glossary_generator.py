"""
glossary_generator.py — Generate MeSH-style glossary JSON from LLM-extracted phrases.

Takes the output of ``llm_phrase_extractor`` (vocabulary + context mapping) and
uses the LLM to group phrases into domains/categories with canonical forms and
synonyms, producing a glossary JSON file compatible with
``OntologyExpander`` (ontology_expander.py).

Usage:
    from glossary_generator import generate_glossary
    generate_glossary(phrase_list, dataset_name, output_path="config/glossary_bioasq.json")
"""

import json, os, re, time, sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter

# Ensure semantic_folding/ is on sys.path for lib import
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from lib import get_logger

logger = get_logger("glossary_gen")

# ── Load API config ──────────────────────────────────────────────────────────

def _load_env(path: str = ".env") -> dict:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    env = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env

_env = _load_env()
LLM_API_KEY = os.environ.get("LLM_API_KEY") or _env.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE") or _env.get("LLM_API_BASE", "https://opencode.ai/zen/v1")
LLM_MODEL = os.environ.get("LLM_MODEL") or _env.get("LLM_MODEL", "big-pickle")


def _call_llm(messages: list, max_tokens: int = 2048,
              temperature: float = 0.1, retries: int = 3) -> Optional[str]:
    """Call the chat-completion API."""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, data=payload, headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            }, method="POST")
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
        except HTTPError as e:
            body = e.read().decode()[:300] if e.fp else str(e)
            logger.warning(f"  [GLOSSARY] attempt {attempt}/{retries}: {e.code} {body}")
            if attempt < retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"  [GLOSSARY] attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


# ── Category detection ───────────────────────────────────────────────────────

KNOWN_DOMAINS = [
    "genetics", "oncology", "cardiovascular", "neurology",
    "pharmacology", "immunology", "cell_biology", "biochemistry",
    "microbiology", "physiology", "pathology", "epidemiology",
    "diagnostics", "surgery", "pediatrics", "general_medical",
]

SYSTEM_PROMPT = """You are a biomedical ontology builder. Group the following biomedical phrases into domains and identify canonical forms and synonyms.

Input: A list of biomedical phrases extracted from a research corpus.

Output: A JSON object with a "domains" key, where each domain contains categories, and each category has canonical terms with their synonyms.

EXAMPLE FORMAT:
{
  "domains": {
    "genetics": {
      "gene_expression": {
        "gene expression": ["transcription", "gene activity"],
        "mutation": ["genetic variant", "allele change"]
      }
    },
    "oncology": {
      "cancer_types": {
        "breast cancer": ["mammary carcinoma", "breast neoplasm"]
      }
    }
  }
}

Rules:
- Group related phrases under shared canonical forms
- Abbreviations go as synonyms (e.g., "miRNA" -> synonym of "microRNA")
- Single-occurrence technical terms can be standalone with empty synonym list
- Every canonical form appears ONLY ONCE across all domains
- Return ONLY valid JSON, no explanation"""


def generate_glossary(
    phrase_list: List[str],
    dataset_name: str,
    output_path: str = None,
    max_phrases: int = 500,
    batch_size: int = 200,
) -> dict:
    """Generate a glossary JSON from a list of extracted phrases.

    Args:
        phrase_list: Unique phrases from LLM extraction.
        dataset_name: Used for description metadata.
        output_path: If set, save glossary to this path.
        max_phrases: Cap on phrases to send to LLM (largest sets get expensive).
        batch_size: Phrases per LLM call.

    Returns:
        Glossary dict in OntologyExpander-compatible format.
    """
    # Deduplicate and normalize
    unique = sorted(set(p.strip().lower() for p in phrase_list if len(p.strip()) >= 3))

    if len(unique) > max_phrases:
        logger.info(f"  [GLOSSARY] truncating {len(unique)} phrases to top {max_phrases} by length (multi-word preferred)")
        # Prefer multi-word and longer phrases
        unique.sort(key=lambda p: (-p.count(" "), -len(p)))
        unique = unique[:max_phrases]

    logger.info(f"  [GLOSSARY] categorizing {len(unique)} phrases")

    # Split into batches
    glossary: dict = {"_description": f"Auto-generated glossary for {dataset_name} from LLM-extracted phrases",
                      "domains": {}}

    for start in range(0, len(unique), batch_size):
        end = min(start + batch_size, len(unique))
        batch = unique[start:end]

        user_prompt = f"Categorize these {len(batch)} biomedical phrases into domains:\n\n"
        for i, phrase in enumerate(batch):
            user_prompt += f"{i}. {phrase}\n"
        user_prompt += "\nReturn JSON with 'domains' as described."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        content = _call_llm(messages, max_tokens=4096, temperature=0.05)
        if not content:
            logger.warning(f"  [GLOSSARY] batch {start//batch_size + 1} returned empty, skipping")
            continue

        # Parse
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            batch_glossary = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"  [GLOSSARY] batch {start//batch_size + 1} unparseable, skipping")
            continue

        # Merge into main glossary
        batch_domains = batch_glossary.get("domains", {})
        for domain_name, categories in batch_domains.items():
            if not isinstance(categories, dict):
                continue
            if domain_name not in glossary["domains"]:
                glossary["domains"][domain_name] = {}
            for category, terms in categories.items():
                if not isinstance(terms, dict):
                    continue
                if category not in glossary["domains"][domain_name]:
                    glossary["domains"][domain_name][category] = {}
                for canonical, synonyms in terms.items():
                    if not isinstance(synonyms, list):
                        synonyms = []
                    glossary["domains"][domain_name][category][canonical] = synonyms

        logger.info(f"  [GLOSSARY] batch {start//batch_size + 1}: "
                     f"{sum(len(v) for v in batch_domains.values())} categories added")

        if end < len(unique):
            time.sleep(0.5)

    # Count total canonical terms
    total_terms = sum(
        len(categories)
        for domain in glossary["domains"].values()
        for categories in domain.values()
    )
    logger.success(f"  [GLOSSARY] generated {total_terms} canonical terms across "
                    f"{len(glossary['domains'])} domains")

    # Save if requested
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(glossary, f, ensure_ascii=False, indent=2)
        logger.success(f"  [GLOSSARY] saved to {out}")

    return glossary


def generate_from_llm_output(
    llm_output_dir: Path,
    dataset_name: str,
    output_path: str = None,
) -> dict:
    """Convenience: load LLM phrase output and generate glossary.

    Args:
        llm_output_dir: Directory containing vocabulary.csv from LLM extractor.
        dataset_name: Dataset name for metadata.
        output_path: If set, save glossary here.

    Returns:
        Glossary dict.
    """
    vocab_path = llm_output_dir / "vocabulary.csv"
    if not vocab_path.exists():
        logger.error(f"  [GLOSSARY] vocabulary not found: {vocab_path}")
        return {"_description": f"Empty glossary for {dataset_name}", "domains": {}}

    phrases = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "," in line:
                phrase = line.split(",")[0].strip()
                if phrase:
                    phrases.append(phrase)

    logger.info(f"  [GLOSSARY] loaded {len(phrases)} phrases from {vocab_path}")

    return generate_glossary(phrases, dataset_name, output_path)


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate glossary JSON from LLM-extracted phrases")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with llm_phrases/vocabulary.csv")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output", type=str, default=None, help="Output path for glossary JSON")
    parser.add_argument("--max-phrases", type=int, default=500)
    args = parser.parse_args()

    generate_from_llm_output(args.input_dir, args.dataset, args.output)
