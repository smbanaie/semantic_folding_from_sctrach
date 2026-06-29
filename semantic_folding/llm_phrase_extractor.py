"""
Domain-Specific Phrase Extractor

Provides an interface for domain-aware phrase extraction, replacing spaCy's
generic noun chunk extraction with curated biomedical / domain terminology.

When the 'model' starts with 'gpt' and openai is available, it calls the
OpenAI API. For any other model (or 'stub'/'local'), it uses a built-in
domain-specific extractor with curated biomedical term lists and patterns.

Usage:
    extractor = LLMPhraseExtractor(domain="biomedical")
    phrases = extractor.extract_phrases(text)

    # Or batch process
    results = extractor.extract_phrases_batch(texts, output_path="phrases.json")
"""

import json
import re
import time
import os
from typing import List, Dict, Optional, Set
from pathlib import Path
from collections import Counter

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from lib import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Curated biomedical term dictionaries
# ═══════════════════════════════════════════════════════════════════════════════

# Common biomedical abbreviations / acronyms (non-exhaustive, high-value)
_BIOMEDICAL_ACRONYMS: Set[str] = {
    # Genes / proteins
    "RET", "GDNF", "EDNRB", "EDN3", "SOX10", "NTN3", "ECE1",
    "EGFR", "EGF", "AREG", "BTC", "EPR", "EPG", "HB-EGF",
    "TGF", "VEGF", "PDGF", "FGF", "IGF", "HGF", "KGF",
    "TP53", "APC", "KRAS", "NRAS", "BRAF", "PIK3CA", "PTEN",
    "BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2",
    "ALK", "ROS1", "MET", "NTRK", "FGFR", "IDH1", "IDH2",
    "HSCR", "MHC", "HLA", "CRP", "ESR", "PR", "HER2",
    "G6PD", "ACE", "APOE", "CFTR", "CYP2D6", "CYP3A4",
    "DNA", "RNA", "mRNA", "tRNA", "rRNA", "miRNA", "siRNA",
    "PCR", "ELISA", "FISH", "CGH", "NGS", "WGS", "WES",
    # Diseases / conditions
    "COPD", "CKD", "CHD", "CAD", "MI", "CVA", "DVT", "PE",
    "T2DM", "NAFLD", "NASH", "IBD", "IBS", "UTI", "STD",
    "HIV", "AIDS", "HBV", "HCV", "HPV", "CMV", "EBV",
    "ADHD", "ASD", "OCD", "PTSD", "MCI", "ALS", "MS",
    "RA", "OA", "SLE", "APS", "PAN", "ANCA",
    "ARDS", "MODS", "SIRS", "DIC", "TTP", "HUS",
    # Drugs / treatments
    "NSAID", "SSRI", "SNRI", "MAOI", "TCA", "ACEi", "ARB",
    "PPI", "H2RA", "SGLT2", "GLP-1", "DPP-4",
    # Anatomy
    "CNS", "PNS", "ANS", "GI", "GU", "CV", "ENT",
}

# Common biomedical multi-word terms (lowercased for matching)
_BIOMEDICAL_TERMS: Set[str] = {
    "hirschsprung disease", "hirschsprung's disease", "hscr",
    "mendelian disorder", "multifactorial disorder",
    "multifactorial inheritance", "sex modified multifactorial",
    "coding sequence mutation", "noncoding mutation",
    "receptor tyrosine kinase", "enteric nervous system",
    "signaling molecule", "cell surface receptor",
    "extracellular matrix", "epidermal growth factor",
    "heparin binding egf", "transforming growth factor",
    "growth factor receptor", "tyrosine kinase inhibitor",
    "monoclonal antibody", "angiogenesis inhibitor",
    "immune checkpoint", "programmed death ligand",
    "stem cell", "cell differentiation", "cell proliferation",
    "gene expression", "transcription factor", "dna repair",
    "cell cycle", "apoptosis pathway", "signal transduction",
    "oxidative stress", "reactive oxygen species",
    "nitric oxide", "free radical", "lipid peroxidation",
    "blood brain barrier", "blood pressure", "heart rate",
    "body mass index", "waist circumference",
    "gestational diabetes", "insulin resistance",
    "glucose tolerance", "fasting glucose", "fasting insulin",
    "homeostasis model assessment", "parathyroid hormone",
    "vitamin d", "vitamin d deficiency",
    "odds ratio", "confidence interval", "statistical significance",
    "randomized controlled trial", "double blind",
    "systematic review", "meta analysis", "cohort study",
    "case control", "cross sectional", "longitudinal study",
    "prospective study", "retrospective study",
    "inflammatory bowel disease", "irritable bowel syndrome",
    "cardiovascular disease", "coronary artery disease",
    "chronic kidney disease", "chronic obstructive pulmonary",
    "type 2 diabetes", "non alcoholic fatty liver",
    "acute respiratory distress", "renal cell carcinoma",
    "hepatocellular carcinoma", "colorectal cancer",
    "breast cancer", "lung cancer", "prostate cancer",
    "non small cell", "small cell lung",
    "sentinel lymph node", "lymph node metastasis",
    "overall survival", "progression free survival",
    "quality of life", "adverse event", "side effect",
    "dose dependent", "concentration dependent",
    "mechanism of action", "mode of action",
    "structure activity relationship", "quantitative structure",
    "minimal inhibitory concentration", "half maximal",
    "area under curve", "maximum concentration",
    "pregnancy test", "renal function", "glomerular filtration",
    "cardiac arrest", "ventricular septal defect",
    "pulmonary artery", "right ventricle", "heart failure",
    "myocardial infarction", "cerebrovascular accident",
    "deep vein thrombosis", "pulmonary embolism",
    "peripheral neuropathy", "neurodegenerative disease",
    "alzheimer disease", "parkinson disease", "huntington disease",
    "amyotrophic lateral sclerosis", "multiple sclerosis",
    "rheumatoid arthritis", "osteoarthritis", "systemic lupus",
    "antiphospholipid syndrome", "vasculitis",
    "graft versus host", "hematopoietic stem cell",
    "health related quality", "patient reported outcome",
    "emergency department", "intensive care unit",
    "primary care", "secondary care", "tertiary care",
    "electronic health record", "electronic medical record",
    "adverse drug reaction", "drug drug interaction",
    "therapeutic drug monitoring", "precision medicine",
    "personalized medicine", "translational research",
    "evidence based medicine", "clinical decision support",
    "genome wide association", "gene expression profiling",
}

# Domain-specific patterns for biomedical entities
# (gene names, mutation notation, chemical formulas, etc.)
_BIOMEDICAL_PATTERNS: List[re.Pattern] = [
    # Human gene symbols (2-6 uppercase letters, optionally with digits)
    re.compile(r'\b[A-Z][A-Z0-9]{1,5}(?:-[A-Z][A-Z0-9]{1,5})?\b'),
    # Mutation notations: p.Gly12Cys, c.123A>G, etc.
    re.compile(r'\b[pcr]\.[A-Za-z]+\d+[A-Za-z]*(?:[>_][A-Za-z]+|\d+)?\b'),
    # Chemical formulas like Ca2+, H2O, CO2
    re.compile(r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+\b'),
    # Protein names: "protein X", "receptor Y"
    re.compile(r'\b(?:protein|receptor|ligand|factor|enzyme|kinase|inhibitor|agonist|antagonist)\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'),
    # Multi-word biomedical terms with hyphens: "T-cell", "B-cell", "E-cadherin"
    re.compile(r'\b[A-Z]-[a-z][a-z]+(?:-[a-z]+)?\b'),
    # Chromosomal notation: 1p/19q, 11q23, Xq28
    re.compile(r'\b\d{1,2}[pq]\d{1,2}(?:\.\d+)?\b'),
    # Allele names: *01, *02, *03
    re.compile(r'\*0\d\b'),
]


def _load_env_file() -> dict:
    """Load .env file from project root or current directory."""
    # Search from project root (where semantic_folding/ lives)
    project_root = Path(__file__).parent.parent
    search_dirs = [project_root] + [Path.cwd()] + list(Path.cwd().parents)
    
    for search_dir in search_dirs:
        env_path = search_dir / ".env"
        if env_path.exists():
            break
    else:
        return {}
    
    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip()
    return env_vars


class LLMPhraseExtractor:
    """
    Extract domain-specific phrases using domain knowledge or an LLM.

    Args:
        model: 'gpt-3.5-turbo' (needs OPENAI_API_KEY), 'gpt-4', or 'stub'/'local'
            for built-in domain-specific extraction.
        domain: Domain context ('biomedical', 'legal', 'technical', etc.)
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        max_retries: Number of retries on API failure
        batch_size: Number of texts to process in parallel
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        domain: str = "biomedical",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        prompt_path: Optional[str] = None,
        max_retries: int = 3,
        batch_size: int = 10,
        force_domain_only: bool = False,
    ):
        self.model = model
        self.domain = domain
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.force_domain_only = force_domain_only
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "")
        self.prompt_path = prompt_path
        
        # If still no credentials, try .env
        if not self.api_key or not self.api_base:
            env_vars = _load_env_file()
            if not self.api_key:
                self.api_key = env_vars.get("LLM_API_KEY", self.api_key)
            if not self.api_base:
                self.api_base = env_vars.get("LLM_API_BASE", self.api_base)
        
        # Load custom prompt if provided
        self._system_prompt = self._load_system_prompt()
        
        if self.force_domain_only or not (self.api_key and REQUESTS_AVAILABLE):
            self.extractor_type = "domain"
            logger.info(f"Domain-specific extractor initialized for '{domain}' domain")
        else:
            self.extractor_type = "openai"
            logger.info(f"LLM extractor initialized: {model} via {self.api_base or 'default API'} for {domain} domain")

    def _load_system_prompt(self) -> str:
        """Load the system prompt from a file or use default."""
        if self.prompt_path and Path(self.prompt_path).exists():
            with open(self.prompt_path, 'r') as f:
                prompt = f.read().strip()
            logger.info(f"Loaded custom prompt from {self.prompt_path}")
            return prompt
        
        # Default prompts by domain
        defaults = {
            "biomedical": "Extract biomedical key terms and phrases from the user's text. "
                         "Return ONLY a flat JSON array of strings. "
                         'Example: ["hirschsprung disease", "mendelian disorder", "coding sequence mutation", "RET"]',
        }
        return defaults.get(self.domain, defaults["biomedical"])

    def extract_phrases(self, text: str, domain: Optional[str] = None) -> List[str]:
        """
        Extract domain-specific phrases from a single text.

        Args:
            text: Input text to extract phrases from
            domain: Override default domain (optional)

        Returns:
            List of extracted phrases (strings)
        """
        domain = domain or self.domain

        if self.extractor_type == "openai":
            return self._extract_phrases_openai(text, domain)
        else:
            return self._extract_phrases_domain(text, domain)

    def _extract_phrases_openai(self, text: str, domain: str) -> List[str]:
        """Extract phrases using OpenAI-compatible API (custom endpoint supported)."""
        prompt_text = self._build_prompt(text, domain)
        
        # For reasoning models (deepseek, etc.), we need a high token budget
        # because they spend tokens on chain-of-thought before responding
        max_tokens = 2000

        for attempt in range(self.max_retries):
            try:
                # Build request URL
                if self.api_base:
                    url = self.api_base.rstrip("/")
                    if not url.endswith("/chat/completions"):
                        url += "/chat/completions"
                else:
                    url = "https://api.openai.com/v1/chat/completions"

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }

                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                # Handle different response formats
                message = result["choices"][0]["message"]
                
                # Try content first
                content = message.get("content", "").strip()
                
                # If empty, try reasoning_content (deepseek models)
                if not content:
                    content = message.get("reasoning_content", "").strip()
                
                # If still empty, try to extract from the raw response
                if not content:
                    # Some models put the response in a different field
                    for key in ["text", "response", "output"]:
                        if key in message and message[key]:
                            content = str(message[key]).strip()
                            break

                if not content:
                    logger.warning(f"[API] Empty response on attempt {attempt+1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return self._extract_phrases_domain(text, domain)

                # Try to parse as JSON list
                try:
                    phrases = json.loads(content)
                    if isinstance(phrases, list):
                        # Normalise: lowercase, strip whitespace
                        phrases = [str(p).strip() for p in phrases if str(p).strip()]
                        # Deduplicate preserving order
                        seen = set()
                        deduped = []
                        for p in phrases:
                            p_lower = p.lower()
                            if p_lower not in seen:
                                seen.add(p_lower)
                                deduped.append(p)
                        logger.debug(f"[API] Extracted {len(deduped)} phrases from {len(text)} chars")
                        return deduped
                    elif isinstance(phrases, dict):
                        # Some APIs wrap in {"phrases": [...]}
                        for val in phrases.values():
                            if isinstance(val, list):
                                return [str(p).strip() for p in val if str(p).strip()]
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    import re as _re
                    json_match = _re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
                    if json_match:
                        try:
                            phrases = json.loads(json_match.group(1))
                            if isinstance(phrases, list):
                                return [str(p).strip() for p in phrases if str(p).strip()]
                        except json.JSONDecodeError:
                            pass

                    # Fallback: line-by-line parsing
                    lines = [
                        line.strip().strip('"\'[],).-* ')
                        for line in content.split('\n')
                        if line.strip() and not line.strip().startswith('"') and not line.strip().startswith('[') and not line.strip().startswith(']') and not line.strip().startswith(',')
                    ]
                    lines = [l for l in lines if l and len(l) >= 2]
                    # Attempt to extract quoted strings
                    quoted = _re.findall(r'"([^"]+)"', content)
                    if quoted:
                        return [q.strip() for q in quoted if q.strip()]
                    if lines:
                        return lines

                    logger.warning(f"[API] Could not parse response: {content[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return self._extract_phrases_domain(text, domain)

            except requests.exceptions.Timeout:
                logger.warning(f"[API] Timeout on attempt {attempt+1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"[API] Request error on attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"[API] Unexpected error on attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("[API] Max retries — falling back to domain extractor")
                    return self._extract_phrases_domain(text, domain)

        return self._extract_phrases_domain(text, domain)

    def _extract_phrases_domain(self, text: str, domain: str) -> List[str]:
        """
        Extract domain-specific phrases using curated term lists and patterns.

        Strategy:
        1. Exact-match curated biomedical multi-word terms (case-insensitive)
        2. Match biomedical acronyms / gene symbols
        3. Apply domain-specific regex patterns
        4. Extract noun-adjective sequences targeting biomedical constructions
        5. Deduplicate and filter

        Returns:
            List of unique extracted phrases
        """
        import re as _re

        phrases: Set[str] = set()
        text_lower = text.lower()

        # ── Strategy 1: Exact multi-word biomedical terms ─────────────────────
        # Try matching full multi-word terms from the curated list
        for term in _BIOMEDICAL_TERMS:
            if term in text_lower:
                phrases.add(term)

        # ── Strategy 2: Acronyms / gene symbols ──────────────────────────────
        # Match uppercase acronyms present in the text
        for acr in _BIOMEDICAL_ACRONYMS:
            # Acronym must appear as a standalone word
            pattern = _re.compile(r'\b' + _re.escape(acr) + r'\b')
            if pattern.search(text):
                phrases.add(acr)

        # ── Strategy 3: Domain-specific patterns ─────────────────────────────
        for pattern_obj in _BIOMEDICAL_PATTERNS:
            for match in pattern_obj.finditer(text):
                matched = match.group().strip()
                if matched and len(matched) >= 2:
                    phrases.add(matched)

        # ── Strategy 4: Biomedical-specific noun phrase patterns ─────────────
        # Pattern: adjective + noun where the noun is biomedical
        # "mendelian disorder", "multifactorial disorder", "recessive mutation"
        biomedical_noun_patterns = [
            # ADJ + medical condition/disorder
            _re.compile(
                r'\b(mendelian|multifactorial|recessive|dominant|x-linked|'
                r'autosomal|genetic|hereditary|congenital|familial|sporadic|'
                r'acute|chronic|malignant|benign|metastatic|refractory)\s+'
                r'(disorder|disease|syndrome|condition|cancer|tumor|infection|'
                r'inheritance|mutation|form|variant)\b',
                _re.IGNORECASE
            ),
            # Pattern: molecular entity + process
            _re.compile(
                r'\b(dna|rna|protein|gene|cell|tissue|receptor|lignant|'
                r'kinase|enzyme|transcription|signaling|pathway|apoptosis)\s+'
                r'[a-z]{3,}(?:ion|ing|ase|tion|sis)\b',
                _re.IGNORECASE
            ),
            # Pattern: gene/protein "expression" / "activity"
            _re.compile(
                r'\b[A-Z]{2,}\s+(expression|activity|mutation|activation|'
                r'inhibition|phosphorylation|methylation|regulation)\b',
            ),
            # Pattern: "X gene" / "X protein" / "X receptor"
            _re.compile(
                r'\b[A-Z][A-Z0-9]{1,5}\s+(gene|protein|receptor|lignant|kinase|enzyme)\b',
            ),
        ]

        for pat in biomedical_noun_patterns:
            for match in pat.finditer(text):
                matched = match.group().strip().lower()
                if matched:
                    phrases.add(matched)

        # ── Strategy 5: Important single-word biomedical terms ───────────────
        # High-value single-word terms
        single_word_terms = {
            "mendelian", "multifactorial", "non-mendelian", "nonmendelian",
            "homozygous", "heterozygous", "hemizygous",
            "genotype", "phenotype", "allele", "locus", "loci",
            "exon", "intron", "promoter", "enhancer", "silencer",
            "transcription", "translation", "replication",
            "methylation", "acetylation", "phosphorylation", "ubiquitination",
            "apoptosis", "necrosis", "autophagy", "senescence",
            "metastasis", "angiogenesis", "carcinogenesis",
            "pathophysiology", "etiology", "pathogenesis", "histology",
            "prognosis", "diagnosis", "screening", "biopsy",
            "genomics", "proteomics", "metabolomics", "transcriptomics",
            "orthologous", "paralogous", "homologous", "conserved",
        }
        for term in single_word_terms:
            if _re.search(r'\b' + _re.escape(term) + r'\b', text_lower):
                phrases.add(term)

        # ── Strategy 6: Gene/protein symbols with numeric suffixes ──────────
        # e.g., "IL-6", "TNF-alpha", "TGF-beta1"
        gene_variant_pattern = _re.compile(
            r'\b[A-Z]{2,}(?:-\d+|[A-Za-z]\d*)\b'
        )
        for match in gene_variant_pattern.finditer(text):
            matched = match.group().strip()
            if matched and len(matched) >= 3:
                phrases.add(matched)

        # ── Strategy 7: Context-dependent multi-word phrases ────────────────
        # Extract meaningful bigrams where first word is domain-specific
        bigram_pattern = _re.compile(
            r'\b(mutation|gene|protein|cell|tissue|blood|serum|plasma|'
            r'urine|renal|hepatic|cardiac|pulmonary|neural|tumor|'
            r'cancer|receptor|signaling|pathway|clinical|patient|'
            r'treatment|therapy|diagnostic|prognostic|molecular|cellular|'
            r'genetic|epigenetic|chromosomal|mitotic|meiotic)\s+'
            r'[a-z]{3,}\b',
            _re.IGNORECASE
        )
        for match in bigram_pattern.finditer(text):
            matched = match.group().strip().lower()
            if matched:
                phrases.add(matched)

        # Convert to list, sort for reproducibility, limit to reasonable max
        result = sorted(phrases)

        # Filter out very short (< 2 chars) or very long (> 50 chars) phrases
        result = [p for p in result if 2 <= len(p) <= 50]

        # Cap at 50 phrases per text (prevents excessive noise)
        if len(result) > 50:
            result = result[:50]

        logger.debug(f"[DOMAIN] Extracted {len(result)} phrases from text ({len(text)} chars)")
        return result

    def _build_prompt(self, text: str, domain: str) -> str:
        """Build the prompt for LLM phrase extraction."""
        return f"""Extract key domain-specific phrases from the following {domain} text.

Focus on:
- Technical terminology
- Domain-specific entities (e.g., medical conditions, proteins, drugs)
- Multi-word expressions
- Acronyms and abbreviations

Return ONLY a JSON list of strings. Example:
["phrase1", "phrase2", "phrase3"]

Text: {text[:1000]}  # Truncate to 1000 chars

JSON list:"""

    def extract_phrases_batch(
        self,
        texts: List[str],
        output_path: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Extract phrases from multiple texts (batch processing).

        Args:
            texts: List of input texts
            output_path: Save intermediate results to this path (for resumability)
            domain: Override default domain

        Returns:
            Dict mapping doc_id -> list of phrases
        """
        domain = domain or self.domain
        results = {}

        # Resume from checkpoint if output_path exists
        if output_path and Path(output_path).exists():
            with open(output_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Resumed from checkpoint: {len(results)} docs already processed")

        for i, text in enumerate(texts):
            doc_id = f"doc_{i:04d}"

            if doc_id in results:
                logger.debug(f"Skipping {doc_id} (already processed)")
                continue

            phrases = self.extract_phrases(text, domain)
            results[doc_id] = phrases

            # Save checkpoint every 10 docs
            if output_path and (i % 10 == 0):
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Checkpoint saved: {i+1}/{len(texts)} docs processed")

            if self.extractor_type == "openai":
                time.sleep(0.1)

        # Final save
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Final results saved to {output_path}")

        return results


def create_llm_vocabulary(
    llm_phrases: Dict[str, List[str]],
    min_freq: int = 1,
    max_doc_freq: int = 0
) -> Dict[str, int]:
    """
    Convert LLM phrase extraction results to vocabulary format.

    Args:
        llm_phrases: Output from extract_phrases_batch()
        min_freq: Minimum document frequency
        max_doc_freq: Maximum document frequency (0 = no limit)

    Returns:
        vocabulary: Dict[phrase] = document frequency
        mapping: Dict[phrase] = list of doc_ids containing that phrase
    """
    from collections import Counter, defaultdict

    phrase_doc_counts = Counter()
    phrase_docs = defaultdict(list)

    for doc_id, phrases in llm_phrases.items():
        for phrase in set(phrases):  # Unique phrases per doc
            phrase_doc_counts[phrase] += 1
            phrase_docs[phrase].append(doc_id)

    # Filter by frequency
    vocabulary = {}
    mapping = {}
    for phrase, count in phrase_doc_counts.items():
        if count >= min_freq and (max_doc_freq == 0 or count <= max_doc_freq):
            vocabulary[phrase] = count
            mapping[phrase] = phrase_docs[phrase]

    logger.info(f"Domain vocabulary: {len(vocabulary)} phrases after frequency filtering")
    return vocabulary, mapping


if __name__ == "__main__":
    # Test
    extractor = LLMPhraseExtractor(domain="biomedical")
    test_text = ("Hirschsprung disease is a mendelian disorder. "
                 "Patients with RET mutations may benefit from treatment. "
                 "The EGFR ligands include EGF and amphiregulin (AREG).")
    phrases = extractor.extract_phrases(test_text)
    print(f"Extracted {len(phrases)} phrases:")
    for p in phrases:
        print(f"  - {p}")
