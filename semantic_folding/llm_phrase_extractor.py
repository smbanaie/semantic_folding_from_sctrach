"""
LLM-Based Domain-Specific Phrase Extractor

This module provides an interface for using LLMs to extract domain-specific
phrases from text, replacing spaCy's generic noun chunk extraction.

Usage:
    extractor = LLMPhraseExtractor(model="gpt-3.5-turbo", domain="biomedical")
    phrases = extractor.extract_phrases(text)
    
    # Or batch process
    results = extractor.extract_phrases_batch(texts, output_path="llm_phrases.json")
"""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from lib import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class LLMPhraseExtractor:
    """
    Extract domain-specific phrases using an LLM.
    
    Args:
        model: LLM model to use ('gpt-3.5-turbo', 'gpt-4', or path to local model)
        domain: Domain context for phrase extraction ('biomedical', 'legal', 'technical', etc.)
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        max_retries: Number of retries on API failure
        batch_size: Number of texts to process in parallel (for future async implementation)
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        domain: str = "biomedical",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        batch_size: int = 10
    ):
        self.model = model
        self.domain = domain
        self.max_retries = max_retries
        self.batch_size = batch_size
        
        if model.startswith("gpt") and OPENAI_AVAILABLE:
            if api_key:
                openai.api_key = api_key
            self.extractor_type = "openai"
            logger.info(f"LLM extractor initialized: {model} for {domain} domain")
        else:
            # Placeholder for local LLM support
            self.extractor_type = "stub"
            logger.warning(f"Model {model} not available — using stub extractor")
            logger.warning("Implement _extract_phrases_local() for local LLM support")
    
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
            return self._extract_phrases_stub(text, domain)
    
    def _extract_phrases_openai(self, text: str, domain: str) -> List[str]:
        """Extract phrases using OpenAI API."""
        prompt = self._build_prompt(text, domain)
        
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse as JSON list
                try:
                    phrases = json.loads(content)
                    if isinstance(phrases, list):
                        return [str(p) for p in phrases]
                except json.JSONDecodeError:
                    # Fallback: parse line-by-line
                    phrases = [line.strip().strip('"\'[]],')
                               for line in content.split('\n') if line.strip()]
                    return phrases
                
            except Exception as e:
                logger.error(f"OpenAI API error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Max retries exceeded — returning empty list")
                    return []
        
        return []
    
    def _extract_phrases_stub(self, text: str, domain: str) -> List[str]:
        """
        Stub implementation — REPLACE with actual LLM call.
        
        This is a placeholder. The user's agent should implement:
        1. OpenAI API call (if using GPT)
        2. Local LLM call (if using Llama/Mistral)
        3. Prompt engineering for domain-specific extraction
        """
        logger.warning("Stub extractor called — implement _extract_phrases_openai() or _extract_phrases_local()")
        
        # Very basic fallback: split by comma, semicolon, or newline
        # This is NOT a real implementation — just prevents crashes
        import re
        # Look for potential phrases (capitalized words, hyphenated terms)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b|\b([a-z]+(?:-[a-z]+)+)\b'
        matches = re.findall(pattern, text)
        phrases = [m[0] or m[1] for m in matches if m[0] or m[1]]
        return phrases[:10]  # Limit to 10 phrases
    
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
            
            # Rate limiting for OpenAI API
            if self.extractor_type == "openai":
                time.sleep(0.1)  # 10 calls/second max for gpt-3.5-turbo
        
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
    
    logger.info(f"LLM vocabulary: {len(vocabulary)} phrases after frequency filtering")
    return vocabulary, mapping


if __name__ == "__main__":
    # Test stub
    extractor = LLMPhraseExtractor(model="stub", domain="biomedical")
    test_text = "Hirschsprung disease is a mendelian disorder. Patients with RANKL secretion issues may benefit from Denosumab treatment."
    phrases = extractor.extract_phrases(test_text)
    print(f"Extracted phrases: {phrases}")
