# Plan: LLM-Based Domain-Specific Phrase Extraction

## Overview

BioASQ fails because spaCy's `en_core_web_sm` (trained on news) doesn't capture biomedical terminology. This plan implements LLM-based phrase extraction to fix that.

## Architecture

### Flag: `--use-llm-phrases`
- **Phase 1 (phrase_extractor.py)**: When set, extract phrases using LLM instead of spaCy
- **Phase 6 (query_processor.py)**: When set, extract query phrases using the same LLM for consistency

### Workflow
1. **Pre-processing**: Run LLM on corpus to extract domain phrases → save to `llm_phrases.json`
2. **Phase 1**: Load `llm_phrases.json` instead of running spaCy
3. **Phase 6**: For each query, call LLM to extract phrases (or load from cache)

## Implementation Plan

### Step 1: Create LLM Phrase Extractor Module
**File**: `semantic_folding/llm_phrase_extractor.py`

```python
class LLMPhraseExtractor:
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        # Support OpenAI API or local LLM
    
    def extract_phrases(self, text, domain="biomedical"):
        """
        Use LLM to extract domain-specific phrases from text.
        
        Prompt template:
        "Extract key domain-specific phrases from the following biomedical text. 
        Focus on: medical conditions, proteins, genes, drugs, procedures.
        Return as JSON list: ['phrase1', 'phrase2', ...]"
        """
        # Call LLM API
        # Parse response
        # Return list of phrases
    
    def extract_phrases_batch(self, texts, domain="biomedical"):
        """Batch process multiple texts."""
```

### Step 2: Modify Phase 1 (phrase_extractor.py)
Add flag `--use-llm-phrases`:
- When set, instantiate `LLMPhraseExtractor`
- Call `extract_phrases_batch()` on entire corpus
- Save results to `outputs/<dataset>_benchmark/llm_phrases.json`
- Use LLM phrases instead of spaCy phrases for vocabulary

### Step 3: Modify Phase 6 (query_processor.py)
Add flag `--use-llm-phrases`:
- When set, instantiate `LLMPhraseExtractor`
- For each query, call `extract_phrases(query)`
- Use LLM phrases for query fingerprint (consistent with index)

### Step 4: Benchmark Methodology
1. Run baseline (spaCy phrases) on BioASQ → MRR=0.288
2. Run with LLM phrases on BioASQ → compare MRR
3. Run on PubMedQA (also biomedical) → check generalization

## Detailed Steps for the Agent

### Step A: Create `llm_phrase_extractor.py`
```python
# semantic_folding/llm_phrase_extractor.py
import json
import openai  # or local LLM client

class LLMPhraseExtractor:
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        self.model = model
        if api_key:
            openai.api_key = api_key
    
    def extract_phrases(self, text, domain="biomedical"):
        prompt = f"""Extract key domain-specific phrases from the following {domain} text.
Focus on technical terminology, entities, and multi-word expressions.
Return as JSON list: ["phrase1", "phrase2", ...]

Text: {text}

JSON list:"""
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        phrases_json = response.choices[0].message.content
        phrases = json.loads(phrases_json)
        return phrases
    
    def extract_phrases_batch(self, texts, domain="biomedical", output_path=None):
        results = {}
        for i, text in enumerate(texts):
            phrases = self.extract_phrases(text, domain)
            results[f"doc_{i}"] = phrases
            
            if output_path and (i % 10 == 0):
                with open(output_path, 'w') as f:
                    json.dump(results, f)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f)
        
        return results
```

### Step B: Modify `phrase_extractor.py`
Add argument:
```python
parser.add_argument('--use-llm-phrases', action='store_true', 
                   help='Use LLM for phrase extraction instead of spaCy')
```

In `main()`:
```python
if args.use_llm_phrases:
    from llm_phrase_extractor import LLMPhraseExtractor
    extractor = LLMPhraseExtractor(model="gpt-3.5-turbo")
    
    # Load corpus
    with open(args.corpus, 'r') as f:
        texts = [line.split(',', 1)[1].strip() for line in f]
    
    # Extract phrases
    llm_phrases = extractor.extract_phrases_batch(texts, domain="biomedical",
                                                   output_path="llm_phrases.json")
    
    # Convert to vocabulary format
    vocabulary = Counter()
    mapping = {}
    for doc_id, phrases in llm_phrases.items():
        for phrase in phrases:
            vocabulary[phrase] += 1
            if phrase not in mapping:
                mapping[phrase] = []
            mapping[phrase].append(doc_id)
    
    # Save
    save_vocabulary(vocabulary, mapping, args.output_dir)
    return  # Skip spaCy extraction
```

### Step C: Modify `query_processor.py`
Add argument:
```python
parser.add_argument('--use-llm-phrases', action='store_true',
                   help='Use LLM for query phrase extraction')
```

In query processing:
```python
if args.use_llm_phrases:
    from llm_phrase_extractor import LLMPhraseExtractor
    extractor = LLMPhraseExtractor(model="gpt-3.5-turbo")
    
    # Extract query phrases
    query_phrases = extractor.extract_phrases(query, domain="biomedical")
else:
    # Original spaCy extraction
    query_phrases = extract_phrases_spacy(query)
```

### Step D: Benchmark Script
**File**: `temp/benchmark_llm_phrases.py`

```python
#!/usr/bin/env python3
"""
Benchmark LLM-based phrase extraction on BioASQ.
Compares: spaCy vs LLM phrases.
"""
import subprocess
import json

# Run baseline (spaCy)
result_baseline = subprocess.run([
    "python", "semantic_folding/dataset_benchmark/generic_benchmark.py",
    "all", "--dataset", "bioasq", "--jsonl", "data/bioasq/converted/bioasq.jsonl",
    "--max-queries", "50", "--no-splade"
], capture_output=True, text=True)

# Run with LLM phrases
result_llm = subprocess.run([
    "python", "semantic_folding/dataset_benchmark/generic_benchmark.py",
    "all", "--dataset", "bioasq", "--jsonl", "data/bioasq/converted/bioasq.jsonl",
    "--max-queries", "50", "--no-splade", "--use-llm-phrases"
], capture_output=True, text=True)

# Compare MRR
baseline_mrr = extract_mrr_from_output(result_baseline.stdout)
llm_mrr = extract_mrr_from_output(result_llm.stdout)

print(f"Baseline (spaCy): MRR={baseline_mrr}")
print(f"LLM phrases:    MRR={llm_mrr}")
print(f"Improvement:     {((llm_mrr - baseline_mrr) / baseline_mrr * 100):.1f}%")
```

## Expected Outputs

### 1. LLM Phrases File
**Path**: `outputs/bioasq_benchmark/llm_phrases.json`
**Format**:
```json
{
  "doc_000": ["Hirschsprung disease", "mendelian disorder", "multifactorial disorder"],
  "doc_001": ["signaling molecules", "ligands", "cell surface receptors"],
  ...
}
```

### 2. Benchmark Comparison
**Path**: `temp/bioasq_llm_comparison.txt`
**Content**:
```
Baseline (spaCy): MRR=0.288
LLM phrases:    MRR=0.XXX
Improvement:     +X.X%
```

### 3. vocabulary Files (replaced)
- `vocabulary.csv` — now contains LLM phrases
- `phrase_context_mapping.json` — remapped to LLM phrases

## Agent Execution Steps

1. **Implement `llm_phrase_extractor.py`** — start with OpenAI API, add local LLM support later
2. **Add `--use-llm-phrases` flag to `phrase_extractor.py`** — test on small corpus first
3. **Add `--use-llm-phrases` flag to `query_processor.py`** — ensure consistency
4. **Run benchmark on BioASQ** — compare spaCy vs LLM
5. **Run benchmark on PubMedQA** — check generalization to other biomedical dataset
6. **Report results** — MRR improvement, phrase quality examples

## Notes

- **Cost**: LLM API calls cost money. For 1,075 BioASQ docs, estimate $0.50-$2.00 with GPT-3.5-turbo.
- **Latency**: LLM extraction is slower than spaCy. Use batching.
- **Quality**: LLM phrases should be more accurate for biomedical terms, but may over-extract. Add frequency filtering.
- **Fallback**: If LLM fails, fall back to spaCy (graceful degradation).

## Success Criteria

- [ ] LLM phrase extraction implemented and tested
- [ ] `--use-llm-phrases` flag works in Phase 1 and Phase 6
- [ ] BioASQ MRR improves from 0.288 (target: >0.35)
- [ ] PubMedQA MRR maintains or improves (verify generalization)
- [ ] Plan documented in `docs/plans/llm_phrase_extraction.md`
