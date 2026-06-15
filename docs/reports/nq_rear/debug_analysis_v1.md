# NQ-REaR Debug Analysis: Query Pipeline Deep Dive

## Executive Summary

Ran queries 1-10 with DEBUG logging. Key finding: **Score compression** is the root cause — all documents score within a narrow range (0.034-0.051), making gold passages indistinguishable from distractors.

## Debug Output Summary

| Query | Gold Rank | Gold Score | Top Score | Gap | Status |
|-------|-----------|------------|-----------|-----|--------|
| 0: "oldest team in nba" | #10 (last) | 0.0342 | 0.0398 | -16% | FAIL |
| 1: "religion china silk route" | #1 | 0.0494 | 0.0494 | 0% | PASS |
| 2: "medical bills great britain" | #1 | 0.0338 | 0.0338 | 0% | PASS |
| 5: "second largest country asia" | #7 | 0.0336 | 0.0383 | -12% | FAIL |
| 6: "zinc to pennies" | #2,#3 | 0.0475 | 0.0486 | -2% | PARTIAL |
| 7: "grey's anatomy chief resident" | NOT in top 10 | — | 0.0513 | — | FAIL |

## Root Cause Analysis

### 1. Score Compression (Critical)

All 990 documents score within a narrow range:
- Query 0: 0.034 - 0.040 (16% spread)
- Query 1: 0.043 - 0.049 (14% spread)
- Query 7: 0.046 - 0.051 (11% spread)

**Why this happens**: 
- Query fingerprints have ~3000-3800 active bits (out of 4096)
- Document fingerprints have similar sparsity
- Cosine similarity between any two sparse vectors is inherently low

### 2. Topic Dilution

When passages share a topic, their fingerprints become nearly identical:

**Query 0: "oldest team in nba"**
- Gold: "Eddie Gottlieb" (barnstorming teams, basketball history)
- Distractor #1: "National Basketball Association" (NBA, championships)
- Both passages are about basketball → similar fingerprints

**Query 7: "grey's anatomy chief resident"**
- Gold: "April Kepner" (Chief Resident, season 7)
- Distractor #1: "The Resident (TV series)" (has "resident" in title)
- Title match boosts distractor above gold

### 3. Missing Specificity

SF captures **topic similarity** but not **relational specificity**:
- "oldest" → not captured (just "NBA" topic)
- "chief resident" → not captured (just "resident" topic)
- "zinc to pennies" → not captured (just "penny" topic)

### 4. No Threshold Filtering

All 990 documents are scored and returned. No early filtering based on:
- Minimum similarity threshold
- Candidate passage pre-filtering
- Topic relevance check

## Pipeline Stage Analysis

### Stage 1: Phrase Extraction
- 100% phrase match rate (not the problem)
- Phrases are well-extracted from queries

### Stage 2: Fingerprint Construction
- Query fingerprint: ~3000-3800 active bits
- Document fingerprint: similar density
- **Issue**: Too many active bits → low discrimination

### Stage 3: Spreading
- Radius=1, decay=0.5
- Adds ~100-200 bits to query fingerprint
- **Issue**: Spreading dilutes specificity further

### Stage 4: Ranking
- Cosine similarity across all 990 documents
- No threshold filtering
- **Issue**: All documents score similarly

## Recommended Improvements

### Priority 1: Hybrid Scoring (Quick Win)
Combine SF fingerprints with BM25 term frequency:
```
final_score = α * SF_score + (1-α) * BM25_score
```
- BM25 captures exact term matching
- SF captures semantic similarity
- Best of both worlds

### Priority 2: Query Term Weighting
Boost rare/distinctive terms:
- "nba" → high weight (rare, specific)
- "team" → low weight (common, generic)
- "oldest" → medium weight (specific but common)

### Priority 3: Threshold Filtering
Set minimum similarity threshold:
- Filter out documents with score < 0.01
- Reduce candidate pool from 990 to ~50-100
- Focus ranking on relevant passages

### Priority 4: Passage Length Normalization
Normalize fingerprints by passage length:
- Short passages get boosted
- Long passages get penalized
- Prevents length bias in ranking

### Priority 5: Title Weighting
Optionally boost title matches:
- If query contains "nba", boost passages with "NBA" in title
- If query contains "grey's anatomy", boost passages with that title
- Simple lexical matching for titles

## Implementation Roadmap

| Improvement | Effort | Expected Gain | Priority |
|-------------|--------|---------------|----------|
| Hybrid scoring | Low | +10-15% MRR | HIGH |
| Query term weighting | Medium | +5-10% MRR | MEDIUM |
| Threshold filtering | Low | +3-5% MRR | MEDIUM |
| Passage length norm | Low | +2-3% MRR | LOW |
| Title weighting | Low | +2-3% MRR | LOW |

## Conclusion

The debug output confirms that **semantic dilution** is the root cause, not vocabulary mismatch. The fix is to combine SF's semantic understanding with BM25's lexical matching via hybrid scoring.
