# Musique API Results Error Analysis

## Summary

**Total Records:** 400  
**Completed:** 0 (0.0%)  
**Failed:** 400 (100.0%)

## Root Cause

**All 400 records failed with the same error:**
```
API error: deepseek/deepseek-v3.1:free is not a valid model ID
```

## Analysis

### Problem Chain:

1. **All models in `free-models.yml` are marked as `inactive`** due to:
   - Rate limit errors: "free-models-per-day" (most common)
   - Invalid model IDs (some models don't exist)
   - Data policy issues (OpenAI models require privacy settings)
   - Connection errors

2. **`load_free_models(active_only=True)` returns empty list** because no models have `status: "active"`

3. **Code falls back to invalid default model:**
   - When `free_models` is empty, `extract_triples.py` line 275 uses: `cfg.model`
   - `cfg.model` = `DEFAULT_OPENROUTER_MODEL` = `"deepseek/deepseek-v3.1:free"`
   - This model ID is **not valid** on OpenRouter

4. **All 400 API calls fail** because the invalid model ID is used

## Error Breakdown from free-models.yml

### Rate Limit Errors (7 models):
- `tngtech/deepseek-r1t2-chimera:free`
- `tngtech/deepseek-r1t-chimera:free`
- `nvidia/nemotron-3-nano-30b-a3b:free`
- `google/gemma-3-27b-it:free`
- `google/gemini-2.0-flash-exp:free`
- `arcee-ai/trinity-mini:free`

**Error:** "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"

### Invalid Model IDs (4 models):
- `deepseek/r1-0528:free`
- `tngtech/r1t-chimera:free`
- `qwen/qwen3-coder-480b-a35b:free`
- `nvidia/nemotron-nano-12b-2-vl:free`

**Error:** "is not a valid model ID"

### Data Policy Issues (2 models):
- `openai/gpt-oss-120b:free`
- `openai/gpt-oss-20b:free`

**Error:** "No endpoints found matching your data policy (Free model publication). Configure: https://openrouter.ai/settings/privacy"

### Other Errors (2 models):
- `z-ai/glm-4.5-air:free`: "Server disconnected"
- `qwen/qwen3-next-80b-a3b-instruct:free`: "Cannot connect to host"
- `meta-llama/llama-3.3-70b-instruct:free`: "Rate limit exceeded: Provider returned error"

## Recommendations

### Immediate Fix:

1. **Update `DEFAULT_OPENROUTER_MODEL`** to a valid model ID that exists on OpenRouter
   - Current: `"deepseek/deepseek-v3.1:free"` ❌ (invalid)
   - Options: Check OpenRouter docs for valid free model IDs

2. **Add validation in `extract_triples.py`:**
   - If `free_models` is empty AND `cfg.model` is invalid, raise a clear error instead of proceeding
   - Warn user that no active models are available

3. **Handle rate limits better:**
   - Wait for rate limits to reset (daily limit)
   - Or add credits to OpenRouter account
   - Or use paid models

### Long-term Solutions:

1. **Update `free-models.yml`** with valid model IDs from OpenRouter's current free models list
2. **Run `test_models.py`** when rate limits reset to mark working models as `active`
3. **Add fallback logic** to try inactive models if all active ones fail (with warnings)
4. **Monitor OpenRouter API** for model availability changes

## Next Steps

1. Fix `DEFAULT_OPENROUTER_MODEL` to a valid model ID
2. Re-run extraction with valid models
3. Or wait for rate limits to reset and re-run `test_models.py`
