import numpy as np
import json
from pathlib import Path

RUN_ID = "20260323_014724"
BASE = Path(f"outputs/{RUN_ID}")
QUERY = "How has language evolved and what does it reveal about cultural and historical human interaction?"

# ── 1. Load phrase fingerprints ──────────────────────────────────────────────
phrase_fp_data = np.load(BASE / "phrase_fingerprints/phrase_fingerprints.npz")
phrase_fps = phrase_fp_data["fingerprints"]          # shape (862, 256)

with open(BASE / "phrase_fingerprints/phrase_fingerprints_meta.json") as f:
    phrase_meta = json.load(f)

# phrase_meta is a dict mapping phrase_text → row_index
# We need to invert it to get row_index → phrase_text
phrase_list = [""] * len(phrase_meta)
for phrase_text, row_idx in phrase_meta.items():
    phrase_list[row_idx] = phrase_text

phrase_index = {p: i for i, p in enumerate(phrase_list)}

# ── 2. Tokenize query and find vocabulary hits ────────────────────────────────
import re
query_tokens = set(re.findall(r'\b[a-z]+\b', QUERY.lower()))
hits = [p for p in phrase_list if p in query_tokens]
print(f"Vocab hits ({len(hits)}): {hits}")

# ── 3. Build query fingerprint (OR of matched phrase fingerprints) ─────────────
query_fp = np.zeros(256, dtype=np.float32)
for phrase in hits:
    idx = phrase_index[phrase]
    query_fp = np.maximum(query_fp, phrase_fps[idx])   # binary OR via max

# spreading_steps=0 → no spreading, use as-is
active_bits = np.count_nonzero(query_fp)
print(f"Query active bits: {active_bits}")
print(f"Query active indices: {np.nonzero(query_fp)[0]}")

# Save it so the original pipeline can use it later
out_dir = BASE / "query_results"
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_dir / "query_fingerprint.npz", fingerprint=query_fp)
print(f"Saved query fingerprint → {out_dir / 'query_fingerprint.npz'}")

# ── 4. Load document fingerprints ────────────────────────────────────────────
doc_data = np.load(BASE / "doc_fingerprints/doc_fingerprints.npz")
doc_fps  = doc_data["fingerprints"]                   # shape (20, 256)
print(f"\nDoc fingerprints shape : {doc_fps.shape}")
print(f"Query fingerprint shape: {query_fp.shape}")

# ── 5. Check if all phrase fingerprints are identical ─────────────────────────
print("\n" + "="*65)
print("PHRASE FINGERPRINT ANALYSIS")
print("="*65)
unique_phrase_fps = len({tuple(phrase_fps[i].nonzero()[0]) for i in range(phrase_fps.shape[0])})
print(f"Total phrase fingerprints: {phrase_fps.shape[0]}")
print(f"Unique fingerprint patterns: {unique_phrase_fps}")

# Show first 5 phrase fingerprints
for i in range(min(5, phrase_fps.shape[0])):
    active = np.nonzero(phrase_fps[i])[0]
    print(f"Phrase {i} ('{phrase_list[i]}'): active bits = {active}")

# ── 6. Manual cosine similarity for ALL docs ─────────────────────────────────
query_norm = np.linalg.norm(query_fp)
scores = {}
query_set = set(np.nonzero(query_fp)[0])

print("\n" + "="*65)
print(f"{'Doc':<6} {'Active':>8} {'Dot':>8} {'DocNorm':>10} {'Cosine':>10} {'Q⊆D?':>6}")
print("="*65)

for i in range(doc_fps.shape[0]):
    d = doc_fps[i].flatten()
    doc_norm  = np.linalg.norm(d)
    dot       = float(np.dot(query_fp, d))
    cosine    = dot / (query_norm * doc_norm) if (query_norm > 0 and doc_norm > 0) else 0.0
    scores[i] = cosine

    doc_set   = set(np.nonzero(d)[0])
    subset    = query_set.issubset(doc_set)
    print(f"Doc {i:<3} {int(np.count_nonzero(d)):>8} {dot:>8.2f} {doc_norm:>10.4f} {cosine:>10.6f} {str(subset):>6}")

# ── 7. Are all document vectors identical? ────────────────────────────────────
print("\n" + "="*65)
print("DOCUMENT IDENTITY CHECK")
all_same = all(np.array_equal(doc_fps[0], doc_fps[i]) for i in range(1, doc_fps.shape[0]))
print(f"All 20 doc vectors identical? {all_same}")
if not all_same:
    unique_rows = len({tuple(doc_fps[i].nonzero()[0]) for i in range(doc_fps.shape[0])})
    print(f"Unique document fingerprint patterns: {unique_rows}")

# ── 8. Ranked results using CORRECT cosine ────────────────────────────────────
print("\n" + "="*65)
print("CORRECT RANKING (manual cosine)")
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for rank, (doc_id, score) in enumerate(ranked[:10], 1):
    print(f"  #{rank:>2}  Doc {doc_id:<3}  score={score:.6f}")
