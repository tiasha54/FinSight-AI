"""
embed_chunks.py
----------------
Vector Database (Step 1: Generate Embeddings)

Reads data/chunks/all_chunks.json (from the ingestion step), generates an
embedding vector for every chunk using BAAI/bge-small-en-v1.5,
and saves the embeddings + metadata to disk.

You are NOT training anything here — bge-small-en-v1.5 is a pre-trained
model. This script just runs it over your chunks to produce vectors.

Run this BEFORE build_faiss_index.py
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# PATH SETUP — matches team folder structure:
#   data/chunks/all_chunks.json   <- input from the ingestion step
#   embeddings/                   <- output of this script
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH    = os.path.join(BASE_DIR, "data", "chunks", "all_chunks.json")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("=" * 60)
print("  AI Finance Analyzer — Embedding Generation")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Load chunks from the ingestion step
# ─────────────────────────────────────────────
print(f"\n[1/4] Loading chunks from {CHUNKS_PATH} ...")

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        f"Could not find {CHUNKS_PATH}. "
        "Regenerate the latest all_chunks.json and place it in data/chunks/."
    )

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"      Loaded {len(chunks)} chunks")

# Sanity check on metadata — fail loudly instead of silently embedding garbage.
# Known issue: the filename parsing put "10K" in the year field for
# every chunk instead of the actual year. Flag it instead of hiding it.
years_seen = set(c.get("year", "unknown") for c in chunks)
if years_seen == {"10K"} or years_seen == {"unknown"}:
    print("      WARNING: every chunk has the same/unknown 'year' field.")
    print("      Filtering search results by year will NOT work until that part is fixed.")
    print("      (filename Apple_2022_10K.pdf -> year should be '2022', currently reads '10K')\n")

# ─────────────────────────────────────────────
# STEP 2: Load the embedding model (pre-trained, not trained by us)
# ─────────────────────────────────────────────
print(f"\n[2/4] Loading embedding model: {MODEL_NAME} ...")
print("      (downloads automatically the first time you run this)")

model = SentenceTransformer(MODEL_NAME)

# ─────────────────────────────────────────────
# STEP 3: Generate embeddings for every chunk
# ─────────────────────────────────────────────
print(f"\n[3/4] Generating embeddings for {len(chunks)} chunks...")
print("      (a few minutes on CPU for ~10k chunks)")

texts = [c["text"] for c in chunks]

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,   # lets us use inner-product = cosine similarity in FAISS
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")  # FAISS requires float32

print(f"      Done. Embedding matrix shape: {embeddings.shape}")

# ─────────────────────────────────────────────
# STEP 4: Save embeddings + metadata to disk
# ─────────────────────────────────────────────
print("\n[4/4] Saving embeddings and metadata...")

np.save(os.path.join(EMBEDDINGS_DIR, "chunk_embeddings.npy"), embeddings)

# Metadata is saved in the SAME ORDER as the embeddings array.
# embeddings[i] <-> metadata[i] <-> chunks[i]. Never sort/filter one without the other.
metadata = [
    {
        "chunk_id":  c["chunk_id"],
        "company":   c["company"],
        "year":      c["year"],
        "source":    c["source"],
        "chunk_num": c["chunk_num"],
        "text":      c["text"],
    }
    for c in chunks
]

with open(os.path.join(EMBEDDINGS_DIR, "chunk_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"      Saved: embeddings/chunk_embeddings.npy  {embeddings.shape}")
print(f"      Saved: embeddings/chunk_metadata.json   ({len(metadata)} records)")

print("\n" + "=" * 60)
print("  EMBEDDING GENERATION COMPLETE")
print("  Next step: run build_faiss_index.py")
print("=" * 60)
