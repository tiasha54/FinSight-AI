"""
build_faiss_index.py
---------------------
Vector Database (Step 2 & 3: Build + Store FAISS Index)

Loads the embeddings from embed_chunks.py and builds a FAISS index
for fast similarity search. Saves everything to vector_db/ so it's
a self-contained handoff folder for the RAG engine.

Run this AFTER embed_chunks.py
"""

import os
import json
import numpy as np
import faiss

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
VECTOR_DB_DIR  = os.path.join(BASE_DIR, "vector_db")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

print("=" * 60)
print("  AI Finance Analyzer — FAISS Index Build")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Load embeddings + metadata
# ─────────────────────────────────────────────
print("\n[1/3] Loading embeddings and metadata...")

embeddings_path = os.path.join(EMBEDDINGS_DIR, "chunk_embeddings.npy")
metadata_path   = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.json")

if not os.path.exists(embeddings_path):
    raise FileNotFoundError("Run embed_chunks.py first to generate chunk_embeddings.npy")

embeddings = np.load(embeddings_path).astype("float32")

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

assert embeddings.shape[0] == len(metadata), (
    f"Mismatch: {embeddings.shape[0]} embeddings vs {len(metadata)} metadata records. "
    "These must always be the same length and same order."
)

dimension = embeddings.shape[1]
print(f"      {embeddings.shape[0]} vectors, dimension = {dimension}")

# ─────────────────────────────────────────────
# STEP 2: Build the FAISS index
# ─────────────────────────────────────────────
print("\n[2/3] Building FAISS index...")

# IndexFlatIP = exact inner-product search.
# Embeddings are normalized (normalize_embeddings=True), so inner product == cosine similarity.
# Fine for ~10k chunks. If the corpus grows past ~100k chunks, switch to IndexIVFFlat for speed.
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"      Index built. Total vectors in index: {index.ntotal}")

# ─────────────────────────────────────────────
# STEP 3: Save index + metadata together (this folder = handoff to the RAG engine)
# ─────────────────────────────────────────────
print("\n[3/3] Saving to vector_db/ ...")

faiss.write_index(index, os.path.join(VECTOR_DB_DIR, "faiss_index.bin"))

with open(os.path.join(VECTOR_DB_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"      Saved: vector_db/faiss_index.bin")
print(f"      Saved: vector_db/metadata.json")

print("\n" + "=" * 60)
print("  FAISS INDEX BUILD COMPLETE")
print(f"  {index.ntotal} chunks are now searchable")
print("  Hand the vector_db/ folder to the RAG engine")
print("=" * 60)
