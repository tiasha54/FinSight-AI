"""
retriever.py
-------------
Vector Database (Step 4: Retrieval function)

This is the file the RAG engine imports for the pipeline.
Loads the FAISS index once, exposes search() for any question.

Usage (from the RAG engine):
    from retriever import search
    results = search("What was Apple's revenue in 2023?", top_k=5)
    results = search("What was Apple's revenue in 2023?", top_k=5, company="Apple")
    results = search("What was Apple's revenue in 2023?", top_k=5, company="Apple", year="2023")
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
MODEL_NAME    = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────
# Load index, metadata, and model ONCE at import time.
# ─────────────────────────────────────────────
print("[retriever] Loading FAISS index and metadata...")

_index_path    = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
_metadata_path = os.path.join(VECTOR_DB_DIR, "metadata.json")

if not os.path.exists(_index_path):
    raise FileNotFoundError(
        "vector_db/faiss_index.bin not found. "
        "Run embed_chunks.py then build_faiss_index.py first."
    )

_index = faiss.read_index(_index_path)

with open(_metadata_path, "r", encoding="utf-8") as f:
    _metadata = json.load(f)

print(f"[retriever] Loaded index with {_index.ntotal} vectors")

print("[retriever] Loading embedding model (for encoding queries)...")
_model = SentenceTransformer(MODEL_NAME)
print("[retriever] Ready.")


def search(query: str, top_k: int = 5, company: str = None, year: str = None):
    """
    Search the vector DB for chunks relevant to `query`.

    Args:
        query:   the user's question
        top_k:   how many chunks to return
        company: optional filter, e.g. "Apple"
        year:    optional filter, e.g. "2023"

    Returns:
        List of dicts sorted by relevance (highest score first):
        {"text": ..., "company": ..., "year": ..., "source": ..., "score": ...}
    """
    # bge models recommend prefixing QUERIES (not documents) with this instruction.
    # Document chunks were embedded WITHOUT this prefix in embed_chunks.py — that
    # asymmetry is intentional and correct for this model family.
    instructed_query = f"Represent this sentence for searching relevant passages: {query}"

    query_vector = _model.encode(
        [instructed_query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Over-fetch when filtering, since the filter discards non-matching results.
    fetch_k = top_k * 10 if (company or year) else top_k
    fetch_k = min(fetch_k, _index.ntotal)

    scores, indices = _index.search(query_vector, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        record = _metadata[idx]

        if company and record["company"].lower() != company.lower():
            continue
        if year and str(record["year"]) != str(year):
            continue

        results.append({
            "text":    record["text"],
            "company": record["company"],
            "year":    record["year"],
            "source":  record["source"],
            "score":   float(score)
        })

        if len(results) >= top_k:
            break

    return results
