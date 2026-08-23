"""
llm_integration.py
-------------------
LLM Integration (Steps 5 & 6 of the RAG Pipeline)

STEP 5: RETRIEVER (Top-K Search)
    - Accepts user query
    - Embeds it and searches FAISS vector DB (retriever.py)
    - Returns top-K most relevant chunks with similarity scores

STEP 6: LLM GENERATION (Augmented)
    - Builds an augmented prompt: Retrieved Context + System Instructions + User Question
    - Sends to LLM backend (Ollama local → OpenAI cloud fallback)
    - Returns structured answer with sources

Supported LLM backends (in priority order):
    1. Ollama  — llama3 / mistral (local, free, private)
    2. OpenAI  — gpt-4o-mini (cloud fallback if Ollama is unavailable)

Usage:
    from llm_integration import FinanceRAG
    rag = FinanceRAG()
    answer = rag.ask("What was Apple's revenue in 2022?")
    print(answer["answer"])
    print(answer["sources"])
"""

import os
import sys
import json
import time
import textwrap
import requests
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these if needed)
# ─────────────────────────────────────────────────────────────

# Ollama local server (default install)
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3")          # or "mistral"

# OpenAI cloud fallback
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")              # set in env or .env file
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# RAG settings
TOP_K             = int(os.getenv("TOP_K", "5"))                  # chunks to retrieve
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))   # cap context length

# ─────────────────────────────────────────────────────────────
# STEP 5: RETRIEVER  — thin wrapper around the retriever
# ─────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = TOP_K,
                    company: Optional[str] = None,
                    year: Optional[str]    = None) -> list[dict]:
    """
    Step 5 — Top-K Retrieval.

    Calls the search() function to find the most relevant
    chunks from the FAISS vector database.

    Args:
        query:   Natural-language question from the user.
        top_k:   Number of chunks to return (default 5).
        company: Optional company filter, e.g. "Apple".
        year:    Optional year filter, e.g. "2022".

    Returns:
        List of chunk dicts: {text, company, year, source, score}
        Sorted by cosine-similarity score, highest first.
    """
    try:
        from retriever import search           # vector search module
    except ImportError:
        raise ImportError(
            "Cannot import 'retriever'. "
            "Make sure retriever.py (and vector_db/) are in the same folder, "
            "or add their path to sys.path before calling retrieve_chunks()."
        )

    print(f"\n{'='*55}")
    print(f"  [STEP 5] RETRIEVER — Top-K Search")
    print(f"{'='*55}")
    print(f"  Query   : {query}")
    print(f"  Top-K   : {top_k}")
    if company: print(f"  Company : {company}")
    if year:    print(f"  Year    : {year}")
    print()

    t0 = time.time()
    chunks = search(query, top_k=top_k, company=company, year=year)
    elapsed = time.time() - t0

    if not chunks:
        print("  [WARNING] No chunks returned. Try broadening filters or check the vector_db.")
        return []

    print(f"  Retrieved {len(chunks)} chunks in {elapsed:.2f}s")
    for i, c in enumerate(chunks, 1):
        print(f"  Chunk {i} (score={c['score']:.4f}) | {c['company']} {c['year']} | {c['source']}")
        preview = c["text"][:120].replace("\n", " ")
        print(f"           \"{preview}...\"")

    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 6 — PART A: PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_augmented_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Step 6a — Build Augmented Prompt.

    Constructs the system prompt and user message that will be sent
    to the LLM. Follows the structure from the architecture diagram:
        System Instructions + Retrieved Context + User Question

    Returns:
        (system_prompt, user_message)  — two strings ready for the LLM.
    """
    # --- System prompt (static instructions for the LLM's role) ---
    system_prompt = textwrap.dedent("""
        You are a professional financial analyst AI assistant.

        You will be given RETRIEVED CONTEXT extracted from 10-K annual reports
        of public companies (Apple, Microsoft, Nvidia, Tesla) and other financial data.

        Your task:
        - Answer the user's question using ONLY the information in the retrieved context.
        - Be precise with numbers, dates, and company names.
        - If the answer is not present in the context, say: "I don't have enough information
          in the retrieved documents to answer this question."
        - Never invent figures or speculate beyond what the context states.
        - Structure your answer clearly: lead with the direct answer, then supporting detail.
        - At the end, list the sources (document names) you used.
    """).strip()

    # --- Build context block from retrieved chunks ---
    context_parts = []
    total_chars   = 0

    for i, chunk in enumerate(chunks, 1):
        label   = f"[Chunk {i} | {chunk['company']} {chunk['year']} | Score: {chunk['score']:.3f}]"
        content = chunk["text"].strip()

        block = f"{label}\n{content}"

        # Guard against exceeding the context window
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                block = block[:remaining] + "... [truncated]"
                context_parts.append(block)
            break

        context_parts.append(block)
        total_chars += len(block)

    context_str = "\n\n".join(context_parts)

    # --- User message (context + question, as the LLM sees it) ---
    user_message = textwrap.dedent(f"""
        RETRIEVED CONTEXT:
        {context_str}

        ---
        USER QUESTION:
        {query}

        Please answer using only the context above.
    """).strip()

    return system_prompt, user_message


# ─────────────────────────────────────────────────────────────
# STEP 6 — PART B: LLM BACKENDS
# ─────────────────────────────────────────────────────────────

def _call_ollama(system_prompt: str, user_message: str, model: str) -> str:
    """
    Send prompt to local Ollama server.
    Supports llama3 and mistral (or any other pulled model).
    """
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp = more factual, less creative
            "num_predict": 1024,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out after 120s. Model may still be loading.")
    except KeyError:
        raise RuntimeError(f"Unexpected Ollama response structure: {response.text[:300]}")


def _call_openai(system_prompt: str, user_message: str, model: str) -> str:
    """
    Send prompt to OpenAI API (cloud fallback).
    Requires OPENAI_API_KEY environment variable.
    """
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Set it with: export OPENAI_API_KEY='sk-...'"
        )

    url     = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens":  1024,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"OpenAI API error: {response.status_code} — {response.text[:200]}")
    except requests.exceptions.Timeout:
        raise TimeoutError("OpenAI request timed out after 60s.")


def call_llm(system_prompt: str, user_message: str,
             preferred_backend: str = "ollama") -> tuple[str, str]:
    """
    Step 6b — Send Augmented Prompt to LLM.

    Tries the preferred backend first; falls back to the other.

    Args:
        system_prompt:     System instructions for the LLM.
        user_message:      Context + user question.
        preferred_backend: "ollama" (default) or "openai".

    Returns:
        (raw_answer, backend_used)
    """
    print(f"\n{'='*55}")
    print(f"  [STEP 6] LLM GENERATION (Augmented)")
    print(f"{'='*55}")

    backends = (
        [("ollama", OLLAMA_MODEL), ("openai", OPENAI_MODEL)]
        if preferred_backend == "ollama"
        else [("openai", OPENAI_MODEL), ("ollama", OLLAMA_MODEL)]
    )

    last_error = None
    for backend, model in backends:
        try:
            print(f"  Trying backend: {backend.upper()} ({model}) ...")
            t0 = time.time()

            if backend == "ollama":
                answer = _call_ollama(system_prompt, user_message, model)
            else:
                answer = _call_openai(system_prompt, user_message, model)

            elapsed = time.time() - t0
            print(f"  ✓ Response received in {elapsed:.1f}s from {backend.upper()}")
            return answer, backend

        except (ConnectionError, EnvironmentError) as e:
            print(f"  ✗ {backend.upper()} unavailable: {e}")
            last_error = e
            continue

        except Exception as e:
            print(f"  ✗ {backend.upper()} error: {e}")
            last_error = e
            continue

    raise RuntimeError(
        f"All LLM backends failed. Last error: {last_error}\n"
        "Make sure either Ollama is running (ollama serve) or OPENAI_API_KEY is set."
    )


# ─────────────────────────────────────────────────────────────
# STEP 6 — PART C: RESPONSE FORMATTER (Step 7 from diagram)
# ─────────────────────────────────────────────────────────────

def format_response(query: str, raw_answer: str,
                    chunks: list[dict], backend: str) -> dict:
    """
    Step 7 — Format the LLM output into a structured response.

    Returns a dict with:
        answer    : clean LLM answer (str)
        sources   : deduplicated list of source documents used
        chunks    : full retrieved chunks (for debugging / UI display)
        metadata  : query, backend, scores, counts
    """
    # Deduplicate sources, preserving order
    seen    = set()
    sources = []
    for c in chunks:
        src = f"{c['company']} {c['year']} ({c['source']})"
        if src not in seen:
            seen.add(src)
            sources.append(src)

    return {
        "answer":   raw_answer,
        "sources":  sources,
        "chunks":   chunks,
        "metadata": {
            "query":         query,
            "backend":       backend,
            "top_k":         len(chunks),
            "avg_score":     round(sum(c["score"] for c in chunks) / len(chunks), 4)
                             if chunks else 0.0,
            "top_score":     round(chunks[0]["score"], 4) if chunks else 0.0,
        }
    }


# ─────────────────────────────────────────────────────────────
# MAIN CLASS: FinanceRAG — the clean public API for this module
# ─────────────────────────────────────────────────────────────

class FinanceRAG:
    """
    High-level interface for the Finance RAG pipeline (Steps 5 & 6).

    This is the one class exposes for downstream use
    or to a UI / API wrapper.

    Example:
        rag = FinanceRAG()
        result = rag.ask("What was Apple's total revenue in FY2022?",
                         company="Apple", year="2022")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(self,
                 top_k:     int = TOP_K,
                 backend:   str = "ollama",
                 ollama_model: str = OLLAMA_MODEL,
                 openai_model: str = OPENAI_MODEL):
        """
        Args:
            top_k:         Number of chunks to retrieve (default 5).
            backend:       Preferred LLM backend: "ollama" or "openai".
            ollama_model:  Ollama model name ("llama3" or "mistral").
            openai_model:  OpenAI model name ("gpt-4o-mini").
        """
        self.top_k         = top_k
        self.backend       = backend
        self.ollama_model  = ollama_model
        self.openai_model  = openai_model

        # Override global model names if custom ones were passed
        global OLLAMA_MODEL, OPENAI_MODEL
        OLLAMA_MODEL = ollama_model
        OPENAI_MODEL = openai_model

        print(f"[FinanceRAG] Initialized")
        print(f"  Backend : {backend} (fallback to the other)")
        print(f"  Ollama  : {ollama_model} @ {OLLAMA_BASE_URL}")
        print(f"  OpenAI  : {openai_model} ({'key set' if OPENAI_API_KEY else 'NO KEY — cloud disabled'})")
        print(f"  Top-K   : {top_k}")

    def ask(self, query:   str,
            company:       Optional[str] = None,
            year:          Optional[str] = None,
            top_k:         Optional[int] = None) -> dict:
        """
        Full RAG pipeline: Question → Retriever → Top Chunks → LLM → Answer.

        Args:
            query:   The user's natural-language finance question.
            company: Optional company filter ("Apple", "Microsoft", "Nvidia", "Tesla").
            year:    Optional year filter ("2022", "2023", "2024").
            top_k:   Override default top_k for this call.

        Returns:
            {
              "answer":   str,
              "sources":  list[str],
              "chunks":   list[dict],
              "metadata": dict
            }
        """
        k = top_k or self.top_k

        # ── STEP 5: Retrieve relevant chunks ──────────────────
        chunks = retrieve_chunks(query, top_k=k, company=company, year=year)

        if not chunks:
            return {
                "answer":   "No relevant documents were found in the vector database for your query. "
                            "Try rephrasing, or check that the FAISS index is built and populated.",
                "sources":  [],
                "chunks":   [],
                "metadata": {"query": query, "top_k": 0, "backend": "none"}
            }

        # ── STEP 6a: Build augmented prompt ───────────────────
        system_prompt, user_message = build_augmented_prompt(query, chunks)

        # ── STEP 6b: Send to LLM ──────────────────────────────
        raw_answer, backend_used = call_llm(
            system_prompt, user_message,
            preferred_backend=self.backend
        )

        # ── STEP 7: Format and return structured response ─────
        result = format_response(query, raw_answer, chunks, backend_used)

        # Pretty-print for CLI usage
        self._print_result(result)

        return result

    # ──────────────────────────────────────────────────────────
    def _print_result(self, result: dict) -> None:
        """Print a clean, readable summary of the RAG result to stdout."""
        meta = result["metadata"]
        print(f"\n{'='*55}")
        print(f"  ANSWER  (via {meta['backend'].upper()})")
        print(f"{'='*55}")
        # Wrap long lines for terminal readability
        wrapped = textwrap.fill(result["answer"], width=70,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

        print(f"\n  SOURCES ({len(result['sources'])} document(s)):")
        for s in result["sources"]:
            print(f"    • {s}")

        print(f"\n  STATS  |  top_k={meta['top_k']}  "
              f"top_score={meta['top_score']}  "
              f"avg_score={meta['avg_score']}")
        print(f"{'='*55}\n")
