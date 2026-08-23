"""
rag_runner.py
--------------
LLM Integration: Command-Line Runner

The simplest way to run the full RAG pipeline end-to-end.

Usage (single question):
    python rag_runner.py "What was Apple's revenue in FY2022?"
    python rag_runner.py "Compare Nvidia and Microsoft gross margins" --top-k 8
    python rag_runner.py "Tesla's net income 2023?" --company Tesla --year 2023
    python rag_runner.py "Revenue?" --backend openai

Usage (interactive chat loop):
    python rag_runner.py

Options:
    --company   Filter results to one company   (Apple | Microsoft | Nvidia | Tesla)
    --year      Filter results to one year       (2022 | 2023 | 2024)
    --top-k     Number of chunks to retrieve    (default: 5)
    --backend   LLM backend to prefer           (ollama | openai)
    --model     Override Ollama model name      (llama3 | mistral)
    --json      Output raw JSON instead of pretty-print
"""

import sys
import os
import json
import argparse

# Allow running from any directory by adding this file's folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_integration import FinanceRAG


BANNER = """
╔══════════════════════════════════════════════════════╗
║       AI Finance Analyzer — RAG Pipeline CLI         ║
║       Step 5 (Retriever) + Step 6 (LLM Gen)         ║
╚══════════════════════════════════════════════════════╝
  Backends : Ollama (llama3 / mistral)  →  OpenAI fallback
  Data     : Apple, Microsoft, Nvidia, Tesla — 10-K 2022-2024
  Type  'exit' or 'quit' to stop the session.
  Type  'help' to see filter commands.
"""

HELP_TEXT = """
FILTER COMMANDS (prefix your question with these):
  company:<name>  e.g.  company:Apple  What was the revenue?
  year:<yyyy>     e.g.  year:2023      What was the revenue?
  top_k:<n>       e.g.  top_k:8        What were the risk factors?

You can combine them:
  company:Nvidia year:2024  What was the gross margin?

Or just type a plain question — no filter needed:
  What was Microsoft's net income in FY2023?
"""


def parse_inline_filters(raw_query: str):
    """
    Extract optional inline filters from the query string.
    Returns (clean_query, company, year, top_k_override).
    Example:
        "company:Apple year:2022 What was revenue?"
        → ("What was revenue?", "Apple", "2022", None)
    """
    import re
    company    = None
    year       = None
    top_k_ovr  = None

    # Extract company:XYZ
    m = re.search(r'\bcompany:(\w+)', raw_query, re.IGNORECASE)
    if m:
        company   = m.group(1).capitalize()
        raw_query = raw_query[:m.start()] + raw_query[m.end():]

    # Extract year:YYYY
    m = re.search(r'\byear:(\d{4})', raw_query, re.IGNORECASE)
    if m:
        year      = m.group(1)
        raw_query = raw_query[:m.start()] + raw_query[m.end():]

    # Extract top_k:N
    m = re.search(r'\btop_k:(\d+)', raw_query, re.IGNORECASE)
    if m:
        top_k_ovr = int(m.group(1))
        raw_query = raw_query[:m.start()] + raw_query[m.end():]

    return raw_query.strip(), company, year, top_k_ovr


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Finance RAG Pipeline — RAG Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("query",    nargs="?",  default=None,
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--company",default=None,
                        help="Filter by company: Apple | Microsoft | Nvidia | Tesla")
    parser.add_argument("--year",   default=None,
                        help="Filter by year: 2022 | 2023 | 2024")
    parser.add_argument("--top-k",  type=int, default=5, dest="top_k",
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--backend",default="ollama",
                        choices=["ollama", "openai"],
                        help="Preferred LLM backend (default: ollama)")
    parser.add_argument("--model",  default=None,
                        help="Override Ollama model (e.g. mistral)")
    parser.add_argument("--json",   action="store_true",
                        help="Output raw JSON (machine-readable)")
    args = parser.parse_args()

    # Build RAG instance
    rag = FinanceRAG(
        top_k   = args.top_k,
        backend = args.backend,
        ollama_model = args.model or "llama3",
    )

    # ── SINGLE-QUERY MODE ────────────────────────────────────
    if args.query:
        result = rag.ask(
            args.query,
            company = args.company,
            year    = args.year,
            top_k   = args.top_k,
        )
        if args.json:
            # Remove chunk text from JSON to keep output readable
            for c in result["chunks"]:
                c["text"] = c["text"][:200] + "..."
            print(json.dumps(result, indent=2))
        return

    # ── INTERACTIVE MODE ─────────────────────────────────────
    print(BANNER)

    while True:
        try:
            raw = input("  ❯ Ask a finance question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            print("  Goodbye!")
            break
        if raw.lower() == "help":
            print(HELP_TEXT)
            continue

        # Parse any inline filters the user typed
        query, company, year, top_k_ovr = parse_inline_filters(raw)

        if not query:
            print("  (No question detected after filters — please include a question.)")
            continue

        try:
            result = rag.ask(
                query,
                company = company or args.company,
                year    = year    or args.year,
                top_k   = top_k_ovr,
            )
        except RuntimeError as e:
            print(f"\n  [ERROR] {e}\n")
            continue


if __name__ == "__main__":
    main()
