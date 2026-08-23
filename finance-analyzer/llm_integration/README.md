# LLM Integration
## Steps 5 (Retriever) & 6 (LLM Generation) of the Finance RAG Pipeline

---

## What This Module Does

```
User Question
     │
     ▼
┌─────────────────────────────────────────────┐
│  STEP 5 — RETRIEVER (Top-K Search)          │
│  • Embed query using BAAI/bge-small-en-v1.5 │
│  • Search FAISS index (the vector_db/ folder) │
│  • Return Top-K most relevant text chunks   │
└─────────────────┬───────────────────────────┘
                  │  Top-K Chunks (with scores)
                  ▼
┌─────────────────────────────────────────────┐
│  STEP 6 — LLM GENERATION (Augmented)        │
│  • Build prompt: System + Context + Question│
│  • Send to Ollama (llama3/mistral) locally  │
│  • Fallback to OpenAI if Ollama unavailable │
│  • Return structured answer + sources       │
└─────────────────────────────────────────────┘
                  │
                  ▼
         Structured Answer
         (answer, sources, metadata)
```

---

## Folder Structure

```
llm_integration/
│
├── llm_integration.py        ← CORE: Steps 5 & 6 implementation
├── rag_runner.py             ← CLI: interactive + single-query runner
├── test_llm_integration.py   ← Tests: unit + live integration
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
└── vector_db/                ← PLACE vector_db/ HERE
    ├── faiss_index.bin
    └── metadata.json
    (+ retriever.py in the same folder)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Ollama (local LLM — recommended)
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (pick one)
ollama pull llama3      # ~4.7 GB — better quality
ollama pull mistral     # ~4.1 GB — faster

# Start the server (runs on http://localhost:11434)
ollama serve
```

### 3. (Optional) OpenAI cloud fallback
```bash
export OPENAI_API_KEY="sk-..."
```
Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

### 4. Link the vector DB files
```bash
# Copy the vector_db/ output folder into this folder
cp -r ../vector_db/output/vector_db/ ./vector_db/
cp ../vector_db/retriever.py  ./retriever.py
```

---

## Usage

### Python API
```python
from llm_integration import FinanceRAG

rag = FinanceRAG()   # uses Ollama llama3 by default

# Basic question
result = rag.ask("What was Apple's total revenue in FY2022?")
print(result["answer"])
print(result["sources"])

# With filters (faster + more accurate)
result = rag.ask(
    "What were the main risk factors?",
    company="Apple",
    year="2022",
    top_k=7
)

# Switch to Mistral
rag_mistral = FinanceRAG(ollama_model="mistral")

# Switch to OpenAI (needs OPENAI_API_KEY)
rag_openai = FinanceRAG(backend="openai")
```

### CLI — Single question
```bash
python rag_runner.py "What was Apple's revenue in FY2022?"

# With filters
python rag_runner.py "What was the net income?" --company Apple --year 2022

# More chunks (better context for complex questions)
python rag_runner.py "Compare iPhone vs Services revenue" --top-k 8

# Use Mistral instead of Llama3
python rag_runner.py "Revenue trends?" --model mistral

# Force OpenAI
python rag_runner.py "What were the risk factors?" --backend openai

# JSON output (for piping to other tools)
python rag_runner.py "Revenue?" --json
```

### CLI — Interactive mode
```bash
python rag_runner.py
```
Then just type questions. You can use inline filters:
```
❯ company:Apple year:2022  What was the iPhone revenue?
❯ company:Nvidia  What was the data center revenue in 2024?
❯ top_k:8  What were all the major risk factors across companies?
```

---

## Running Tests

```bash
# Unit tests only (no FAISS index or Ollama needed)
python test_llm_integration.py

# With real FAISS retriever
python test_llm_integration.py --real-retriever

# With real Ollama LLM
python test_llm_integration.py --real-llm

# Full end-to-end (both required)
python test_llm_integration.py --real-retriever --real-llm
```

---

## Configuration (Environment Variables)

| Variable           | Default                   | Description                       |
|--------------------|---------------------------|-----------------------------------|
| `OLLAMA_BASE_URL`  | `http://localhost:11434`  | Ollama server URL                 |
| `OLLAMA_MODEL`     | `llama3`                  | Model name (`llama3` / `mistral`) |
| `OPENAI_API_KEY`   | *(unset)*                 | OpenAI key — enables cloud backup |
| `OPENAI_MODEL`     | `gpt-4o-mini`             | OpenAI model name                 |
| `TOP_K`            | `5`                       | Default number of chunks          |
| `MAX_CONTEXT_CHARS`| `6000`                    | Context size cap (token safety)   |

---

## Output Format

```python
{
  "answer": "Apple's total revenue in fiscal year 2022 was $394.3 billion...",

  "sources": [
    "Apple 2022 (Apple_2022_10K.pdf)",
    "Apple 2023 (Apple_2023_10K.pdf)"
  ],

  "chunks": [
    {
      "text":    "Apple Inc. reported total net sales of $394.3 billion...",
      "company": "Apple",
      "year":    "2022",
      "source":  "Apple_2022_10K.pdf",
      "score":   0.9234
    },
    ...
  ],

  "metadata": {
    "query":      "What was Apple's total revenue in FY2022?",
    "backend":    "ollama",
    "top_k":      5,
    "top_score":  0.9234,
    "avg_score":  0.8712
  }
}
```

---

## How It Connects to the Team

| Stage   | Responsibility              | Output handed to the RAG engine |
|---------|------------------------------|----------------------------------|
| Data collection | Data ingestion + chunking   | `all_chunks.json`             |
| Vector DB | Embeddings + FAISS index    | `vector_db/` + `retriever.py` |
| **LLM integration** | **LLM integration** | `llm_integration.py` (this)   |

The `FinanceRAG` class is the handoff to any UI, API, or notebook layer that follows.
