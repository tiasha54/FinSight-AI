# FinSight-AI

A RAG (Retrieval-Augmented Generation) pipeline that answers questions over
company 10-K filings (Apple, Microsoft, Nvidia, Tesla) and personal/household
finance datasets.

```
Raw PDFs / CSVs
      │
      ▼
┌─────────────────────────┐
│ Data Collection          │  Collect & clean 10-K PDFs and finance CSVs,
│                          │  extract + chunk text → chunks/all_chunks.json
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│ Embeddings / Vector DB   │  Embed chunks (BAAI/bge-small-en-v1.5),
│                          │  build FAISS index → vector_db/
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│ RAG Engine               │  Retrieve top-k chunks, build prompt,
│                          │  query Ollama/OpenAI → structured answer
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│ Streamlit UI              │  Chat-style frontend wrapping FinanceRAG
└─────────────────────────┘
```

## Repo structure

```
finance-analyzer/
├── data_collection/   # PDF/CSV ingestion, cleaning
├── vector_db/         # Embedding + FAISS index building, retriever
├── llm_integration/   # RAG query engine (Ollama/OpenAI) + CLI + tests
├── streamlit_app/     # Streamlit frontend (in progress)
└── docs/              # Project workflow doc
```

## Status

| Module | Status |
|--------|--------|
| Data collection & ingestion | Done |
| Embeddings & FAISS index | Done |
| LLM integration (retriever + generation) | Done |
| Streamlit app | In progress — see `streamlit_app/app.py` |

## Setup

```bash
pip install -r requirements.txt
```

See each module folder for specific instructions
(`llm_integration/README.md` has the most detail: Ollama setup,
CLI usage, output format, etc.). See `data_collection/data/` for
notes on the full raw datasets (not committed — see below).

## Data

Only small samples of raw/cleaned CSVs are committed
(`data_collection/data/samples/`). Full datasets and the 10-K PDFs
(Apple/Microsoft/Nvidia/Tesla, 2020–2025) are too large for git — download
them from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar), and
regenerate `chunks/` and `vector_db/` locally:

```bash
python data_collection/ingestion.py
python vector_db/embed_chunks.py
python vector_db/build_faiss_index.py
```

## Running the RAG pipeline

```bash
python llm_integration/rag_runner.py "What was Apple's revenue in FY2022?"
```

## Running the Streamlit app (once built)

```bash
streamlit run streamlit_app/app.py
```
