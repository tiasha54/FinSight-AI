"""
ingestion.py
------------
Data Ingestion Pipeline
Reads all PDFs from reports/ folder, extracts text,
cleans it, chunks it, and saves to chunks/ folder for the embedding step.
"""

import os
import json
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHUNKS_DIR  = os.path.join(BASE_DIR, "chunks")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# Create output folders if they don't exist
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


print("=" * 60)
print("  AI Finance Analyzer — PDF Ingestion Pipeline")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 1: Get all PDFs from reports/ folder
# ─────────────────────────────────────────────
print("\n[1/5] Scanning reports/ folder for PDFs...")
pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")]

if len(pdf_files) == 0:
    print("      ERROR: No PDFs found in reports/ folder!")
    print("      Please add your PDF files and run again.")
    exit()

print(f"      Found {len(pdf_files)} PDF files:")
for f in pdf_files:
    print(f"        - {f}")


# ─────────────────────────────────────────────
# STEP 2: Extract text from each PDF
# ─────────────────────────────────────────────
print("\n[2/5] Extracting text from PDFs...")

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file page by page."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        except Exception as e:
            print(f"        WARNING: Could not read page {page_num} — {e}")
    return full_text

extracted_data = {}

for pdf_file in pdf_files:
    pdf_path = os.path.join(REPORTS_DIR, pdf_file)
    print(f"      Processing: {pdf_file}")
    text = extract_text_from_pdf(pdf_path)
    if text.strip():
        extracted_data[pdf_file] = text
        print(f"        Extracted {len(text)} characters")
    else:
        print(f"        WARNING: No text extracted from {pdf_file}")

print(f"\n      Successfully extracted text from {len(extracted_data)} PDFs")


# ─────────────────────────────────────────────
# STEP 3: Clean extracted text
# ─────────────────────────────────────────────
print("\n[3/5] Cleaning extracted text...")

def clean_text(text):
    """Clean raw PDF text."""
    # Remove extra whitespace and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers (standalone numbers)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove special characters but keep finance symbols
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\$\%\/\&]', '', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

cleaned_data = {}
for filename, text in extracted_data.items():
    cleaned_data[filename] = clean_text(text)
    print(f"      Cleaned: {filename}")

print("      Done")


# ─────────────────────────────────────────────
# STEP 4: Chunk the cleaned text
# Each chunk = 500 characters, 100 overlap
# ─────────────────────────────────────────────
print("\n[4/5] Chunking text for the embedding step...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

all_chunks = []

for filename, text in cleaned_data.items():
    # Extract company name and year from filename
    # Expected format: apple_10k_2023.pdf
    parts = filename.replace(".pdf", "").split("_")
    company = parts[0] if len(parts) > 0 else "unknown"
    year    = parts[2] if len(parts) > 2 else "unknown"

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id"  : f"{company}_{year}_{i}",
            "company"   : company,
            "year"      : year,
            "source"    : filename,
            "chunk_num" : i,
            "text"      : chunk
        })

    print(f"      {filename} → {len(chunks)} chunks")

print(f"\n      Total chunks created: {len(all_chunks)}")


# ─────────────────────────────────────────────
# STEP 5: Save chunks to chunks/ folder
# ─────────────────────────────────────────────
print("\n[5/5] Saving chunks...")

# Save all chunks in one JSON file for the embedding step
output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"      Saved: chunks/all_chunks.json")

# Also save a summary in data/ folder
summary = {
    "total_pdfs"   : len(extracted_data),
    "total_chunks" : len(all_chunks),
    "companies"    : list(set(c["company"] for c in all_chunks)),
    "years"        : list(set(c["year"] for c in all_chunks)),
    "files"        : list(extracted_data.keys())
}

summary_path = os.path.join(DATA_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"      Saved: data/summary.json")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  INGESTION COMPLETE")
print("=" * 60)
print(f"\n  PDFs processed : {len(extracted_data)}")
print(f"  Total chunks   : {len(all_chunks)}")
print(f"  Companies      : {summary['companies']}")
print(f"  Years          : {summary['years']}")
print(f"\n  Output for the embedding step → chunks/all_chunks.json")
print(f"  Summary        → data/summary.json")
print("\n  Ingestion complete. chunks/all_chunks.json is ready for embedding.")