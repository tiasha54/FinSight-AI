"""
test_retrieval.py
-------------------
Vector Database (Test Script)

Verifies the retrieval pipeline actually works before handing off
to the RAG engine. Run this after build_faiss_index.py.

This checks:
  1. The index loads and has the expected number of vectors
  2. A generic finance question returns results
  3. A company-filtered question only returns that company's chunks
  4. A year-filtered question only returns that year's chunks
     (will show a clear FAIL until a known year-field bug is fixed)
  5. Scores are sorted highest-to-lowest
"""

from retriever import search, _index, _metadata

print("=" * 60)
print("  Retrieval Test Suite")
print("=" * 60)

passed = 0
failed = 0

def check(label, condition):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if condition:
        passed += 1
    else:
        failed += 1


# ── Test 1: Index loaded correctly ──────────────────────────
print("\n[Test 1] Index sanity check")
check("Index has at least 1000 vectors", _index.ntotal > 1000)
check("Metadata length matches index size", len(_metadata) == _index.ntotal)


# ── Test 2: Basic search returns results ────────────────────
print("\n[Test 2] Generic query returns results")
results = search("What are the main risk factors?", top_k=5)
check("Returns 5 results", len(results) == 5)
check("Each result has required fields",
      all(set(r.keys()) >= {"text", "company", "year", "source", "score"} for r in results))


# ── Test 3: Company filter works ────────────────────────────
print("\n[Test 3] Company-filtered query")
results = search("What was the total revenue?", top_k=5, company="Apple")
check("Returns results", len(results) > 0)
check("All results are from Apple",
      all(r["company"].lower() == "apple" for r in results))


# ── Test 4: Year filter (expected to fail until that bug is fixed) ──
print("\n[Test 4] Year-filtered query")
results = search("What was the total revenue?", top_k=5, company="Apple", year="2022")
if len(results) == 0:
    print("  [FAIL] No results for company=Apple, year=2022")
    print("         This is EXPECTED if a known year-field bug isn't fixed yet")
    print("         (every chunk's 'year' currently reads '10K' instead of the real year)")
    failed += 1
else:
    check("All results are from year 2022", all(str(r["year"]) == "2022" for r in results))


# ── Test 5: Results are sorted by score, descending ─────────
print("\n[Test 5] Results sorted by relevance")
results = search("What is the company's debt level?", top_k=5)
scores = [r["score"] for r in results]
check("Scores are in descending order", scores == sorted(scores, reverse=True))


# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    print("\n  Review FAIL lines above. The year-filter failure is a known")
    print("  upstream issue in the ingestion script, not a bug in retriever.py.")
