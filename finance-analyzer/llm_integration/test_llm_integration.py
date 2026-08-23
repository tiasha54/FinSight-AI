"""
test_llm_integration.py
------------------------
LLM Integration: Test Suite

Tests Step 5 (Retriever) and Step 6 (LLM Generation) independently
and together. Designed to run even WITHOUT the FAISS index or live LLM
by using mock stubs where needed.

Run all tests:
    python test_llm_integration.py

Run with real FAISS index (requires the vector_db/ folder/):
    python test_llm_integration.py --real-retriever

Run with live Ollama (requires ollama serve + llama3):
    python test_llm_integration.py --real-llm

Run full end-to-end:
    python test_llm_integration.py --real-retriever --real-llm
"""

import sys
import os
import json
import argparse
import unittest
from unittest.mock import patch, MagicMock

# Allow running from this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_integration import (
    build_augmented_prompt,
    format_response,
    retrieve_chunks,
    call_llm,
    FinanceRAG,
    MAX_CONTEXT_CHARS,
)

# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "text":    "Apple Inc. reported total net sales of $394.3 billion for the fiscal year ended September 24, 2022, representing a 7.8% increase compared to the prior year.",
        "company": "Apple",
        "year":    "2022",
        "source":  "Apple_2022_10K.pdf",
        "score":   0.92,
    },
    {
        "text":    "iPhone net sales were $205.5 billion in fiscal year 2022, representing approximately 52% of total net revenue, compared to $191.97 billion in fiscal year 2021.",
        "company": "Apple",
        "year":    "2022",
        "source":  "Apple_2022_10K.pdf",
        "score":   0.88,
    },
    {
        "text":    "Services revenue for fiscal year 2022 was $78.1 billion, an increase of $7.6 billion or 10.8% from the prior year.",
        "company": "Apple",
        "year":    "2022",
        "source":  "Apple_2022_10K.pdf",
        "score":   0.84,
    },
]

SAMPLE_QUERY   = "What was Apple's total revenue in FY2022?"
SAMPLE_ANSWER  = "Apple's total revenue in fiscal year 2022 was $394.3 billion, a 7.8% increase year-over-year."


# ─────────────────────────────────────────────────────────────
# TEST SUITE
# ─────────────────────────────────────────────────────────────

class TestStep5_Retriever(unittest.TestCase):
    """Tests for Step 5: Retriever (Top-K Search)."""

    def test_retrieve_chunks_calls_search(self):
        """retrieve_chunks() should call the search() with correct args."""
        mock_search = MagicMock(return_value=SAMPLE_CHUNKS)
        mock_module = MagicMock()
        mock_module.search = mock_search

        with patch.dict("sys.modules", {"retriever": mock_module}):
            # Force re-import so our mock is used
            import importlib
            import llm_integration
            original_retrieve = llm_integration.retrieve_chunks

            # Patch at module level
            with patch("llm_integration.retrieve_chunks", wraps=original_retrieve):
                with patch.dict("sys.modules", {"retriever": mock_module}):
                    pass  # retriever is already mocked above

        # Direct call with mock
        with patch.dict("sys.modules", {"retriever": mock_module}):
            result = retrieve_chunks(SAMPLE_QUERY, top_k=3, company="Apple", year="2022")

        mock_search.assert_called_once_with(SAMPLE_QUERY, top_k=3, company="Apple", year="2022")
        self.assertEqual(len(result), 3)

    def test_retrieve_chunks_returns_list(self):
        """retrieve_chunks() should return a list of dicts with expected keys."""
        mock_module = MagicMock()
        mock_module.search = MagicMock(return_value=SAMPLE_CHUNKS)

        with patch.dict("sys.modules", {"retriever": mock_module}):
            result = retrieve_chunks(SAMPLE_QUERY, top_k=3)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        required_keys = {"text", "company", "year", "source", "score"}
        for chunk in result:
            self.assertTrue(required_keys.issubset(chunk.keys()),
                            f"Chunk missing keys: {required_keys - chunk.keys()}")

    def test_retrieve_chunks_empty_returns_empty_list(self):
        """retrieve_chunks() should return [] when no results found."""
        mock_module = MagicMock()
        mock_module.search = MagicMock(return_value=[])

        with patch.dict("sys.modules", {"retriever": mock_module}):
            result = retrieve_chunks("totally unrelated query", top_k=5)

        self.assertEqual(result, [])

    def test_retrieve_chunks_missing_retriever_raises(self):
        """retrieve_chunks() raises ImportError if retriever module is absent."""
        import importlib
        import llm_integration
        # Remove retriever from sys.modules if present
        sys.modules.pop("retriever", None)
        with self.assertRaises(ImportError):
            llm_integration.retrieve_chunks(SAMPLE_QUERY, top_k=3)

    def test_scores_descending(self):
        """Scores should be descending (retriever contract)."""
        chunks_ordered = sorted(SAMPLE_CHUNKS, key=lambda c: c["score"], reverse=True)
        scores = [c["score"] for c in chunks_ordered]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestStep6a_PromptBuilder(unittest.TestCase):
    """Tests for Step 6a: Augmented Prompt Construction."""

    def test_build_returns_two_strings(self):
        """build_augmented_prompt() should return (str, str)."""
        sys_p, user_m = build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)
        self.assertIsInstance(sys_p, str)
        self.assertIsInstance(user_m, str)

    def test_system_prompt_contains_role(self):
        """System prompt must describe the LLM's role as a financial analyst."""
        sys_p, _ = build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)
        self.assertIn("financial analyst", sys_p.lower())

    def test_system_prompt_contains_fallback_instruction(self):
        """System prompt must tell LLM to say 'I don't have enough information'."""
        sys_p, _ = build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)
        self.assertIn("don't have enough information", sys_p)

    def test_user_message_contains_query(self):
        """User message must include the original query."""
        _, user_m = build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)
        self.assertIn(SAMPLE_QUERY, user_m)

    def test_user_message_contains_chunk_text(self):
        """User message must embed chunk text from retrieved results."""
        _, user_m = build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)
        self.assertIn("394.3 billion", user_m)   # from chunk 1 text

    def test_user_message_contains_chunk_labels(self):
        """User message should label each chunk with company + year."""
        _, user_m = build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)
        self.assertIn("Apple", user_m)
        self.assertIn("2022",  user_m)

    def test_context_length_capped(self):
        """Context must not exceed MAX_CONTEXT_CHARS to avoid token overflow."""
        very_long_chunks = [
            {**c, "text": "X" * 5000}
            for c in SAMPLE_CHUNKS
        ]
        _, user_m = build_augmented_prompt(SAMPLE_QUERY, very_long_chunks)
        self.assertLessEqual(len(user_m), MAX_CONTEXT_CHARS + 500,
                             "User message is suspiciously long — context cap may not be working")

    def test_empty_chunks_handled(self):
        """build_augmented_prompt() must not crash with zero chunks."""
        try:
            sys_p, user_m = build_augmented_prompt(SAMPLE_QUERY, [])
            self.assertIsInstance(sys_p,  str)
            self.assertIsInstance(user_m, str)
        except Exception as e:
            self.fail(f"Crashed with empty chunks: {e}")


class TestStep6b_LLMBackends(unittest.TestCase):
    """Tests for Step 6b: LLM Backend calls."""

    def _build_prompt(self):
        return build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)

    def test_call_llm_returns_string_and_backend(self):
        """call_llm() should return (str, str) on success."""
        sys_p, user_m = self._build_prompt()

        with patch("llm_integration._call_ollama", return_value=SAMPLE_ANSWER):
            answer, backend = call_llm(sys_p, user_m, preferred_backend="ollama")

        self.assertIsInstance(answer,  str)
        self.assertIsInstance(backend, str)
        self.assertGreater(len(answer), 0)
        self.assertEqual(backend, "ollama")

    def test_call_llm_falls_back_to_openai(self):
        """call_llm() should try OpenAI if Ollama is unreachable."""
        sys_p, user_m = self._build_prompt()

        with patch("llm_integration._call_ollama",
                   side_effect=ConnectionError("Ollama not running")):
            with patch("llm_integration._call_openai", return_value=SAMPLE_ANSWER):
                with patch("llm_integration.OPENAI_API_KEY", "sk-fake"):
                    answer, backend = call_llm(sys_p, user_m, preferred_backend="ollama")

        self.assertEqual(backend, "openai")
        self.assertEqual(answer, SAMPLE_ANSWER)

    def test_call_llm_raises_if_all_fail(self):
        """call_llm() should raise RuntimeError if both backends fail."""
        sys_p, user_m = self._build_prompt()

        with patch("llm_integration._call_ollama",
                   side_effect=ConnectionError("No Ollama")):
            with patch("llm_integration._call_openai",
                       side_effect=EnvironmentError("No API key")):
                with self.assertRaises(RuntimeError):
                    call_llm(sys_p, user_m)

    def test_openai_preferred_when_specified(self):
        """When backend='openai', OpenAI should be tried first."""
        sys_p, user_m = self._build_prompt()
        call_order    = []

        def fake_openai(*_): call_order.append("openai"); return SAMPLE_ANSWER
        def fake_ollama(*_): call_order.append("ollama"); return SAMPLE_ANSWER

        with patch("llm_integration._call_openai", side_effect=fake_openai):
            with patch("llm_integration._call_ollama", side_effect=fake_ollama):
                with patch("llm_integration.OPENAI_API_KEY", "sk-fake"):
                    call_llm(sys_p, user_m, preferred_backend="openai")

        self.assertEqual(call_order[0], "openai",
                         "OpenAI should be called first when backend='openai'")


class TestStep6c_ResponseFormatter(unittest.TestCase):
    """Tests for format_response() / Step 7."""

    def test_format_response_keys(self):
        """format_response() must return dict with required keys."""
        result = format_response(SAMPLE_QUERY, SAMPLE_ANSWER, SAMPLE_CHUNKS, "ollama")
        required = {"answer", "sources", "chunks", "metadata"}
        self.assertTrue(required.issubset(result.keys()))

    def test_answer_preserved(self):
        """Answer text must be passed through unchanged."""
        result = format_response(SAMPLE_QUERY, SAMPLE_ANSWER, SAMPLE_CHUNKS, "ollama")
        self.assertEqual(result["answer"], SAMPLE_ANSWER)

    def test_sources_deduplicated(self):
        """Sources list should not have duplicates."""
        # Two chunks from the same source
        result = format_response(SAMPLE_QUERY, SAMPLE_ANSWER, SAMPLE_CHUNKS, "ollama")
        self.assertEqual(len(result["sources"]), len(set(result["sources"])))

    def test_sources_contains_company_and_year(self):
        """Each source string should mention company and year."""
        result = format_response(SAMPLE_QUERY, SAMPLE_ANSWER, SAMPLE_CHUNKS, "ollama")
        for src in result["sources"]:
            self.assertIn("Apple", src)
            self.assertIn("2022",  src)

    def test_metadata_backend(self):
        """Metadata must record which backend was used."""
        result = format_response(SAMPLE_QUERY, SAMPLE_ANSWER, SAMPLE_CHUNKS, "openai")
        self.assertEqual(result["metadata"]["backend"], "openai")

    def test_metadata_scores(self):
        """Metadata must include top_score and avg_score."""
        result = format_response(SAMPLE_QUERY, SAMPLE_ANSWER, SAMPLE_CHUNKS, "ollama")
        self.assertIn("top_score", result["metadata"])
        self.assertIn("avg_score", result["metadata"])
        self.assertAlmostEqual(result["metadata"]["top_score"], 0.92)


class TestFinanceRAGClass(unittest.TestCase):
    """Integration tests for the FinanceRAG class (mocked backends)."""

    def _make_rag(self):
        return FinanceRAG(top_k=3, backend="ollama")

    def test_ask_returns_structured_result(self):
        """FinanceRAG.ask() must return a dict with answer, sources, chunks."""
        rag = self._make_rag()

        mock_retriever = MagicMock()
        mock_retriever.search = MagicMock(return_value=SAMPLE_CHUNKS)

        with patch.dict("sys.modules", {"retriever": mock_retriever}):
            with patch("llm_integration._call_ollama", return_value=SAMPLE_ANSWER):
                result = rag.ask(SAMPLE_QUERY, company="Apple", year="2022")

        self.assertIn("answer",   result)
        self.assertIn("sources",  result)
        self.assertIn("chunks",   result)
        self.assertIn("metadata", result)
        self.assertEqual(result["answer"], SAMPLE_ANSWER)

    def test_ask_no_chunks_returns_graceful_message(self):
        """FinanceRAG.ask() must return a helpful message if no chunks found."""
        rag = self._make_rag()
        mock_retriever = MagicMock()
        mock_retriever.search = MagicMock(return_value=[])

        with patch.dict("sys.modules", {"retriever": mock_retriever}):
            result = rag.ask("something completely irrelevant")

        self.assertIn("No relevant documents", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_ask_passes_filters(self):
        """FinanceRAG.ask() must pass company and year filters to retriever."""
        rag = self._make_rag()
        mock_retriever = MagicMock()
        mock_search = MagicMock(return_value=SAMPLE_CHUNKS)
        mock_retriever.search = mock_search

        with patch.dict("sys.modules", {"retriever": mock_retriever}):
            with patch("llm_integration._call_ollama", return_value=SAMPLE_ANSWER):
                rag.ask(SAMPLE_QUERY, company="Apple", year="2022", top_k=3)

        mock_search.assert_called_once_with(
            SAMPLE_QUERY, top_k=3, company="Apple", year="2022"
        )


# ─────────────────────────────────────────────────────────────
# LIVE / INTEGRATION TESTS (opt-in via CLI flags)
# ─────────────────────────────────────────────────────────────

class LiveRetrieverTest(unittest.TestCase):
    """Runs against the real FAISS index — only if --real-retriever is passed."""

    def test_real_retrieval(self):
        results = retrieve_chunks("What was Apple's total revenue in 2022?",
                                  top_k=5, company="Apple")
        self.assertGreater(len(results), 0,
                           "Real retriever returned no results")
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         "Scores not in descending order")
        print(f"\n  [LIVE] Retrieved {len(results)} chunks")
        print(f"  Top chunk score: {results[0]['score']:.4f}")
        print(f"  Preview: {results[0]['text'][:120]}...")


class LiveLLMTest(unittest.TestCase):
    """Runs against real Ollama — only if --real-llm is passed."""

    def test_real_llm_call(self):
        sys_p, user_m = build_augmented_prompt(SAMPLE_QUERY, SAMPLE_CHUNKS)
        answer, backend = call_llm(sys_p, user_m, preferred_backend="ollama")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 20, "LLM returned suspiciously short answer")
        print(f"\n  [LIVE] Backend: {backend}")
        print(f"  Answer preview: {answer[:200]}...")


class LiveEndToEndTest(unittest.TestCase):
    """Full pipeline test — requires both FAISS index and Ollama."""

    def test_full_pipeline(self):
        rag    = FinanceRAG(top_k=5, backend="ollama")
        result = rag.ask("What was Apple's total revenue in FY2022?",
                         company="Apple", year="2022")
        self.assertIn("answer",  result)
        self.assertGreater(len(result["answer"]), 20)
        self.assertGreater(len(result["sources"]), 0)
        print(f"\n  [E2E] Answer: {result['answer'][:200]}")
        print(f"  [E2E] Sources: {result['sources']}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine — Test Suite")
    parser.add_argument("--real-retriever", action="store_true",
                        help="Run live retrieval tests (needs vector_db/)")
    parser.add_argument("--real-llm",       action="store_true",
                        help="Run live LLM tests (needs ollama serve + llama3)")
    args, remaining = parser.parse_known_args()

    # Build test suite
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Always run unit tests
    for cls in [TestStep5_Retriever, TestStep6a_PromptBuilder,
                TestStep6b_LLMBackends, TestStep6c_ResponseFormatter,
                TestFinanceRAGClass]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Conditionally add live tests
    if args.real_retriever and not args.real_llm:
        suite.addTests(loader.loadTestsFromTestCase(LiveRetrieverTest))
    if args.real_llm and not args.real_retriever:
        suite.addTests(loader.loadTestsFromTestCase(LiveLLMTest))
    if args.real_retriever and args.real_llm:
        suite.addTests(loader.loadTestsFromTestCase(LiveEndToEndTest))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*55)
    if result.wasSuccessful():
        print("  ALL TESTS PASSED ✓")
    else:
        print(f"  {len(result.failures)} FAILURES  {len(result.errors)} ERRORS")
    print("="*55)

    sys.exit(0 if result.wasSuccessful() else 1)
