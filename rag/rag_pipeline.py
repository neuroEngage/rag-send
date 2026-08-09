"""
RAG Pipeline Orchestrator — Yelp

Single entry point that wires together:
    Retriever → Context Builder → Generator → Answer + Sources

Usage (terminal):
    python -m rag.rag_pipeline "Why are customers unhappy with this restaurant?"

Usage (from code):
    from rag.rag_pipeline import answer_question
    result = answer_question("Best pizza in Philadelphia?")
    print(result["answer"])
    print(result["sources"])
"""

import sys
import os
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.query_rag import YelpRAGRetriever
from rag.context_builder import build_context
from rag.generator import generate_answer

# Module-level retriever (loaded once, reused across calls)
_retriever = None


def _get_retriever() -> YelpRAGRetriever:
    """Lazy-loads the retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = YelpRAGRetriever()
    return _retriever


def answer_question(
    question: str,
    doc_type: str = "review",
    city: str = None,
    min_stars: float = None,
    sentiment: str = None,
    business_id: str = None,
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """
    Full RAG pipeline: question → FAISS → context → LLM → answer + sources.

    Args:
        question:    The user's natural language question.
        doc_type:    "review", "business", or "all" (default: "review").
        city:        Optional city filter (e.g. "Philadelphia").
        min_stars:   Optional minimum star rating filter.
        sentiment:   Optional sentiment filter ("positive" / "negative").
        business_id: Scope retrieval to a single business (critical for Owner Copilot).
        top_k:       Number of documents to retrieve (default: 5).
        llm_model:   OpenAI model to use (default: "gpt-4o-mini").

    Returns:
        {
            "answer":  str  — LLM-generated answer,
            "sources": list — retrieved results with metadata,
            "context": str  — formatted context passed to the LLM,
        }
    """
    retriever = _get_retriever()

    # 1. Retrieve — pass business_id so FAISS is scoped to the right business
    results = retriever.search(
        query_text=question,
        doc_type=doc_type,
        city=city,
        min_stars=min_stars,
        sentiment=sentiment,
        business_id=business_id,
        top_k=top_k,
    )

    # 2. Build context
    context = build_context(results)

    # 3. Generate answer
    answer = generate_answer(question=question, context=context, model=llm_model)

    return {
        "answer": answer,
        "sources": results,
        "context": context,
    }


# ---------------------------------------------------------------------------
# Terminal test runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What do customers say about the food quality and service?"

    print("\n" + "=" * 70)
    print("  Yelp RAG Pipeline -- Terminal Test")
    print("=" * 70)
    print(f"  QUESTION: {test_question}")
    print("=" * 70)

    output = answer_question(test_question, top_k=5)

    print("\n[AI ANSWER]")
    print("-" * 70)
    print(output["answer"])

    print("\n[SOURCES]")
    print("-" * 70)
    for i, src in enumerate(output["sources"], 1):
        biz = src.get("business_name", "Unknown")
        city = src.get("city", "")
        stars = src.get("stars") or src.get("business_rating", "?")
        sentiment = src.get("sentiment", "")
        doc_type = src.get("document_type", "").upper()
        score = src.get("score", 0.0)
        print(f"  [{i}] {doc_type} | {biz} ({city}) | {stars} stars | {sentiment} | sim={score:.4f}")
