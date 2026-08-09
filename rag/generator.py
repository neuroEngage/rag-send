"""
Generator — Yelp RAG Pipeline

Single responsibility: take a question + context string and return an
LLM-generated answer via the OpenAI API.

Does NOT:
- search FAISS
- query DuckDB
- load embeddings
- handle Streamlit UI
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Strict RAG system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Yelp Business Intelligence Assistant.

Answer the user's question using ONLY the provided Yelp information.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the retrieved information is insufficient, explicitly say so.
4. Distinguish facts from recommendations.
5. Keep the answer concise and useful.
6. When making a claim about customer opinions, refer to the provided reviews.
7. Use bullet points or numbered lists where appropriate for clarity."""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """USER QUESTION:
{question}

RETRIEVED YELP INFORMATION:
{context}"""


def generate_answer(question: str, context: str, model: str = "gpt-4o-mini") -> str:
    """
    Calls the OpenAI Chat API with the question and retrieved context.

    Args:
        question: The user's natural language question.
        context:  Pre-formatted context string from context_builder.build_context().
        model:    OpenAI model to use (default: gpt-4o-mini — fast & cheap).

    Returns:
        The LLM-generated answer as a string.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. "
            "Create a .env file with OPENAI_API_KEY=your_key_here"
        )

    client = OpenAI(api_key=api_key)

    user_message = USER_PROMPT_TEMPLATE.format(question=question, context=context)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,   # Low temperature → more factual, less creative
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()
