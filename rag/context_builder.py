"""
Context Builder — Yelp RAG Pipeline

Converts raw FAISS retrieval results into a clean, structured text block
that the LLM can reason over.

Responsibility: formatting only.
No FAISS calls. No DB calls. No Streamlit.
"""


def build_context(results: list) -> str:
    """
    Takes a list of retrieved review/business dicts from YelpRAGRetriever.search()
    and returns a formatted context string for the LLM.

    Args:
        results: List of result dicts from the retriever.

    Returns:
        A clean, formatted string ready to be injected into the prompt.
    """
    if not results:
        return "No relevant Yelp information was found for this query."

    context_parts = []

    review_count = 0
    business_count = 0

    for result in results:
        doc_type = result.get("document_type", "review")

        if doc_type == "review":
            review_count += 1
            biz_name = result.get("business_name", "Unknown Business")
            city = result.get("city", "")
            state = result.get("state", "")
            stars = result.get("stars", "N/A")
            sentiment = result.get("sentiment", "N/A").capitalize() if result.get("sentiment") else "N/A"
            text = result.get("document_text", "").strip()
            review_date = result.get("review_date", result.get("date", "N/A"))
            location = ", ".join(filter(None, [city, state]))

            block = (
                f"REVIEW {review_count}\n"
                f"Business: {biz_name}\n"
                f"City: {location}\n"
                f"Rating: {stars}/5\n"
                f"Sentiment: {sentiment}\n"
                f"Date: {review_date}\n\n"
                f"Review:\n{text}"
            )
            context_parts.append(block)

        elif doc_type == "business":
            business_count += 1
            biz_name = result.get("business_name", "Unknown Business")
            city = result.get("city", "")
            state = result.get("state", "")
            category = result.get("primary_category", "N/A")
            rating = result.get("business_rating", "N/A")
            price_range = result.get("price_range", "N/A")
            text = result.get("document_text", "").strip()
            location = ", ".join(filter(None, [city, state]))

            block = (
                f"BUSINESS PROFILE {business_count}\n"
                f"Name: {biz_name}\n"
                f"Category: {category}\n"
                f"Location: {location}\n"
                f"Rating: {rating}/5\n"
                f"Price Range: {price_range}\n\n"
                f"Description:\n{text}"
            )
            context_parts.append(block)

    separator = "\n\n" + ("─" * 60) + "\n\n"
    return separator.join(context_parts)
