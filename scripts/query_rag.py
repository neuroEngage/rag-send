"""
Script for querying the Yelp Gold Layer RAG pipeline with vector similarity search & metadata filtering.
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
from sentence_transformers import SentenceTransformer
import config

class YelpRAGRetriever:
    """
    RAG Retriever class managing FAISS vector search and metadata filtering.
    """
    def __init__(self):
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        self.biz_index = None
        self.biz_meta = []
        self.rev_index = None
        self.rev_meta = []

        self._load_indices()

    def _load_indices(self):
        """Loads FAISS indices and metadata pickleg files if available."""
        if os.path.exists(config.BUSINESS_INDEX_PATH) and os.path.exists(config.BUSINESS_META_PATH):
            print(f"Loading business FAISS index: {config.BUSINESS_INDEX_PATH}...")
            self.biz_index = faiss.read_index(str(config.BUSINESS_INDEX_PATH))
            with open(config.BUSINESS_META_PATH, "rb") as f:
                self.biz_meta = pickle.load(f)

        if os.path.exists(config.REVIEW_INDEX_PATH) and os.path.exists(config.REVIEW_META_PATH):
            print(f"Loading review FAISS index: {config.REVIEW_INDEX_PATH}...")
            self.rev_index = faiss.read_index(str(config.REVIEW_INDEX_PATH))
            with open(config.REVIEW_META_PATH, "rb") as f:
                self.rev_meta = pickle.load(f)

    def search(
        self,
        query_text,
        doc_type="all",
        city=None,
        min_stars=None,
        sentiment=None,
        business_id=None,
        top_k=5
    ):
        """
        Executes semantic vector similarity search with optional metadata filtering.
        """
        query_vec = self.model.encode([query_text], normalize_embeddings=True).astype(np.float32)
        results = []

        # 1. Search Business Index
        if (doc_type in ["all", "business"]) and self.biz_index:
            scores, indices = self.biz_index.search(query_vec, top_k * 3)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.biz_meta):
                    continue
                meta = self.biz_meta[idx]
                
                # Apply metadata filters
                if city and meta.get("city", "").lower() != city.lower():
                    continue
                if min_stars and meta.get("business_rating", 0) < min_stars:
                    continue

                results.append({
                    "score": float(score),
                    "document_id": meta.get("document_id"),
                    "document_type": "business",
                    "business_name": meta.get("business_name"),
                    "city": meta.get("city"),
                    "state": meta.get("state"),
                    "primary_category": meta.get("primary_category"),
                    "business_rating": meta.get("business_rating"),
                    "price_range": meta.get("price_range"),
                    "document_text": meta.get("document_text")
                })

        # 2. Search Review Index
        if (doc_type in ["all", "review"]) and self.rev_index:
            fetch_limit = min(top_k * 500 if business_id else top_k * 10, self.rev_index.ntotal)
            scores, indices = self.rev_index.search(query_vec, fetch_limit)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.rev_meta):
                    continue
                meta = self.rev_meta[idx]

                # Apply metadata filters
                if business_id and meta.get("business_id") != business_id:
                    continue
                if city and meta.get("city", "").lower() != city.lower():
                    continue
                if min_stars and meta.get("stars", 0) < min_stars:
                    continue
                if sentiment and meta.get("sentiment", "").lower() != sentiment.lower():
                    continue

                results.append({
                    "score": float(score),
                    "document_id": meta.get("document_id"),
                    "document_type": "review",
                    "business_id": meta.get("business_id"),
                    "business_name": meta.get("business_name"),
                    "city": meta.get("city"),
                    "state": meta.get("state"),
                    "primary_category": meta.get("primary_category"),
                    "stars": meta.get("stars"),
                    "sentiment": meta.get("sentiment"),
                    "review_date": meta.get("review_date"),
                    "document_text": meta.get("document_text")
                })

            # Fallback for business_id: if FAISS top-K didn't find enough matches for this business
            if business_id and len(results) < top_k:
                existing_ids = {r["document_id"] for r in results}
                for meta in self.rev_meta:
                    if meta.get("business_id") == business_id and meta.get("document_id") not in existing_ids:
                        # Apply additional filters if specified
                        if min_stars and meta.get("stars", 0) < min_stars:
                            continue
                        if sentiment and meta.get("sentiment", "").lower() != sentiment.lower():
                            continue
                        results.append({
                            "score": 0.500,
                            "document_id": meta.get("document_id"),
                            "document_type": "review",
                            "business_id": meta.get("business_id"),
                            "business_name": meta.get("business_name"),
                            "city": meta.get("city"),
                            "state": meta.get("state"),
                            "primary_category": meta.get("primary_category"),
                            "stars": meta.get("stars"),
                            "sentiment": meta.get("sentiment"),
                            "review_date": meta.get("review_date"),
                            "document_text": meta.get("document_text")
                        })
                        if len(results) >= top_k:
                            break

        # Sort combined results by similarity score descending
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return results

def query_rag_pipeline(query, city=None, min_stars=None, sentiment=None, doc_type="all", top_k=5):
    """
    Query interface function.
    """
    retriever = YelpRAGRetriever()
    results = retriever.search(
        query_text=query,
        doc_type=doc_type,
        city=city,
        min_stars=min_stars,
        sentiment=sentiment,
        top_k=top_k
    )

    print(f"\n=======================================================")
    print(f"   Yelp RAG Query Search Results: '{query}'")
    print(f"=======================================================")
    if not results:
        print("No matching documents found.")
        return results

    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Score: {res['score']:.4f} | Type: {res['document_type'].upper()} | Business: {res.get('business_name')} ({res.get('city')}, {res.get('state')})")
        print(f"    Primary Category: {res.get('primary_category')}")
        if res['document_type'] == 'business':
            print(f"    Rating: {res.get('business_rating')} stars | Price Range: {res.get('price_range')}")
        else:
            print(f"    Review Rating: {res.get('stars')} stars | Sentiment: {res.get('sentiment')}")
        print("    --- Text Excerpt ---")
        excerpt = res['document_text'].replace('\n', ' | ')[:200]
        print(f"    {excerpt}...")

    return results

if __name__ == "__main__":
    query_text = "Great pizza restaurants with outdoor seating"
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    query_rag_pipeline(query_text)
