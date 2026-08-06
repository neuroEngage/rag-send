"""
Configuration settings and paths for Yelp RAG system.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SILVER_DIR = DATA_DIR / "01_silver"
GOLD_DIR = DATA_DIR / "02_gold"
CHUNKS_DIR = DATA_DIR / "03_chunks"

MODELS_DIR = BASE_DIR / "models"

# Dataset Paths
SILVER_BUSINESS_PATH = SILVER_DIR / "business"
SILVER_REVIEW_PATH = SILVER_DIR / "review"

GOLD_BUSINESS_PATH = GOLD_DIR / "business_documents"
GOLD_REVIEW_PATH = GOLD_DIR / "review_documents"

CHUNKS_BUSINESS_PATH = CHUNKS_DIR / "business_chunks.parquet"
CHUNKS_REVIEW_PATH = CHUNKS_DIR / "review_chunks.parquet"

# Vector Store & Embedding Model Config
FAISS_INDEX_DIR = MODELS_DIR / "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector Index File Paths
BUSINESS_INDEX_PATH = FAISS_INDEX_DIR / "business_index.faiss"
BUSINESS_META_PATH = FAISS_INDEX_DIR / "business_meta.pkl"

REVIEW_INDEX_PATH = FAISS_INDEX_DIR / "review_index.faiss"
REVIEW_META_PATH = FAISS_INDEX_DIR / "review_meta.pkl"

