"""
Script for building embeddings and FAISS index for Yelp RAG System from Gold Layer Parquet datasets.
"""

import sys
import os
import time
import pickle
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import faiss
from sentence_transformers import SentenceTransformer
import config

def build_index_for_dataset(
    dataset_name,
    parquet_glob_path,
    index_path,
    meta_path,
    model,
    batch_size=256,
    max_records=50000
):
    """
    Reads Parquet dataset in batches, computes dense embeddings, builds FAISS index, and persists index + metadata.
    """
    print(f"\n=======================================================")
    print(f"   Building FAISS Index for: {dataset_name}")
    print(f"=======================================================")
    print(f"Parquet Input: {parquet_glob_path}")
    print(f"Index Output:  {index_path}")
    print(f"Meta Output:   {meta_path}")

    con = duckdb.connect()
    
    # Check record count
    total_records = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_glob_path}')").fetchone()[0]
    records_to_process = min(total_records, max_records)
    print(f"Found {total_records:,} total records. Processing up to {records_to_process:,} records...")

    # Query columns
    query = f"""
    SELECT * FROM read_parquet('{parquet_glob_path}') LIMIT {records_to_process}
    """
    df = con.execute(query).df()
    con.close()

    if df.empty:
        print(f"[WARNING] No data found for {dataset_name}. Skipping index creation.")
        return

    # Prepare document texts and metadata items
    texts = df["document_text"].fillna("").tolist()
    
    # Store metadata dictionaries for each vector ID
    metadata_list = []
    for idx, row in df.iterrows():
        meta = row.to_dict()
        # Convert timestamp objects to string for clean serialization
        if "last_updated" in meta:
            meta["last_updated"] = str(meta["last_updated"])
        if "review_date" in meta:
            meta["review_date"] = str(meta["review_date"])
        metadata_list.append(meta)

    # Initialize FAISS Index (Inner Product with Normalized L2 for Cosine Similarity)
    embedding_dim = config.EMBEDDING_DIM
    index = faiss.IndexFlatIP(embedding_dim)

    print(f"--> Encoding {len(texts):,} documents in batches of {batch_size}...")
    start_time = time.time()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Generate embeddings
        embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embeddings.astype(np.float32))
        
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(texts):
            processed = min(i + batch_size, len(texts))
            print(f"    - Processed {processed:,} / {len(texts):,} documents...")

    embeddings_matrix = np.vstack(all_embeddings)
    index.add(embeddings_matrix)

    elapsed = time.time() - start_time
    print(f"[OK] Encoded {index.ntotal:,} vectors in {elapsed:.2f} seconds.")

    # Create destination directories
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Persist FAISS Index and Metadata
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata_list, f)

    print(f"[SUCCESS] Persisted FAISS index ({os.path.getsize(index_path):,} bytes) & Metadata ({os.path.getsize(meta_path):,} bytes).\n")

def main(max_records_per_dataset=50000):
    """
    Main execution function for building Gold Layer FAISS RAG indices.
    """
    print(f"Loading SentenceTransformer embedding model: {config.EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    biz_glob = str(config.GOLD_BUSINESS_PATH / "*.parquet").replace("\\", "/")
    rev_glob = str(config.GOLD_REVIEW_PATH / "*.parquet").replace("\\", "/")

    # Build Business Documents Index
    build_index_for_dataset(
        dataset_name="Business Documents",
        parquet_glob_path=biz_glob,
        index_path=config.BUSINESS_INDEX_PATH,
        meta_path=config.BUSINESS_META_PATH,
        model=model,
        max_records=max_records_per_dataset
    )

    # Build Review Documents Index
    build_index_for_dataset(
        dataset_name="Review Documents",
        parquet_glob_path=rev_glob,
        index_path=config.REVIEW_INDEX_PATH,
        meta_path=config.REVIEW_META_PATH,
        model=model,
        max_records=max_records_per_dataset
    )

if __name__ == "__main__":
    max_recs = 50000
    if len(sys.argv) > 1:
        try:
            max_recs = int(sys.argv[1])
        except ValueError:
            pass
    main(max_records_per_dataset=max_recs)
