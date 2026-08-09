import sys
import duckdb
import pickle
import faiss
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config

gold_biz = str(config.GOLD_BUSINESS_PATH / "*.parquet").replace("\\", "/")
gold_rev = str(config.GOLD_REVIEW_PATH / "*.parquet").replace("\\", "/")
meta_biz_path = config.BUSINESS_META_PATH
index_biz_path = config.BUSINESS_INDEX_PATH

con = duckdb.connect()

print("==================================================")
print(" 1. GOLD BUSINESS DOCUMENT (PARQUET)")
print("==================================================")
biz_doc = con.execute(f"SELECT document_id, business_name, city, state, primary_category, business_rating, document_text FROM read_parquet('{gold_biz}') LIMIT 1").fetchone()
print(f"Document ID: {biz_doc[0]}")
print(f"Business:    {biz_doc[1]} ({biz_doc[2]}, {biz_doc[3]})")
print(f"Category:    {biz_doc[4]} | Rating: {biz_doc[5]} stars")
print("--- Document Text ---")
print(biz_doc[6])

print("\n==================================================")
print(" 2. GOLD REVIEW DOCUMENT (PARQUET)")
print("==================================================")
rev_doc = con.execute(f"SELECT document_id, review_id, business_name, city, state, stars, sentiment, document_text FROM read_parquet('{gold_rev}') LIMIT 1").fetchone()
print(f"Document ID: {rev_doc[0]}")
print(f"Review ID:   {rev_doc[1]}")
print(f"Business:    {rev_doc[2]} ({rev_doc[3]}, {rev_doc[4]})")
print(f"Rating:      {rev_doc[5]} stars | Sentiment: {rev_doc[6]}")
print("--- Document Text ---")
print(rev_doc[7])

print("\n==================================================")
print(" 3. STORED FAISS VECTOR METADATA ENTRY (PICKLE)")
print("==================================================")
if meta_biz_path.exists():
    with open(meta_biz_path, "rb") as f:
        meta_list = pickle.load(f)
    print(f"Total FAISS Vector Metadata Items: {len(meta_list):,}")
    print("Record #0 Keys & Sample Values:")
    sample = meta_list[0]
    for k, v in sample.items():
        if k == "document_text":
            print(f"  {k}: {str(v)[:100]}...")
        else:
            print(f"  {k}: {v}")

print("\n==================================================")
print(" 4. ACTUAL DENSE EMBEDDING VECTOR (FAISS INDEX)")
print("==================================================")
if index_biz_path.exists():
    index = faiss.read_index(str(index_biz_path))
    print(f"Total Vectors in FAISS Index: {index.ntotal:,}")
    print(f"Embedding Vector Dimensions:  {index.d} dimensions")
    
    # Reconstruct vector for record #0
    vec_0 = index.reconstruct(0)
    print(f"Vector #0 Data Type:         {vec_0.dtype}")
    print(f"Vector #0 Shape:             {vec_0.shape}")
    print(f"Vector #0 Norm (L2 Length):  {np.linalg.norm(vec_0):.4f}")
    print("\n--- Vector #0 Raw Numerical Values (First 20 dimensions out of 384) ---")
    print(np.round(vec_0[:20], 6))

con.close()

