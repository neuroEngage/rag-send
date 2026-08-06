# Notebook Execution & Cell Mapping: `03_generate_embeddings.ipynb`

> **Module Title**: Generate Dense Vector Embeddings

> **Source Notebook**: [03_generate_embeddings.ipynb](file:///c:/Users/shank/Downloads/yelp_project/notebooks/03_generate_embeddings.ipynb)

---

## Cell-by-Cell Code & Output Mapping

### Markdown Section 1
# Notebook 03: Generate Dense Vector Embeddings
This notebook loads `sentence-transformers/all-MiniLM-L6-v2` locally and generates dense 384-dimensional vector embeddings for chunked Yelp business and review documents.

---

### Code Cell #1
```python
import time
import pickle
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(r"c:/Users/shank/Downloads/yelp_project")
GOLD_BIZ_PATH = ROOT_DIR / "gold_layer" / "rag" / "business_documents"
GOLD_REV_PATH = ROOT_DIR / "gold_layer" / "rag" / "review_documents"
FAISS_DIR = ROOT_DIR / "yelp-rag" / "models" / "faiss_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading Embedding Model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"[OK] Model Loaded. Vector Embedding Dimension: {embedding_dim}")
```

**Exact Runtime Output:**
```text
Loading Embedding Model: sentence-transformers/all-MiniLM-L6-v2...
[OK] Model Loaded. Vector Embedding Dimension: 384
```

---

### Code Cell #2
```python
con = duckdb.connect()
biz_glob = str(GOLD_BIZ_PATH / "*.parquet").replace("\\", "/")

df_biz = con.execute(f"""
    SELECT document_id, business_name, city, state, primary_category, business_rating, price_range, document_text
    FROM read_parquet('{biz_glob}')
    LIMIT 2000
""").df()

texts_biz = df_biz["document_text"].fillna("").tolist()
print(f"Encoding {len(texts_biz):,} business documents in batches...")

start_t = time.time()
embeddings_biz = model.encode(texts_biz, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
elapsed = time.time() - start_t

embeddings_biz_matrix = np.array(embeddings_biz, dtype=np.float32)

print(f"=======================================================")
print(f"       BUSINESS EMBEDDINGS GENERATION COMPLETED        ")
print(f"=======================================================")
print(f"Total Vectors Generated: {embeddings_biz_matrix.shape[0]:,}")
print(f"Vector Dimension:        {embeddings_biz_matrix.shape[1]}")
print(f"Encoding Time:           {elapsed:.2f} seconds")
print(f"Embedding Matrix Data:   {embeddings_biz_matrix.dtype}")
```

**Exact Runtime Output:**
```text
[ERROR]: IO Error: No files found that match the pattern "c:/Users/shank/Downloads/yelp_project/gold_layer/rag/business_documents/*.parquet"

LINE 3:     FROM read_parquet('c:/Users/shank/Downloads/yelp_project/gold_layer...
                 ^
```

---

### Code Cell #3
```python
rev_glob = str(GOLD_REV_PATH / "*.parquet").replace("\\", "/")

df_rev = con.execute(f"""
    SELECT document_id, review_id, business_id, business_name, city, state, stars, sentiment, document_text
    FROM read_parquet('{rev_glob}')
    LIMIT 2000
""").df()

texts_rev = df_rev["document_text"].fillna("").tolist()
print(f"Encoding {len(texts_rev):,} review documents in batches...")

start_t = time.time()
embeddings_rev = model.encode(texts_rev, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
elapsed = time.time() - start_t

embeddings_rev_matrix = np.array(embeddings_rev, dtype=np.float32)

print(f"=======================================================")
print(f"        REVIEW EMBEDDINGS GENERATION COMPLETED         ")
print(f"=======================================================")
print(f"Total Vectors Generated: {embeddings_rev_matrix.shape[0]:,}")
print(f"Vector Dimension:        {embeddings_rev_matrix.shape[1]}")
print(f"Encoding Time:           {elapsed:.2f} seconds")
con.close()
```

**Exact Runtime Output:**
```text
[ERROR]: IO Error: No files found that match the pattern "c:/Users/shank/Downloads/yelp_project/gold_layer/rag/review_documents/*.parquet"

LINE 3:     FROM read_parquet('c:/Users/shank/Downloads/yelp_project/gold_layer...
                 ^
```

---

### Code Cell #4
```python
# Persist embeddings and metadata objects locally
biz_meta = df_biz.to_dict(orient="records")
rev_meta = df_rev.to_dict(orient="records")

with open(FAISS_DIR / "biz_embeddings.pkl", "wb") as f:
    pickle.dump({"matrix": embeddings_biz_matrix, "meta": biz_meta}, f)

with open(FAISS_DIR / "rev_embeddings.pkl", "wb") as f:
    pickle.dump({"matrix": embeddings_rev_matrix, "meta": rev_meta}, f)

print(f"[SUCCESS] Persisted business & review embedding matrices + metadata to {FAISS_DIR}")
```

**Exact Runtime Output:**
```text
[ERROR]: name 'df_biz' is not defined
```

---
