# Notebook Execution & Cell Mapping: `04_build_and_query_rag.ipynb`

> **Module Title**: Build FAISS Vector Store & Semantic RAG Search

> **Source Notebook**: [04_build_and_query_rag.ipynb](file:///c:/Users/shank/Downloads/yelp_project/notebooks/04_build_and_query_rag.ipynb)

---

## Cell-by-Cell Code & Output Mapping

### Markdown Section 1
# Notebook 04: Build FAISS Vector Store & Semantic RAG Search
This notebook constructs local FAISS `IndexFlatIP` indices for business and review vector spaces and executes semantic search queries with metadata filtering.

---

### Code Cell #1
```python
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
FAISS_DIR = PROJECT_ROOT / 'models' / 'faiss_index'

biz_meta_file = FAISS_DIR / "biz_embeddings.pkl" if (FAISS_DIR / "biz_embeddings.pkl").exists() else FAISS_DIR / "business_meta.pkl"
rev_meta_file = FAISS_DIR / "rev_embeddings.pkl" if (FAISS_DIR / "rev_embeddings.pkl").exists() else FAISS_DIR / "review_meta.pkl"

with open(biz_meta_file, "rb") as f:
    biz_data = pickle.load(f)

with open(rev_meta_file, "rb") as f:
    rev_data = pickle.load(f)

if isinstance(biz_data, dict) and "matrix" in biz_data:
    biz_matrix, biz_meta = biz_data["matrix"], biz_data["meta"]
    dim = biz_matrix.shape[1]
    biz_index = faiss.IndexFlatIP(dim)
    biz_index.add(biz_matrix)
else:
    biz_meta = biz_data
    biz_index = faiss.read_index(str(FAISS_DIR / "business_index.faiss"))
    dim = biz_index.d

if isinstance(rev_data, dict) and "matrix" in rev_data:
    rev_matrix, rev_meta = rev_data["matrix"], rev_data["meta"]
    rev_index = faiss.IndexFlatIP(dim)
    rev_index.add(rev_matrix)
else:
    rev_meta = rev_data
    rev_index = faiss.read_index(str(FAISS_DIR / "review_index.faiss"))

print("=======================================================")
print("               FAISS INDEX BUILD METRICS               ")
print("=======================================================")
print(f"Business Index Total Vectors: {biz_index.ntotal:,}")
print(f"Review Index Total Vectors:   {rev_index.ntotal:,}")
print(f"Vector Dimensions:            {dim}")
```

**Exact Runtime Output:**
```text
=======================================================
               FAISS INDEX BUILD METRICS               
=======================================================
Business Index Total Vectors: 2,000
Review Index Total Vectors:   2,000
Vector Dimensions:            384
```

---

### Code Cell #2
```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def search_rag(query_text, doc_type="all", city=None, min_stars=None, top_k=3):
    query_vec = model.encode([query_text], normalize_embeddings=True).astype(np.float32)
    results = []

    if doc_type in ["all", "business"]:
        scores, indices = biz_index.search(query_vec, top_k * 5)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(biz_meta): continue
            m = biz_meta[idx]
            if city and m.get("city", "").lower() != city.lower(): continue
            rating = m.get("business_rating") or m.get("stars") or 0
            if min_stars and rating < min_stars: continue
            results.append({
                "score": float(score),
                "type": "business",
                "name": m.get("business_name") or m.get("name"),
                "city": m.get("city"),
                "rating": rating,
                "text": m.get("document_text", "")
            })

    if doc_type in ["all", "review"]:
        scores, indices = rev_index.search(query_vec, top_k * 5)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(rev_meta): continue
            m = rev_meta[idx]
            if city and m.get("city", "").lower() != city.lower(): continue
            rating = m.get("stars") or m.get("business_rating") or 0
            if min_stars and rating < min_stars: continue
            results.append({
                "score": float(score),
                "type": "review",
                "name": m.get("business_name"),
                "city": m.get("city"),
                "rating": rating,
                "text": m.get("document_text", "")
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results

print("[OK] Search function initialized.")
```

**Exact Runtime Output:**
```text
[OK] Search function initialized.
```

---

### Code Cell #3
```python
query_1 = "Top rated Italian restaurants with delicious pasta and outdoor seating"
print(f"--> QUERY 1: '{query_1}'\n")

res1 = search_rag(query_1, doc_type="all", top_k=3)
for i, r in enumerate(res1, 1):
    print(f"[{i}] Score: {r['score']:.4f} | Type: {r['type'].upper()} | Name: {r['name']} ({r['city']}) | Rating: {r['rating']} star(s)")
    print("    Text Excerpt:")
    clean_text = str(r['text']).replace('\n', ' ')
    print(f"    {clean_text[:180]}...")
    print("-" * 75)
```

---

### Code Cell #4
```python
query_2 = "Great pizza place with fast service and friendly staff"
print(f"--> QUERY 2: '{query_2}'\n")

res2 = search_rag(query_2, doc_type="review", top_k=3)
for i, r in enumerate(res2, 1):
    print(f"[{i}] Score: {r['score']:.4f} | Type: {r['type'].upper()} | Name: {r['name']} ({r['city']}) | Rating: {r['rating']} star(s)")
    print("    Text Excerpt:")
    clean_text = str(r['text']).replace('\n', ' ')
    print(f"    {clean_text[:180]}...")
    print("-" * 75)
```

---
