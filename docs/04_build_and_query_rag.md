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

ROOT_DIR = Path(r"c:/Users/shank/Downloads/yelp_project")
FAISS_DIR = ROOT_DIR / "yelp-rag" / "models" / "faiss_index"

with open(FAISS_DIR / "biz_embeddings.pkl", "rb") as f:
    biz_data = pickle.load(f)

with open(FAISS_DIR / "rev_embeddings.pkl", "rb") as f:
    rev_data = pickle.load(f)

biz_matrix, biz_meta = biz_data["matrix"], biz_data["meta"]
rev_matrix, rev_meta = rev_data["matrix"], rev_data["meta"]

dim = biz_matrix.shape[1]

# Initialize FAISS Inner Product (Cosine Similarity) Indices
biz_index = faiss.IndexFlatIP(dim)
biz_index.add(biz_matrix)

rev_index = faiss.IndexFlatIP(dim)
rev_index.add(rev_matrix)

print(f"=======================================================")
print(f"               FAISS INDEX BUILD METRICS               ")
print(f"=======================================================")
print(f"Business Index Total Vectors: {biz_index.ntotal:,}")
print(f"Review Index Total Vectors:   {rev_index.ntotal:,}")
print(f"Vector Dimensions:            {dim}")
```

**Exact Runtime Output:**
```text
[ERROR]: [Errno 2] No such file or directory: 'c:\\Users\\shank\\Downloads\\yelp_project\\yelp-rag\\models\\faiss_index\\biz_embeddings.pkl'
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
        scores, indices = biz_index.search(query_vec, top_k * 3)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(biz_meta): continue
            m = biz_meta[idx]
            if city and m.get("city", "").lower() != city.lower(): continue
            if min_stars and m.get("business_rating", 0) < min_stars: continue
            results.append({
                "score": float(score), "type": "business", "name": m.get("business_name"),
                "city": m.get("city"), "rating": m.get("business_rating"), "text": m.get("document_text")
            })

    if doc_type in ["all", "review"]:
        scores, indices = rev_index.search(query_vec, top_k * 3)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(rev_meta): continue
            m = rev_meta[idx]
            if city and m.get("city", "").lower() != city.lower(): continue
            if min_stars and m.get("stars", 0) < min_stars: continue
            results.append({
                "score": float(score), "type": "review", "name": m.get("business_name"),
                "city": m.get("city"), "rating": m.get("stars"), "text": m.get("document_text")
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
    print(f"[{i}] Score: {r['score']:.4f} | Type: {r['type'].upper()} | Name: {r['name']} ({r['city']}) | Rating: {r['rating']} ⭐")
    print("    Text Excerpt:")
    print(f"    {r['text'].replace('\n', ' | ')[:180]}...")
    print("-" * 75)
```

**Exact Runtime Output:**
```text
[ERROR]: f-string expression part cannot include a backslash (<string>, line 8)
```

---

### Code Cell #4
```python
query_2 = "Great pizza place with fast service and friendly staff"
print(f"--> QUERY 2: '{query_2}'\n")

res2 = search_rag(query_2, doc_type="review", top_k=3)
for i, r in enumerate(res2, 1):
    print(f"[{i}] Score: {r['score']:.4f} | Type: {r['type'].upper()} | Name: {r['name']} ({r['city']}) | Rating: {r['rating']} ⭐")
    print("    Text Excerpt:")
    print(f"    {r['text'].replace('\n', ' | ')[:180]}...")
    print("-" * 75)
```

**Exact Runtime Output:**
```text
[ERROR]: f-string expression part cannot include a backslash (<string>, line 8)
```

---
