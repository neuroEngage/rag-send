# ⭐ Yelp RAG System & Business Intelligence Engine

High-performance, local Retrieval-Augmented Generation (RAG) system and interactive Streamlit web dashboard built on the Yelp dataset using **DuckDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, **FAISS vector search**, and **Streamlit**.

---

## 🌟 Key Features & Dual-Portal Architecture

### 1. 👤 Consumer Search & Discovery Portal
* **Continuous Multi-Turn Chat:** Interactive AI chatbot with conversation memory (`st.session_state`) for natural language business discovery and recommendations.
* **Smart Filters:** Filter search by City, Star Rating threshold, and Review Sentiment.
* **Interactive Result Cards:** Displays business metadata, rating stars, similarity scores, and document excerpts.

### 2. 💼 Business Owner Intelligence Dashboard
* **Searchable Business Selector:** Choose or search any business (e.g., *Acme Oyster House*, *Oceana Grill*, *Commander's Palace*, *Cappy's Pizza*, *Grimaldi's Pizzeria*).
* **Executive KPI Summary:** Displays Average Star Rating, Total Review Volume, % Positive Sentiment, and % Negative Complaints.
* **🤖 AI Owner Copilot:** AI assistant tuned specifically to the selected business's review corpus to answer questions (*"What are top complaints?"*, *"What dishes do customers praise?"*, *"How can I raise my rating?"*).
* **📈 Sentiment & Pain Point Analysis:** Side-by-side view of critical negative complaints vs top positive reviews.
* **📝 Feedback Archive:** Interactive SQL review table log.

---

## 📊 Dataset Metrics

* **Gold Business Profiles:** `150,346` businesses (`data/02_gold/business_documents/`)
* **Gold Customer Reviews:** `6,990,280` customer reviews (`data/02_gold/review_documents/`)
* **Businesses with Reviews:** `150,346` (100%)
* **Average Reviews per Business:** `46.49` reviews
* **Dense Embedding Dimension:** `384` dimensions (`sentence-transformers/all-MiniLM-L6-v2`)

---

## 📁 Project Structure

```
yelp_project/
├── data/
│   ├── 01_silver/                 # Raw Parquet input datasets
│   ├── 02_gold/                   # Gold Layer RAG document parquet files
│   └── 03_chunks/                 # Text chunks
├── models/
│   └── faiss_index/               # FAISS vector indices & metadata pickles
├── notebooks/
│   ├── 01_read_gold_documents.ipynb
│   ├── 02_chunk_documents.ipynb
│   ├── 03_generate_embeddings.ipynb
│   └── 04_build_and_query_rag.ipynb
├── scripts/
│   ├── prepare_data.py            # Gold Layer ETL script (DuckDB Streaming)
│   ├── build_rag.py               # FAISS index construction script
│   ├── query_rag.py               # Hybrid Retriever Engine (SQL + FAISS)
│   └── glue_gold_rag_etl.py       # AWS Glue PySpark Cloud ETL script
├── app.py                         # Streamlit Dual-Portal Web App
├── config.py                      # Centralized configuration & paths
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup & Activation
```powershell
# Activate Python Virtual Environment
.\.venv\Scripts\Activate.ps1
```

### 2. Generate Gold Layer RAG Datasets
```powershell
python scripts/prepare_data.py
```

### 3. Build FAISS Vector Index
```powershell
# Build FAISS vector index (e.g., 50,000 documents)
python scripts/build_rag.py 50000
```

### 4. Launch Streamlit Application
```powershell
python -m streamlit run app.py --server.port 8501 --server.fileWatcherType none
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.
