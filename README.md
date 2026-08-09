# ⭐ Yelp RAG System & Business Intelligence Engine

High-performance, local Retrieval-Augmented Generation (RAG) system and interactive Streamlit web dashboard built on the Yelp dataset using **DuckDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, **FAISS Vector Search**, **OpenAI GPT-4o-mini**, and **Streamlit**.

---

## 🌟 Key Features & Dual-Portal Architecture

### 1. 👤 Consumer Search & Discovery Portal
* **Continuous Multi-Turn Chat:** Interactive AI assistant with session memory for natural language discovery (*"Best authentic Italian pizza with good wine in Philadelphia"*).
* **AI Recommendation Engine:** Generates factual summaries synthesized from retrieved business profiles and review evidence.
* **Smart Filters:** Filter vector search by City, Star Rating threshold, and Review Sentiment.
* **Interactive Evidence Cards:** Displays business metadata, star ratings, cosine similarity scores, and review document excerpts.

### 2. 💼 Business Owner Intelligence Dashboard & AI Copilot
* **Searchable Business Selector:** Instant auto-complete lookup across 150,000+ businesses (e.g. *Acme Oyster House*, *Oceana Grill*, *Commander's Palace*, *Cappy's Pizza*).
* **Executive KPI Summary:** Displays Average Star Rating, Total Review Volume, % Positive Sentiment, and % Critical Complaints.
* **🤖 AI Owner Copilot:** AI assistant scoped strictly to the selected business's review corpus via `business_id` filtering (*"What are top complaints?"*, *"What dishes do customers praise?"*, *"How can I improve ratings?"*).
* **📈 Sentiment & Pain Point Analysis:** Ordered side-by-side view showing positive praise reviews FIRST, followed by critical customer complaints SECOND.
* **📝 Feedback Archive:** Interactive SQL review table log powered by DuckDB.

---

## 🏗️ RAG Pipeline Architecture

```
User Query ──► [Hybrid Retriever] ──► [Context Builder] ──► [LLM Generator] ──► Structured Output
                    │                      │                     │
                    ├── FAISS (Vector)     └── Pre-formats       └── OpenAI GPT-4o-mini
                    └── DuckDB (Metadata)      retrieved docs        (Factual response)
```

1. **Retriever (`scripts/query_rag.py`)**: Computes dense 384d vector embeddings and executes cosine similarity search over FAISS indices (`IndexFlatIP`). Supports strict business scoping (`business_id`), city, star rating, and sentiment filtering.
2. **Context Builder (`rag/context_builder.py`)**: Formats retrieved FAISS results into clean, structured context blocks for the LLM.
3. **Generator (`rag/generator.py`)**: Directs `gpt-4o-mini` with a strict system prompt to synthesize factual responses using ONLY the retrieved Yelp context without hallucinating outside facts.
4. **Pipeline Orchestrator (`rag/rag_pipeline.py`)**: Serves as the single unified entry point (`answer_question()`) wiring retriever, context builder, and generator.

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
├── app.py                         # Entrypoint delegating to yelp_streamlit_ui/app.py
├── config.py                      # Centralized configuration & file paths
├── test.py                        # Diagnostic & verification test runner
├── requirements.txt               # Project dependencies
├── .env                           # API keys (OPENAI_API_KEY)
├── rag/                           # RAG Pipeline Package
│   ├── __init__.py
│   ├── context_builder.py         # Context formatter for LLM prompts
│   ├── generator.py               # OpenAI LLM generation layer
│   └── rag_pipeline.py            # End-to-end pipeline orchestrator
├── yelp_streamlit_ui/             # Streamlit Application
│   └── app.py                     # Yelp Dual-Portal Web UI (Cookbook Design System)
├── scripts/                       # Core Pipeline Scripts
│   ├── prepare_data.py            # Gold Layer ETL (DuckDB Direct Streaming)
│   ├── build_rag.py               # Standard single-process FAISS builder
│   ├── build_rag_fast.py          # High-performance parallel CPU FAISS builder (INT8 + Multi-Process)
│   ├── query_rag.py               # Hybrid Retriever Engine (SQL + FAISS)
│   └── utils.py                   # Utilities
├── docs/                          # Execution Guides & Notebook Documentation
│   ├── README.md                  # Documentation Index
│   ├── 01_read_gold_documents.md  # Gold dataset inspection docs
│   ├── 02_chunk_documents.md     # Document chunking docs
│   ├── 03_generate_embeddings.md # Embedding generation docs
│   ├── 04_build_and_query_rag.md  # Vector store & RAG search docs
│   └── 05_high_performance_cpu_pipeline.md # High-performance CPU pipeline guide
├── notebooks/                     # Interactive Jupyter Notebooks
│   ├── 01_read_gold_documents.ipynb
│   ├── 02_chunk_documents.ipynb
│   ├── 03_generate_embeddings.ipynb
│   └── 04_build_and_query_rag.ipynb
├── data/ (ignored)                # Parquet datasets (Silver, Gold, Chunks)
└── models/ (ignored)              # FAISS vector indices & metadata pickles
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup & API Configuration
```powershell
# Activate Python Virtual Environment
.\.venv\Scripts\Activate.ps1

# Create .env file with your OpenAI API Key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. Generate Gold Layer RAG Datasets
```powershell
python scripts/prepare_data.py
```

### 3. Build FAISS Vector Index (High-Performance Parallel CPU)
```powershell
# Benchmark run for 50,000 documents (~4-5 mins)
python scripts/build_rag_fast.py 50000

# Full dataset run for 100% of all 7.14M documents (~1.5-2 hours)
python scripts/build_rag_fast.py 0
```

### 4. Test RAG Pipeline from Terminal (CLI)
```powershell
python -m rag.rag_pipeline "What are the best seafood places in New Orleans?"
```

### 5. Launch Streamlit Dual-Portal Application
```powershell
python -m streamlit run app.py --server.port 8501 --server.fileWatcherType none
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📑 Detailed Documentation & Benchmarks

For in-depth explanations, notebook mappings, and performance benchmarks, see the **[Docs Directory](docs/README.md)**:
* 📄 **[High-Performance CPU Pipeline Architecture](docs/05_high_performance_cpu_pipeline.md)**
* 📄 **[Gold Document Schema Inspection](docs/01_read_gold_documents.md)**
* 📄 **[FAISS Vector Store Construction](docs/04_build_and_query_rag.md)**
