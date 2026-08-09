# Yelp RAG System — Notebook Execution & Cell Documentation Index

This directory contains cell-by-cell code documentation and exact captured outputs for every notebook and high-performance pipeline in the Yelp RAG architecture.

## Documentation Index

1. **[01_read_gold_documents.md](01_read_gold_documents.md)**
   - Inspects Gold Layer Parquet datasets (`business_documents`, `review_documents`) via DuckDB.
   - Maps document schemas, total row counts, and sample formatted `document_text` fields.

2. **[02_chunk_documents.md](02_chunk_documents.md)**
   - Implements text chunking with character windowing (`CHUNK_SIZE = 1000`, `OVERLAP = 200`).
   - Maps chunk generation logic, document explosions, unique `chunk_id` assignments, and metrics.

3. **[03_generate_embeddings.md](03_generate_embeddings.md)**
   - Loads `sentence-transformers/all-MiniLM-L6-v2` locally.
   - Encodes text documents into normalized 384-dimensional dense vectors and maps vector matrix properties.

4. **[04_build_and_query_rag.md](04_build_and_query_rag.md)**
   - Builds FAISS `IndexFlatIP` vector search indices locally.
   - Maps semantic similarity search queries, metadata filtering (city, rating, sentiment), and top matching result snippets.

5. **[05_high_performance_cpu_pipeline.md](05_high_performance_cpu_pipeline.md)**
   - Architecture guide for high-performance multi-process parallel CPU vector store construction.
   - Explains INT8 dynamic quantization, DuckDB Parquet streaming, and zero-copy FAISS sub-index merging for 7.14M documents.
