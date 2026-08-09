"""
High-Performance Local CPU Vector Store Builder for Yelp RAG Pipeline.

Utilizes PyTorch INT8 Dynamic Quantization, DuckDB Parquet Streaming,
and Multi-Process Parallel Partitioning across CPU cores.
"""

import sys
import os
import time
import shutil
import pickle
import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import faiss
import torch
from sentence_transformers import SentenceTransformer
import config

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _worker_process_partition(
    worker_id,
    parquet_glob,
    offset,
    total_to_process,
    batch_size,
    temp_dir_str
):
    """
    Worker function executed in separate process per CPU core.
    """
    # Limit internal PyTorch threads per worker to prevent CPU over-subscription
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    temp_dir = Path(temp_dir_str)
    
    # Load model in worker process from local cache
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, local_files_only=True)

    con = duckdb.connect()
    
    query = f"""
    SELECT * FROM read_parquet('{parquet_glob}') LIMIT {total_to_process} OFFSET {offset}
    """
    df = con.execute(query).df()
    con.close()

    if df.empty:
        return

    texts = df["document_text"].fillna("").tolist()
    metadata_list = []
    for _, row in df.iterrows():
        meta = row.to_dict()
        if "last_updated" in meta:
            meta["last_updated"] = str(meta["last_updated"])
        if "review_date" in meta:
            meta["review_date"] = str(meta["review_date"])
        metadata_list.append(meta)

    # Encode texts in batches with live progress updates
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        emb = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(emb.astype(np.float32))
        
        processed = min(i + batch_size, len(texts))
        if processed % 1280 == 0 or processed == len(texts):
            print(f"  [Worker #{worker_id}] Encoded {processed:,} / {len(texts):,} documents...")

    embeddings = np.vstack(all_embeddings)

    # Build local FAISS sub-index
    sub_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    sub_index.add(embeddings)

    # Save partition artifacts
    sub_idx_path = temp_dir / f"part_{worker_id}.faiss"
    sub_meta_path = temp_dir / f"part_{worker_id}.pkl"

    faiss.write_index(sub_index, str(sub_idx_path))
    with open(sub_meta_path, "wb") as f:
        pickle.dump(metadata_list, f)

def build_fast_index_for_dataset(
    dataset_name,
    parquet_glob_path,
    index_path,
    meta_path,
    max_records=0,
    batch_size=128
):
    """
    Orchestrates multi-process parallel vector embedding & FAISS index building.
    """
    print(f"\n=======================================================")
    print(f"   [FAST-CPU] Building FAISS Index for: {dataset_name}")
    print(f"=======================================================")
    print(f"Input Glob:   {parquet_glob_path}")
    print(f"Index Output: {index_path}")
    print(f"Meta Output:  {meta_path}")

    con = duckdb.connect()
    total_records = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_glob_path}')").fetchone()[0]
    con.close()

    if max_records > 0:
        records_to_process = min(total_records, max_records)
    else:
        records_to_process = total_records

    print(f"Total dataset records: {total_records:,}. Processing: {records_to_process:,} records...")

    if records_to_process == 0:
        print("[WARNING] No records found. Skipping index creation.")
        return

    num_workers = min(mp.cpu_count(), 4)
    records_per_worker = int(np.ceil(records_to_process / num_workers))
    print(f"Parallel Workers: {num_workers} processes (~{records_per_worker:,} records/worker)")

    temp_dir = config.FAISS_INDEX_DIR / f"_temp_{dataset_name.lower().replace(' ', '_')}"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    processes = []

    for w_id in range(num_workers):
        offset = w_id * records_per_worker
        limit = min(records_per_worker, records_to_process - offset)
        if limit <= 0:
            break

        p = mp.Process(
            target=_worker_process_partition,
            args=(w_id, parquet_glob_path, offset, limit, batch_size, str(temp_dir))
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Master Index Merge
    master_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    master_metadata = []

    for w_id in range(len(processes)):
        sub_idx_path = temp_dir / f"part_{w_id}.faiss"
        sub_meta_path = temp_dir / f"part_{w_id}.pkl"

        if sub_idx_path.exists() and sub_meta_path.exists():
            sub_idx = faiss.read_index(str(sub_idx_path))
            master_index.merge_from(sub_idx)

            with open(sub_meta_path, "rb") as f:
                sub_meta = pickle.load(f)
                master_metadata.extend(sub_meta)

    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(master_index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(master_metadata, f)

    shutil.rmtree(temp_dir, ignore_errors=True)
    elapsed = time.time() - start_time

    rate = master_index.ntotal / elapsed if elapsed > 0 else 0
    print(f"[SUCCESS] Processed {master_index.ntotal:,} vectors in {elapsed:.2f} seconds ({rate:.1f} docs/sec).")
    print(f"Persisted FAISS index ({os.path.getsize(index_path):,} bytes) & Metadata ({os.path.getsize(meta_path):,} bytes).\n")

def main():
    parser = argparse.ArgumentParser(description="High-Performance Local CPU FAISS Indexer")
    parser.add_argument(
        "max_records",
        type=int,
        nargs="?",
        default=0,
        help="Max records per dataset (0 = process full dataset)"
    )
    args = parser.parse_args()

    biz_glob = str(config.GOLD_BUSINESS_PATH / "*.parquet").replace("\\", "/")
    rev_glob = str(config.GOLD_REVIEW_PATH / "*.parquet").replace("\\", "/")

    print(f"=== Starting High-Performance Local CPU FAISS Vector Builder ===")
    
    # 1. Business Documents
    build_fast_index_for_dataset(
        dataset_name="Business Documents",
        parquet_glob_path=biz_glob,
        index_path=config.BUSINESS_INDEX_PATH,
        meta_path=config.BUSINESS_META_PATH,
        max_records=args.max_records
    )

    # 2. Review Documents
    build_fast_index_for_dataset(
        dataset_name="Review Documents",
        parquet_glob_path=rev_glob,
        index_path=config.REVIEW_INDEX_PATH,
        meta_path=config.REVIEW_META_PATH,
        max_records=args.max_records
    )

if __name__ == "__main__":
    main()
