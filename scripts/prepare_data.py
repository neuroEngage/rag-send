"""
Yelp Gold Layer ETL Script for RAG Dataset Generation.

Transforms Silver Layer datasets (business & review parquet) into Gold Layer RAG documents:
- gold_layer/rag/business_documents/
- gold_layer/rag/review_documents/
"""

import sys
import os
from pathlib import Path
import time
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def process_gold_layer():
    start_time = time.time()
    
    # Destination output paths
    gold_biz_path = config.GOLD_BUSINESS_PATH
    gold_rev_path = config.GOLD_REVIEW_PATH

    # Ensure target output directories exist
    gold_biz_path.mkdir(parents=True, exist_ok=True)
    gold_rev_path.mkdir(parents=True, exist_ok=True)

    silver_biz_glob = str((config.SILVER_BUSINESS_PATH / "*.parquet")).replace("\\", "/")
    silver_rev_glob = str((config.SILVER_REVIEW_PATH / "*.parquet")).replace("\\", "/")


    print("=== Starting Yelp Gold Layer RAG Dataset Generation ===")
    print(f"Silver Business Input: {silver_biz_glob}")
    print(f"Silver Review Input:   {silver_rev_glob}")
    print(f"Gold Business Output:  {gold_biz_path}")
    print(f"Gold Review Output:    {gold_rev_path}\n")

    con = duckdb.connect()
    # Configure streaming and memory limits for low memory footprint
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA threads=4")

    # -------------------------------------------------------------
    # 1. PROCESS BUSINESS DOCUMENTS
    # -------------------------------------------------------------
    print("--> Processing business_documents...")

    out_biz_file = str(gold_biz_path / "business_documents.parquet").replace("\\", "/")

    biz_copy_query = f"""
    COPY (
        SELECT
            'doc_biz_' || business_id AS document_id,
            business_id,
            name AS business_name,
            address,
            city,
            state,
            postal_code,
            latitude,
            longitude,
            trim(string_split(categories, ',')[1]) AS primary_category,
            string_split(categories, ',') AS category_list,
            stars AS business_rating,
            review_count,
            attributes_restaurantspricerange2 AS price_range,
            is_open,
            (
                COALESCE('Monday: ' || hours_monday || ' | ', '') ||
                COALESCE('Tuesday: ' || hours_tuesday || ' | ', '') ||
                COALESCE('Wednesday: ' || hours_wednesday || ' | ', '') ||
                COALESCE('Thursday: ' || hours_thursday || ' | ', '') ||
                COALESCE('Friday: ' || hours_friday || ' | ', '') ||
                COALESCE('Saturday: ' || hours_saturday || ' | ', '') ||
                COALESCE('Sunday: ' || hours_sunday, '')
            ) AS hours,
            (
                COALESCE('creditcards=' || attributes_businessacceptscreditcards || ', ', '') ||
                COALESCE('parking=' || attributes_businessparking || ', ', '') ||
                COALESCE('pricerange=' || attributes_restaurantspricerange2 || ', ', '') ||
                COALESCE('delivery=' || attributes_restaurantsdelivery || ', ', '') ||
                COALESCE('takeout=' || attributes_restaurantstakeout || ', ', '') ||
                COALESCE('outdoorseating=' || attributes_outdoorseating || ', ', '') ||
                COALESCE('wifi=' || attributes_wifi, '')
            ) AS business_features,
            (
                'Business Name: ' || COALESCE(name, 'Unknown') ||
                '\nPrimary Category: ' || COALESCE(trim(string_split(categories, ',')[1]), 'N/A') ||
                '\nCategories: ' || COALESCE(categories, 'N/A') ||
                '\nLocation: ' || COALESCE(address, '') || ', ' || COALESCE(city, '') || ', ' || COALESCE(state, '') || ' ' || COALESCE(postal_code, '') ||
                '\nCoordinates: Lat ' || COALESCE(CAST(latitude AS VARCHAR), 'N/A') || ', Lon ' || COALESCE(CAST(longitude AS VARCHAR), 'N/A') ||
                '\nRating: ' || COALESCE(CAST(stars AS VARCHAR), 'N/A') || ' stars (' || COALESCE(CAST(review_count AS VARCHAR), '0') || ' reviews)' ||
                '\nPrice Range: ' || COALESCE(attributes_restaurantspricerange2, 'N/A') ||
                '\nOperating Status: ' || CASE WHEN is_open = 1 THEN 'Open' ELSE 'Closed' END ||
                '\nHours: ' || COALESCE(
                    COALESCE('Monday: ' || hours_monday || ' | ', '') ||
                    COALESCE('Tuesday: ' || hours_tuesday || ' | ', '') ||
                    COALESCE('Wednesday: ' || hours_wednesday || ' | ', '') ||
                    COALESCE('Thursday: ' || hours_thursday || ' | ', '') ||
                    COALESCE('Friday: ' || hours_friday || ' | ', '') ||
                    COALESCE('Saturday: ' || hours_saturday || ' | ', '') ||
                    COALESCE('Sunday: ' || hours_sunday, ''), 'N/A') ||
                '\nFeatures: ' || COALESCE(
                    COALESCE('creditcards=' || attributes_businessacceptscreditcards || ', ', '') ||
                    COALESCE('parking=' || attributes_businessparking || ', ', '') ||
                    COALESCE('pricerange=' || attributes_restaurantspricerange2 || ', ', '') ||
                    COALESCE('delivery=' || attributes_restaurantsdelivery || ', ', '') ||
                    COALESCE('takeout=' || attributes_restaurantstakeout || ', ', '') ||
                    COALESCE('outdoorseating=' || attributes_outdoorseating || ', ', '') ||
                    COALESCE('wifi=' || attributes_wifi, ''), 'N/A')
            ) AS document_text,
            'business' AS document_type,
            now() AS last_updated
        FROM read_parquet('{silver_biz_glob}')
    ) TO '{out_biz_file}' (FORMAT PARQUET)
    """
    con.execute(biz_copy_query)
    count_biz = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_biz_file}')").fetchone()[0]
    print(f"[OK] Streamed {out_biz_file} ({count_biz:,} rows)")

    # -------------------------------------------------------------
    # 2. PROCESS REVIEW DOCUMENTS (Streaming Direct to Disk)
    # -------------------------------------------------------------
    print("\n--> Processing review_documents (streaming to disk)...")

    out_rev_file = str(gold_rev_path / "review_documents.parquet").replace("\\", "/")

    rev_copy_query = f"""
    COPY (
        SELECT
            'doc_rev_' || r.review_id AS document_id,
            r.review_id,
            r.business_id,
            r.user_id,
            b.name AS business_name,
            b.city,
            b.state,
            trim(string_split(b.categories, ',')[1]) AS primary_category,
            r.date AS review_date,
            r.stars,
            CASE
                WHEN r.stars >= 4 THEN 'positive'
                WHEN r.stars = 3 THEN 'neutral'
                ELSE 'negative'
            END AS sentiment,
            length(COALESCE(r.text, '')) AS review_length,
            (
                'Review for ' || COALESCE(b.name, 'Unknown Business') ||
                ' (' || COALESCE(b.city, '') || ', ' || COALESCE(b.state, '') || ')' ||
                '\nCategory: ' || COALESCE(trim(string_split(b.categories, ',')[1]), 'N/A') ||
                '\nReview Rating: ' || COALESCE(CAST(r.stars AS VARCHAR), 'N/A') || ' stars' ||
                '\nDate: ' || COALESCE(CAST(r.date AS VARCHAR), 'N/A') ||
                '\nSentiment: ' || CASE WHEN r.stars >= 4 THEN 'positive' WHEN r.stars = 3 THEN 'neutral' ELSE 'negative' END ||
                '\nContent: ' || COALESCE(r.text, '')
            ) AS document_text,
            'review' AS document_type,
            now() AS last_updated
        FROM read_parquet('{silver_rev_glob}') r
        LEFT JOIN (
            SELECT business_id, name, city, state, categories
            FROM read_parquet('{silver_biz_glob}')
        ) b ON r.business_id = b.business_id
    ) TO '{out_rev_file}' (FORMAT PARQUET)
    """
    con.execute(rev_copy_query)
    count_rev = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_rev_file}')").fetchone()[0]
    print(f"[OK] Streamed {out_rev_file} ({count_rev:,} rows)")

    con.close()
    elapsed = time.time() - start_time
    print(f"\n=== Gold Layer Dataset Generation Completed Successfully in {elapsed:.2f} seconds! ===")

if __name__ == "__main__":
    process_gold_layer()
