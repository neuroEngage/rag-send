# Notebook Execution & Cell Mapping: `01_read_gold_documents.ipynb`

> **Module Title**: Read & Inspect Gold Layer Documents

> **Source Notebook**: [01_read_gold_documents.ipynb](../notebooks/01_read_gold_documents.ipynb)

---

## Cell-by-Cell Code & Output Mapping

### Markdown Section 1
# Notebook 01: Read & Inspect Gold Layer Documents
This notebook reads the Gold Layer Parquet datasets (`business_documents` and `review_documents`) locally, inspecting document schemas, record counts, and document text structures.

---

### Code Cell #1
```python
import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

# Setup paths relative to project root
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR

GOLD_BIZ_PATH = PROJECT_ROOT / 'data' / '02_gold' / 'business_documents'
GOLD_REV_PATH = PROJECT_ROOT / 'data' / '02_gold' / 'review_documents'

biz_glob = str(GOLD_BIZ_PATH / '*.parquet').replace('\\', '/')
rev_glob = str(GOLD_REV_PATH / '*.parquet').replace('\\', '/')

# Initialize DuckDB connection
con = duckdb.connect()

print(f'Project Root:  {PROJECT_ROOT}')
print(f'Business Path: {GOLD_BIZ_PATH}')
print(f'Review Path:   {GOLD_REV_PATH}')
```

**Exact Runtime Output:**
```text
Project Root:  C:\Users\shank\Downloads\yelp_project
Business Path: C:\Users\shank\Downloads\yelp_project\data\02_gold\business_documents
Review Path:   C:\Users\shank\Downloads\yelp_project\data\02_gold\review_documents
```

---

### Code Cell #2
```python
print("--> Business Documents Schema & Sample Rows:")
biz_df = con.execute(f"""
    SELECT 
        document_id, business_name, city, state, 
        primary_category, business_rating, review_count, price_range, is_open
    FROM read_parquet('{biz_glob}')
    LIMIT 5
""").df()

display(biz_df)
```

**Exact Runtime Output:**
```text
--> Business Documents Schema & Sample Rows:
                      document_id           business_name  ... price_range is_open
0  doc_biz_--30_8IhuyMHbSOcNWd6DQ           Action Karate  ...        None       1
1  doc_biz_--FcbSxK1AoEtEAxOgBaCw        Victory Car Wash  ...        None       1
2  doc_biz_--SJXpAa0E-GCp2smaHf0A              Winn Dixie  ...           2       1
3  doc_biz_--ZVrH2X2QXBFdCilbirsw   Chris's Sandwich Shop  ...           1       0
4  doc_biz_--a_r_w1HTsOY-fagPeNKg  Binford Farmers Market  ...           3       0
```

---

### Code Cell #3
```python
print("--> Sample Business Document Text:")
sample_biz_text = con.execute(f"SELECT document_text FROM read_parquet('{biz_glob}') LIMIT 1").fetchone()[0]
print("------------------ DOCUMENT TEXT ------------------")
print(sample_biz_text)
print("---------------------------------------------------")
```

**Exact Runtime Output:**
```text
--> Sample Business Document Text:
------------------ DOCUMENT TEXT ------------------
Business Name: Action Karate
Primary Category: Trainers
Categories: Trainers, Active Life, Fitness & Instruction, Karate, Martial Arts
Location: 2235 York Rd, Jamison, PA 18929
Coordinates: Lat 40.2553619, Lon -75.0883992
Rating: 3.5 stars (9 reviews)
Price Range: N/A
Operating Status: Open
Hours: 
Features: creditcards=True, 
---------------------------------------------------
```

---

### Code Cell #4
```python
print("--> Review Documents Schema & Sample Rows:")
rev_df = con.execute(f"""
    SELECT 
        document_id, review_id, business_name, city, state, 
        stars, sentiment, review_date
    FROM read_parquet('{rev_glob}')
    LIMIT 5
""").df()

display(rev_df)
```

**Exact Runtime Output:**
```text
--> Review Documents Schema & Sample Rows:
                      document_id               review_id  ... sentiment review_date
0  doc_rev_LJlThM2hOOKBWZAj9YqWVw  LJlThM2hOOKBWZAj9YqWVw  ...  negative  2018-05-30
1  doc_rev_LJlUzK-5RniNx7lm88AoTw  LJlUzK-5RniNx7lm88AoTw  ...  negative  2021-03-20
2  doc_rev_LJl_stLAdy-0ETC0Pcm17w  LJl_stLAdy-0ETC0Pcm17w  ...   neutral  2012-06-23
3  doc_rev_LJlen_gDOedc_fanRySlMg  LJlen_gDOedc_fanRySlMg  ...  positive  2019-09-11
4  doc_rev_LJlk6gRJrkevHTFK8w2ptw  LJlk6gRJrkevHTFK8w2ptw  ...  negative  2020-01-25
```

---

### Code Cell #5
```python
print("--> Sample Review Document Text:")
sample_rev_text = con.execute(f"SELECT document_text FROM read_parquet('{rev_glob}') LIMIT 1").fetchone()[0]
print("------------------ DOCUMENT TEXT ------------------")
print(sample_rev_text)
print("---------------------------------------------------")
con.close()
```

**Exact Runtime Output:**
```text
--> Sample Review Document Text:
------------------ DOCUMENT TEXT ------------------
Review for Homewood Suites by Hilton New Orleans French Quarter (New Orleans, LA)
Category: Hotels
Review Rating: 2.0 stars
Date: 2018-05-30
Sentiment: negative
Content: I love this place in the beginning. We stayed tor 8 days...
---------------------------------------------------
```

---
