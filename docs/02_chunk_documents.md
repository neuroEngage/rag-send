# Notebook Execution & Cell Mapping: `02_chunk_documents.ipynb`

> **Module Title**: Document Chunking & Text Preparation

> **Source Notebook**: [02_chunk_documents.ipynb](file:///c:/Users/shank/Downloads/yelp_project/notebooks/02_chunk_documents.ipynb)

---

## Cell-by-Cell Code & Output Mapping

### Markdown Section 1
# Notebook 02: Document Chunking & Text Preparation
This notebook implements text chunking for Gold Layer business and review documents using windowed character segmentation (`CHUNK_SIZE=1000`, `OVERLAP=200`).

---

### Code Cell #1
```python
import duckdb
import pandas as pd
from pathlib import Path

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR

GOLD_BIZ_PATH = PROJECT_ROOT / 'data' / '02_gold' / 'business_documents'
GOLD_REV_PATH = PROJECT_ROOT / 'data' / '02_gold' / 'review_documents'
CHUNKS_DIR = PROJECT_ROOT / 'data' / '03_chunks'
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
OVERLAP = 200
STEP = CHUNK_SIZE - OVERLAP

print(f'Chunk Configuration:')
print(f'  - Chunk Size: {CHUNK_SIZE} characters')
print(f'  - Overlap:    {OVERLAP} characters')
print(f'  - Step Size:  {STEP} characters')
```

**Exact Runtime Output:**
```text
Chunk Configuration:
  - Chunk Size: 1000 characters
  - Overlap:    200 characters
  - Step Size:  800 characters
```

---

### Code Cell #2
```python
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    if not text or not isinstance(text, str):
        return []
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return chunks

# Test chunking on sample document
sample_doc = "Yelp RAG Document Chunking Test. " * 50
sample_chunks = chunk_text(sample_doc, chunk_size=100, overlap=20)
print(f"Sample Input Length:  {len(sample_doc)} chars")
print(f"Total Chunks Created: {len(sample_chunks)}")
print(f"Chunk #0: {sample_chunks[0][:60]}...")
print(f"Chunk #1: {sample_chunks[1][:60]}...")
```

**Exact Runtime Output:**
```text
Sample Input Length:  1650 chars
Total Chunks Created: 21
Chunk #0: Yelp RAG Document Chunking Test. Yelp RAG Document Chunking ...
Chunk #1: ent Chunking Test. Yelp RAG Document Chunking Test. Yelp RAG...
```

---

### Code Cell #3
```python
con = duckdb.connect()
biz_glob = str(GOLD_BIZ_PATH / "*.parquet").replace("\\", "/")

biz_df = con.execute(f"SELECT document_id, business_name, city, state, primary_category, business_rating, document_text FROM read_parquet('{biz_glob}') LIMIT 2000").df()

biz_chunk_records = []
for idx, row in biz_df.iterrows():
    doc_id = row["document_id"]
    text = row["document_text"]
    chunks = chunk_text(text)
    for c_idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{c_idx}"
        biz_chunk_records.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "chunk_number": c_idx,
            "business_name": row["business_name"],
            "city": row["city"],
            "state": row["state"],
            "primary_category": row["primary_category"],
            "business_rating": row["business_rating"],
            "chunk_text": chunk,
            "document_type": "business"
        })

biz_chunks_df = pd.DataFrame(biz_chunk_records)
print("=======================================================")
print("           BUSINESS DOCUMENT CHUNKING METRICS         ")
print("=======================================================")
print(f"Total Documents Processed: {len(biz_df):,}")
print(f"Total Chunks Generated:    {len(biz_chunks_df):,}")
print(f"Avg Chunks per Document:   {len(biz_chunks_df)/len(biz_df):.2f}")
display(biz_chunks_df.head(5))
```

**Exact Runtime Output:**
```text
=======================================================
           BUSINESS DOCUMENT CHUNKING METRICS         
=======================================================
Total Documents Processed: 2,000
Total Chunks Generated:    2,000
Avg Chunks per Document:   1.00
                        chunk_id                    document_id  chunk_number          business_name         city state       primary_category  business_rating                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             chunk_text document_type
doc_biz_--30_8IhuyMHbSOcNWd6DQ_0 doc_biz_--30_8IhuyMHbSOcNWd6DQ             0          Action Karate      Jamison    PA               Trainers              3.5                                                                                                                                                                                                                                                                                                                                                                                                      Business Name: Action Karate\nPrimary Category: Trainers\nCategories: Trainers, Active Life, Fitness & Instruction, Karate, Martial Arts\nLocation: 2235 York Rd, Jamison, PA 18929\nCoordinates: Lat 40.2553619, Lon -75.0883992\nRating: 3.5 stars (9 reviews)\nPrice Range: N/A\nOperating Status: Open\nHours: \nFeatures: creditcards=True,       business
doc_biz_--FcbSxK1AoEtEAxOgBaCw_0 doc_biz_--FcbSxK1AoEtEAxOgBaCw             0       Victory Car Wash    Riverview    FL               Car Wash              3.5                                                                                                                                                                                                                                                                                                                                                                                                                                             Business Name: Victory Car Wash\nPrimary Category: Car Wash\nCategories: Car Wash, Automotive\nLocation: 3820 US-301 S, Riverview, FL 33578\nCoordinates: Lat 27.9143642, Lon -82.3477533\nRating: 3.5 stars (40 reviews)\nPrice Range: N/A\nOperating Status: Open\nHours: \nFeatures: creditcards=True,       business
doc_biz_--SJXpAa0E-GCp2smaHf0A_0 doc_biz_--SJXpAa0E-GCp2smaHf0A             0             Winn Dixie    Riverview    FL                Grocery              2.5                                                                                                                                           Business Name: Winn Dixie\nPrimary Category: Grocery\nCategories: Grocery, Beer, Wine & Spirits, Food\nLocation: 10667 Big Bend Rd, Riverview, FL 33579\nCoordinates: Lat 27.7913331653, Lon -82.3322398549\nRating: 2.5 stars (13 reviews)\nPrice Range: 2\nOperating Status: Open\nHours: Monday: 7:0-23:0 | Tuesday: 7:0-23:0 | Wednesday: 7:0-23:0 | Thursday: 7:0-23:0 | Friday: 7:0-23:0 | Saturday: 7:0-23:0 | Sunday: 7:0-23:0\nFeatures: creditcards=True, parking={'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}, pricerange=2, delivery=True, takeout=True,       business
doc_biz_--ZVrH2X2QXBFdCilbirsw_0 doc_biz_--ZVrH2X2QXBFdCilbirsw             0  Chris's Sandwich Shop      Ardmore    PA American (Traditional)              4.5 Business Name: Chris's Sandwich Shop\nPrimary Category: American (Traditional)\nCategories: American (Traditional), Restaurants, Pizza, Sandwiches, Meat Shops, Food, Wraps, Delis, Specialty Food, Salad\nLocation: 1531 W Wynnewood Rd, Ardmore, PA 19003\nCoordinates: Lat 39.9972988, Lon -75.2922074\nRating: 4.5 stars (32 reviews)\nPrice Range: 1\nOperating Status: Closed\nHours: Monday: 11:0-21:0 | Tuesday: 11:0-21:0 | Wednesday: 11:0-21:0 | Thursday: 11:0-21:0 | Friday: 11:0-21:0 | Saturday: 11:0-21:0 | Sunday: 11:0-21:0\nFeatures: creditcards=True, parking={'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}, pricerange=1, delivery=False, takeout=True, outdoorseating=False, wifi=u'free'      business
doc_biz_--a_r_w1HTsOY-fagPeNKg_0 doc_biz_--a_r_w1HTsOY-fagPeNKg             0 Binford Farmers Market Indianapolis    IN         Farmers Market              4.0                                                                                                                                                                                                                                                                                                   Business Name: Binford Farmers Market\nPrimary Category: Farmers Market\nCategories: Farmers Market, Food\nLocation: 62nd and Binford Blvd, Indianapolis, IN 46250\nCoordinates: Lat 39.8686498404, Lon -86.0839500278\nRating: 4.0 stars (5 reviews)\nPrice Range: 3\nOperating Status: Closed\nHours: Saturday: 8:0-13:0 | \nFeatures: parking={'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}, pricerange=3,       business
```

---

### Code Cell #4
```python
rev_glob = str(GOLD_REV_PATH / "*.parquet").replace("\\", "/")

rev_df = con.execute(f"SELECT document_id, review_id, business_id, business_name, city, state, stars, sentiment, document_text FROM read_parquet('{rev_glob}') LIMIT 2000").df()

rev_chunk_records = []
for idx, row in rev_df.iterrows():
    doc_id = row["document_id"]
    text = row["document_text"]
    chunks = chunk_text(text)
    for c_idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{c_idx}"
        rev_chunk_records.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "chunk_number": c_idx,
            "business_name": row["business_name"],
            "city": row["city"],
            "state": row["state"],
            "stars": row["stars"],
            "sentiment": row["sentiment"],
            "chunk_text": chunk,
            "document_type": "review"
        })

rev_chunks_df = pd.DataFrame(rev_chunk_records)
print("=======================================================")
print("            REVIEW DOCUMENT CHUNKING METRICS          ")
print("=======================================================")
print(f"Total Documents Processed: {len(rev_df):,}")
print(f"Total Chunks Generated:    {len(rev_chunks_df):,}")
print(f"Avg Chunks per Document:   {len(rev_chunks_df)/len(rev_df):.2f}")
display(rev_chunks_df.head(5))
con.close()
```

**Exact Runtime Output:**
```text
=======================================================
            REVIEW DOCUMENT CHUNKING METRICS          
=======================================================
Total Documents Processed: 2,000
Total Chunks Generated:    2,524
Avg Chunks per Document:   1.26
                        chunk_id                    document_id  chunk_number                                        business_name          city state  stars sentiment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         chunk_text document_type
doc_rev_LJlThM2hOOKBWZAj9YqWVw_0 doc_rev_LJlThM2hOOKBWZAj9YqWVw             0 Homewood Suites by Hilton New Orleans French Quarter   New Orleans    LA    2.0  negative Review for Homewood Suites by Hilton New Orleans French Quarter (New Orleans, LA)\nCategory: Hotels\nReview Rating: 2.0 stars\nDate: 2018-05-30\nSentiment: negative\nContent: I love this place in the beginning. We stayed tor 8 days. Hot breakfast 7 days a week and happy hour M-Thursday included    What's not to like. However, as the week, progressed there were definite problems. \nThe breakfast deteriorated.  There were always empty stations.  No coffee, no eggs, no bread products and they ran out of CATSUP today. \nThe room... the shower was clogged and in order to turn it on you got wet because the door opens on the far side and you have to step in to turn it on.  \nThe sheets on the bed were too small so every night the sheets became a tangled mess because they came undone. Also, the sheets were never changed in in fact some days we had to call for maid service at 4:00 because the room hasn't been done. \n\nThe check in staff was great until the last day. Grumpy had a new meaning. I asked for        review
doc_rev_LJlThM2hOOKBWZAj9YqWVw_1 doc_rev_LJlThM2hOOKBWZAj9YqWVw             1 Homewood Suites by Hilton New Orleans French Quarter   New Orleans    LA    2.0  negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               e never changed in in fact some days we had to call for maid service at 4:00 because the room hasn't been done. \n\nThe check in staff was great until the last day. Grumpy had a new meaning. I asked for water ( they have a coffee and tea station set up).  The lady at checkin told me there was a drinking fountain behind and around the corner of the bathrooms   \nThis was definitely mot the way i wanted to end my trip.        review
doc_rev_LJlUzK-5RniNx7lm88AoTw_0 doc_rev_LJlUzK-5RniNx7lm88AoTw             0                            Popeyes Louisiana Kitchen Wesley Chapel    FL    1.0  negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Review for Popeyes Louisiana Kitchen (Wesley Chapel, FL)\nCategory: Fast Food\nReview Rating: 1.0 stars\nDate: 2021-03-20\nSentiment: negative\nContent: What are they deboning the chicken from scratch? There's at least 5 employees in the lobby.. The car line in the drive thru goes all the way out to the Main Street and the line is hardly moving at all. It just doesn't make sense- avoid this restaurant at all costs! Unless you're obliged to eat here and it's the only thing that is open. This is my first and last time for sure.        review
doc_rev_LJl_stLAdy-0ETC0Pcm17w_0 doc_rev_LJl_stLAdy-0ETC0Pcm17w             0             Molly Maguire's Irish Restaurant And Pub  Phoenixville    PA    3.0   neutral                                                                                                                                                                              Review for Molly Maguire's Irish Restaurant And Pub (Phoenixville, PA)\nCategory: Sandwiches\nReview Rating: 3.0 stars\nDate: 2012-06-23\nSentiment: neutral\nContent: Stopped in Friday after my daughters dance recital.  Started with the Irish nachos, waffle fries with cheese, ham, tomatoes with buffalo sour cream.  Really crispy and hot fries, tomatoes were really good.  Cheese got the fries a little soggy after sitting for a few minutes but really good.  I had the fish and chips.  Awesome, crisp but not greasy, great tartar sauce.  Not a big fan of the chips, no flavor and a little too done.  My wife had the cod sandwich.  Fish was good but the bun was enormous, too big for the fish.  My daughter had the Molly burger.  Pretty good but she filled up on the Irish nachos.  Overall was good, would like to come back for more fun.        review
doc_rev_LJlen_gDOedc_fanRySlMg_0 doc_rev_LJlen_gDOedc_fanRySlMg             0                The Coffee House at Second and Bridge      Franklin    TN    4.0  positive                                                                                                                                                                                                                                                                                                                                                                                                                                    Review for The Coffee House at Second and Bridge (Franklin, TN)\nCategory: Food\nReview Rating: 4.0 stars\nDate: 2019-09-11\nSentiment: positive\nContent: Cute little coffee shop nooked away in the heart of Franklin. The construction right now makes it a hassle, but the Coffee House proves worthy of that little extra effort to get it. \n\nThey have a wide array of food and drink and cute little tables and rooms to relax, read a book, do some work, and catch up with friends. \n\nI recommend their salads and you cannot beat their crepes. Give them a try the next time you're in the area!        review
```

---
