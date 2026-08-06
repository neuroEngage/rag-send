# AWS Cloud Deployment Guide for Yelp RAG

This guide explains how to deploy the end-to-end Yelp RAG pipeline directly on AWS using **AWS Glue**, **Amazon S3**, **Amazon Bedrock**, and **Amazon OpenSearch Serverless**.

---

## 1. AWS Architecture Overview

```
 ┌────────────────┐     ┌─────────────────────┐     ┌──────────────────────────────────┐
 │ AWS Glue ETL   │ ──> │ Amazon S3 Gold      │ ──> │ Amazon Bedrock Knowledge Bases   │
 │ (PySpark Job)  │     │ (Parquet Data)      │     │  - Embedding: Titan Text v2      │
 └────────────────┘     └─────────────────────┘     │  - Store: OpenSearch Serverless  │
                                                    └────────────────┬─────────────────┘
                                                                     │
                                                                     ▼
                                                    ┌──────────────────────────────────┐
                                                    │ AWS Bedrock RetrieveAndGenerate  │
                                                    │  - Model: Claude 3 Haiku / Titan │
                                                    └──────────────────────────────────┘
```

---

## Step 1: Execute AWS Glue ETL Job

1. Upload [`glue_gold_rag_etl.py`](file:///c:/Users/shank/Downloads/yelp_project/yelp-rag/scripts/glue_gold_rag_etl.py) to S3:
   ```bash
   aws s3 cp yelp-rag/scripts/glue_gold_rag_etl.py s3://<your-bucket-name>/scripts/glue_gold_rag_etl.py
   ```

2. Create & Run AWS Glue PySpark Job:
   - **Job Type**: Spark (PySpark 3.4 / Glue 4.0)
   - **IAM Role**: `AWSGlueServiceRole` with read/write access to `<your-bucket-name>`
   - **Script Path**: `s3://<your-bucket-name>/scripts/glue_gold_rag_etl.py`
   - **Job Arguments**:
     - `--INPUT_SILVER_PATH` = `s3://<your-bucket-name>/data/silver_layer/`
     - `--OUTPUT_GOLD_PATH` = `s3://<your-bucket-name>/gold_layer/rag/`

3. Output created in S3:
   - `s3://<your-bucket-name>/gold_layer/rag/business_documents/`
   - `s3://<your-bucket-name>/gold_layer/rag/review_documents/`

---

## Step 2: Create Amazon Bedrock Knowledge Base

1. Navigate to **Amazon Bedrock** Console > **Knowledge Bases** > **Create Knowledge Base**.
2. Select **Data Source**: Amazon S3. Point to `s3://<your-bucket-name>/gold_layer/rag/`.
3. Choose **Embedding Model**: `Amazon Titan Text Embeddings v2` (`amazon.titan-embed-text-v2`).
4. Select **Vector Store**: **Quick create a new vector store** (Amazon OpenSearch Serverless collection automatically managed).
5. Click **Sync Data Source**. Bedrock automatically ingests the `document_text` fields from Parquet and indexes vectors in OpenSearch.

---

## Step 3: Querying the Cloud RAG (Boto3 Python Script)

Create and run the following Python snippet to query your Cloud RAG using AWS Bedrock:

```python
import boto3

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

def query_cloud_rag(prompt, knowledge_base_id):
    response = bedrock_agent_runtime.retrieve_and_generate(
        input={'text': prompt},
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledge_base_id,
                'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
            }
        }
    )
    
    print("=== Generated Answer ===")
    print(response['output']['text'])
    
    print("\n=== Retrived Source Documents ===")
    for citation in response.get('citations', []):
        for ref in citation.get('retrievedReferences', []):
            print(f"- {ref['content']['text'][:150]}...")

if __name__ == "__main__":
    KB_ID = "<YOUR_BEDROCK_KNOWLEDGE_BASE_ID>"
    query_cloud_rag("What are the best pizza places in Las Vegas with good reviews?", KB_ID)
```
