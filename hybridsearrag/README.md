# Hybrid Search RAG

## What this does
Implements a hybrid retrieval pipeline that combines **dense semantic search** and **sparse keyword search (BM25)** using Pinecone and LangChain. This approach retrieves documents that are both semantically similar and lexically relevant to the query, outperforming either method alone.

## Why Hybrid Search?
- **Dense only**: finds semantically similar results but may miss exact keyword matches
- **Sparse (BM25) only**: finds exact keyword matches but misses paraphrases and synonyms
- **Hybrid**: combines both signals for more robust retrieval across varied query types

## Pipeline Steps
1. **Initialise Pinecone** — create a serverless index with `dotproduct` metric (required for sparse value support)
2. **Dense embeddings** — HuggingFace `all-MiniLM-L6-v2` generates 384-dimensional dense vectors
3. **Sparse encoder** — `BM25Encoder` fits TF-IDF weights on the corpus and encodes sparse vectors
4. **Index documents** — both dense and sparse vectors stored together in Pinecone
5. **Hybrid retrieval** — `PineconeHybridSearchRetriever` queries both and merges results

## Demo
Three travel sentences are indexed and queried with both semantic and keyword-style questions to demonstrate the hybrid retrieval advantage.

## Key Libraries
- `langchain_community` — `PineconeHybridSearchRetriever`
- `pinecone` — serverless vector database
- `pinecone_text` — `BM25Encoder` for sparse encoding
- `langchain_huggingface` — `HuggingFaceEmbeddings`

## Requirements
- Pinecone API key (set as `pinecone_api_key` in `.env`)
- HuggingFace token (set as `HF_TOKEN` in `.env`)

## How to Run
```bash
pip install -r ../requirements.txt
# Add Pinecone and HuggingFace keys to .env
jupyter notebook
```
