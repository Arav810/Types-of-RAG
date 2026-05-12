# Baseline RAG Pipeline

## What this does
Implements a standard Retrieval-Augmented Generation pipeline using LangChain. A user question is answered using content retrieved from source documents rather than the LLM's training data alone.

## Data Sources
- `speech.txt` — MLK "I Have a Dream" speech (plain text)
- Web page — Lilian Weng's blog post on LLM-powered agents (loaded via `WebBaseLoader`)
- `attention.pdf` — "Attention Is All You Need" (Vaswani et al., 2017) — the original Transformer paper

## Pipeline Steps
1. **Ingest** — load documents from text, web, or PDF sources
2. **Chunk** — split into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`) using `RecursiveCharacterTextSplitter`
3. **Embed** — generate vector embeddings using Ollama `nomic-embed-text`
4. **Store** — index embeddings in ChromaDB or FAISS vector store
5. **Retrieve** — similarity search returns the most relevant chunks for a query
6. **Generate** — Ollama `llama2` generates a grounded answer using retrieved context

## Key Libraries
- `langchain_community` — document loaders, vector stores
- `langchain_ollama` — local LLM and embedding inference
- `langchain` — retrieval chain, document chain

## How to Run
```bash
pip install -r ../requirements.txt
# Ensure Ollama is running locally with llama2 and nomic-embed-text models pulled
jupyter notebook
```

## Output
Natural language answers grounded in the source documents, with the retrieval chain traceable back to specific document chunks.
