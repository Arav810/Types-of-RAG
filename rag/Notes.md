# RAG Notebook — Design Notes & Observations

## Design Decisions
- **Chunk size = 1000, overlap = 200**: Balances retrieval granularity with context preservation. Too small = insufficient context for the LLM; too large = less precise similarity search. Overlap of 200 ensures sentences split across boundaries are not lost.
- **RecursiveCharacterTextSplitter**: Splits on natural boundaries (paragraphs → sentences → words), preserving semantic coherence within chunks better than fixed-size splitting.
- **Two vector stores compared — ChromaDB vs FAISS**: ChromaDB supports persistence across sessions; FAISS is faster for in-memory prototyping. Both tested on the same dataset for comparison.
- **Local LLM (Ollama llama2)**: Keeps all inference local — no external API calls, important for sensitive document contexts.

## ChromaDB vs FAISS — Observations
Both returned the same top result for the attention paper query, confirming that embedding quality drives retrieval more than vector store choice at this dataset size. FAISS was faster; ChromaDB is better for production use where re-indexing on every run would be expensive.

## Limitations & Next Steps
- Single retrieval strategy (dense only) — hybrid search combining BM25 + dense would improve keyword-heavy queries (see `hybridsearrag/`)
- No quantitative evaluation metric — adding RAGAS would allow systematic comparison of chunking strategies
- Graph RAG in `graph/` enables relational queries impossible with vector retrieval alone
