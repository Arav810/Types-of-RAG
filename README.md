# Types of RAG — LangChain Implementations

A hands-on collection of Retrieval-Augmented Generation (RAG) pipeline implementations built with LangChain, exploring different retrieval architectures from baseline to hybrid and graph-based approaches.

Each folder is a self-contained implementation demonstrating a specific RAG variant, allowing direct comparison of retrieval strategies, performance trade-offs, and use cases.

---

##  Project Structure

```
Types-of-RAG/
│
├── rag/                   # Baseline RAG — text + PDF document retrieval using FAISS and ChromaDB
├── hybridsearrag/         # Hybrid Search RAG — dense + sparse retrieval using Pinecone + BM25
├── graph/                 # Graph RAG — knowledge graph construction and querying with Neo4j
├── groq/                  # Groq-accelerated RAG — LLM inference with Groq API
├── api/                   # FastAPI backend serving RAG pipeline responses
├── chatbot/               # Streamlit-based chatbot frontend
│
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

##  RAG Variants Explained

### 1. Baseline RAG (`rag/`)
Standard RAG pipeline demonstrating:
- **Multi-source document ingestion**: plain text files (`speech.txt`), web pages (via `WebBaseLoader`), and PDFs (`attention.pdf` — the original Transformer paper)
- **Document chunking**: `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`
- **Embeddings**: Ollama `nomic-embed-text` embeddings
- **Vector stores**: both **ChromaDB** and **FAISS** implementations compared side-by-side
- **Retrieval chain**: LangChain `create_retrieval_chain` with `create_stuff_documents_chain`
- **LLM**: Ollama `llama2` for local inference

**Key learning**: foundational RAG pipeline — how documents are loaded, split, embedded, stored, and retrieved to answer questions grounded in source material.

---

### 2. Hybrid Search RAG (`hybridsearrag/`)
Advanced retrieval combining **dense vector search** and **sparse keyword search (BM25)**:
- **Dense embeddings**: HuggingFace `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Sparse encoder**: `BM25Encoder` from `pinecone_text` — TF-IDF based keyword matching
- **Vector store**: Pinecone serverless index (dotproduct metric, supports sparse values)
- **Retriever**: `PineconeHybridSearchRetriever` combining both retrieval signals

**Key learning**: hybrid search returns better results than either dense or sparse alone — semantic similarity (dense) catches meaning, BM25 (sparse) catches exact keyword matches. Demonstrated on a set of travel sentences with comparative queries.

---

### 3. Graph RAG (`graph/`)
Knowledge graph-based retrieval and querying using Neo4j:
- **Graph construction**: `LLMGraphTransformer` converts unstructured text (Elon Musk biography) into graph nodes and relationships automatically
- **Graph database**: Neo4j AuraDB with Cypher query language
- **Movie dataset**: loaded via CSV from a public dataset — nodes: `Person`, `Movie`, `Genre`; relationships: `ACTED_IN`, `DIRECTED`, `IN_GENRE`
- **Natural language to Cypher**: `GraphCypherQAChain` translates English questions into Cypher queries automatically
- **LLM**: Groq `llama3-8b-8192` for fast inference

**Key learning**: graph-structured retrieval enables multi-hop relational queries impossible in standard RAG — e.g. "Who directed a movie starring X that is in genre Y?" Graph RAG is superior for structured, relational data.

---

##  Tech Stack

| Component | Technology |
|---|---|
| RAG Framework | LangChain |
| LLMs | Ollama (llama2), Groq (llama3-8b-8192) |
| Dense Embeddings | Ollama nomic-embed-text, HuggingFace all-MiniLM-L6-v2 |
| Sparse Retrieval | BM25 (pinecone_text) |
| Vector Stores | FAISS, ChromaDB, Pinecone |
| Graph Database | Neo4j |
| API Backend | FastAPI |
| Frontend | Streamlit |
| Containerisation | Docker Compose |

---

##  Getting Started

### Prerequisites
- Python 3.11+
- Ollama installed locally (for baseline RAG)
- API keys: Pinecone, Groq, Neo4j AuraDB, LangChain (set in `.env`)

### Installation

```bash
git clone https://github.com/Arav810/Types-of-RAG.git
cd Types-of-RAG
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the root directory:

```
LANGCHAIN_API_KEY=your_langchain_key
GROQ_API_KEY=your_groq_key
pinecone_api_key=your_pinecone_key
HF_TOKEN=your_huggingface_token
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

### Running Each Variant

**Baseline RAG:**
```bash
cd rag
jupyter notebook
# Open and run the notebook cell by cell
```

**Hybrid Search RAG:**
```bash
cd hybridsearrag
jupyter notebook
# Requires Pinecone index to be created (handled in notebook)
```

**Graph RAG:**
```bash
cd graph
jupyter notebook
# Requires Neo4j AuraDB connection
```

---

##  Retrieval Architecture Comparison

| Approach | Strength | Weakness | Best For |
|---|---|---|---|
| Baseline (FAISS) | Simple, fast, local | Only semantic similarity | General Q&A on documents |
| Hybrid (Pinecone) | Semantic + keyword match | Requires cloud setup | Mixed query types |
| Graph (Neo4j) | Relational, multi-hop | Complex setup | Structured/relational data |

---

##  What I Learned

- Baseline RAG is fast to implement but struggles with exact keyword queries
- Hybrid search (BM25 + dense) consistently outperforms either method alone for mixed query types
- Graph RAG enables entirely new query types — relational reasoning across entities — that vector retrieval cannot support
- LangChain's abstraction layer allows swapping retrieval backends with minimal code changes, making comparative evaluation straightforward

---

##  See Also

- [`Chat-bot-using-RAG`](https://github.com/Arav810/Chat-bot-using-RAG/tree/Qdrant) — Full-stack multimodal RAG chatbot with FastAPI + Streamlit + Docker
- [`langgraph-project-collection`](https://github.com/Arav810/langgraph-project-collection) — Multi-agent LLM systems using LangGraph

---

##  Author

**Arav Chauhan**  
MSc Data Science & Analytics, University of Leeds  
[LinkedIn](https://linkedin.com/in/aravchauhan) | [GitHub](https://github.com/Arav810)
