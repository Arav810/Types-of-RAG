# Graph RAG — Neo4j Knowledge Graph

## What this does
Implements a Graph-based RAG pipeline using Neo4j as the knowledge store. Unlike vector RAG which retrieves similar text chunks, Graph RAG stores and queries structured relationships between entities — enabling multi-hop relational reasoning.

## Two Approaches Demonstrated

### 1. Automatic Graph Construction (LLMGraphTransformer)
- Unstructured text (Elon Musk biography) is passed to `LLMGraphTransformer`
- The LLM automatically extracts entities (nodes) and relationships (edges)
- The resulting knowledge graph is stored in Neo4j

### 2. Structured Dataset (Movie Graph)
- A movie dataset is loaded into Neo4j via CSV
- Nodes: `Person`, `Movie`, `Genre`
- Relationships: `ACTED_IN`, `DIRECTED`, `IN_GENRE`
- Natural language questions are translated to Cypher queries via `GraphCypherQAChain`

## Why Graph RAG?
Vector RAG cannot answer relational questions like:
- "Which directors have worked with actor X across multiple genres?"
- "What movies connect actor A to actor B within 2 hops?"

Graph RAG handles these natively through graph traversal.

## Pipeline Steps
1. **Connect** to Neo4j AuraDB
2. **Construct graph** — either via `LLMGraphTransformer` (unstructured text) or direct CSV load
3. **Refresh schema** — LangChain reads the graph schema automatically
4. **Natural language to Cypher** — `GraphCypherQAChain` generates and executes Cypher queries
5. **Answer** — LLM formulates a natural language response from query results

## Key Libraries
- `langchain_community.graphs` — `Neo4jGraph`
- `langchain_experimental.graph_transformers` — `LLMGraphTransformer`
- `langchain.chains` — `GraphCypherQAChain`
- `langchain_groq` — `ChatGroq` for fast LLM inference

## Requirements
- Neo4j AuraDB credentials (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` in `.env`)
- Groq API key (`GROQ_API_KEY` in `.env`)

## How to Run
```bash
pip install -r ../requirements.txt
# Add Neo4j and Groq credentials to .env
jupyter notebook
```

## Example Queries
- `"Who was the director of the movie Casino?"` → `Martin Scorsese`
- `"What genre is GoldenEye?"` → `Adventure, Action, Thriller`
