# Vector Database-Powered RAG Pipeline

## Project Objective
Build a Retrieval-Augmented Generation (RAG) pipeline using LangChain and ChromaDB, with a focus on vector database concepts: embeddings, HNSW indexing, cosine similarity, and Top-K retrieval.

## Team Responsibilities

### Person 1 (Celesta Maria): Document Processing & Embeddings
- Document loading
- Text preprocessing
- Text chunking
- Embedding generation

### Person 2 (Shalika): Vector Database & Retrieval
- ChromaDB setup
- Vector storage
- Similarity search
- Top-K retrieval

### Person 3 (Daved Tawdros): RAG, UI & Evaluation
- RAG pipeline
- LLM integration
- Prompt engineering
- User interface
- Evaluation

## Technology Stack
- Python
- LangChain
- ChromaDB (HNSW index, cosine space)
- Hugging Face embeddings: all-MiniLM-L6-v2 (Sentence Transformers)
- Groq API with `openai/gpt-oss-120b` for LLM inference
- Streamlit

## How It Works
1. The PDF is loaded and split into chunks (37 chunks).
2. Each chunk is embedded with all-MiniLM-L6-v2.
3. Embeddings, text, and metadata (source, page) are stored in a ChromaDB collection configured with `"hnsw:space": "cosine"`.
4. A user question is embedded with the same model.
5. ChromaDB uses its HNSW index to return the Top-K (k=3) nearest chunks, with cosine distances.
6. The chunks are passed as context to the Groq LLM, which answers with citations, or says it cannot find the answer.

## Vector Database Concepts
- **HNSW**: an approximate nearest-neighbor index built as a layered graph. It searches the graph instead of comparing the query to every vector, which scales to large collections.
- **Cosine distance**: `1 - cosine similarity`. Lower distance means a more similar chunk.
- **Top-K retrieval**: returns the K closest chunks, always K, even if some are not relevant.

## Setup
1. Clone the repository and create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root (see `.env.example`):
   ```
   GROQ_API_KEY=gsk_your_own_key
   GROQ_MODEL=openai/gpt-oss-120b
   ```
   Get a free key at console.groq.com. Never commit `.env`.
3. Make sure the vector database exists in `data/` (see Notes).

## Run
Evaluation:
```
python src/evaluation/test_queries.py
```
Streamlit app:
```
python -m streamlit run app.py
```
Top-K comparison (k = 1, 3, 5):
```
python -m src.evaluation.compare_k
```

## Evaluation Results
The evaluation script passes 4/4 tests and prints the cosine distance of each retrieved chunk.

- The best match for "What database is used for vector storage?" had distance 0.289 (similarity 0.711).
- Top-K always returns K chunks, even weak matches.
- An unanswerable question had closer distances (0.50 to 0.55) than some good answers (0.59 to 0.66), so a distance cutoff alone cannot detect missing evidence. The prompt's fallback instruction handles it.

## Notes
- Page numbers from the PDF loader are 0-based; the app and prompt show them +1.
- The vector store is currently stored in `data/chroma_db_test`.