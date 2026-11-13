# Introduction to Vector Databases

## 1. What is a vector?
A vector is an ordered list of numbers. In machine learning, an embedding model converts an item such as a sentence, paragraph, image, or audio segment into a numeric vector. The vector is designed to represent useful features or meaning from the input.

For example, an embedding model may represent “How do I reset my password?” and “I forgot my login password” with vectors that are close to one another, even though the wording differs.

## 2. What is a vector database?
A vector database stores, indexes, and searches vector embeddings. It commonly stores each vector together with an identifier, the original text or a reference to it, and metadata such as source filename, page number, category, or date.

A vector database is useful when an application needs to find items by semantic similarity rather than only exact keyword matches.

## 3. Why use one?
Traditional keyword search is effective when users know the exact terms. Vector search can find conceptually related content even when the query and document use different wording. Common applications include semantic search, recommendation, image retrieval, duplicate detection, and retrieval-augmented generation (RAG).

## 4. What is stored?
A typical record contains:
- an ID identifying the record;
- an embedding vector;
- source content, often a text chunk;
- metadata describing the source or record.

A vector database does not normally turn an entire PDF file into one single vector. A document is usually extracted, divided into chunks, and each chunk is embedded and stored as a separate record.

## 5. Vector database in a RAG system
During ingestion, text chunks are embedded and stored. At question time, the user's query is embedded using a compatible embedding model. The database searches for nearby vectors and returns their associated text. The retrieved text is then provided to a language model as context.

## 6. Important limitations
Vector similarity is not the same as factual correctness. A nearest result can still be irrelevant. Retrieval quality depends on document quality, chunking, the embedding model, index settings, distance metric, and query design. Applications should evaluate retrieval and provide a fallback when evidence is insufficient.

## References
- Chroma concepts: https://cookbook.chromadb.dev/core/concepts/
- OpenAI embeddings guide: https://developers.openai.com/api/docs/guides/embeddings
