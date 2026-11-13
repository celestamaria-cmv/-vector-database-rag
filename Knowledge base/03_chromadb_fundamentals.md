# ChromaDB Fundamentals

## 1. What is ChromaDB?
Chroma is an open-source database designed for storing and querying embeddings. It can store vectors together with text documents and metadata, making it useful for semantic search and RAG applications.

## 2. Collections
A collection is a logical group of records that an application queries. A collection can contain IDs, documents, metadata, and embeddings. For a course project, a collection might be named `vector_db_knowledge`.

## 3. Records
A record commonly includes:
- `id`: a unique identifier;
- `document`: the text chunk;
- `embedding`: the numeric vector;
- `metadata`: source details such as filename, page, topic, or chunk number.

Chroma's terminology uses “document” to mean the stored text content, which may be a chunk from a larger file.

## 4. Adding and querying data
During ingestion, the application creates embeddings and adds records to a collection. At query time, it submits a query or query embedding and requests the nearest records. Chroma returns matching records and associated information, including distances.

## 5. Persistence
A persistent Chroma client stores database data on disk so that records can be reused after the application restarts. The application must point to the same persistence directory and collection when it reopens the database.

## 6. Metadata
Metadata supports source tracking and filtering. Useful fields for this project include:
- `source`: original filename or URL;
- `page`: source page number;
- `chunk_id`: position within the source;
- `topic`: topic label.

Metadata values should be simple supported values, and the application should validate required fields before insertion.

## 7. Important implementation note
Chroma versions have changed configuration APIs. Current Chroma documentation describes index settings through collection configuration, while older examples use metadata keys such as `hnsw:space`. Check the installed Chroma version and use the matching API rather than mixing examples from different versions.

## References
- Chroma concepts: https://cookbook.chromadb.dev/core/concepts/
- Chroma collections: https://cookbook.chromadb.dev/core/collections/
- Chroma configuration: https://cookbook.chromadb.dev/core/configuration/
