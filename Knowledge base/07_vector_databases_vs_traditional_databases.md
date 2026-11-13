# Vector Databases vs Traditional Databases

## 1. Different primary purposes
Traditional relational databases are designed for structured data and operations such as filtering, joins, transactions, and exact comparisons. Vector databases are designed to store vector embeddings and retrieve records by vector similarity.

## 2. Example query difference
A traditional SQL query might find records where `category = 'networking'` or where a product price is below a specified value. A vector search might find passages semantically related to “how to make a database search faster” even if those exact words do not appear.

## 3. They are complementary
A vector database does not replace every traditional database. Many applications use both:
- a relational database for users, orders, permissions, and transactional records;
- a vector index for semantic retrieval;
- metadata filters to narrow vector search by attributes.

## 4. Strengths of vector databases
- semantic retrieval based on embeddings;
- nearest-neighbour search over high-dimensional vectors;
- association of vectors with documents and metadata;
- useful retrieval component for RAG and recommendation systems.

## 5. Limitations and design considerations
- embedding generation adds processing cost;
- approximate search can trade exactness for speed;
- quality depends on the embedding model and source data;
- vector similarity does not establish truth;
- updates, deletion, access control, and metadata design still need careful application logic.

## 6. Summary
Use a traditional database when the core requirement is structured transactional data and exact queries. Use vector search when the core requirement is finding semantically similar content. In many systems, the best architecture combines both.

## References
- Chroma concepts: https://cookbook.chromadb.dev/core/concepts/
- OpenAI embeddings guide: https://developers.openai.com/api/docs/guides/embeddings
