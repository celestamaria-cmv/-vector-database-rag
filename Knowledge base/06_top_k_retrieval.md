# Top-K Retrieval

## 1. Definition
Top-K retrieval returns up to K nearest records for a query. If K is 3, the retriever requests the three closest candidates according to the configured metric and index.

## 2. Example
A user asks: “What is HNSW indexing?” The query is embedded, compared with stored vectors, and the retriever requests the top three matches. The returned chunks may explain HNSW, ANN search, or Chroma's index configuration.

## 3. Choosing K
A small K can reduce irrelevant context and keep prompts short, but may omit useful evidence. A larger K can provide broader context, but may add redundant or unrelated chunks and increase processing cost. K should be selected through testing.

## 4. Top-K does not guarantee relevance
Top-K usually ranks the nearest available candidates. It does not guarantee that every returned chunk is relevant or that the answer exists in the collection. If the collection contains no good match, the database can still return weak candidates.

## 5. Evaluate different K values
For this project, compare K = 1, 3, and 5 using the same set of test questions. Record:
- whether a useful source chunk appears;
- whether the top result is relevant;
- answer correctness and faithfulness;
- response time;
- whether extra chunks improve or harm the answer.

## 6. Retrieval metadata
Return source filename, page number, and chunk identifier with each result. This allows the application to display citations and helps the team inspect retrieval failures.

## References
- Chroma query concepts: https://cookbook.chromadb.dev/core/concepts/
- Sentence Transformers semantic search: https://www.sbert.net/docs/sentence_transformer/usage/usage.html
