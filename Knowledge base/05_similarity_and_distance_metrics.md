# Similarity Search and Distance Metrics

## 1. Similarity search
Similarity search compares a query embedding with stored embeddings and returns the closest candidates. The meaning of “closest” depends on the distance or similarity metric configured for the collection.

## 2. Cosine similarity
Cosine similarity measures the angle between two vectors:

cosine_similarity(A, B) = (A · B) / (||A|| ||B||)

For non-zero vectors, a larger cosine similarity means the vectors point in more similar directions. It is commonly used for text embeddings.

## 3. Cosine distance
Some databases return cosine distance rather than cosine similarity. A common definition is:

cosine_distance = 1 - cosine_similarity

Under this definition, lower distance indicates greater similarity. For example, a distance of 0.20 is a closer match than 0.70. Do not describe a returned distance as a similarity score without checking the database's metric definition.

## 4. Euclidean distance
Euclidean distance is the straight-line distance between two points in vector space. Smaller values indicate closer vectors. The exact ranking can differ from cosine distance when vector magnitudes vary.

## 5. Inner product
The inner product (dot product) combines vector direction and magnitude. Its interpretation depends on whether vectors are normalized and on the database's implementation.

## 6. Choosing a metric
The appropriate metric depends on the embedding model, whether vectors are normalized, and the task. For this project, the collection is intended to use cosine distance. Verify the actual collection configuration rather than assuming it.

## 7. Practical warning
Distance alone is not a reliable test of whether a question is answerable. A database can return the nearest available records even when none contains the answer. Use relevance evaluation and a grounded LLM fallback in addition to retrieval scores.

## References
- Chroma collections and distance configuration: https://cookbook.chromadb.dev/core/collections/
- OpenAI embeddings guide: https://developers.openai.com/api/docs/guides/embeddings
