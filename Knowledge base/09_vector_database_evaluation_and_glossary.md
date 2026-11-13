# Vector Database Evaluation and Glossary

## 1. What should be evaluated?
A vector database RAG project should evaluate retrieval separately from answer generation.

### Retrieval evaluation
- **Relevance:** Are the retrieved chunks related to the question?
- **Recall at K:** Do the top K results include the expected relevant chunks?
- **Ranking:** Are the most useful chunks near the top?
- **Latency:** How long does retrieval take?
- **Metadata correctness:** Are source and page details accurate?

### Answer evaluation
- **Faithfulness:** Are claims supported by retrieved context?
- **Answer relevance:** Does the response answer the question?
- **Citation accuracy:** Do citations point to the correct source?
- **Abstention:** Does the system admit when the evidence is insufficient?

## 2. Suggested test set
Create a balanced test set containing:
1. Questions directly answered by one source.
2. Questions requiring information from multiple chunks.
3. Questions using paraphrases rather than exact document wording.
4. Questions outside the knowledge base.
5. Ambiguous questions.

Run the same questions with K = 1, 3, and 5. Keep the embedding model and collection unchanged so the comparison is fair.

## 3. Glossary
- **ANN:** Approximate nearest-neighbour search; finds close vectors efficiently without guaranteeing exhaustive exact search.
- **Embedding:** Numeric representation of input data.
- **HNSW:** Hierarchical Navigable Small World graph index for approximate nearest-neighbour search.
- **Metadata:** Descriptive key-value information associated with a record.
- **Vector:** Ordered numeric array.
- **Vector database:** Database designed to store, index, and search vector embeddings.
- **Cosine similarity:** Measure based on the angle between two vectors.
- **Cosine distance:** Often defined as one minus cosine similarity; lower is closer.
- **Top-K:** The K highest-ranked or nearest results returned by a retrieval operation.
- **Recall:** Fraction of relevant items successfully retrieved from the expected relevant set.
- **Chunk:** A segment of a larger document used as a unit for embedding and retrieval.
- **Persistence:** Saving database data so it can be reused after restarting an application.

## 4. Suggested demo questions
- What is a vector database?
- What information is stored in a ChromaDB record?
- How does HNSW find approximate nearest neighbours?
- What does a cosine distance of 0.2 mean compared with 0.7?
- Why can Top-K retrieval return irrelevant chunks?
- Why should the same embedding model be used for documents and queries?
- What is the difference between a vector database and a relational database?
- Who won the 2024 football championship? (Use as an out-of-scope test; the system should not invent an answer based on this knowledge base.)

## References
- Chroma concepts: https://cookbook.chromadb.dev/core/concepts/
- HNSW paper: https://pubmed.ncbi.nlm.nih.gov/30602420/
- Sentence Transformers model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
