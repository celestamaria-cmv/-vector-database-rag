# HNSW Indexing

## 1. What does an index do?
A vector index helps a database find vectors close to a query vector. Without an index, a system may need to compare the query with every stored vector. That exhaustive approach becomes expensive as the collection grows.

## 2. What is HNSW?
HNSW stands for Hierarchical Navigable Small World. It is an approximate nearest-neighbour (ANN) search method based on a multi-layer graph. Each node represents a vector, and graph links connect selected neighbouring nodes.

## 3. How search works conceptually
1. Search begins in an upper, sparse graph layer.
2. The algorithm moves through links toward nodes closer to the query.
3. It descends through lower layers, which contain more nodes and finer detail.
4. It explores candidate neighbours and returns a ranked set of approximate nearest matches.

The hierarchy allows the search to navigate large vector collections without exhaustively comparing every vector.

## 4. Approximate does not mean guaranteed exact
HNSW is an approximate nearest-neighbour method. It aims to provide a useful balance between search speed, memory use, and recall. Depending on index parameters and data distribution, it may not return the mathematically exact nearest neighbours every time.

## 5. Recall and performance
Recall measures how many of the true nearest neighbours are found by the approximate search. More search effort can improve recall but may increase latency. Index construction and search parameters affect this trade-off.

## 6. HNSW in ChromaDB
Chroma uses HNSW as a vector index for supported single-node configurations. The collection's distance space and HNSW parameters affect how vectors are indexed and searched. The exact configuration API depends on the Chroma version.

## References
- Original HNSW paper: https://pubmed.ncbi.nlm.nih.gov/30602420/
- IEEE paper page: https://ieeexplore.ieee.org/abstract/document/8594636/
- Chroma configuration: https://cookbook.chromadb.dev/core/configuration/
