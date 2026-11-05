from src.embeddings.embedding_model import get_embedding_model
from src.vector_store.chroma_db import get_collection
from src.vector_store.search import similarity_search

QUERY = "What database is used for vector storage?"

model = get_embedding_model()
collection = get_collection()

for k in (1, 3, 5):
    r = similarity_search(collection, QUERY, model, k=k)
    print(f"\nk={k}")
    for rank, (dist, meta) in enumerate(zip(r["distances"][0], r["metadatas"][0]), 1):
        print(f"  #{rank} distance={dist:.4f} similarity={1 - dist:.4f} page={meta.get('page')}")