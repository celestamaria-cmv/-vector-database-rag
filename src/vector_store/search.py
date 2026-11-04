def similarity_search(collection, query, model, k=3):
    query_embedding=model.embed_query(query)
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=k
    )
    return results