import chromadb
from langchain_core.documents import Document

def create_vector_store(chunks, embeddings,persist_directory="data/chroma_db",collection_name="rag_knowledge_base"):
    # Check that every chunk has an embedding
    if len(chunks)!=len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")
   
    # Create a persistent ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or access a collection
    collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
    )

    # Extract text and metadata from the chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Create unique IDs for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Store everything in ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    return collection
