import os
import chromadb
from langchain_core.documents import Document

# Absolute path so ingestion, evaluation, and Streamlit always use the same database
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db_test")


def create_vector_store(chunks, embeddings, persist_directory=DEFAULT_DB_PATH, collection_name="rag_knowledge_base"):
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    texts = [chunk.page_content for chunk in chunks]
    metas_list = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metas_list
    )
    return collection


def get_collection(persist_directory=DEFAULT_DB_PATH, collection_name="rag_knowledge_base"):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection