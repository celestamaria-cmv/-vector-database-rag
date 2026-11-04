from src.document_processing.load_documents import load_all_documents
from src.embeddings.embedding_model import process_documents
from src.vector_store.chroma_db import create_vector_store
from src.vector_store.search import similarity_search


# STEP 1: Load documents
documents = load_all_documents(
    pdf_paths=["data/pdfs/vector_database_rag_group_project_plan (1).pdf"],
    web_urls=[],
    youtube_urls=[]
)

print("Documents loaded:", len(documents))


# STEP 2: Generate chunks and embeddings
chunks, embeddings, model = process_documents(documents)

print("Chunks generated:", len(chunks))
print("Embeddings generated:", len(embeddings))


# STEP 3: Store chunks and embeddings in ChromaDB
collection = create_vector_store(
    chunks,
    embeddings,
    persist_directory="data/chroma_db_test"
)

print("Data successfully stored in ChromaDB!")


# STEP 4: Search for similar documents
query = "What is the purpose of this RAG project?"

results = similarity_search(
    collection,
    query,
    model,
    k=3
)


# STEP 5: Display the top 3 search results
print("\nTOP 3 SEARCH RESULTS")

for i, text in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print(text)