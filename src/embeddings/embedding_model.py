from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.document_processing.load_documents import load_all_documents
from src.document_processing.chunking import split_documents
# --------------------------------------------------
# 1. LOAD EMBEDDING MODEL
# --------------------------------------------------
def get_embedding_model():
    """
    Load Sentence Transformers embedding model.
    Model: all-MiniLM-L6-v2
    Dimensions: 384
    Device: CPU
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
# --------------------------------------------------
# 2. GENERATE EMBEDDINGS
# --------------------------------------------------
def generate_embeddings(
    chunks: list[Document],
    model
):
    """Convert document chunks into numerical vectors."""
    texts = [chunk.page_content for chunk in chunks]
    return model.embed_documents(texts)
# --------------------------------------------------
# 3. DOCUMENT PROCESSING PIPELINE
# --------------------------------------------------
def process_documents(documents: list[Document]):
    print("\n========== DOCUMENT PROCESSING ==========")
    # STEP 1: CHUNKING
    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )
    print("Original documents:", len(documents))
    print("Generated chunks:", len(chunks))
    if not chunks:
        raise ValueError("No chunks generated.")
    # STEP 2: EMBEDDING MODEL
    print("\nLoading embedding model...")
    model = get_embedding_model()
    # STEP 3: GENERATE EMBEDDINGS
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(chunks, model)
    # STEP 4: VALIDATION
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding count mismatch.")
    if any(len(vector) != 384 for vector in embeddings):
        raise ValueError("Expected 384-dimensional embeddings.")
    print("Total embeddings:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))
    # STEP 5: SOURCE CLASSIFICATION
    print("\n========== SOURCE VALIDATION ==========")
    source_counts = {
        "PDF": 0,
        "Website": 0,
        "YouTube": 0,
        "Markdown": 0,
        "Other": 0,
    }
    for chunk in chunks:
        source_type = chunk.metadata.get("type")
        if source_type == "pdf":
            source_counts["PDF"] += 1
        elif source_type == "website":
            source_counts["Website"] += 1
        elif source_type == "youtube_transcript":
            source_counts["YouTube"] += 1
        elif source_type == "markdown":
            source_counts["Markdown"] += 1
        else:
            source_counts["Other"] += 1
    for source, count in source_counts.items():
        print(f"{source} chunks: {count}")
    print("\nEmbedding validation successful.")
    return chunks, embeddings, model
# --------------------------------------------------
# 4. MAIN TEST
# --------------------------------------------------
if __name__ == "__main__":
    print("\n========== LOADING KNOWLEDGE BASE ==========")
    documents = load_all_documents(
        pdf_paths=[
            "data/pdfs/vector_database_rag_group_project_plan (1).pdf"
        ],
        web_urls=[
            "https://www.pinecone.io/learn/retrieval-augmented-generation/"
        ],
        youtube_urls=[
            "https://www.youtube.com/watch?v=pINxGpEPsKE"
        ],
        markdown_paths=[
            "knowledge_base/01_introduction_to_vector_databases.md",
            "knowledge_base/02_embeddings_and_text_vectors.md",
            "knowledge_base/03_chromadb_fundamentals.md",
            "knowledge_base/04_hnsw_indexing.md",
            "knowledge_base/05_similarity_and_distance_metrics.md",
            "knowledge_base/06_top_k_retrieval.md",
            "knowledge_base/07_vector_databases_vs_traditional_databases.md",
            "knowledge_base/08_vector_databases_in_rag.md",
            "knowledge_base/09_vector_database_evaluation_and_glossary.md",
        ],
    )
    # PROCESS DOCUMENTS
    chunks, embeddings, model = process_documents(documents)
    # DISPLAY RESULTS
    print("\n========== FINAL RESULTS ==========")
    print("\nFirst chunk:")
    print(chunks[0].page_content[:200])
    print("\nMetadata:")
    print(chunks[0].metadata)
    print("\nFirst embedding (first 5 values):")
    print(embeddings[0][:5])
    print("\nTotal documents:", len(documents))
    print("Total chunks:", len(chunks))
    print("Total embeddings:", len(embeddings))
    print("\nPerson 1 pipeline completed successfully.")