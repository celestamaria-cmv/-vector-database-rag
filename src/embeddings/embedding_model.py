
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.document_processing.load_documents import load_all_documents
from src.document_processing.chunking import split_documents


# --------------------------------------------------
# 1. LOAD EMBEDDING MODEL
# --------------------------------------------------

def get_embedding_model():
    """
    Load the local Sentence Transformers embedding model.

    Model: all-MiniLM-L6-v2
    Dimensions: 384
    Device: CPU
    """

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return model


# --------------------------------------------------
# 2. GENERATE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(chunks: list[Document], model):
    """
    Convert document chunks into numerical vectors.
    Each chunk produces one embedding.
    """

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.embed_documents(texts)

    return embeddings


# --------------------------------------------------
# 3. COMPLETE DOCUMENT PROCESSING PIPELINE
# --------------------------------------------------

def process_documents(documents: list[Document]):
    """
    Complete Person 1 pipeline:

    Documents -> Chunks -> Embeddings

    Returns:
        chunks: List of LangChain Document objects
        embeddings: List of numerical vectors
        model: Embedding model instance
    """

    print("\n========== DOCUMENT PROCESSING ==========")

    # Step 1: Split documents into chunks
    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    print("Total original documents:", len(documents))
    print("Total chunks generated:", len(chunks))

    if not chunks:
        raise ValueError("No chunks were generated from the documents.")

    # Step 2: Load embedding model
    print("\nLoading embedding model...")

    model = get_embedding_model()

    # Step 3: Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(chunks, model)

    print("Total embeddings:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))

    # Step 4: Validate embeddings
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Mismatch between number of chunks and embeddings."
        )

    if any(len(vector) != 384 for vector in embeddings):
        raise ValueError(
            "Unexpected embedding dimensions. Expected 384."
        )

    print("\nEmbedding validation successful.")

    # Step 5: Validate source metadata
    print("\n========== METADATA VALIDATION ==========")

    pdf_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("page") is not None
    ]

    web_chunks = [
        chunk for chunk in chunks
        if (
            chunk.metadata.get("source", "").startswith("https://")
            and chunk.metadata.get("video_id") is None
        )
    ]

    youtube_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("type") == "youtube_transcript"
    ]

    print("PDF chunks:", len(pdf_chunks))
    print("Web chunks:", len(web_chunks))
    print("YouTube chunks:", len(youtube_chunks))

    classified_total = (
        len(pdf_chunks) + len(web_chunks) + len(youtube_chunks)
    )

    print("Total classified chunks:", classified_total)

    if classified_total != len(chunks):
        print("Warning: Some chunks could not be classified.")

    print("\n========== PROCESSING COMPLETED ==========")

    return chunks, embeddings, model


# --------------------------------------------------
# 4. TEST THE COMPLETE PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    # Load all three document types
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
    )

    # Process documents and generate embeddings
    chunks, embeddings, model = process_documents(documents)

    # Display sample results
    print("\n========== FINAL RESULTS ==========")

    print("\nFirst chunk:")
    print(chunks[0].page_content[:200])

    print("\nFirst chunk metadata:")
    print(chunks[0].metadata)

    print("\nFirst embedding (first 5 values):")
    print(embeddings[0][:5])

    print("\nTotal documents:", len(documents))
    print("Total chunks:", len(chunks))
    print("Total embeddings:", len(embeddings))

    print("\nPerson 1 pipeline test completed successfully.")
