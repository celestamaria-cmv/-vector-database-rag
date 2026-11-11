import os
import re
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
)
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
os.environ["USER_AGENT"] = (
    "vector-database-rag-student-project/1.0"
)
# --------------------------------------------------
# 1. PDF DOCUMENT LOADER
# --------------------------------------------------
def load_pdf(file_path: str) -> list[Document]:
    """Load PDF pages as LangChain Documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    for document in documents:
        document.metadata["type"] = "pdf"
        document.metadata["source"] = path.name
    print(f"Loaded {len(documents)} PDF pages.")
    return documents
# --------------------------------------------------
# 2. WEB ARTICLE LOADER
# --------------------------------------------------
def load_web_article(url: str) -> list[Document]:
    """Load a web page as LangChain Documents."""
    loader = WebBaseLoader(web_path=url)
    documents = loader.load()
    for document in documents:
        document.metadata["type"] = "website"
        document.metadata["source"] = url
    print(f"Loaded {len(documents)} web documents.")
    return documents
# --------------------------------------------------
# 3. YOUTUBE TRANSCRIPT LOADER
# --------------------------------------------------
def load_youtube_transcript(
    video_url: str
) -> list[Document]:
    """
    Load a YouTube transcript directly using
    youtube-transcript-api.
    """
    match = re.search(
        r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        video_url,
    )
    if not match:
        raise ValueError("Invalid YouTube URL")
    video_id = match.group(1)
    api = YouTubeTranscriptApi()
    transcript = api.fetch(
        video_id,
        languages=["en"],
    )
    transcript_text = " ".join(
        snippet.text for snippet in transcript
    )
    if not transcript_text.strip():
        raise ValueError("YouTube transcript is empty")
    document = Document(
        page_content=transcript_text,
        metadata={
            "source": video_url,
            "video_id": video_id,
            "type": "youtube_transcript",
            "language": "en",
        },
    )
    print("Loaded YouTube transcript successfully.")
    print("Transcript characters:", len(transcript_text))
    return [document]
# --------------------------------------------------
# 4. MARKDOWN DOCUMENT LOADER
# --------------------------------------------------
def load_markdown(
    file_path: str
) -> list[Document]:
    """
    Load Markdown knowledge-base documents.
    Used for the Vector Database educational
    documents in the knowledge_base folder.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {file_path}"
        )
    loader = TextLoader(
        str(path),
        encoding="utf-8",
    )
    documents = loader.load()
    for document in documents:
        document.metadata["type"] = "markdown"
        document.metadata["source"] = path.name
    print(f"Loaded Markdown: {path.name}")
    return documents
# --------------------------------------------------
# 5. LOAD ALL MARKDOWN DOCUMENTS
# --------------------------------------------------
def load_markdown_folder(
    folder_path: str = "knowledge base"
) -> list[Document]:
    """
    Automatically load all Markdown files
    from the knowledge_base folder.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"Knowledge base folder not found: {folder_path}"
        )
    markdown_files = sorted(folder.glob("*.md"))
    if not markdown_files:
        raise ValueError(
            "No Markdown files found in knowledge_base."
        )
    all_documents = []
    for file_path in markdown_files:
        print(f"\nLoading Markdown: {file_path.name}")
        documents = load_markdown(str(file_path))
        all_documents.extend(documents)
    print(
        f"\nTotal Markdown documents loaded: "
        f"{len(all_documents)}"
    )
    return all_documents
# --------------------------------------------------
# 6. UNIFIED DOCUMENT LOADER
# --------------------------------------------------
def load_all_documents(
    pdf_paths: list[str] | None = None,
    web_urls: list[str] | None = None,
    youtube_urls: list[str] | None = None,
    markdown_paths: list[str] | None = None,
) -> list[Document]:
    """
    Load documents from multiple sources:
    1. PDF files
    2. Web articles
    3. YouTube transcripts
    4. Markdown files
    Returns all documents as one list.
    """
    all_documents = []
    pdf_paths = pdf_paths or []
    web_urls = web_urls or []
    youtube_urls = youtube_urls or []
    markdown_paths = markdown_paths or []
    # PDF DOCUMENTS
    for pdf_path in pdf_paths:
        print(f"\nLoading PDF: {pdf_path}")
        try:
            all_documents.extend(
                load_pdf(pdf_path)
            )
        except Exception as error:
            print(f"PDF loading failed: {error}")
    # WEB ARTICLES
    for url in web_urls:
        print(f"\nLoading web article: {url}")
        try:
            all_documents.extend(
                load_web_article(url)
            )
        except Exception as error:
            print(f"Website loading failed: {error}")
    # YOUTUBE VIDEOS
    for url in youtube_urls:
        print(f"\nLoading YouTube video: {url}")
        try:
            all_documents.extend(
                load_youtube_transcript(url)
            )
        except Exception as error:
            print(f"YouTube loading failed: {error}")
    # MARKDOWN DOCUMENTS
    for path in markdown_paths:
        print(f"\nLoading Markdown: {path}")
        try:
            all_documents.extend(
                load_markdown(path)
            )
        except Exception as error:
            print(f"Markdown loading failed: {error}")
    print("\n===================================")
    print("UNIFIED DOCUMENT LOADING COMPLETED")
    print("===================================")
    print("Total documents loaded:", len(all_documents))
    return all_documents
# --------------------------------------------------
# 7. MAIN TEST
# --------------------------------------------------
if __name__ == "__main__":
    documents = load_all_documents(
        # PDF DOCUMENTS
        pdf_paths=[
            "data/pdfs/vector_database_rag_group_project_plan (1).pdf"
        ],
        # VECTOR DATABASE WEBSITES
        web_urls=[
            "https://www.pinecone.io/learn/retrieval-augmented-generation/",
        ],
        # VECTOR DATABASE YOUTUBE VIDEOS
        youtube_urls=[
            "https://www.youtube.com/watch?v=pINxGpEPsKE",
        ],
        # NINE VECTOR DATABASE MARKDOWN DOCUMENTS
        markdown_paths=[
            "knowledge base/01_introduction_to_vector_databases.md",
            "knowledge base/02_embeddings_and_text_vectors.md",
            "knowledge base/03_chromadb_fundamentals.md",
            "knowledge base/04_hnsw_indexing.md",
            "knowledge base/05_similarity_and_distance_metrics.md",
            "knowledge base/06_top_k_retrieval.md",
            "knowledge base/07_vector_databases_vs_traditional_databases.md",
            "knowledge base/08_vector_databases_in_rag.md",
            "knowledge base/09_vector_database_evaluation_and_glossary.md",
        ],
    )
    print("\n========== DOCUMENT LOADER TEST ==========")
    print("Total documents loaded:", len(documents))
    for index, document in enumerate(documents):
        print(f"\nDocument {index + 1}")
        print("\nContent:")
        print(document.page_content[:200])
        print("\nMetadata:")
        print(document.metadata)
    print("\n========== LOADING COMPLETED ==========")