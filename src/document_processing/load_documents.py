import os
import re
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

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

    print(f"Loaded {len(documents)} PDF pages.")

    return documents


# --------------------------------------------------
# 2. WEB ARTICLE LOADER
# --------------------------------------------------

def load_web_article(url: str) -> list[Document]:
    """Load a web page as LangChain Documents."""

    loader = WebBaseLoader(web_path=url)
    documents = loader.load()

    print(f"Loaded {len(documents)} web documents.")

    return documents


# --------------------------------------------------
# 3. YOUTUBE TRANSCRIPT LOADER
# --------------------------------------------------

def load_youtube_transcript(video_url: str) -> list[Document]:
    """
    Load a YouTube transcript directly using
    youtube-transcript-api.
    """

    # Extract YouTube video ID from URL
    match = re.search(
        r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        video_url,
    )

    if not match:
        raise ValueError("Invalid YouTube URL")

    video_id = match.group(1)

    # Fetch English transcript
    api = YouTubeTranscriptApi()
    transcript = api.fetch(
        video_id,
        languages=["en"],
    )

    # Combine transcript segments into text
    transcript_text = " ".join(
        snippet.text for snippet in transcript
    )

    # Convert transcript into LangChain Document
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
# 4. UNIFIED DOCUMENT LOADER
# --------------------------------------------------

def load_all_documents(
    pdf_paths: list[str],
    web_urls: list[str],
    youtube_urls: list[str],
) -> list[Document]:
    """
    Load documents from PDF, web articles and YouTube.
    Return all documents as a single list.
    """

    all_documents = []

    # Load PDF files
    for pdf_path in pdf_paths:
        print(f"\nLoading PDF: {pdf_path}")
        all_documents.extend(load_pdf(pdf_path))

    # Load web articles
    for url in web_urls:
        print(f"\nLoading web article: {url}")
        all_documents.extend(load_web_article(url))

    # Load YouTube transcripts
    for url in youtube_urls:
        print(f"\nLoading YouTube video: {url}")
        all_documents.extend(load_youtube_transcript(url))

    return all_documents


# --------------------------------------------------
# 5. MAIN TEST
# --------------------------------------------------

if __name__ == "__main__":

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

    print("\n========== UNIFIED DOCUMENT LOADER ==========")

    print("Total documents loaded:", len(documents))

    # Display document content and metadata
    for index, document in enumerate(documents):

        print(f"\nDocument {index + 1}")

        print("\nContent:")
        print(document.page_content[:150])

        print("\nMetadata:")
        print(document.metadata)

    print("\n========== LOADING COMPLETED ==========")
