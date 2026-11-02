
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:

    """
    Split documents into smaller chunks while preserving metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    return chunks



if __name__ == "__main__":

    from src.document_processing.load_documents import load_pdf

    pdf_path = (
        "data/pdfs/"
        "vector_database_rag_group_project_plan (1).pdf"
    )

    documents = load_pdf(pdf_path)

    experiments = [
        {"chunk_size": 300, "chunk_overlap": 50},
        {"chunk_size": 500, "chunk_overlap": 100},
        {"chunk_size": 1000, "chunk_overlap": 200},
    ]

    print("CHUNKING EXPERIMENT RESULTS")

    for experiment in experiments:

        chunks = split_documents(
            documents,
            chunk_size=experiment["chunk_size"],
            chunk_overlap=experiment["chunk_overlap"],
        )

        print("\n------------------------------")
        print("Chunk size:", experiment["chunk_size"])
        print("Overlap:", experiment["chunk_overlap"])
        print("Total chunks:", len(chunks))

        print("First chunk preview:")
        print(chunks[0].page_content[:200])
