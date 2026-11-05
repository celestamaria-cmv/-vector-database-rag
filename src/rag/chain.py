import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.embeddings.embedding_model import get_embedding_model
from src.vector_store.chroma_db import get_collection
from src.vector_store.search import similarity_search

load_dotenv(find_dotenv(), override=True)


def get_rag_chain():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key.startswith("gsk_"):
        raise RuntimeError("GROQ_API_KEY missing or malformed. Check your .env file.")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        groq_api_key=key,
    )

    template = """You are an expert AI assistant. Answer the user's question using ONLY the provided context below. 
If the answer cannot be found in the context, explicitly state: "I cannot find the answer in the provided documents." 
Always cite the source document and page number where applicable based on the metadata.

Context:
{context}

User Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    return llm, prompt


def format_docs_from_results(results):
    formatted = []
    documents = []

    if not results or "documents" not in results or not results["documents"][0]:
        return "", []

    docs_list = results["documents"][0]
    metas_list = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs_list)
    dists_list = results["distances"][0] if results.get("distances") else [None] * len(docs_list)

    for content, metadata, dist in zip(docs_list, metas_list, dists_list):
        metadata = dict(metadata or {})
        if dist is not None:
            metadata["distance"] = dist

        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page", "Unknown Page")
        # PDF loader pages start at 0, so show human page numbers to the LLM
        display_page = page + 1 if isinstance(page, int) else page

        documents.append(Document(page_content=content, metadata=metadata))
        formatted.append(f"Source: {source} (Page: {display_page})\nContent: {content}")

    return "\n\n---\n\n".join(formatted), documents


def ask_rag(query: str, k: int = 3):
    model = get_embedding_model()
    collection = get_collection()
    raw_results = similarity_search(collection=collection, query=query, model=model, k=k)
    context_text, retrieved_docs = format_docs_from_results(raw_results)

    llm, prompt = get_rag_chain()
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": query})

    return response, retrieved_docs