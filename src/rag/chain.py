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

    
    template = """
You are an expert assistant specializing in vector databases,
embeddings, ChromaDB, HNSW indexing, similarity search, and RAG.

Your task is to answer the user's question using the retrieved context.

Instructions:

1. Read ALL the provided context carefully.
2. Identify information relevant to the user's question.
3. Combine information from multiple context chunks when necessary.
4. Understand the meaning of the question, not just exact keyword matches.
5. The user may ask questions using different wording or technical terminology.
   Match concepts by meaning rather than exact words.
6. Explain technical concepts clearly and provide examples
   when the context supports them.
7. Do not invent facts or use information that is not supported by the context.
8. If the context contains only partial information, explain what can
   be answered and clearly identify what is missing.
9. If the answer genuinely cannot be found in the provided context,
   respond exactly:
   "I cannot find the answer in the provided documents."
10. Cite the source document and page number whenever available.

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""
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


def ask_rag(query: str, k: int = 5):
    model = get_embedding_model()
    collection = get_collection()
    raw_results = similarity_search(collection=collection, query=query, model=model, k=k)
    context_text, retrieved_docs = format_docs_from_results(raw_results)

    llm, prompt = get_rag_chain()
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": query})

    return response, retrieved_docs