import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from src.rag.chain import ask_rag

# Load environment variables (.env wins over stale shell variables)
load_dotenv(find_dotenv(), override=True)

# Page configuration
st.set_page_config(
    page_title="Vector Database-Powered RAG Pipeline",
    page_icon="🤖",
    layout="centered"
)

st.title("📚 RAG Knowledge Assistant")
st.markdown("Ask questions about your documents, powered by **LangChain**, **ChromaDB**, and **Groq**.")

# Check for Groq API Key
if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ Groq API key not found! Please set your `GROQ_API_KEY` in a `.env` file.")

# User query input box
query = st.text_input(
    "Enter your question about the project plan or documents:",
    placeholder="e.g., What are Person 1's tasks?"
)

if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Searching vector database and generating answer..."):
            try:
                # Run RAG pipeline
                answer, source_docs = ask_rag(query, k=3)

                if not source_docs:
                    st.warning(
                        "No chunks were retrieved. The vector database may be empty. "
                        "Run the ingestion script first."
                    )

                # Display Answer
                st.subheader("💡 Answer")
                st.write(answer)

                # Display Sources / Citations in an Expander
                with st.expander("🔍 View Retrieved Source Chunks & Citations"):
                    if not source_docs:
                        st.info("No relevant sources were retrieved.")
                    else:
                        for i, doc in enumerate(source_docs, 1):
                            source = doc.metadata.get("source", "Unknown Source")
                            page = doc.metadata.get("page", "N/A")
                            if isinstance(page, int):
                                page += 1
                            dist = doc.metadata.get("distance")

                            st.markdown(f"**Source {i}:** `{source}` | **Page:** `{page}`")
                            if dist is not None:
                                st.caption(
                                    f"Cosine distance: {dist:.4f} | "
                                    f"Cosine similarity: {1 - dist:.4f}"
                                )
                            st.text(doc.page_content)
                            st.divider()

            except Exception as e:
                st.error(f"An error occurred: {e}")