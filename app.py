# ============================================================
# VECTOR DATABASE-POWERED RAG PIPELINE
# Streamlit application
#
# Technologies:
# - Streamlit: User interface
# - LangChain: LLM integration
# - ChromaDB: Vector database
# - Hugging Face: Embeddings
# - Groq: LLM inference
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import time

import streamlit as st

from dotenv import load_dotenv, find_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.rag.chain import ask_rag


# ------------------------------------------------------------
# 2. LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

# Load API keys and model configuration from the .env file.
# override=True ensures the .env values take priority over
# old environment variables.

load_dotenv(find_dotenv(), override=True)


# ------------------------------------------------------------
# 3. PAGE CONFIGURATION
# ------------------------------------------------------------

# Configure the Streamlit page before displaying any UI.

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# 4. CUSTOM CSS
# ------------------------------------------------------------

# Custom CSS makes the application look like a professional
# AI dashboard rather than a default Streamlit application.

st.markdown("""
<style>

/* Main application background */

.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

/* Main title */

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

/* Subtitle */

.subtitle {
    color: #94a3b8;
    font-size: 15px;
}

/* Metric cards */

.metric-card {
    background: #172033;
    border: 1px solid #263449;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 10px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    color: #22d3ee;
    font-size: 26px;
    font-weight: 800;
    margin-top: 5px;
}

/* Answer containers */

.answer-card {
    background: #172033;
    border: 1px solid #263449;
    border-radius: 12px;
    padding: 20px;
    min-height: 180px;
    line-height: 1.7;
}

/* Section headings */

.section-title {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Small information text */

.info-text {
    color: #94a3b8;
    font-size: 13px;
}

/* Buttons */

.stButton > button {
    background-color: #0891b2;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 700;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #06b6d4;
    color: white;
}

/* Input fields */

.stTextInput input {
    background-color: #172033;
    color: white;
    border-radius: 8px;
}

/* Divider */

hr {
    border-color: #263449;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 5. API KEY VALIDATION
# ------------------------------------------------------------

# Check whether the Groq API key is configured.
# Without this key, neither the base LLM nor RAG can run.

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not GROQ_API_KEY.startswith("gsk_"):

    st.error(
        "Groq API key is missing or invalid. "
        "Please configure GROQ_API_KEY in your .env file."
    )

    st.stop()


# ------------------------------------------------------------
# 6. INITIALIZE GROQ LLM
# ------------------------------------------------------------

@st.cache_resource
def get_base_llm():

    """
    Creates and caches the Groq language model.

    The base LLM is used to answer questions WITHOUT
    retrieving information from the vector database.

    Caching avoids creating a new model object every time
    the Streamlit interface refreshes.
    """

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )


# ------------------------------------------------------------
# 7. BASE LLM RESPONSE
# ------------------------------------------------------------

def get_base_answer(question):

    """
    Generates an answer directly from the Groq LLM.

    Important:
    No query embedding is generated.
    No ChromaDB search is performed.
    No project documents are retrieved.

    This represents the BEFORE RETRIEVAL response.
    """

    llm = get_base_llm()

    start_time = time.perf_counter()

    response = llm.invoke(
        [HumanMessage(content=question)]
    )

    end_time = time.perf_counter()

    # Calculate response time in seconds.

    response_time = end_time - start_time

    return response.content, response_time


# ------------------------------------------------------------
# 8. RAG RESPONSE
# ------------------------------------------------------------

def get_rag_answer(question):

    """
    Executes the existing RAG pipeline.

    The ask_rag function performs:

    1. Load the embedding model.
    2. Connect to ChromaDB.
    3. Convert the question into an embedding.
    4. Search for relevant chunks.
    5. Retrieve the Top-K documents.
    6. Format the retrieved context.
    7. Send context and question to Groq.
    8. Generate the final answer.

    Returns:
        answer: Generated RAG response.
        documents: Retrieved source documents.
        response_time: Total RAG execution time.
    """

    start_time = time.perf_counter()

    answer, documents = ask_rag(
        question,
        k=5
    )

    end_time = time.perf_counter()

    response_time = end_time - start_time

    return answer, documents, response_time


# ------------------------------------------------------------
# 9. FORMAT TIME
# ------------------------------------------------------------

def format_time(seconds):

    """
    Converts seconds into a readable time format.

    Example:
        1.25 seconds -> 1.25 s
    """

    return f"{seconds:.2f} s"


# ------------------------------------------------------------
# 10. DISPLAY METRIC CARD
# ------------------------------------------------------------

def display_metric(label, value):
    """
    Displays a reusable metric card.

    Used for:
        Base LLM time
        RAG total time
        Retrieved chunks
        Top similarity
    """
    st.metric(label=label, value=value)
# ------------------------------------------------------------
# 11. DISPLAY RETRIEVED SOURCES
# ------------------------------------------------------------

def display_sources(documents):

    """
    Displays all chunks retrieved from ChromaDB.

    For each chunk, we show:

    - Source document
    - Page number
    - Cosine distance
    - Cosine similarity
    - Retrieved text

    This makes the retrieval process transparent.
    """

    if not documents:

        st.warning("No source chunks were retrieved.")

        return

    for index, doc in enumerate(documents, start=1):

        metadata = doc.metadata or {}

        source = metadata.get(
            "source",
            "Unknown Source"
        )

        page = metadata.get(
            "page",
            "N/A"
        )

        # PDF page indexes normally start at zero.
        # Convert them into human-readable page numbers.

        if isinstance(page, int):
            page += 1

        distance = metadata.get("distance")

        st.markdown(f"### Source {index}")

        st.caption(
            f"Document: {source} | Page: {page}"
        )

        if distance is not None:

            similarity = 1 - distance

            st.write(
                f"Cosine distance: {distance:.4f}"
            )

            st.write(
                f"Cosine similarity: {similarity:.4f}"
            )

            st.progress(
                max(0.0, min(1.0, float(similarity)))
            )

        st.text(doc.page_content)

        st.divider()


# ------------------------------------------------------------
# 12. HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 RAG Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Vector Database-Powered Retrieval Augmented Generation
    <br>
    LangChain · ChromaDB · Hugging Face · Groq
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ------------------------------------------------------------
# 13. SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.title("⚙️ System Information")

    st.success("Groq API Connected")

    st.info(
        f"LLM Model: {GROQ_MODEL}"
    )

    st.info(
        "Embedding Model: all-MiniLM-L6-v2"
    )

    st.info(
        "Vector Database: ChromaDB"
    )

    st.info(
        "Index: HNSW"
    )

    st.info(
        "Distance Metric: Cosine"
    )

    st.info(
        "Default Top-K: 5"
    )

    st.divider()

    st.markdown("### Pipeline")

    st.markdown("""
    1. User Query
    2. Query Embedding
    3. ChromaDB Retrieval
    4. Top-K Selection
    5. Context Formatting
    6. Groq LLM
    7. Final Answer
    """)


# ------------------------------------------------------------
# 14. USER INPUT
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">Ask Your Question</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "Enter your question about vector databases:",
    placeholder="Example: How does HNSW indexing work?"
)


# ------------------------------------------------------------
# 15. GENERATE ANSWERS
# ------------------------------------------------------------

if st.button(
    "🚀 Compare Base LLM vs RAG",
    use_container_width=True
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            # ------------------------------------------------
            # PHASE 1: BASE LLM
            # ------------------------------------------------

            st.markdown("## 1. Before Retrieval")

            with st.spinner(
                "Generating answer using Base Groq LLM..."
            ):

                base_answer, base_time = get_base_answer(
                    question
                )

            st.success("Base LLM response generated.")

            # ------------------------------------------------
            # PHASE 2: RAG PIPELINE
            # ------------------------------------------------

            st.markdown("## 2. After Retrieval")

            with st.spinner(
                "Embedding query, searching ChromaDB "
                "and generating RAG answer..."
            ):

                rag_answer, source_docs, rag_time = get_rag_answer(
                    question
                )

            st.success("RAG response generated.")

            # ------------------------------------------------
            # PHASE 3: PERFORMANCE METRICS
            # ------------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-title">📊 Performance Metrics</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                display_metric(
                    "Base LLM Time",
                    format_time(base_time)
                )

            with col2:

                display_metric(
                    "RAG Total Time",
                    format_time(rag_time)
                )

            with col3:

                display_metric(
                    "Retrieved Chunks",
                    str(len(source_docs))
                )

            with col4:

                if source_docs:

                    scores = [
                        1 - doc.metadata["distance"]
                        for doc in source_docs
                        if doc.metadata.get("distance") is not None
                    ]

                    top_similarity = (
                        max(scores) if scores else 0
                    )

                else:

                    top_similarity = 0

                display_metric(
                    "Top Similarity",
                    f"{top_similarity:.4f}"
                )

            # ------------------------------------------------
            # PHASE 4: ANSWER COMPARISON
            # ------------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-title">⚖️ Before vs After Comparison</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("🧠 Base LLM")

                st.caption(
                    "Before retrieval · No project context"
                )

                st.markdown(
                    f"""
                    <div class="answer-card">
                    {base_answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric(
                    "Response Time",
                    format_time(base_time)
                )

            with col2:

                st.subheader("📚 RAG Pipeline")

                st.caption(
                    "After retrieval · Uses project context"
                )

                st.markdown(
                    f"""
                    <div class="answer-card">
                    {rag_answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric(
                    "Response Time",
                    format_time(rag_time)
                )

            # ------------------------------------------------
            # PHASE 5: TIME COMPARISON
            # ------------------------------------------------

            st.divider()

            st.markdown(
                "### ⏱️ Execution Time Comparison"
            )

            st.bar_chart(
                {
                    "Execution Time (seconds)": {
                        "Base LLM": base_time,
                        "RAG Pipeline": rag_time
                    }
                }
            )

            if rag_time > base_time:

                difference = rag_time - base_time

                st.info(
                    f"RAG took {difference:.2f} seconds longer "
                    "than the base LLM in this run. "
                    "This includes retrieval and generation."
                )

            # ------------------------------------------------
            # PHASE 6: RETRIEVED CONTEXT
            # ------------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-title">🔍 Retrieved Source Chunks & Citations</div>',
                unsafe_allow_html=True
            )

            with st.expander(
                "View Retrieved Documents",
                expanded=True
            ):

                display_sources(source_docs)

        except Exception as error:

            st.error(
                f"An error occurred while running the pipeline: {error}"
            )

            st.exception(error)


# ------------------------------------------------------------
# 16. FOOTER
# ------------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:13px;">

    Vector Database-Powered RAG Pipeline
    <br>
    Built with LangChain · ChromaDB · Groq · Streamlit

    </div>
    """,
    unsafe_allow_html=True
)