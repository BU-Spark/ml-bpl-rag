#!/usr/bin/env python3
import os
import json
import logging
import streamlit as st
import psycopg2
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# --- Import Unbundled Modules ---
# We import individual steps to support the unbundled pipeline structure
try:
    from RAG import (
        rephrase_and_expand_query,
        extract_filters_with_llm,
        retrieve_from_pg,
        rerank,
        generate_catalog_summary,
    )
except ImportError as e:
    st.error(
        f"❌ Critical Error: Could not import pipeline modules. Ensure the 'RAG' package is in the same directory. ({e})"
    )
    st.stop()


# --- Page Config & Styling ---
st.set_page_config(
    page_title="BPL Archives Chatbot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stAppHeader {background-color: #1871bd;}
    .main .block-container {padding-top: 2rem;}
    h1 {color: #1871bd;}
    .stChatInput {border-color: #1871bd;}
    </style>
""",
    unsafe_allow_html=True,
)

load_dotenv()

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False
if "error_state" not in st.session_state:
    st.session_state.error_state = None  # {"query": str, "error": str}

# --- Sidebar: Developer Options ---
with st.sidebar:
    st.markdown("### 🛠 Developer Settings")
    st.session_state.dev_mode = st.toggle(
        "Enable Developer Mode", value=st.session_state.dev_mode
    )

    if st.session_state.dev_mode:
        st.divider()
        if "db_conn" in st.session_state and not st.session_state.db_conn.closed:
            st.success("🟢 DB Connected")
        else:
            st.warning("🔴 DB Disconnected")


# --- Core Functions ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def load_llm():
    if running_in_docker():
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")

        return ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )



def get_db_conn():
    if "db_conn" not in st.session_state or st.session_state.db_conn.closed:
        try:
            st.session_state.db_conn = psycopg2.connect(
                host=os.getenv("PGHOST"),
                port=os.getenv("PGPORT"),
                database=os.getenv("PGDATABASE"),
                user=os.getenv("PGUSER"),
                password=os.getenv("PGPASSWORD"),
                sslmode=os.getenv("PGSSLMODE", "prefer"),
            )
            st.session_state.db_conn.autocommit = True
        except Exception as e:
            st.error(f"Database Connection Failed: {e}")
            st.stop()
    return st.session_state.db_conn


def running_in_docker() -> bool:
    """Detect if we're inside a Docker container."""
    # Hugging Face Spaces always runs inside Docker
    return os.path.exists("/.dockerenv") or os.getenv("SPACE_ID") is not None


def process_message(query: str):
    llm = st.session_state.llm
    embeddings = st.session_state.embeddings
    conn = get_db_conn()

    # --- STEP 1: Query Expansion ---
    # Now returns a dictionary with 'text', 'improved', 'expanded'
    expansion_result = rephrase_and_expand_query(query, llm)
    expanded_query = expansion_result["text"]

    # Visualization: Query Expansion
    if st.session_state.dev_mode:
        with st.sidebar:
            st.subheader("🔍 RAG Logic Debug")
            with st.expander("🧠 Query Expansion", expanded=True):
                st.markdown("**Original:**")
                st.info(query)

                st.markdown("**Improved (Core):**")
                st.success(expansion_result["improved"])

                if expansion_result["expanded"]:
                    st.markdown("**Expanded (Context):**")
                    st.info(expansion_result["expanded"])

                st.caption(
                    "ℹ️ Improved and Expanded are combined for the final search vector."
                )

    # --- STEP 2: Filter Extraction (On Expanded Query) ---
    filters = extract_filters_with_llm(expanded_query, llm)

    # Visualization: Filters
    if st.session_state.dev_mode:
        with st.sidebar:
            with st.expander("🎯 Metadata Filters", expanded=True):
                st.json(filters.model_dump(), expanded=True)

    # --- STEP 3: Retrieval (Pass Pre-calculated Filters) ---
    # We pass 'filters' here so retrieval_from_pg DOES NOT call the LLM again.
    retrieved_docs, _ = retrieve_from_pg(
        conn, embeddings, expanded_query, llm, k=100, filters=filters
    )

    if not retrieved_docs:
        return "No documents found for your query.", []

    # --- STEP 4: Reranking ---
    reranked_docs = rerank(retrieved_docs, expanded_query, top_k=10)

    if not reranked_docs:
        return "No relevant items found after reranking.", []

    # --- STEP 5: Summarization ---
    context_text = "\n\n".join(d.page_content for d in reranked_docs if d.page_content)
    summary = generate_catalog_summary(llm, expanded_query, context_text)

    return summary, reranked_docs


def display_error_with_retry(error_message: str, query: str):
    """Display error message with OpenAI-like retry button."""
    error_container = st.container(border=True)
    with error_container:
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown(
                f"""
            <div style="padding: 12px; background-color: #fee; border-left: 4px solid #c33; border-radius: 4px;">
                <strong>❌ Error:</strong> {error_message}
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            if st.button(
                "🔄 Retry", key=f"retry_{id(error_message)}", use_container_width=True
            ):
                st.session_state.error_state = None
                st.rerun()


def display_sources(sources: List):
    if not sources:
        return
    st.markdown("### 📚 Referenced Archives")

    seen = set()
    unique_sources = []
    for doc in sources:
        key = doc.metadata.get("source", str(doc.metadata))
        if key not in seen:
            seen.add(key)
            unique_sources.append(doc)

    for doc in unique_sources:
        try:
            metadata = doc.metadata
            source_id = metadata.get("source", "Unknown")
            title = metadata.get("title_info_primary_tsi", "Untitled")
            doc_url = f"https://www.digitalcommonwealth.org/search/{source_id}"

            with st.expander(f"📄 {title} (ID: {source_id})", expanded=False):
                content_preview = (
                    doc.page_content[:300] + "..."
                    if doc.page_content
                    else "No text content available."
                )
                st.markdown(f"**Preview:** {content_preview}")
                st.markdown(f"[🔗 View Original Source]({doc_url})")
        except Exception as e:
            logger.warning(f"Error displaying document: {e}")


# --- Main UI ---
def main():
    # 1. RENDER UI ELEMENTS FIRST
    st.title("Boston Public Library Archives 🏛️")
    st.caption(
        "Explore history through the Digital Commonwealth collection. Ask about photographs, manuscripts, maps, and more."
    )

    # 2. LOAD RESOURCES
    llm, embeddings, conn = load_llm(), load_embeddings(), get_db_conn()
    st.session_state.llm = llm
    st.session_state.embeddings = embeddings

    # Suggested Queries
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        if col1.button("📸 Old Boston Photos"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "Show me photographs of Boston streets in the 1920s.",
                }
            )
            st.rerun()
        if col2.button("⚾ Baseball History"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "Find pictures of the Boston Red Sox and Fenway Park from the early 1900s.",
                }
            )
            st.rerun()
        if col3.button("🗺️ Civil War Maps"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "Show me maps of the United States from the Civil War era.",
                }
            )
            st.rerun()

    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"
        ):
            st.markdown(msg["content"])
            if msg.get("sources"):
                display_sources(msg["sources"])

    # Input Handling
    user_input = st.chat_input("Type your research question here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

    # Logic Loop
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        query_text = st.session_state.messages[-1]["content"]

        # Check if we're retrying after an error
        is_retry = (
            st.session_state.error_state
            and st.session_state.error_state.get("query") == query_text
        )

        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.status("🧠 Searching Archives...", expanded=True) as status:
                    st.write("🔍 Analyzing query & extracting filters...")

                    response, sources = process_message(query_text)

                    st.write("📚 Retrieving and re-ranking documents...")
                    st.write("✍️ Generating summary...")
                    status.update(
                        label="✅ Answer Ready", state="complete", expanded=False
                    )

                st.markdown(response)
                display_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "sources": sources}
                )

                # Clear error state on successful retry
                st.session_state.error_state = None

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing query: {error_msg}")

                # Store error state for retry
                st.session_state.error_state = {"query": query_text, "error": error_msg}

                # Display error with retry button
                display_error_with_retry(
                    f"Failed to process your query: {error_msg}", query_text
                )

    st.markdown("---")
    st.caption("Built with LangChain + Streamlit + PostgreSQL (pgvector).")
    st.caption(
        "Access digitized photographs, manuscripts, audio, and other historical materials through natural-language search."
    )


if __name__ == "__main__":
    main()
