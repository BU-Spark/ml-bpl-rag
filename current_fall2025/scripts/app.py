#!/usr/bin/env python3
import os
import streamlit as st
import psycopg2
import logging
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from RAG import RAG

# ------------------------------------------------------------------------------
# 🌎 INITIAL SETUP
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Boston Public Library Chatbot (pgvector)",
    page_icon="🤖",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# ------------------------------------------------------------------------------
# 🧠 MODEL & EMBEDDINGS (safe to cache globally)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_embeddings():
    logger.info("Loading HuggingFace embedding model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_llm():
    logger.info("Initializing OpenAI Chat model...")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ------------------------------------------------------------------------------
# 🧩 PER-SESSION POSTGRES CONNECTION (thread-safe)
# ------------------------------------------------------------------------------
def get_db_conn():
    """Each Streamlit user session gets its own psycopg2 connection."""
    if "db_conn" not in st.session_state or st.session_state.db_conn.closed:
        st.session_state.db_conn = psycopg2.connect(
            host=os.getenv("PGHOST"),
            port=os.getenv("PGPORT"),
            database=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            sslmode=os.getenv("PGSSLMODE", "prefer"),
        )
        st.session_state.db_conn.autocommit = True
        logger.info("✅ PostgreSQL connection established for this session.")
    return st.session_state.db_conn


def close_db_conn():
    """Cleanly close DB connection when session ends."""
    if "db_conn" in st.session_state and not st.session_state.db_conn.closed:
        st.session_state.db_conn.close()
        logger.info("🔒 Closed PostgreSQL connection for this session.")



# ------------------------------------------------------------------------------
# 🔧 APP INITIALIZATION
# ------------------------------------------------------------------------------
def initialize_all():
    """Load LLM, embeddings, and DB with spinner + notifications."""
    with st.spinner("🔄 Initializing models and database... this may take up to 1 minute (first load only)"):
        if "llm" not in st.session_state:
            st.session_state.llm = load_llm()

        if "embeddings" not in st.session_state:
            st.session_state.embeddings = load_embeddings()

        conn = get_db_conn()

        st.success("✅ Models and database ready!")
        return st.session_state.llm, st.session_state.embeddings, conn


# ------------------------------------------------------------------------------
# 💬 RAG QUERY PROCESSOR
# ------------------------------------------------------------------------------
def process_message(query: str) -> Tuple[str, List]:
    """Run full retrieval-augmented generation pipeline."""
    llm = st.session_state.llm
    embeddings = st.session_state.embeddings
    conn = get_db_conn()

    logger.info(f"Processing query: {query}")
    response, sources = RAG(llm, conn, embeddings, query=query)
    return response, sources


# ------------------------------------------------------------------------------
# 🧾 SOURCE DISPLAY (simplified, no image/audio)
# ------------------------------------------------------------------------------
def display_sources(sources: List) -> None:
    """Show retrieved documents and preview content."""
    if not sources:
        st.info("No sources available for this response.")
        return

    st.subheader("📚 Sources")
    for doc in sources:
        try:
            metadata = doc.metadata
            source_id = metadata.get("source", "Unknown")
            title = metadata.get("title_info_primary_tsi", "Untitled")
            doc_url = f"https://www.digitalcommonwealth.org/search/{source_id}"

            with st.expander(f"📄 {title} (ID: {source_id})"):
                st.markdown(f"**Preview:** {doc.page_content[:300]}...")
                st.markdown(f"[🔗 View Original Source]({doc_url})")

        except Exception as e:
            logger.warning(f"Error displaying document: {e}")
            st.error("⚠️ Error displaying this source.")


# ------------------------------------------------------------------------------
# 🚀 MAIN STREAMLIT APP
# ------------------------------------------------------------------------------
def main():
    st.title("📖 Digital Commonwealth Chatbot (Postgres + LangChain RAG)")

    # Initialize system
    llm, embeddings, conn = initialize_all()

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "num_sources" not in st.session_state:
        st.session_state.num_sources = 10

    # Show past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask about Boston Public Library archives...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching archives and generating answer..."):
                response, sources = process_message(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_sources(sources[:st.session_state.num_sources])

    st.markdown("---")
    st.caption("Built with LangChain + Streamlit + PostgreSQL pgvector + OpenAI")


if __name__ == "__main__":
    main()