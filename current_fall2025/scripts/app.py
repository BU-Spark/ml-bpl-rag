#!/usr/bin/env python3
import os
import shutil
import logging
import streamlit as st
import psycopg2
from dotenv import load_dotenv
from typing import List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from RAG import RAG
from image_scraper import DigitalCommonwealthScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 🧩 Streamlit Page Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Boston Public Library RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------------------------------------
# ⚡ Cached Resources
# ------------------------------------------------------------------------------
@st.cache_resource
def load_env():
    load_dotenv()
    return True

@st.cache_resource
def load_embeddings():
    logger.info("🔢 Loading HuggingFace embeddings...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_llm():
    logger.info("🧠 Initializing LLM (OpenAI gpt-4o-mini)...")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=2)

@st.cache_resource
def connect_db():
    logger.info("🗄️ Connecting to PostgreSQL database...")
    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )
    logger.info("✅ Connected to PostgreSQL (pgvector backend)")
    return conn

# ------------------------------------------------------------------------------
# 🧠 Initialize all components with spinner + notifications
# ------------------------------------------------------------------------------
def initialize_all() -> Tuple[ChatOpenAI, HuggingFaceEmbeddings, psycopg2.extensions.connection]:
    with st.spinner("🔄 Initializing models and database... please wait ~30–60 seconds on first load."):
        load_env()
        llm = load_llm()
        embeddings = load_embeddings()
        conn = connect_db()
    st.success("✅ All systems ready! You can start asking questions.")
    return llm, embeddings, conn

# ------------------------------------------------------------------------------
# 📚 Display Retrieved Sources
# ------------------------------------------------------------------------------
def display_sources(sources: List) -> None:
    """Display retrieved sources with metadata, image previews, and links."""
    if not sources:
        st.info("No sources found for this query.")
        return

    st.subheader("📚 Sources Retrieved")

    for doc in sources:
        try:
            metadata = doc.metadata or {}
            source = metadata.get("source", "Unknown Source")
            title = metadata.get("title_info_primary_tsi", "Untitled")
            format_type = str(metadata.get("format", "")).lower()

            expander_title = f"🔊 {title}" if "audio" in format_type else f"📄 {title}"
            with st.expander(expander_title):
                st.markdown(f"**Snippet:** {doc.page_content[:400]}...")
                doc_url = metadata.get("URL") or f"https://www.digitalcommonwealth.org/search/{source}"
                st.markdown(f"**Source ID:** {source}")
                st.markdown(f"**Format:** {format_type if format_type else 'N/A'}")
                st.markdown(f"[Open in Digital Commonwealth ↗️]({doc_url})")

                # Display image if available
                scraper = DigitalCommonwealthScraper()
                images = scraper.extract_images(doc_url)[:1]
                if images:
                    output_dir = "downloaded_images"
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
                    downloaded_files = scraper.download_images(images)
                    st.image(
                        downloaded_files,
                        width=400,
                        caption=[img.get("alt", "Image") for img in images],
                    )

        except Exception as e:
            logger.warning(f"[display_sources] Error: {e}")
            st.error("Error displaying one of the sources.")

# ------------------------------------------------------------------------------
# 💬 Query Handler
# ------------------------------------------------------------------------------
def process_message(query: str, llm: ChatOpenAI, conn, embeddings: HuggingFaceEmbeddings) -> Tuple[str, List]:
    try:
        with st.spinner("🧠 Thinking and retrieving relevant materials..."):
            response, sources = RAG(llm=llm, conn=conn, embeddings=embeddings, query=query)
        return response, sources
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return f"Error processing message: {e}", []

# ------------------------------------------------------------------------------
# 🚀 Main Streamlit App
# ------------------------------------------------------------------------------
def main():
    st.title("📚 Boston Public Library RAG Chatbot 🤖")
    st.caption("Ask about historical events, archives, or images in the Digital Commonwealth collection.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "num_sources" not in st.session_state:
        st.session_state.num_sources = 10

    llm, embeddings, conn = initialize_all()

    with st.expander("⚙️ Settings"):
        st.session_state.num_sources = st.number_input(
            "Number of Sources to Display",
            min_value=1,
            max_value=100,
            value=st.session_state.num_sources,
            step=1,
        )

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            response, sources = process_message(user_input, llm, conn, embeddings)
            if isinstance(response, str):
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_sources(sources[:st.session_state.num_sources])

    # Footer
    st.markdown("---")
    st.caption(
        "Built with LangChain + Streamlit + PostgreSQL (pgvector) for the Boston Public Library’s Digital Commonwealth collections."
    )
    st.caption(
        "Access digitized photographs, manuscripts, audio, and other historical materials directly through natural-language search."
    )

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
