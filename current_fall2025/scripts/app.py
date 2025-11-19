#!/usr/bin/env python3
import os
import json
import streamlit as st
import psycopg2
import logging
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from RAG import RAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Boston Public Library Chatbot", page_icon="🤖", layout="wide")
load_dotenv()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

def get_db_conn():
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
    return st.session_state.db_conn

def close_db_conn():
    if "db_conn" in st.session_state and not st.session_state.db_conn.closed:
        st.session_state.db_conn.close()

def initialize_all():
    with st.spinner("🔄 Initializing models and database..."):
        if "llm" not in st.session_state:
            st.session_state.llm = load_llm()
        if "embeddings" not in st.session_state:
            st.session_state.embeddings = load_embeddings()
        conn = get_db_conn()
        st.success("✅ Models and database ready!")
        return st.session_state.llm, st.session_state.embeddings, conn

def process_message(query: str) -> Tuple[str, List]:
    llm = st.session_state.llm
    embeddings = st.session_state.embeddings
    conn = get_db_conn()
    response, sources = RAG(llm, conn, embeddings, query=query)
    return response, sources

def display_sources(sources: List) -> None:
    if not sources:
        return
    for doc in sources:
        try:
            metadata = doc.metadata
            source_id = metadata.get("source", "Unknown")
            title = metadata.get("title_info_primary_tsi", "Untitled")
            doc_url = f"https://www.digitalcommonwealth.org/search/{source_id}"
            with st.expander(f"📄 {title} (ID: {source_id})", expanded=False):
                st.markdown(f"**Preview:** {doc.page_content[:300]}...")
                st.markdown(f"[🔗 View Original Source]({doc_url})")
        except Exception as e:
            logger.warning(f"Error displaying document: {e}")

def dedup_sources(sources: List) -> List:
    seen = {}
    for doc in sources:
        key = json.dumps(doc.metadata, sort_keys=True)
        if key not in seen:
            seen[key] = doc
    return list(seen.values())

# ============================
# Developer Mode + Runtime Logs (ADD ONLY)
# ============================

# ---- Developer Mode Toggle ----
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False

with st.sidebar:
    st.header("🛠 Developer Options")
    st.session_state.dev_mode = st.checkbox(
        "Enable Developer Mode",
        value=st.session_state.dev_mode
    )


# ---- Persistent Hidden Log Placeholder ----
if "log_hidden_placeholder" not in st.session_state:
    st.session_state.log_hidden_placeholder = st.empty()


# ---- Visible Placeholder When Dev Mode ON ----
if st.session_state.dev_mode:
    with st.sidebar:
        st.subheader("📟 Runtime Logs")
        visible_log_placeholder = st.empty()
else:
    visible_log_placeholder = None


# ---- Streamlit Log Handler Class ----
class DevViewLogHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.buffer = ""

    def emit(self, record):
        msg = self.format(record)
        self.buffer += msg + "\n"

        # Print to sidebar (if visible)
        if self.placeholder:
            self.placeholder.code(self.buffer)

        # ALSO print to terminal
        print(msg)


# ---- Attach Handler Once ----
if "dev_log_handler" not in st.session_state:
    handler = DevViewLogHandler(st.session_state.log_hidden_placeholder)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    st.session_state.dev_log_handler = handler

# ---- Update Handler Target ----
if st.session_state.dev_mode and visible_log_placeholder:
    st.session_state.dev_log_handler.placeholder = visible_log_placeholder
else:
    st.session_state.dev_log_handler.placeholder = st.session_state.log_hidden_placeholder


def main():
    st.title("📚 Boston Public Library RAG Chatbot 🤖")
    st.caption("Ask about historical events, archives, or images in the Digital Commonwealth collection.")
    llm, embeddings, conn = initialize_all()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                display_sources(msg["sources"])

    user_input = st.chat_input("Type your question here...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching archives and generating answer..."):
                response, sources = process_message(user_input)
                sources = dedup_sources(sources)
                st.markdown(response)
                display_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "sources": sources}
                )
        # ============================
        # ADD: Developer Debug Info
        # ============================
        debug_info = {
            "query": user_input,
            "response_preview": response[:500],
            "num_sources": len(sources),
            "source_ids": [d.metadata.get("source") for d in sources],
        }
        if st.session_state.dev_mode:
            with st.expander("🛠 Developer Debug Info", expanded=False):
                st.json(debug_info)


            st.markdown("---")
            st.caption("Built with LangChain + Streamlit + PostgreSQL (pgvector).")
            st.caption("Access digitized photographs, manuscripts, audio, and other historical materials through natural-language search.")

if __name__ == "__main__":
    main()
