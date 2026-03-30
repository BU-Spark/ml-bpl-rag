#!/usr/bin/env python3
"""
Streamlit application for the BGE-M3 hybrid RAG + GraphRAG system.

Startup
-------
    streamlit run current_spring2026/BGE_hybrid_retrieval/app.py

Environment variables (set in .env or the shell):
    OPENAI_API_KEY
    BGE_USE_CROSS_ENCODER=1   (optional — enables cross-encoder reranking)
    BGE_INDEX_DIR             (optional — path to FAISS indexes, defaults to indexes/1940)
    BGE_GRAPHRAG_DIR          (optional — path to GraphRAG indexes, defaults to graphrag_index/)
    BGE_USE_GRAPHRAG=1        (optional — enable GraphRAG path, default 1)
    BGE_GRAPHRAG_FULL=1       (optional — use full GraphRAG search vs fast similarity, default 0)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from embedder import BGEM3Embedder
from graph_retrieval import GraphRAGStore, load_graph_store, load_all_graph_stores
from pipeline import RAG
from reranking import BGECrossEncoder
from retrieval import load_indexes

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BPL Hybrid RAG — BGE-M3 + GraphRAG",
    page_icon="📚",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).parent / "indexes" / "1940"
DEFAULT_GRAPHRAG_DIR = Path(__file__).parent / "graphrag_index"


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading BGE-M3 embedder…")
def _load_embedder() -> BGEM3Embedder:
    return BGEM3Embedder(use_fp16=True)


@st.cache_resource(show_spinner="Loading BGE cross-encoder…")
def _load_cross_encoder() -> Optional[BGECrossEncoder]:
    if os.getenv("BGE_USE_CROSS_ENCODER", "0") == "1":
        return BGECrossEncoder(use_fp16=True)
    return None


@st.cache_resource(show_spinner="Loading FAISS indexes…")
def _load_indexes(index_dir: str):
    return load_indexes(Path(index_dir))


@st.cache_resource(show_spinner="Loading GraphRAG indexes…")
def _load_graph_store(graphrag_dir: str) -> Optional[GraphRAGStore]:
    """Load the first available GraphRAG store (or a merged view)."""
    gdir = Path(graphrag_dir)
    if not gdir.exists():
        logger.info(f"GraphRAG directory not found: {gdir}")
        return None

    stores = load_all_graph_stores(gdir)
    if not stores:
        logger.info("No GraphRAG stores found.")
        return None

    # For now, use the first available year's store
    # (a future enhancement could merge across years)
    first_year = next(iter(stores))
    logger.info(f"Using GraphRAG store for year {first_year}")
    return stores[first_year]


def initialize_session() -> None:
    """Load all resources into st.session_state on first run."""
    load_dotenv()

    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=60,
            max_retries=2,
        )

    if "embedder" not in st.session_state:
        st.session_state.embedder = _load_embedder()

    if "cross_encoder" not in st.session_state:
        st.session_state.cross_encoder = _load_cross_encoder()

    if "indexes" not in st.session_state:
        index_dir = os.getenv("BGE_INDEX_DIR", str(DEFAULT_INDEX_DIR))
        try:
            (
                st.session_state.chunk_index,
                st.session_state.title_index,
                st.session_state.chunk_meta,
                st.session_state.title_meta,
                st.session_state.doc_chunks_map,
                st.session_state.sparse_index,
            ) = _load_indexes(index_dir)
            st.session_state.indexes = True
        except Exception as e:
            st.error(f"Failed to load FAISS indexes from {index_dir}: {e}")
            st.stop()

    # GraphRAG store
    if "graph_store" not in st.session_state:
        graphrag_dir = os.getenv("BGE_GRAPHRAG_DIR", str(DEFAULT_GRAPHRAG_DIR))
        try:
            st.session_state.graph_store = _load_graph_store(graphrag_dir)
            if st.session_state.graph_store:
                logger.info("GraphRAG store loaded successfully.")
            else:
                logger.info("No GraphRAG store available — running without GraphRAG.")
        except Exception as e:
            logger.warning(f"Failed to load GraphRAG store: {e}")
            st.session_state.graph_store = None

    if "num_sources" not in st.session_state:
        st.session_state.num_sources = 10

    if "use_cross_encoder" not in st.session_state:
        st.session_state.use_cross_encoder = False

    if "use_graphrag" not in st.session_state:
        st.session_state.use_graphrag = os.getenv("BGE_USE_GRAPHRAG", "1") == "1"

    if "graphrag_full_search" not in st.session_state:
        st.session_state.graphrag_full_search = os.getenv("BGE_GRAPHRAG_FULL", "0") == "1"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_sources(sources: List) -> None:
    if not sources:
        st.info("No sources available for this response.")
        return

    st.subheader("Sources")
    for doc in sources:
        try:
            meta = doc.metadata
            retrieval_type = meta.get("retrieval_type", "standard")

            # GraphRAG community source
            if retrieval_type == "graphrag_community_summary":
                community_id = meta.get("community_id", "")
                score = meta.get("community_score", meta.get("rank", 0))
                label = f"Knowledge Graph Community {community_id}"

                with st.expander(f"🔗 {label}"):
                    if doc.page_content:
                        st.markdown(f"**Community Summary:** {doc.page_content[:500]}…")
                    st.caption(f"Source: GraphRAG | Score: {score:.4f}")
                continue

            # Standard RAG source
            newspaper = meta.get("newspaper", "Unknown Publication")
            issue_date = meta.get("issue_date", "")
            source_url = meta.get("source_url", "")
            source_id = meta.get("source", "")

            label = f"{newspaper}"
            if issue_date:
                label += f" — {issue_date}"

            with st.expander(f"📄 {label}"):
                if doc.page_content:
                    st.markdown(f"**Content preview:** {doc.page_content[:300]}…")

                if source_url:
                    st.markdown(f"**URL:** {source_url}")
                elif source_id:
                    st.markdown(
                        f"**URL:** https://ark.digitalcommonwealth.org/ark:/50959/{source_id}"
                    )

                st.markdown(f"**Source ID:** `{source_id}`")

                topics = meta.get("topics", [])
                if topics:
                    st.markdown(f"**Topics:** {', '.join(topics)}")

                geography = meta.get("geography", [])
                if geography:
                    st.markdown(f"**Geography:** {', '.join(geography)}")

                dense = meta.get("dense_score")
                if dense is not None:
                    st.caption(f"Dense score: {dense:.4f}")

        except Exception as e:
            logger.warning(f"Error displaying source: {e}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📚 BPL Digital Commonwealth — BGE-M3 Hybrid RAG + GraphRAG")

    graphrag_available = (
        hasattr(st.session_state, "graph_store")
        and st.session_state.get("graph_store") is not None
    )
    caption_parts = [
        "Hybrid dense + sparse retrieval with RRF fusion.",
    ]
    if graphrag_available:
        caption_parts.append("GraphRAG community clusters enabled.")
    caption_parts.append("Powered by BAAI/bge-m3, FAISS, Microsoft GraphRAG, and OpenAI.")
    st.caption(" ".join(caption_parts))

    initialize_session()

    # ── Settings panel ────────────────────────────────────────────────────
    if st.button("⚙️ Settings"):
        st.session_state.show_settings = not st.session_state.show_settings

    if st.session_state.show_settings:
        with st.container():
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state.num_sources = st.number_input(
                    "Sources to display",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.num_sources,
                    step=1,
                )
            with col2:
                if st.session_state.cross_encoder is not None:
                    st.session_state.use_cross_encoder = st.checkbox(
                        "Use BGE cross-encoder reranking (slower, more accurate)",
                        value=st.session_state.use_cross_encoder,
                    )
                else:
                    st.info(
                        "Cross-encoder disabled. "
                        "Set BGE_USE_CROSS_ENCODER=1 to enable."
                    )
            with col3:
                graph_store = st.session_state.get("graph_store")
                if graph_store is not None:
                    st.session_state.use_graphrag = st.checkbox(
                        "Enable GraphRAG (community clusters)",
                        value=st.session_state.use_graphrag,
                    )
                    st.session_state.graphrag_full_search = st.checkbox(
                        "Full GraphRAG search (LLM-based, slower)",
                        value=st.session_state.graphrag_full_search,
                    )
                else:
                    st.info(
                        "GraphRAG not available. "
                        "Run graph_indexer.py first to build the knowledge graph."
                    )

            if st.button("❌ Close Settings"):
                st.session_state.show_settings = False
            st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about the Boston Public Library newspaper collections…")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            spinner_msg = "Searching collections"
            if st.session_state.use_graphrag and st.session_state.get("graph_store"):
                spinner_msg += " (dense+sparse+title RAG + GraphRAG in parallel)"
            else:
                spinner_msg += " (dense + sparse + title retrieval + RRF)"
            spinner_msg += "…"

            with st.spinner(spinner_msg):
                response, sources = RAG(
                    llm=st.session_state.llm,
                    embedder=st.session_state.embedder,
                    query=user_input,
                    chunk_index=st.session_state.chunk_index,
                    title_index=st.session_state.title_index,
                    chunk_meta=st.session_state.chunk_meta,
                    title_meta=st.session_state.title_meta,
                    doc_chunks_map=st.session_state.doc_chunks_map,
                    sparse_index=st.session_state.get("sparse_index"),
                    graph_store=st.session_state.get("graph_store"),
                    use_graphrag=st.session_state.use_graphrag,
                    graphrag_full_search=st.session_state.graphrag_full_search,
                    top_k=st.session_state.num_sources,
                    use_cross_encoder=st.session_state.use_cross_encoder,
                    cross_encoder=st.session_state.cross_encoder,
                )

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            display_sources(sources[: st.session_state.num_sources])

    st.markdown("---")
    st.caption(
        "Built with BGE-M3 · FAISS · RRF · Microsoft GraphRAG · LangChain · Streamlit · OpenAI"
    )


if __name__ == "__main__":
    main()
