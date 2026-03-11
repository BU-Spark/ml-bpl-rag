#!/usr/bin/env python3
"""
Streamlit app for the historical newspaper RAG pipeline.

Workflow:
  1. Run build_newspaper_index.py once to build the FAISS indexes.
  2. Launch this app:  streamlit run newspaper_app.py

Required environment variables (.env or shell export):
    OPENAI_API_KEY      — for GPT-4o-mini answer generation
    COHERE_API_KEY      — for Cohere cross-encoder reranking

Optional:
    INDEX_DIR           — path to FAISS indexes directory (default: indexes/1940)
    E5_MODEL            — E5 model name (default: intfloat/e5-base-v2)
    SPACY_MODEL         — spaCy model name (default: en_core_web_sm)
    RAG_THRESHOLD       — similarity threshold for Stage 1 gate (default: 0.30)
    RAG_TOP_K           — articles retrieved per query (default: 10)
    RAG_RERANK_TOP      — articles kept after reranking (default: 5)
    COHERE_WEIGHT       — weight of Cohere score in hybrid (default: 0.70)
    NER_WEIGHT          — weight of NER TF-IDF score in hybrid (default: 0.30)
"""

import os
import sys
import logging
from pathlib import Path

import cohere
import spacy
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from newspaper_RAG import newspaper_rag, load_indexes

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

INDEX_DIR      = os.getenv("INDEX_DIR",      "indexes/1940")
E5_MODEL       = os.getenv("E5_MODEL",       "intfloat/e5-base-v2")
SPACY_MODEL    = os.getenv("SPACY_MODEL",    "en_core_web_sm")
THRESHOLD      = float(os.getenv("RAG_THRESHOLD",  "0.30"))
TOP_K          = int(os.getenv("RAG_TOP_K",        "10"))
RERANK_TOP     = int(os.getenv("RAG_RERANK_TOP",   "5"))
COHERE_WEIGHT  = float(os.getenv("COHERE_WEIGHT",  "0.70"))
NER_WEIGHT     = float(os.getenv("NER_WEIGHT",     "0.30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BPL Newspaper RAG",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stAppHeader {background-color: #1871bd;}
    h1 {color: #1871bd;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading E5 embedding model...")
def load_e5_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading E5 model '{E5_MODEL}' on {device}...")
    return SentenceTransformer(E5_MODEL, device=device)


@st.cache_resource(show_spinner="Loading spaCy NER model...")
def load_spacy() -> spacy.language.Language:
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        st.error(
            f"spaCy model '{SPACY_MODEL}' not found. "
            f"Run:  python -m spacy download {SPACY_MODEL}"
        )
        st.stop()


@st.cache_resource(show_spinner="Loading LLM...")
def load_llm() -> ChatOpenAI:
    # Prefer OpenRouter if that key is present, otherwise fall back to OpenAI
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key     = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        logger.info("Using OpenRouter API.")
        return ChatOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    elif openai_key:
        logger.info("Using OpenAI API.")
        return ChatOpenAI(
            api_key=openai_key,
            model="gpt-4o-mini",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    else:
        st.error(
            "No LLM API key found. Set either OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file."
        )
        st.stop()


@st.cache_resource(show_spinner="Connecting to Cohere...")
def load_cohere() -> cohere.ClientV2:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        st.error("COHERE_API_KEY environment variable is not set.")
        st.stop()
    return cohere.ClientV2(api_key=api_key)


@st.cache_resource(show_spinner="Loading FAISS indexes...")
def load_faiss_indexes():
    index_path = Path(INDEX_DIR)
    if not index_path.exists():
        st.error(
            f"Index directory '{INDEX_DIR}' not found. "
            "Run build_newspaper_index.py first."
        )
        st.stop()
    return load_indexes(INDEX_DIR)


# ── Display helpers ───────────────────────────────────────────────────────────

def display_sources(articles) -> None:
    """Render reranked articles as expandable source cards."""
    if not articles:
        return

    st.markdown("### 📰 Source Articles")
    seen = set()
    for art in articles:
        if art.doc_id in seen:
            continue
        seen.add(art.doc_id)

        label = f"{art.newspaper}  ·  {art.issue_date}"
        with st.expander(f"📄 {label}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                preview = art.full_text[:400].replace("\n", " ") + "..."
                st.markdown(f"**Preview:** {preview}")
                if art.topics:
                    st.markdown(f"**Topics:** {', '.join(art.topics)}")
                if art.geography:
                    st.markdown(f"**Geography:** {', '.join(art.geography)}")
            with col2:
                st.metric("Retrieval", f"{art.retrieval_score:.3f}")
                st.metric("Rerank",    f"{art.combined_score:.3f}")
            st.markdown(f"[🔗 View Original]({art.source_url})")

            if st.session_state.dev_mode:
                with st.expander("🔬 Score breakdown", expanded=False):
                    st.json({
                        "cohere_score":   round(art.cohere_score, 4),
                        "ner_score":      round(art.ner_score, 4),
                        "combined_score": round(art.combined_score, 4),
                    })


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.session_state.dev_mode = st.toggle(
            "Developer mode", value=st.session_state.dev_mode
        )
        if st.session_state.dev_mode:
            st.divider()
            st.caption(f"Index: `{INDEX_DIR}`")
            st.caption(f"E5 model: `{E5_MODEL}`")
            st.caption(f"Threshold: `{THRESHOLD}`")
            st.caption(f"Top-K: `{TOP_K}` → rerank `{RERANK_TOP}`")
            st.caption(f"Weights: Cohere {COHERE_WEIGHT} / NER {NER_WEIGHT}")

    # Header
    st.title("Boston Evening Transcript Archive 📰")
    st.caption(
        "Ask questions about historical news from 1940. "
        "Powered by E5 dense retrieval · Cohere reranking · NER TF-IDF · GPT-4o-mini."
    )

    # Load all resources (cached after first call)
    e5       = load_e5_model()
    nlp      = load_spacy()
    llm      = load_llm()
    cohere_c = load_cohere()
    title_index, full_index, title_meta, chunk_meta, doc_chunks_map = load_faiss_indexes()

    # Suggested queries
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        cols = st.columns(3)
        suggestions = [
            ("🌍 WWII Coverage",    "What was reported about the war in Europe in early 1940?"),
            ("🏛️ Boston Politics",  "What political events happened in Boston in 1940?"),
            ("⚾ Sports News",      "What Boston sports news was covered in January 1940?"),
        ]
        for col, (label, q) in zip(cols, suggestions):
            if col.button(label):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("articles"):
                display_sources(msg["articles"])

    # Input
    user_input = st.chat_input("Ask a question about the 1940 Boston Evening Transcript...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

    # Process last user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        query = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.status("🔍 Searching the archive...", expanded=True) as status:
                    st.write("Embedding query and searching Title/Summary index...")

                    answer, reranked = newspaper_rag(
                        query=query,
                        title_index=title_index,
                        full_index=full_index,
                        title_meta=title_meta,
                        chunk_meta=chunk_meta,
                        doc_chunks_map=doc_chunks_map,
                        e5_model=e5,
                        llm=llm,
                        cohere_client=cohere_c,
                        nlp=nlp,
                        threshold=THRESHOLD,
                        top_k=TOP_K,
                        rerank_top=RERANK_TOP,
                        cohere_weight=COHERE_WEIGHT,
                        ner_weight=NER_WEIGHT,
                    )

                    st.write("Hybrid reranking and generating answer...")
                    status.update(label="✅ Done", state="complete", expanded=False)

                st.markdown(answer)
                display_sources(reranked)

                # Developer: show retrieval pipeline details
                if st.session_state.dev_mode and reranked:
                    with st.sidebar:
                        st.divider()
                        st.subheader("🔬 Pipeline Debug")
                        with st.expander("Reranked articles", expanded=True):
                            for a in reranked:
                                st.markdown(
                                    f"**{a.issue_date}**  "
                                    f"ret={a.retrieval_score:.3f}  "
                                    f"cohere={a.cohere_score:.3f}  "
                                    f"ner={a.ner_score:.3f}  "
                                    f"combined=**{a.combined_score:.3f}**"
                                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "articles": reranked,
                })

            except Exception as exc:
                logger.exception("Error processing query")
                st.error(f"❌ Error: {exc}")

    st.markdown("---")
    st.caption(
        "Pipeline: E5 dense retrieval · Two-stage (Title/Summary → Full Article) · "
        "Cohere + NER TF-IDF hybrid reranking · GPT-4o-mini"
    )


if __name__ == "__main__":
    main()
