import streamlit as st
import sys
import os

# Add project root to path so we can import pipeline modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_query, PipelineResult
from retrieval.retriever import RetrievedDocument

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Commonwealth · BPL Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --cream:    #F7F3EC;
    --ink:      #1A1410;
    --sepia:    #7A5C3A;
    --gold:     #C8973A;
    --rust:     #A63D2F;
    --muted:    #8A7B6A;
    --border:   #D9CFC2;
    --card-bg:  #FFFDF9;
    --tag-bg:   #EDE5D8;
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--cream);
    color: var(--ink);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 960px; }

/* ── Masthead ── */
.masthead {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2.5rem;
}
.masthead-eyebrow {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--sepia);
    margin-bottom: 0.6rem;
}
.masthead-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3rem;
    font-weight: 400;
    color: var(--ink);
    line-height: 1.15;
    margin: 0 0 0.5rem;
}
.masthead-title em {
    font-style: italic;
    color: var(--sepia);
}
.masthead-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Search box wrapper ── */
.search-wrapper {
    background: var(--card-bg);
    border: 1.5px solid var(--border);
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(26,20,16,0.06);
}
.search-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--sepia);
    margin-bottom: 0.5rem;
}

/* ── Example query pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
.pill {
    background: var(--tag-bg);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.3rem 0.75rem;
    font-size: 0.78rem;
    color: var(--sepia);
    cursor: pointer;
    font-style: italic;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.8rem 0;
}

/* ── Context banner ── */
.context-banner {
    background: linear-gradient(135deg, #FDF6E8 0%, #F5EDD8 100%);
    border-left: 4px solid var(--gold);
    border-radius: 0 4px 4px 0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 2rem;
    font-size: 0.95rem;
    line-height: 1.65;
    color: var(--ink);
}
.context-banner strong { color: var(--sepia); }

/* ── Results header ── */
.results-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 1.2rem;
}
.results-count {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    color: var(--ink);
}
.results-count span { color: var(--sepia); font-style: italic; }
.results-meta {
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.06em;
}

/* ── Result card ── */
.result-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
}
.result-card:hover {
    border-color: var(--gold);
    box-shadow: 0 4px 18px rgba(200,151,58,0.12);
}
.card-type-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--rust);
    border: 1px solid var(--rust);
    border-radius: 2px;
    padding: 0.15rem 0.5rem;
    margin-bottom: 0.6rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.3rem;
    line-height: 1.3;
}
.card-meta {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.7rem;
    line-height: 1.5;
}
.card-snippet {
    font-size: 0.88rem;
    color: #3D3228;
    line-height: 1.65;
    margin-bottom: 0.8rem;
}
.card-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.card-tag {
    background: var(--tag-bg);
    font-size: 0.72rem;
    color: var(--sepia);
    padding: 0.2rem 0.55rem;
    border-radius: 2px;
    border: 1px solid var(--border);
}
.card-link {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--rust);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-decoration: none;
    border-bottom: 1px solid transparent;
}
.card-link:hover { border-bottom-color: var(--rust); }

/* ── Relevance score bar ── */
.score-row { display: flex; align-items: center; gap: 0.6rem; margin-top: 0.6rem; }
.score-label { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
.score-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--tag-bg);
    border-radius: 2px;
    overflow: hidden;
    max-width: 120px;
}
.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--gold), var(--rust));
    border-radius: 2px;
}
.score-val { font-size: 0.7rem; color: var(--sepia); font-weight: 600; }

/* ── No results ── */
.no-results {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}
.no-results-icon { font-size: 3rem; margin-bottom: 1rem; }
.no-results-title { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: var(--ink); margin-bottom: 0.5rem; }

/* ── Footer ── */
.bpl-footer {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.bpl-footer strong { color: var(--sepia); }
</style>
""", unsafe_allow_html=True)


# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "What happened in Boston in 1900?",
    "Find photographs of Greece",
    "Show me circus posters",
    "Victorian era correspondence",
    "Boston Traveler newspaper 1900",
    "Women's suffrage documents",
]

import re
def linkify_citations(text: str, num_docs: int) -> str:
    """Replace [N] with clickable spans that scroll to result cards."""
    def replace(match):
        n = int(match.group(1))
        if 1 <= n <= num_docs:
            return (
                f'<a href="javascript:void(0)" '
                f'onclick="document.getElementById(\'result-{n}\').scrollIntoView({{behavior:\'smooth\'}})" '
                f'style="color:var(--rust);font-weight:600;cursor:pointer;">[{n}]</a>'
            )
        return match.group(0)
    import re
    return re.sub(r'\[(\d+)\]', replace, text)


# ── Helper: format a RetrievedDocument into card fields ──────────────────────
def format_card(doc: RetrievedDocument) -> dict:
    # Determine document type label
    topics = doc.topics or []
    title_lower = (doc.title or "").lower()

    if any(t.lower() in ["photograph", "photography", "photographs"] for t in topics):
        doc_type = "Photograph"
    elif any(t.lower() in ["map", "maps", "cartography"] for t in topics):
        doc_type = "Map"
    elif any(w in title_lower for w in ["traveler", "globe", "herald", "gazette", "journal", "tribune"]):
        doc_type = "Newspaper"
    elif any(t.lower() in ["correspondence", "manuscript", "letter", "papers"] for t in topics):
        doc_type = "Manuscript"
    else:
        doc_type = "Document"

    # Date string
    if doc.issue_date:
        date_str = doc.issue_date
    elif doc.year:
        date_str = str(doc.year[0])
    else:
        date_str = "Date unknown"

    snippet = doc.best_chunk_text[:300] if doc.best_chunk_text else ""
    tags = list(set((doc.topics or []) + (doc.geography or [])))[:5]
    if doc.best_chunk_text:
        # Has chunk text = individual item
        url = f"https://www.digitalcommonwealth.org/search/commonwealth:{doc.ark_id}"
    else:
        # No chunk text = collection-level metadata record
        url = f"https://www.digitalcommonwealth.org/collections/commonwealth:{doc.ark_id}"

    return {
        "type":       doc_type,
        "title":      doc.title or "Untitled",
        "date":       date_str,
        "collection": doc.institution or "Boston Public Library",
        "snippet":    snippet,
        "tags":       tags,
        "score":      round(doc.final_score, 2),
        "url":        url,
    }


# ── Session state ─────────────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "searched" not in st.session_state:
    st.session_state.searched = False
if "context" not in st.session_state:
    st.session_state.context = ""
if "latency_ms" not in st.session_state:
    st.session_state.latency_ms = 0


# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">Boston Public Library · Digital Commonwealth</div>
    <h1 class="masthead-title">Search the <em>Archive</em></h1>
    <p class="masthead-sub">
        Ask anything in plain language — explore photographs, maps, newspapers,
        manuscripts, and more from Massachusetts history.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Search box ────────────────────────────────────────────────────────────────
st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="search-label">Natural Language Query</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    query_input = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder='e.g. "Find photographs of Boston Harbor from the 1800s"',
        value=st.session_state.query,
        key="query_box",
    )
with col_btn:
    search_clicked = st.button("Search →", use_container_width=True, type="primary")

# Example query pills
st.markdown('<div class="search-label" style="margin-top:1rem;">Try an example</div>', unsafe_allow_html=True)
pill_cols = st.columns(3)
for i, example in enumerate(EXAMPLE_QUERIES):
    with pill_cols[i % 3]:
        if st.button(f'"{example}"', key=f"pill_{i}", use_container_width=True):
            st.session_state.query = example
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ── Handle search ─────────────────────────────────────────────────────────────
active_query = st.session_state.query if st.session_state.query else query_input

if search_clicked and query_input.strip():
    st.session_state.query = query_input.strip()
    active_query = query_input.strip()

if active_query and (search_clicked or st.session_state.query):
    with st.spinner("Searching the archive…"):
        try:
            result: PipelineResult = run_query(active_query)
            cards = [format_card(doc) for doc in result.documents]
            st.session_state.results   = cards
            st.session_state.context   = result.generation.response
            st.session_state.latency_ms = result.latency_ms
            st.session_state.searched  = True
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.session_state.searched = False

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.searched and st.session_state.results is not None:
    results = st.session_state.results
    context = st.session_state.context
    latency = st.session_state.latency_ms

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if results:
        # Context banner
        context_with_links = linkify_citations(context, len(results))
        st.markdown(f"""
        <div class="context-banner">
            <strong>About these results —</strong> {context_with_links}
        </div>
        """, unsafe_allow_html=True)

        # Results header
        st.markdown(f"""
        <div class="results-header">
            <div class="results-count">
                Found <span>{len(results)} items</span> for "{st.session_state.query}"
            </div>
            <div class="results-meta">Ranked by relevance · {latency}ms · Digital Commonwealth BPL Subset</div>
        </div>
        """, unsafe_allow_html=True)

        # Result cards
        for i, r in enumerate(results, start=1):
            score_pct = min(int(r["score"] * 100), 100)
            bar_width = score_pct
            tags_html = ''.join(f'<span class="card-tag">{t}</span>' for t in r["tags"])
            st.markdown(f"""
            <div class="result-card" id="result-{i+1}">
                <div class="card-type-badge">{r['type']}</div>
                <div class="card-title">{r['title']}</div>
                <div class="card-meta">
                    {r['date']} &nbsp;·&nbsp; {r['collection']}
                </div>
                <div class="card-snippet">{r['snippet']}</div>
                <div class="card-tags">{tags_html}</div>
                <div class="score-row">
                    <span class="score-label">Relevance</span>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                    <span class="score-val">{score_pct}%</span>
                    &nbsp;&nbsp;
                    <a class="card-link" href="{r['url']}" target="_blank">View in Digital Commonwealth ↗</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="no-results">
            <div class="no-results-icon">🗂️</div>
            <div class="no-results-title">No matching materials found</div>
            <p>Try rephrasing your query, or use one of the example searches above.<br/>
            The full collection spans photographs, maps, newspapers, manuscripts, and more.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bpl-footer">
    <strong>Boston Public Library</strong> · Digital Commonwealth · BPL RAG Search<br>
    A natural language search prototype built with Retrieval-Augmented Generation.<br>
    Results are drawn from digitized items in the BPL subset of Digital Commonwealth.
</div>
""", unsafe_allow_html=True)