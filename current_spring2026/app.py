import streamlit as st
import sys
import os
import html
import re
import json
import uuid
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_query, PipelineResult
from retrieval.retriever import RetrievedDocument
from retrieval.refine import refine_with_user_query
from evaluation.feedback import log_feedback

st.set_page_config(
    page_title="Digital Commonwealth · BPL Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --cream:   #1A1410;
    --ink:     #F7F3EC;
    --sepia:   #C8A96E;
    --gold:    #C8973A;
    --rust:    #E07050;
    --muted:   #A89880;
    --border:  #3D3028;
    --card-bg: #221A12;
    --tag-bg:  #2E2218;
}

html, body, [class*="css"],
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stBottom"],
[data-testid="stHeader"],
.stApp, .main {
    background-color: var(--cream) !important;
    color: var(--ink) !important;
    font-family: 'Source Sans 3', sans-serif;
}
.block-container {
    background-color: var(--cream) !important;
    padding-top: 0.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 960px !important;
}

/* Pull the first widget block up closer to the masthead divider */
.block-container > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Text input — remove blue focus background */
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextInput"] input:active {
    background-color: #2E2218 !important;
    color: #F7F3EC !important;
    -webkit-text-fill-color: #F7F3EC !important;
    -webkit-box-shadow: none !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stTextInput"]) {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* Smooth scroll — target Streamlit's actual scroll container */
html, body, [data-testid="stAppViewContainer"], .main, .block-container,
section[data-testid="stMain"], div[data-testid="stAppViewBlockContainer"] {
    scroll-behavior: smooth !important;
}

/* Full text expander */
details.full-text-expander {
    margin-top: 0.6rem;
    border-top: 1px solid var(--border);
    padding-top: 0.6rem;
}
details.full-text-expander summary {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sepia);
    cursor: pointer;
    user-select: none;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
details.full-text-expander summary::-webkit-details-marker { display: none; }
details.full-text-expander summary::before {
    content: "▶";
    font-size: 0.6rem;
    transition: transform 0.2s;
    display: inline-block;
}
details.full-text-expander[open] summary::before {
    transform: rotate(90deg);
}
details.full-text-expander summary:hover { color: var(--gold); }
.full-text-body {
    margin-top: 0.75rem;
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.75;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
    padding-right: 0.5rem;
}
.full-text-body::-webkit-scrollbar { width: 4px; }
.full-text-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.result-card:target {
    border-color: #C8973A !important;
    box-shadow: 0 0 0 4px rgba(200,151,58,0.35) !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}

div[data-testid="stButton"] button[kind="primary"] {
    background-color: var(--gold) !important;
    border: none !important;
    border-radius: 4px !important;
    color: #fff !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    width: 100% !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: var(--sepia) !important;
}

div[data-testid="stButton"] button[kind="secondary"] {
    background: var(--tag-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    color: var(--sepia) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.78rem !important;
    font-style: italic !important;
    padding: 0.28rem 0.75rem !important;
    width: 100% !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: var(--gold) !important;
    border-color: var(--gold) !important;
    color: #fff !important;
    font-style: normal !important;
}

.masthead {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2.5rem;
}
.masthead-eyebrow {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--sepia); margin-bottom: 0.6rem;
}
.masthead-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3rem; font-weight: 400;
    color: var(--ink); line-height: 1.15; margin: 0 0 0.5rem;
}
.masthead-title em { font-style: italic; color: var(--gold); }
.masthead-sub {
    font-size: 1rem; color: var(--muted); font-weight: 300;
    max-width: 540px; margin: 0 auto !important;
    display: block; text-align: center !important; line-height: 1.6;
}

.search-wrapper {
    background: var(--card-bg);
    border: 1.5px solid var(--border);
    border-radius: 4px;
    padding: 1.4rem 1.6rem 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(26,20,16,0.06);
}
.search-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--sepia); margin-bottom: 0.4rem;
}

.divider { border: none; border-top: 1px solid var(--border); margin: 1.8rem 0; }

.context-banner {
    background: #2A1E0E;
    border-left: 4px solid var(--gold);
    border-radius: 0 4px 4px 0;
    padding: 1.2rem 1.5rem; margin-bottom: 2rem;
    font-size: 0.95rem; line-height: 1.65; color: var(--ink);
}
.context-banner strong { color: var(--gold); }
.citation-link {
    color: var(--gold); font-weight: 700;
    text-decoration: none; border-bottom: 1px dotted var(--gold);
}
.citation-link:hover { border-bottom-style: solid; color: var(--sepia); }

.results-header {
    display: flex; align-items: baseline;
    justify-content: space-between; margin-bottom: 1.2rem;
}
.results-count { font-family: 'Playfair Display', serif; font-size: 1.35rem; color: var(--ink); }
.results-count span { color: var(--gold); font-style: italic; }
.results-meta { font-size: 0.78rem; color: var(--muted); letter-spacing: 0.06em; }

.result-card {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 1.4rem 1.6rem;
    margin-bottom: 1rem; scroll-margin-top: 80px;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.result-card:hover { border-color: var(--gold); box-shadow: 0 4px 18px rgba(200,151,58,0.12); }
.result-card.citation-highlight {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 4px rgba(200,151,58,0.4) !important;
}
.card-type-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--rust); border: 1px solid var(--rust);
    border-radius: 2px; padding: 0.15rem 0.5rem; margin-bottom: 0.6rem;
}
.card-title {
    font-family: 'Playfair Display', serif; font-size: 1.1rem;
    font-weight: 600; color: var(--ink); margin-bottom: 0.3rem; line-height: 1.3;
}
.card-meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.7rem; line-height: 1.5; }
.card-snippet { font-size: 0.88rem; color: #C8B89A; line-height: 1.65; margin-bottom: 0.8rem; }
.card-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.card-tag {
    background: var(--tag-bg); font-size: 0.72rem; color: var(--sepia);
    padding: 0.2rem 0.55rem; border-radius: 2px; border: 1px solid var(--border);
}
.card-link {
    font-size: 0.78rem; font-weight: 600; color: var(--gold);
    letter-spacing: 0.06em; text-transform: uppercase;
    text-decoration: none; border-bottom: 1px solid transparent;
}
.card-link:hover { border-bottom-color: var(--gold); }
.score-row { display: flex; align-items: center; gap: 0.6rem; margin-top: 0.6rem; }
.score-label { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
.score-bar-bg { flex:1; height:4px; background:var(--border); border-radius:2px; overflow:hidden; max-width:120px; }
.score-bar-fill { height:100%; background:linear-gradient(90deg,var(--gold),var(--rust)); border-radius:2px; }
.score-val { font-size: 0.7rem; color: var(--sepia); font-weight: 600; }

.no-results { text-align:center; padding:4rem 2rem; color:var(--muted); }
.no-results-icon { font-size:3rem; margin-bottom:1rem; }
.no-results-title { font-family:'Playfair Display',serif; font-size:1.4rem; color:var(--ink); margin-bottom:0.5rem; }

/* Align form columns to bottom so button sits flush with input */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    align-items: flex-end !important;
}

/* Pagination buttons */
div[data-testid="stButton"] button[kind="secondary"].pagination {
    border-radius: 4px !important;
}

.bpl-footer {
    text-align:center; padding:2.5rem 1rem 1rem;
    border-top:1px solid var(--border); margin-top:3rem;
    font-size:0.78rem; color:var(--muted); letter-spacing:0.05em;
}
.bpl-footer strong { color: var(--sepia); }

/* Thumbs up/down feedback buttons (per-result).
   - Empty state (kind=secondary): outlined icon, dark pill, muted color.
   - Filled state (kind=primary): gold background, filled icon, white.
   Material Symbols supports fill via font-variation-settings. */
div[data-testid="stButton"] button:has(span[class*="material-symbols"]) {
    border-radius: 6px !important;
    font-style: normal !important;
    padding: 0.35rem 0.5rem !important;
    min-height: 0 !important;
}
div[data-testid="stButton"] button[kind="secondary"]:has(span[class*="material-symbols"]) {
    background: var(--tag-bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
}
div[data-testid="stButton"] button[kind="secondary"]:has(span[class*="material-symbols"]):hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}
div[data-testid="stButton"] button[kind="primary"]:has(span[class*="material-symbols"]) {
    background: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    color: #fff !important;
}
div[data-testid="stButton"] button[kind="secondary"] span[class*="material-symbols"] {
    font-variation-settings: 'FILL' 0 !important;
    font-size: 1.15rem !important;
}
div[data-testid="stButton"] button[kind="primary"] span[class*="material-symbols"] {
    font-variation-settings: 'FILL' 1 !important;
    font-size: 1.15rem !important;
}

/* Hide Streamlit's built-in "Press Cmd+Enter to apply" hint on text inputs */
[data-testid="InputInstructions"],
[data-testid="stTextAreaInstructions"],
[data-testid="stWidgetInstructions"],
div[data-testid="stTextArea"] small,
div[data-testid="stTextInput"] small {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "What happened in Boston in 1900?",
    "Find photographs of Greece",
    "Show me circus posters",
    "Victorian era correspondence",
    "Boston Traveler newspaper 1900",
    "Women's suffrage documents",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def linkify_citations(text: str, num_docs: int) -> str:
    """Use plain anchor hrefs — browser handles scroll natively, no JS needed."""
    def replace(match):
        n = int(match.group(1))
        if 1 <= n <= num_docs:
            return (
                '<a class="citation-link" href="#result-' + str(n) + '">'
                '[' + str(n) + ']</a>'
            )
        return match.group(0)
    return re.sub(r'\[(\d+)\]', replace, text)

def format_card(doc: RetrievedDocument) -> dict:
    topics = doc.topics or []
    title_lower = (doc.title or "").lower()
    if any(t.lower() in ["photograph","photography","photographs"] for t in topics):
        doc_type = "Photograph"
    elif any(t.lower() in ["map","maps","cartography"] for t in topics):
        doc_type = "Map"
    elif any(w in title_lower for w in ["traveler","globe","herald","gazette","journal","tribune"]):
        doc_type = "Newspaper"
    elif any(t.lower() in ["correspondence","manuscript","letter","papers"] for t in topics):
        doc_type = "Manuscript"
    else:
        doc_type = "Document"
    date_str = doc.issue_date or (str(doc.year[0]) if doc.year else "Date unknown")
    snippet   = doc.best_chunk_text[:300] if doc.best_chunk_text else ""
    full_text = doc.best_chunk_text if doc.best_chunk_text else ""
    tags     = list(set((doc.topics or []) + (doc.geography or [])))[:5]
    url = (
        f"https://www.digitalcommonwealth.org/search/commonwealth:{doc.ark_id}"
        if doc.best_chunk_text
        else f"https://www.digitalcommonwealth.org/collections/commonwealth:{doc.ark_id}"
    )
    thumbnail_url = (
        f"https://iiif.digitalcommonwealth.org/iiif/2/{doc.exemplary_image_id}/full/400,/0/default.jpg"
        if doc.exemplary_image_id and doc.exemplary_image_id.strip() else ""
    )
    return {
        "type": doc_type, "title": doc.title or "Untitled",
        "date": date_str, "collection": doc.institution or "Boston Public Library",
        "snippet": snippet, "full_text": full_text, "tags": tags,
        "score": round(doc.final_score, 2), "url": url, "thumbnail": thumbnail_url,
    }

def build_card_html(r: dict, i: int) -> str:
    score_pct = min(int(r["score"] * 100), 100)
    tags_html = "".join(
        '<span class="card-tag">' + html.escape(t) + '</span>' for t in r["tags"]
    )
    thumb = (
        '<img src="' + r["thumbnail"] + '" '
        'style="width:100%;max-height:200px;object-fit:cover;'
        'border-radius:4px;margin-bottom:0.8rem;" />'
        if r.get("thumbnail", "").startswith("https://") else ""
    )
    # Full text expander — only when doc has more text than the snippet
    full_text = r.get("full_text", "")
    if full_text and len(full_text) > 300:
        expander = (
            '<details class="full-text-expander">'
            '<summary>Full Text</summary>'
            '<div class="full-text-body">' + html.escape(full_text) + '</div>'
            '</details>'
        )
    else:
        expander = ""

    return (
        '<div class="result-card" id="result-' + str(i) + '">'
        + thumb
        + '<div class="card-type-badge">' + html.escape(r["type"]) + '</div>'
        + '<div class="card-title">' + html.escape(r["title"]) + '</div>'
        + '<div class="card-meta">'
          + html.escape(r["date"]) + ' &nbsp;·&nbsp; ' + html.escape(r["collection"])
          + '</div>'
        + '<div class="card-snippet">' + html.escape(r["snippet"]) + '</div>'
        + expander
        + '<div class="card-tags">' + tags_html + '</div>'
        + '<div class="score-row">'
          + '<span class="score-label">Relevance</span>'
          + '<div class="score-bar-bg">'
            + '<div class="score-bar-fill" style="width:' + str(score_pct) + '%"></div>'
            + '</div>'
          + '<span class="score-val">' + str(score_pct) + '%</span>'
          + '&nbsp;&nbsp;'
          + '<a class="card-link" href="' + r["url"] + '" target="_blank">'
            + 'View in Digital Commonwealth ↗</a>'
          + '</div>'
        + '</div>'
    )


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("query", ""), ("results", None), ("searched", False),
    ("context", ""), ("latency_ms", 0), ("_last_ran", ""), ("page", 0),
    ("query_id", None),                # Postgres query_logs.id of last run
    ("docs", []),                      # raw RetrievedDocument list (for refine)
    ("thumbs", {}),                    # {ark_id: "up"|"down"} for current results
    ("missing_text", ""),              # last "missed the point" comment
    ("refined_with", []),              # last set of follow-up queries shown
    ("_scroll_to_top", False),         # set after refine succeeds, consumed below
]:
    if k not in st.session_state:
        st.session_state[k] = v

# Per-browser-session UUID — anonymous, stable across reruns of this tab.
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())



# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">Boston Public Library · Digital Commonwealth</div>
    <h1 class="masthead-title">Search the <em>Archive</em></h1>
    <p class="masthead-sub">
        Ask anything in plain language. Explore photographs, maps, newspapers,
        manuscripts, and more from Massachusetts history.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Search box ────────────────────────────────────────────────────────────────
st.markdown('<div class="search-label">Natural Language Query</div>', unsafe_allow_html=True)
with st.form(key="search_form", border=False):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        typed = st.text_input(
            "Natural Language Query",
            value=st.session_state.query,
            placeholder='e.g. "Find photographs of Boston Harbor from the 1800s"',
            label_visibility="collapsed",
            key="text_input_box",
        )
    with col_btn:
        search_clicked = st.form_submit_button(
            "Search →", type="primary", use_container_width=True
        )

st.markdown(
    '<div class="search-label" style="margin-top:0.5rem;margin-bottom:0.4rem;">'
    'Try an example</div>',
    unsafe_allow_html=True,
)
pill_clicked = None
row1 = st.columns(3)
row2 = st.columns(3)
for i, q in enumerate(EXAMPLE_QUERIES):
    col = row1[i] if i < 3 else row2[i - 3]
    with col:
        if st.button(q, key=f"pill_{i}", use_container_width=True):
            pill_clicked = q



# ── Determine active query ────────────────────────────────────────────────────
if pill_clicked:
    st.session_state.query = pill_clicked
elif search_clicked and typed.strip():
    st.session_state.query = typed.strip()

active_query = st.session_state.query.strip()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if active_query and active_query != st.session_state["_last_ran"]:
    st.session_state["_last_ran"]    = active_query
    st.session_state["page"]         = 0
    st.session_state["thumbs"]       = {}
    st.session_state["missing_text"] = ""
    st.session_state["refined_with"] = []
    with st.spinner("Searching the archive…"):
        try:
            result: PipelineResult = run_query(
                active_query,
                session_id = st.session_state["session_id"],
            )
            cards = [format_card(doc) for doc in result.documents]
            st.session_state.results    = cards
            st.session_state.docs       = result.documents
            st.session_state.query_id   = result.query_id
            st.session_state.context    = result.generation.response
            st.session_state.latency_ms = result.latency_ms
            st.session_state.searched   = True
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
        # ── Refined-search banner (only when a refine just ran) ───────────
        if st.session_state["refined_with"]:
            chips = " · ".join(html.escape(q) for q in st.session_state["refined_with"])
            st.markdown(
                '<div class="context-banner">'
                '<strong>Refined search.</strong> ' + chips + '</div>',
                unsafe_allow_html=True,
            )

        # ── Scroll to top after a refinement (one-shot) ───────────────────
        if st.session_state["_scroll_to_top"]:
            st.session_state["_scroll_to_top"] = False
            components.html(
                """
                <script>
                  (function() {
                    const doc = window.parent.document;
                    const main = doc.querySelector('section[data-testid="stMain"]')
                              || doc.querySelector('[data-testid="stAppViewContainer"]')
                              || doc.scrollingElement
                              || doc.body;
                    main.scrollTo({ top: 0, behavior: 'smooth' });
                    window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                  })();
                </script>
                """,
                height=0,
            )

        context_with_links = linkify_citations(context, len(results))
        st.markdown(
            '<div class="context-banner"><strong>About these results.</strong> '
            + context_with_links + '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="results-header">'
            + '<div class="results-count">Found <span>' + str(len(results))
            + ' items</span> for &ldquo;' + html.escape(st.session_state.query)
            + '&rdquo;</div>'
            + '<div class="results-meta">Ranked by relevance &nbsp;·&nbsp; '
            + str(latency) + 'ms &nbsp;·&nbsp; Digital Commonwealth BPL Subset</div>'
            + '</div>',
            unsafe_allow_html=True,
        )

        # ── Citation scroll ───────────────────────────────────────────────────
        # After a ?scroll_to=N reload, results re-render from session_state,
        # then this components.html block scrolls and highlights the target card.


        # ── Pagination ────────────────────────────────────────────────────
        PAGE_SIZE = 10
        total_pages = max(1, (len(results) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = st.session_state["page"]

        start = page * PAGE_SIZE
        end   = start + PAGE_SIZE
        page_results = results[start:end]

        page_docs = st.session_state.docs[start:end]
        for i, (r, doc) in enumerate(zip(page_results, page_docs), start=start + 1):
            st.markdown(build_card_html(r, i), unsafe_allow_html=True)

            # ── Feedback row (thumbs only) ────────────────────────────────
            current = st.session_state["thumbs"].get(doc.ark_id)
            fb_cols = st.columns([1, 1, 10])
            with fb_cols[0]:
                up_type = "primary" if current == "up" else "secondary"
                if st.button(":material/thumb_up:",
                             key=f"up_{i}_{doc.ark_id}",
                             type=up_type, use_container_width=True,
                             help="Helpful result"):
                    st.session_state["thumbs"][doc.ark_id] = "up"
                    log_feedback(
                        query_id   = st.session_state["query_id"],
                        ark_id     = doc.ark_id,
                        signal     = "up",
                        session_id = st.session_state["session_id"],
                        raw_query  = st.session_state["query"],
                    )
                    st.rerun()
            with fb_cols[1]:
                down_type = "primary" if current == "down" else "secondary"
                if st.button(":material/thumb_down:",
                             key=f"down_{i}_{doc.ark_id}",
                             type=down_type, use_container_width=True,
                             help="Not what I wanted"):
                    st.session_state["thumbs"][doc.ark_id] = "down"
                    log_feedback(
                        query_id   = st.session_state["query_id"],
                        ark_id     = doc.ark_id,
                        signal     = "down",
                        session_id = st.session_state["session_id"],
                        raw_query  = st.session_state["query"],
                    )
                    st.rerun()

        # ── Pagination controls ───────────────────────────────────────────────
        if total_pages > 1:
            st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
            pcol_prev, pcol_info, pcol_next = st.columns([1, 2, 1])
            with pcol_prev:
                if page > 0:
                    if st.button("← Previous", key="prev_page", use_container_width=True):
                        st.session_state["page"] -= 1
                        st.rerun()
            with pcol_info:
                st.markdown(
                    '<div style="text-align:center;padding-top:0.5rem;font-size:0.82rem;'
                    'color:var(--muted);font-family:sans-serif;">'
                    'Page ' + str(page + 1) + ' of ' + str(total_pages)
                    + ' &nbsp;&middot;&nbsp; ' + str(len(results)) + ' total results'
                    + '</div>',
                    unsafe_allow_html=True,
                )
            with pcol_next:
                if page < total_pages - 1:
                    if st.button("Next →", key="next_page", use_container_width=True):
                        st.session_state["page"] += 1
                        st.rerun()

        # ── Human-in-the-loop: user-typed refinement ──────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="search-label">Didn\'t find any relevant results?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.85rem;color:var(--muted);margin-bottom:0.6rem;">'
            'Refine your search. Be more specific about what you want.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Refine your search",
            key="missing_text",
            height=110,
            placeholder="e.g. photographs of JFK as a senator in 1958, not newspaper clippings",
            label_visibility="collapsed",
        )
        if st.button("Refine search", key="refine_btn",
                     use_container_width=True):
            user_text = st.session_state["missing_text"].strip()
            if not user_text:
                st.info("Type a refined query before clicking refine.")
            else:
                # Log as 'missing' feedback so the team can see what queries
                # the original retrieval was failing on.
                log_feedback(
                    query_id   = st.session_state["query_id"],
                    ark_id     = "",
                    signal     = "missing",
                    comment    = user_text,
                    session_id = st.session_state["session_id"],
                    raw_query  = st.session_state["query"],
                )
                with st.spinner("Searching with your refined query…"):
                    try:
                        merged, follow_ups, _child_ids = refine_with_user_query(
                            original_query   = st.session_state["query"],
                            original_results = st.session_state.docs,
                            user_query       = user_text,
                            top_k            = max(50, len(st.session_state.docs)),
                            session_id       = st.session_state["session_id"],
                            parent_query_id  = st.session_state["query_id"],
                        )
                        if follow_ups:
                            st.session_state.docs           = merged
                            st.session_state.results        = [format_card(d) for d in merged]
                            st.session_state.refined_with   = follow_ups
                            st.session_state["page"]        = 0
                            st.session_state["_scroll_to_top"] = True
                            st.rerun()
                        else:
                            st.warning("Refinement search failed. Try again.")
                    except Exception as e:
                        st.error(f"Refine failed: {e}")

    else:
        st.markdown(
            '<div class="no-results">'
            '<div class="no-results-icon">🗂️</div>'
            '<div class="no-results-title">No matching materials found</div>'
            '<p>Try rephrasing your query, or use one of the example searches above.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="bpl-footer">'
    '<strong>Boston Public Library</strong> · Digital Commonwealth · BPL RAG Search<br>'
    'A natural language search prototype built with Retrieval-Augmented Generation.<br>'
    'Results are drawn from digitized items in the BPL subset of Digital Commonwealth.'
    '</div>',
    unsafe_allow_html=True,
)