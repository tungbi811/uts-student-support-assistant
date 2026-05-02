import re
import sys
import urllib.parse
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, "src")
from rag import load_config, load_vectorstore, build_rag_chain

st.set_page_config(
    page_title="UTS Student Support",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

[data-testid="stAppViewContainer"] {
    background: #f5f6fa;
}

.uts-header {
    background: #00467F;
    padding: 1rem 2rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
}
.uts-header h1 {
    color: white;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
}
.uts-header p {
    color: #b0c8e0;
    font-size: 0.85rem;
    margin: 0.25rem 0 0;
}

/* Sample question buttons */
div[data-testid="stButton"] button {
    font-size: 0.8rem;
    padding: 0.3rem 0.6rem;
    border-radius: 1rem;
    border: 1px solid #d1d5db;
    background: white;
    color: #374151;
    white-space: normal;
    text-align: left;
    height: auto;
}
div[data-testid="stButton"] button:hover {
    border-color: #00A9CE;
    color: #00467F;
    background: #f0f8ff;
}

/* Citation badge with tooltip */
.cite-wrap {
    position: relative;
    display: inline-block;
}
.cite-wrap a.cite {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #00467F;
    color: white !important;
    border-radius: 4px;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 1px 5px;
    margin: 0 1px;
    vertical-align: super;
    line-height: 1;
    text-decoration: none;
    cursor: pointer;
}
.cite-wrap a.cite:hover {
    background: #00A9CE;
}
.cite-tooltip {
    visibility: hidden;
    opacity: 0;
    background: #1e293b;
    color: #e2e8f0;
    font-size: 0.72rem;
    padding: 6px 10px;
    border-radius: 6px;
    position: absolute;
    bottom: 150%;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
    max-width: 340px;
    overflow: hidden;
    text-overflow: ellipsis;
    z-index: 9999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: opacity 0.15s ease;
    pointer-events: none;
}
.cite-wrap:hover .cite-tooltip {
    visibility: visible;
    opacity: 1;
}

/* Force dark text for readability on light background */
[data-testid="stChatMessage"] {
    color: #1f2937 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #1f2937 !important;
}
/* Keep tooltip text white on dark background */
.cite-tooltip,
[data-testid="stChatMessage"] .cite-tooltip {
    color: #e2e8f0 !important;
}
/* Keep citation badge text white */
.cite-wrap a.cite,
[data-testid="stChatMessage"] .cite-wrap a.cite {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


def build_citation_map(docs):
    """Build {chunk_number: (url, fragment_url)} for all retrieved chunks."""
    citation_map = {}
    for i, doc in enumerate(docs, start=1):
        url = doc.metadata["url"]
        snippet = urllib.parse.quote(doc.page_content.strip()[:120])
        citation_map[i] = (url, f"{url}#:~:text={snippet}")
    return citation_map


def remap_citations(answer, citation_map):
    """
    Renumber citations sequentially by order of first appearance,
    collapsing multiple chunk numbers that share the same URL into one number.
    e.g. if chunks [1],[3],[5] all point to url_a → all become [1].
    Returns (remapped_answer, remapped_citation_map).
    """
    # Collect cited chunk numbers in order of first appearance
    cited_order = []
    seen_nums = set()
    for m in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', answer):
        for n in [int(x.strip()) for x in m.group(1).split(",")]:
            if n not in seen_nums:
                cited_order.append(n)
                seen_nums.add(n)

    # Build remap: chunk_num → sequential_num, deduplicating by URL
    url_to_seq = {}
    remap = {}
    new_citation_map = {}
    seq = 1
    for orig in cited_order:
        if orig not in citation_map:
            continue
        url, frag = citation_map[orig]
        if url not in url_to_seq:
            url_to_seq[url] = seq
            new_citation_map[seq] = (url, frag)
            seq += 1
        remap[orig] = url_to_seq[url]

    # Rewrite answer, removing duplicates within the same bracket
    def replace(m):
        nums = [int(x.strip()) for x in m.group(1).split(",")]
        seq_nums = list(dict.fromkeys(remap[n] for n in nums if n in remap))
        return f"[{', '.join(str(s) for s in seq_nums)}]" if seq_nums else m.group(0)

    remapped_answer = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace, answer)

    # Move citations that sit before a sentence-ending period to after it
    # "text [1]." → "text.[1]"  |  "text [1] [2]." → "text.[1][2]"
    remapped_answer = re.sub(
        r'(\s*(?:\[[\d,\s]+\]\s*)+)([.!?])',
        lambda m: m.group(2) + re.sub(r'\s+', '', m.group(1)),
        remapped_answer
    )

    return remapped_answer, new_citation_map


def _cite_badge(n, citation_map):
    if n not in citation_map:
        return f"[{n}]"
    url, fragment_url = citation_map[n]
    display = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return (
        f'<span class="cite-wrap">'
        f'<a class="cite" href="{fragment_url}" target="_blank">{n}</a>'
        f'<span class="cite-tooltip">{display}</span>'
        f'</span>'
    )


def render_citations(text, citation_map):
    """Replace [N] or [N, M, ...] with hoverable badges linking to source pages."""
    def replace(m):
        numbers = [int(x.strip()) for x in m.group(1).split(",")]
        return "".join(_cite_badge(n, citation_map) for n in numbers)
    return re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace, text)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="uts-header">
    <h1>UTS Student Support Assistant</h1>
    <p>Ask me anything about UTS policies, enrolment, assessments, and more.</p>
</div>
""", unsafe_allow_html=True)

# ── Load chain ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    config = load_config()
    vectorstore = load_vectorstore(config)
    chain, retriever = build_rag_chain(vectorstore, config)
    return chain, retriever

chain, _ = load_chain()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

SAMPLE_QUESTIONS = [
    "How do I apply for special consideration?",
    "What is the late penalty for assignments?",
    "What counts as academic misconduct?",
    "How do I withdraw from a subject?",
    "How do I appeal a grade?",
]

# ── Sample questions (shown only before first message) ────────────────────────
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, question in zip(cols, SAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("citation_map"):
            st.markdown(
                render_citations(msg["content"], msg["citation_map"]),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
elif prompt := st.chat_input("Ask a question about UTS..."):
    pass
else:
    prompt = None

# ── Handle new message ────────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_history = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages[:-1]
        ]

        with st.status("Searching UTS policies...", expanded=False):
            answer, docs = chain({"question": prompt, "chat_history": chat_history})

        citation_map = build_citation_map(docs)
        answer, citation_map = remap_citations(answer, citation_map)
        st.markdown(render_citations(answer, citation_map), unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,      # remapped answer (with [1],[2],[3]...)
        "citation_map": citation_map,
    })
    st.rerun()
