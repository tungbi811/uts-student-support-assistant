import re
import sys
import urllib.parse
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

/* Sample question & suggestion buttons */
div[data-testid="stButton"] {
    width: fit-content !important;
}
div[data-testid="stButton"] button {
    font-size: 0.8rem;
    padding: 0.35rem 1.2rem;
    border-radius: 1.5rem;
    border: 1px solid #d1d5db;
    background: white;
    color: #374151;
    white-space: nowrap;
    text-align: left;
    height: auto;
    width: fit-content !important;
    line-height: 1.4;
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
</style>
""", unsafe_allow_html=True)


def build_citation_map(docs):
    citation_map = {}
    for i, doc in enumerate(docs, start=1):
        url = doc.metadata["url"]
        snippet = urllib.parse.quote(doc.page_content.strip()[:120])
        citation_map[i] = (url, f"{url}#:~:text={snippet}")
    return citation_map


def remap_citations(answer, citation_map):
    cited_order = []
    seen_nums = set()
    for m in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', answer):
        for n in [int(x.strip()) for x in m.group(1).split(",")]:
            if n not in seen_nums:
                cited_order.append(n)
                seen_nums.add(n)

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

    def replace(m):
        nums = [int(x.strip()) for x in m.group(1).split(",")]
        seq_nums = list(dict.fromkeys(remap[n] for n in nums if n in remap))
        return f"[{', '.join(str(s) for s in seq_nums)}]" if seq_nums else m.group(0)

    remapped_answer = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace, answer)
    remapped_answer = re.sub(r'(\[\d+\])(\s*\1)+', r'\1', remapped_answer)
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
    def replace(m):
        numbers = [int(x.strip()) for x in m.group(1).split(",")]
        return "".join(_cite_badge(n, citation_map) for n in numbers)
    return re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace, text)


def render_suggestions(questions, key_suffix=""):
    """Render suggestions: pair short questions on the same row, long ones solo."""
    SHORT = 45  # characters threshold to share a row
    i = 0
    while i < len(questions):
        q = questions[i]
        next_q = questions[i + 1] if i + 1 < len(questions) else None
        if next_q and len(q) <= SHORT and len(next_q) <= SHORT:
            cols = st.columns([len(q), len(next_q)])
            for col, question in zip(cols, [q, next_q]):
                if col.button(f"↳ {question}", key=f"sug_{key_suffix}_{question}", use_container_width=False):
                    st.session_state.pending_question = question
                    st.rerun()
            i += 2
        else:
            if st.button(f"↳ {q}", key=f"sug_{key_suffix}_{q}", use_container_width=False):
                st.session_state.pending_question = q
                st.rerun()
            i += 1


SUGGESTION_PROMPT = PromptTemplate.from_template(
    """Based on this Q&A about UTS student policies, suggest 3 short follow-up questions a student might ask next.
Return only the 3 questions, one per line, no numbering, no bullets, no extra text.

Question: {question}
Answer: {answer}

3 follow-up questions:"""
)


def generate_suggestions(question, answer, config):
    llm = ChatOpenAI(model=config["llm"]["model"], temperature=0.5)
    result = (SUGGESTION_PROMPT | llm | StrOutputParser()).invoke({
        "question": question,
        "answer": answer,
    })
    suggestions = [s.strip() for s in result.strip().split("\n") if s.strip()]
    return suggestions[:3]


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
    return chain, retriever, config

chain, _, config = load_chain()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

SAMPLE_QUESTIONS = [
    "How do I apply for a leave of absence?",
    "How do I get my student ID card?",
    "What support is available for students with a disability?",
    "How do I enrol in subjects for next semester?",
    "What is the maximum study load per session?",
]

# ── Sample questions (shown only before first message) ────────────────────────
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    SHORT = 45
    i = 0
    while i < len(SAMPLE_QUESTIONS):
        q = SAMPLE_QUESTIONS[i]
        next_q = SAMPLE_QUESTIONS[i + 1] if i + 1 < len(SAMPLE_QUESTIONS) else None
        if next_q and len(q) <= SHORT and len(next_q) <= SHORT:
            cols = st.columns([len(q), len(next_q)])
            for col, question in zip(cols, [q, next_q]):
                if col.button(question, key=f"sample_{question}", use_container_width=False):
                    st.session_state.pending_question = question
                    st.rerun()
            i += 2
        else:
            if st.button(q, key=f"sample_{q}", use_container_width=False):
                st.session_state.pending_question = q
                st.rerun()
            i += 1

# ── Chat history ──────────────────────────────────────────────────────────────
last_idx = len(st.session_state.messages) - 1
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("citation_map"):
            st.markdown(
                render_citations(msg["content"], msg["citation_map"]),
                unsafe_allow_html=True,
            )
            if i == last_idx and msg.get("suggestions"):
                render_suggestions(msg["suggestions"], key_suffix=str(i))
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
            suggestions = generate_suggestions(prompt, answer, config)

        citation_map = build_citation_map(docs)
        answer, citation_map = remap_citations(answer, citation_map)
        st.markdown(render_citations(answer, citation_map), unsafe_allow_html=True)

        if suggestions:
            render_suggestions(suggestions, key_suffix="new")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citation_map": citation_map,
        "suggestions": suggestions,
    })
    st.rerun()
