import sys
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, "src")

from rag import load_config, load_vectorstore, build_rag_chain

st.set_page_config(page_title="UTS Student Support Assistant", page_icon="🎓")
st.title("UTS Student Support Assistant")
st.caption("Ask me anything about UTS policies, enrolment, assessments, and more.")


@st.cache_resource
def load_chain():
    config = load_config()
    vectorstore = load_vectorstore(config)
    chain, retriever = build_rag_chain(vectorstore, config)
    return chain, retriever


chain, retriever = load_chain()

SAMPLE_QUESTIONS = [
    "How do I apply for special consideration?",
    "What is the late penalty for assignments?",
    "What counts as academic misconduct?",
    "How do I withdraw from a subject?",
    "How do I appeal a grade?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, question in zip(cols, SAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for url in msg["sources"]:
                    st.markdown(f"- {url}")

if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
elif prompt := st.chat_input("Ask a question..."):
    pass
else:
    prompt = None

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

        with st.status("Thinking...", expanded=False):
            answer = chain({"question": prompt, "chat_history": chat_history})
            docs = retriever.invoke(prompt)
            sources = list(dict.fromkeys(doc.metadata["url"] for doc in docs))

        st.markdown(answer)
        with st.expander("Sources"):
            for url in sources:
                st.markdown(f"- {url}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
