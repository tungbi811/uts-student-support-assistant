import yaml
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONDENSE_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that can be understood without the conversation history. If the follow-up is already standalone, return it unchanged.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""


ANSWER_TEMPLATE = """You are a UTS Student Support Assistant.
Use ONLY the context below to answer. Do not use any outside knowledge.
If the context contains a relevant answer, provide it clearly.
Only say "I couldn't find that in UTS policy — please check uts.edu.au directly." if the context contains absolutely nothing relevant.
Do NOT say this if you have already provided an answer.

Context:
{context}

Question: {question}

Answer:"""


def load_llm(config):
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embeddings(config):
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    elif provider == "ollama":
        return OllamaEmbeddings(model=model)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def load_vectorstore(config):
    embeddings = get_embeddings(config)
    vectorstore = FAISS.load_local(
        config["vectorstore"]["index_path"],
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vectorstore loaded successfully.")
    return vectorstore


def format_chat_history(messages):
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def build_rag_chain(vectorstore, config):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config["retriever"]["k"]}
    )

    llm = load_llm(config)

    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_TEMPLATE
    )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=ANSWER_TEMPLATE
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata['url']}\n{doc.page_content}"
            for doc in docs
        )

    def run(inputs):
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        if chat_history:
            history_str = format_chat_history(chat_history)
            question = (condense_prompt | llm | StrOutputParser()).invoke({
                "chat_history": history_str,
                "question": question
            })

        docs = retriever.invoke(question)
        context = format_docs(docs)

        return (answer_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })

    return run, retriever


def ask(chain, retriever, question, chat_history=None):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    answer = chain({"question": question, "chat_history": chat_history or []})
    print(f"Answer: {answer}")

    docs = retriever.invoke(question)
    print("\nSources:")
    seen = set()
    for doc in docs:
        url = doc.metadata["url"]
        if url not in seen:
            print(f"  - {url}")
            seen.add(url)

    return answer


if __name__ == "__main__":
    config = load_config()
    vectorstore = load_vectorstore(config)
    chain, retriever = build_rag_chain(vectorstore, config)

    questions = [
        "How do I apply for special consideration at UTS?",
        "What is the late penalty for assignments?",
        "What counts as academic misconduct?",
        "How do I withdraw from a subject?",
        "What is the maximum study load per session?",
        "How do I apply for a leave of absence?",
        "What happens if I fail a subject twice?",
        "How are final grades calculated?",
        "What support is available for students with a disability?",
        "How do I appeal a grade or assessment decision?",
    ]

    for q in questions:
        ask(chain, retriever, q)
