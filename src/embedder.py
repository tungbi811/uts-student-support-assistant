import json
import os
import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_chunks(input_path="data/chunks.json"):
    with open(input_path, "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


def get_embeddings(config):
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    elif provider == "ollama":
        return OllamaEmbeddings(model=model)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def build_vectorstore(chunks, config):
    print("Building vectorstore...")

    docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "url": chunk["url"],
                "fetched_at": chunk["fetched_at"]
            }
        )
        for chunk in chunks
    ]

    embeddings = get_embeddings(config)
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


def save_vectorstore(vectorstore, output_path="data/faiss_index"):
    vectorstore.save_local(output_path)
    print(f"Saved vectorstore to {output_path}")


if __name__ == "__main__":
    config = load_config()
    chunks = load_chunks()
    vectorstore = build_vectorstore(chunks, config)
    save_vectorstore(vectorstore, config["vectorstore"]["index_path"])
    print("\nDone! Vectorstore is ready.")