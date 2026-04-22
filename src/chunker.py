import json
import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_pages(input_path="data/raw_pages.json"):
    with open(input_path, "r") as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages from {input_path}")
    return pages

def chunk_pages(pages, embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95):
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    chunks = []
    for i, page in enumerate(pages):
        if i % 100 == 0:
            print(f"  Chunking page {i+1}/{len(pages)}...")
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "url": page["url"],
                "fetched_at": page["fetched_at"]
            })

    return chunks

def save_chunks(chunks, output_path="data/chunks.json"):
    seen = set()
    unique = [c for c in chunks if not (c["text"] in seen or seen.add(c["text"]))]
    if len(unique) < len(chunks):
        print(f"Removed {len(chunks) - len(unique)} duplicate chunks")
    with open(output_path, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"Saved {len(unique)} chunks to {output_path}")

if __name__ == "__main__":
    config = load_config()

    embeddings_cfg = config["embeddings"]
    if embeddings_cfg["provider"] == "openai":
        embeddings = OpenAIEmbeddings(model=embeddings_cfg["model"])
    else:
        raise ValueError(f"SemanticChunker currently only supports openai embeddings. Got: {embeddings_cfg['provider']}")

    chunker_cfg = config.get("chunker", {})
    breakpoint_type = chunker_cfg.get("breakpoint_threshold_type", "percentile")
    breakpoint_amount = chunker_cfg.get("breakpoint_threshold_amount", 95)

    pages = load_pages()

    print(f"\n--- Semantic Chunking (breakpoint={breakpoint_type}, threshold={breakpoint_amount}) ---")
    chunks = chunk_pages(pages, embeddings, breakpoint_type, breakpoint_amount)
    save_chunks(chunks)

    # Preview
    print("\n--- Preview of first 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  URL: {chunk['url']}")
        print(f"  Length: {len(chunk['text'])} chars")
        print(f"  Text: {chunk['text'][:150]}...")
