import json
import yaml
import argparse
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def chunk_pages_semantic(pages, embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95):
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    chunks = []
    for i, page in enumerate(pages):
        if i % 100 == 0:
            print(f"  Chunking page {i+1}/{len(pages)}...")
        for split in splitter.split_text(page["text"]):
            chunks.append({"text": split, "url": page["url"], "fetched_at": page["fetched_at"]})
    return chunks

def chunk_pages_fixed(pages, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for i, page in enumerate(pages):
        if i % 100 == 0:
            print(f"  Chunking page {i+1}/{len(pages)}...")
        for split in splitter.split_text(page["text"]):
            chunks.append({"text": split, "url": page["url"], "fetched_at": page["fetched_at"]})
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         default="semantic", choices=["semantic", "fixed"],
                        help="Chunking strategy: 'semantic' (default) or 'fixed' size")
    parser.add_argument("--chunk-size",   type=int, default=500,
                        help="Characters per chunk for fixed mode (default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                        help="Overlap between chunks for fixed mode (default: 50)")
    parser.add_argument("--output",       default=None,
                        help="Output path for chunks JSON (default: from config)")
    args = parser.parse_args()

    config = load_config()
    pages  = load_pages()

    output_path = args.output or config.get("chunker", {}).get("output_path", "data/chunks.json")

    if args.mode == "fixed":
        print(f"\n--- Fixed Chunking (chunk_size={args.chunk_size}, overlap={args.chunk_overlap}) ---")
        chunks = chunk_pages_fixed(pages, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    else:
        embeddings_cfg = config["embeddings"]
        if embeddings_cfg["provider"] != "openai":
            raise ValueError(f"SemanticChunker only supports openai embeddings. Got: {embeddings_cfg['provider']}")
        embeddings = OpenAIEmbeddings(model=embeddings_cfg["model"])
        chunker_cfg = config.get("chunker", {})
        breakpoint_type   = chunker_cfg.get("breakpoint_threshold_type", "percentile")
        breakpoint_amount = chunker_cfg.get("breakpoint_threshold_amount", 95)
        print(f"\n--- Semantic Chunking (breakpoint={breakpoint_type}, threshold={breakpoint_amount}) ---")
        chunks = chunk_pages_semantic(pages, embeddings, breakpoint_type, breakpoint_amount)

    save_chunks(chunks, output_path)

    print("\n--- Preview of first 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  URL: {chunk['url']}")
        print(f"  Length: {len(chunk['text'])} chars")
        print(f"  Text: {chunk['text'][:150]}...")
