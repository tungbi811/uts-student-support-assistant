import json
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_pages(input_path="data/raw_pages.json"):
    with open(input_path, "r") as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages from {input_path}")
    return pages

def chunk_pages(pages, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "url": page["url"],
                "fetched_at": page["fetched_at"]
            })

    return chunks

def save_chunks(chunks, output_path="data/chunks.json"):
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    config = load_config()
    pages = load_pages()

    print("\n--- Chunking ---")
    chunks = chunk_pages(
        pages,
        chunk_size=config["chunker"]["chunk_size"],
        chunk_overlap=config["chunker"]["chunk_overlap"],
    )
    save_chunks(chunks)

    # Preview
    print("\n--- Preview of first 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  URL: {chunk['url']}")
        print(f"  Length: {len(chunk['text'])} chars")
        print(f"  Text: {chunk['text'][:150]}...")