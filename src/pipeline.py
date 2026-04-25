"""
Run the full ingestion pipeline: scrape → chunk → embed
"""
import yaml
from scraper import crawl, save_pages
from chunker import load_pages, chunk_pages, save_chunks
from embedder import load_config, load_chunks, build_vectorstore, save_vectorstore, get_embeddings


def run_pipeline(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("Step 1/3: Scraping")
    print("=" * 50)
    pages = crawl(config["scraper"])
    save_pages(pages, config["scraper"]["output_path"])

    print("\n" + "=" * 50)
    print("Step 2/3: Chunking")
    print("=" * 50)
    pages = load_pages(config["scraper"]["output_path"])
    embeddings = get_embeddings(config)
    chunks = chunk_pages(
        pages,
        embeddings=embeddings,
        breakpoint_threshold_type=config["chunker"]["breakpoint_threshold_type"],
        breakpoint_threshold_amount=config["chunker"]["breakpoint_threshold_amount"],
    )
    save_chunks(chunks)

    print("\n" + "=" * 50)
    print("Step 3/3: Embedding")
    print("=" * 50)
    chunks = load_chunks()
    vectorstore = build_vectorstore(chunks, config)
    save_vectorstore(vectorstore, config["vectorstore"]["index_path"])

    print("\n" + "=" * 50)
    print("Pipeline complete. Ready to query.")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
