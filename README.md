# UTS Student Support Assistant

A RAG-based chatbot that answers student questions about UTS policies, enrolment, assessments, and more — grounded strictly in scraped UTS web pages.

## Architecture

```
scraper.py → chunker.py → embedder.py → [FAISS index]
                                               ↓
                                           rag.py ← app.py (Streamlit UI)
```

## Setup

**Prerequisites:** Python 3.12+, [Poetry](https://python-poetry.org)

**1. Clone the repository**
```bash
git clone <repo-url>
cd uts-student-support-assistant
```

**2. Install dependencies**
```bash
poetry install
```

**3. Set up environment variables**
```bash
cp .env.example .env
```
Open `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-openai-api-key-here
```

**4. Run the ingestion pipeline** (scrape UTS pages, chunk, and embed)
```bash
poetry run python src/pipeline.py
```

This will take a few minutes. It scrapes up to 150 UTS pages, chunks the text, and builds the FAISS vector index.

**5. Launch the chatbot**
```bash
poetry run streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Other Commands

Run each pipeline step individually:
```bash
poetry run python src/scraper.py
poetry run python src/chunker.py
poetry run python src/embedder.py
```

Run evaluation:
```bash
poetry run python src/evaluate.py
```

### Run evaluation
```bash
python src/evaluate.py
```
Results are saved to `data/eval_results.json`.

## Configuration

All settings are in `config/config.yaml`:

| Section | Key | Description |
|---|---|---|
| `scraper` | `seed_urls` | Starting URLs for the crawler |
| `scraper` | `max_pages` | Max pages to crawl |
| `scraper` | `allowed_prefixes` | URL path filter (`[]` = all of uts.edu.au) |
| `chunker` | `chunk_size` | Characters per chunk |
| `chunker` | `chunk_overlap` | Overlap between chunks |
| `embeddings` | `provider` | `openai` or `ollama` |
| `embeddings` | `model` | Embedding model name |
| `llm` | `provider` | `openai` or `ollama` |
| `llm` | `model` | LLM model name |
| `retriever` | `k` | Number of chunks retrieved per query |

> **Note:** Changing `embeddings.provider` or `embeddings.model` requires rebuilding the index by re-running `pipeline.py`.

## Evaluation

Edit `data/test_set.json` to add your own questions and reference answers, then run `evaluate.py`. Metrics reported:

- **Correctness** — does the answer match the reference? (LLM-as-judge, 0–1)
- **Faithfulness** — is the answer grounded in retrieved context? (LLM-as-judge, 0–1)
- **Source hit rate** — did the retriever return the expected source URL? (0–1)
