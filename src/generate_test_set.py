"""
Generate a test set from crawled chunks using the LLM.

Usage:
    python src/generate_test_set.py --n 20 --output data/test_set.json

Steps:
  1. Randomly samples N chunks from chunks.json
  2. For each chunk, asks the LLM to generate a realistic student question + reference answer
  3. Saves to test_set.json for manual review
"""
import json
import random
import argparse
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

GENERATE_PROMPT = PromptTemplate.from_template("""You are helping build an evaluation dataset for a UTS student support chatbot.

Given the following chunk of text from the UTS website, generate ONE realistic question a UTS student might ask, and a concise reference answer based strictly on the text.

URL: {url}
Text: {text}

Reply in this exact JSON format (no markdown, no code block):
{{
  "question": "...",
  "reference": "...",
  "expected_source": "{domain}"
}}

Rules:
- The question must be answerable from the text
- The reference answer must be factual and concise (1-3 sentences)
- Do NOT include information not present in the text
- The question should sound like a real student asking
- Do NOT generate questions about specific named individuals (students, staff, alumni, or any real person)
- Only generate questions about UTS policies, procedures, services, requirements, or general information
- If the text is primarily about a specific person's story or experience, reply with exactly: {{"skip": true}}""")


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_chunks(path="data/chunks.json"):
    with open(path) as f:
        return json.load(f)


def extract_domain(url):
    """Extract a short identifier from URL for expected_source."""
    parts = url.rstrip("/").split("/")
    return parts[-1] if parts[-1] else parts[-2]


def generate_test_set(n=20, output_path="data/test_set.json", seed=42):
    config = load_config()
    chunks = load_chunks()

    random.seed(seed)
    pool = random.sample(chunks, len(chunks))  # shuffle all, draw as needed

    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
    chain = GENERATE_PROMPT | llm | StrOutputParser()

    results = []
    failed = 0
    pool_idx = 0

    while len(results) < n and pool_idx < len(pool):
        chunk = pool[pool_idx]
        pool_idx += 1
        print(f"[{len(results)+1}/{n}] {chunk['url']}")
        try:
            raw = chain.invoke({
                "url": chunk["url"],
                "text": chunk["text"][:1000],
                "domain": extract_domain(chunk["url"]),
            })
            item = json.loads(raw)
            if item.get("skip"):
                print(f"  Skipped (personal story)")
                continue
            item["url"] = chunk["url"]
            results.append(item)
            print(f"  Q: {item['question']}")
        except Exception as e:
            print(f"  Failed: {e}")
            failed += 1

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGenerated {len(results)} items ({failed} failed) → {output_path}")
    print("Review the file and correct any wrong reference answers before running evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of test items to generate")
    parser.add_argument("--output", default="data/test_set.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    generate_test_set(n=args.n, output_path=args.output, seed=args.seed)
