import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import os
import time
import yaml
from collections import deque
from urllib.parse import urljoin, urlparse

SKIP_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".zip")


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["scraper"]


def is_allowed(url, allowed_prefixes):
    parsed = urlparse(url)
    if parsed.netloc not in ("www.uts.edu.au", "uts.edu.au"):
        return False
    if any(parsed.path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    # If no prefixes specified, allow all pages on the domain
    if not allowed_prefixes:
        return True
    return any(parsed.path.startswith(prefix) for prefix in allowed_prefixes)


def scrape_page(url, allowed_prefixes):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None, []

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract links before removing tags
        links = []
        for a in soup.find_all("a", href=True):
            absolute = urljoin(url, a["href"])
            clean = absolute.split("#")[0].split("?")[0].rstrip("/")
            if clean and is_allowed(clean, allowed_prefixes):
                links.append(clean)

        # Remove noise
        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if len(text) < 200:
            return None, links

        page = {
            "url": url,
            "text": text,
            "fetched_at": datetime.now().isoformat()
        }
        return page, links

    except Exception as e:
        print(f"  Failed: {e}")
        return None, []


def crawl(config):
    seed_urls = config["seed_urls"]
    allowed_prefixes = config.get("allowed_prefixes") or []
    max_pages = config["max_pages"]
    max_depth = config["max_depth"]
    delay = config["delay"]

    visited = set()
    queue = deque()

    for url in seed_urls:
        clean = url.rstrip("/")
        queue.append((clean, 0))
        visited.add(clean)

    pages = []

    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()

        print(f"[{len(pages)+1}/{max_pages}] depth={depth} {url}")
        page, links = scrape_page(url, allowed_prefixes)

        if page:
            pages.append(page)
            print(f"  OK — {len(page['text'])} chars, {len(links)} links found")
        else:
            print(f"  Skipped")

        if depth < max_depth:
            for link in links:
                if link not in visited:
                    visited.add(link)
                    queue.append((link, depth + 1))

        time.sleep(delay)

    print(f"\nCrawl complete — {len(pages)} pages collected.")
    return pages


def save_pages(pages, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pages, f, indent=2)
    print(f"Saved {len(pages)} pages to {output_path}")


if __name__ == "__main__":
    config = load_config()
    pages = crawl(config)
    save_pages(pages, config["output_path"])

    if pages:
        print("\n--- Preview of first page ---")
        print(f"URL: {pages[0]['url']}")
        print(f"Fetched at: {pages[0]['fetched_at']}")
        print(f"Text preview: {pages[0]['text'][:500]}")
