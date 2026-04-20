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


def is_allowed(url, allowed_prefixes, skip_keywords=None):
    parsed = urlparse(url)
    if parsed.netloc not in ("www.uts.edu.au", "uts.edu.au"):
        return False
    if any(parsed.path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    if skip_keywords and any(kw in parsed.path for kw in skip_keywords):
        return False
    # If no prefixes specified, allow all pages on the domain
    if not allowed_prefixes:
        return True
    return any(parsed.path.startswith(prefix) for prefix in allowed_prefixes)


def scrape_page(url, allowed_prefixes, skip_keywords=None, timeout=30):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
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
            if clean and is_allowed(clean, allowed_prefixes, skip_keywords):
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


def _state_path(output_path):
    base, _ = os.path.splitext(output_path)
    return base + "_state.json"


def _save_state(pages, queue, failed, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pages, f, indent=2)
    state = {"queue": list(queue), "failed": list(failed)}
    with open(_state_path(output_path), "w") as f:
        json.dump(state, f)


def _load_state(output_path):
    """Return (pages, queue, failed) from disk, or ([], None, set()) if no checkpoint."""
    if not os.path.exists(output_path):
        return [], None, set()
    with open(output_path) as f:
        pages = json.load(f)
    state_file = _state_path(output_path)
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        queue = deque([tuple(item) for item in state["queue"]])
        failed = set(state.get("failed", []))
    else:
        queue = None
        failed = set()
    return pages, queue, failed


def crawl(config):
    seed_urls = config["seed_urls"]
    allowed_prefixes = config.get("allowed_prefixes") or []
    skip_keywords = config.get("skip_keywords") or []
    max_pages = config["max_pages"]
    max_depth = config["max_depth"]
    delay = config["delay"]
    timeout = config.get("timeout", 30)
    output_path = config["output_path"]
    checkpoint_every = config.get("checkpoint_every", 10)

    pages, saved_queue, failed = _load_state(output_path)
    visited = {p["url"] for p in pages} | failed

    if saved_queue is not None:
        # Re-filter queue against current allowed_prefixes/skip_keywords
        queue = deque(
            (url, depth) for url, depth in saved_queue
            if is_allowed(url, allowed_prefixes, skip_keywords)
        )
        for url, _ in queue:
            visited.add(url)
        # Add any new seed URLs not yet visited
        for url in seed_urls:
            clean = url.rstrip("/")
            if clean not in visited:
                queue.appendleft((clean, 0))
                visited.add(clean)
        print(f"Resuming from checkpoint — {len(pages)} pages collected, {len(queue)} URLs pending, {len(failed)} failed.")
    else:
        queue = deque()
        for url in seed_urls:
            clean = url.rstrip("/")
            queue.append((clean, 0))
            visited.add(clean)

    try:
        while queue and (max_pages is None or len(pages) < max_pages):
            url, depth = queue.popleft()

            page_label = f"{len(pages)+1}" + (f"/{max_pages}" if max_pages else "")
            depth_label = f"depth={depth}" + ("" if max_depth is None else f"/{max_depth}")
            print(f"[{page_label}] {depth_label} {url}")
            page, links = scrape_page(url, allowed_prefixes, skip_keywords, timeout)

            if page:
                pages.append(page)
                print(f"  OK — {len(page['text'])} chars, {len(links)} links found, {len(queue)} in queue")
                if len(pages) % checkpoint_every == 0:
                    _save_state(pages, queue, failed, output_path)
                    print(f"  [checkpoint] saved {len(pages)} pages, {len(queue)} URLs pending")
            else:
                failed.add(url)
                print(f"  Skipped")

            if max_depth is None or depth < max_depth:
                for link in links:
                    if link not in visited:
                        visited.add(link)
                        queue.append((link, depth + 1))

            time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\nInterrupted — saving {len(pages)} pages, {len(queue)} pending URLs, {len(failed)} failed...")
        _save_state(pages, queue, failed, output_path)
        raise

    # Clean up state file on successful completion
    state_file = _state_path(output_path)
    if os.path.exists(state_file):
        os.remove(state_file)
    if failed:
        print(f"{len(failed)} URLs skipped (errors/no content) — not retried on resume.")

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
