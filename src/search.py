import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from urllib.parse import urlparse
import time
import random
import argparse
import json
from tqdm import tqdm
# List of domains to ignore by default
DEFAULT_IGNORED_DOMAINS = [
    'facebook.com', 'fb.com',
    'x.com', 'twitter.com',
    'linkedin.com',
    'youtube.com',
    'bsky.app', 'bluesky.app',
    'vimeo.com',
    'instagram.com'
]

import sys
import json
import subprocess
import time
import random
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

DEFAULT_IGNORED_DOMAINS = [
    "facebook.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "instagram.com",
]

# --------------------------------------------------
# Google Search (SUBPROCESS SAFE)
# --------------------------------------------------

def google_search(
    query,
    num_results=5,
    min_chars=32,
    ignored_domains=DEFAULT_IGNORED_DOMAINS,
    verbose=True,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    if verbose:
        logger.info(f"[Search] {query}")

    results = []
    urls_processed = 0

    try:
        cmd = [
            sys.executable,
            "./src/ddgs_search_once.py",        
            query,
            str(2 * num_results),
        ]

        out = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.STDOUT,
        )

        search_results = json.loads(out)

        for r in search_results:
            url = r.get("href") or r.get("url")
            if not url:
                continue

            urls_processed += 1
            domain = urlparse(url).netloc.lower()

            if any(
                domain == ignored or domain.endswith("." + ignored)
                for ignored in ignored_domains
            ):
                continue

            if url.lower().endswith(".pdf"):
                continue

            text = extract_text_from_url(url, verbose=False)

            if text and len(text) >= min_chars:
                results.append({
                    "url": url,
                    "text": text,
                    "length": len(text),
                })

            time.sleep(random.uniform(1.0, 3.0))

            if len(results) >= num_results:
                break

    except subprocess.CalledProcessError as e:
        logger.error(f"DDGS subprocess failed:\n{e.output}")
    except Exception as e:
        logger.exception(f"Search error: {e}")

    if verbose:
        logger.info(
            f"Processed {urls_processed} URLs, kept {len(results)}"
        )

    return results

# --------------------------------------------------
# HTML Content Extraction
# --------------------------------------------------

def extract_text_from_url(url, verbose=True):
    session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": url.split("/", 3)[0] + "//" + url.split("/", 3)[2],
    }

    try:
        base_url = headers["Referer"]
        session.get(base_url, headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 403:
            if verbose:
                print(f"403 Forbidden: {url}")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        return "\n".join(lines)

    except Exception as e:
        if verbose:
            print(f"Extraction error ({url}): {e}")
        return ""

# --------------------------------------------------
# Batch Processing
# --------------------------------------------------

def process_queries_from_file(
    query_file,
    num_results=20,
    min_chars=32,
    ignored_domains=DEFAULT_IGNORED_DOMAINS,
    verbose=True,
):
    logger = logging.getLogger(__name__)
    all_results = {}

    try:
        with open(query_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]


        for query in tqdm(queries):
            results = google_search(
                query=query,
                num_results=num_results,
                min_chars=min_chars,
                ignored_domains=ignored_domains,
                verbose=verbose,
                logger=logger,
            )
            all_results[query] = results

    except Exception as e:
        if verbose:
            print(f"Batch error: {e}")

    return all_results

def main():
    parser = argparse.ArgumentParser(description='Search Google and download text from result pages')
    parser.add_argument('--query-file', type=str, default='./data/issue-bench.txt', help='File containing search queries (one per line)')
    parser.add_argument('--results', type=int, default=20, help='Number of results to retrieve per query (default: 10)')
    parser.add_argument('--min-chars', type=int, default=32, help='Minimum characters required for a result (default: 1000)')
    parser.add_argument('--output', type=str, default='./data/issue_bench_search_results.json', help='Output file name (default: search_results.json)')
    args = parser.parse_args()

    all_results = process_queries_from_file(args.query_file, args.results, args.min_chars, DEFAULT_IGNORED_DOMAINS)
    with open(args.output, 'w', encoding='utf-8') as f:
        json_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'queries': {}
        }
        #tqdm
        
        for query, results in all_results.items():
            json_data['queries'][query] = results
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Completed! Results for {len(all_results)} queries saved to {args.output} in JSON format")

if __name__ == "__main__":
    main()