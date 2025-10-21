import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from urllib.parse import urlparse
import time
import random
import argparse
import json

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

def google_search(query, num_results=10, min_chars=32, ignored_domains=None, verbose=True):

    if verbose:
        print(f"Searching Google for: {query}")
    results = []
    urls_processed = 0

    try:
        with DDGS() as ddgs:
            search_results = [r for r in ddgs.text(query, max_results=2 * num_results)]
            for r in search_results:
                url = r["href"]
                print (f"Debug: Retrieved URL: {url}")
                urls_processed += 1
                if verbose:
                    print(f"Found URL ({urls_processed}): {url}")

                # Check if URL is from an ignored domain
                domain = urlparse(url).netloc.lower()
                if any(domain == ignored or domain.endswith("." + ignored) for ignored in ignored_domains):
                    if verbose:
                        print(f"✗ Skipped: URL from ignored domain: {url}")
                    continue

                # Skip PDF files
                if url.lower().endswith('.pdf'):
                    if verbose:
                        print(f"✗ Skipped: PDF file: {url}")
                    continue

                # Extract text immediately
                if verbose:
                    print(f"Extracting content from: {url}")
                text = extract_text_from_url(url, verbose=verbose)

                # Check if the content meets the minimum character requirement
                if text and len(text) >= min_chars:
                    results.append({
                        'url': url,
                        'text': text,
                        'length': len(text)
                    })
                    if verbose:
                        print(f"✓ Added to results ({len(text)} characters)")
                else:
                    if verbose:
                        if text:
                            print(f"✗ Skipped: Content too short ({len(text)} characters, minimum is {min_chars})")
                        else:
                            print(f"✗ Skipped: Failed to extract content")

                if len(results) >= num_results:
                    print (f"Reached desired number of results: {num_results}")
                    break


    except Exception as e:
        if verbose:
            print(f"Error during Google search: {e}")

    if verbose:
        print(f"\nProcessed {urls_processed} URLs, found {len(results)} with sufficient content")
    return results

def extract_text_from_url(url, verbose=True):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Download the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        if verbose:
            print(f"Error extracting text from {url}: {e}")
        return ""

def process_queries_from_file(query_file, num_results=20, min_chars=32, ignored_domains=None, verbose=True):

    all_results = {}

    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        if verbose:
            print(f"Found {len(queries)} queries in {query_file}")

        for i, query in enumerate(queries):
            results = google_search(query, num_results, min_chars, ignored_domains, verbose)
            all_results[query] = results

            # Add a delay between queries to avoid being blocked
            if i < len(queries) - 1:  
                delay = random.uniform(2.0, 5.0)
                if verbose:
                    print(f"Waiting {delay:.1f} seconds before next query...")
                time.sleep(delay)

        return all_results

    except Exception as e:
        if verbose:
            print(f"Error processing queries from file: {e}")
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Search Google and download text from result pages')
    parser.add_argument('--query-file', type=str, default='./data/novelty-bench.txt', help='File containing search queries (one per line)')
    parser.add_argument('--results', type=int, default=20, help='Number of results to retrieve per query (default: 10)')
    parser.add_argument('--min-chars', type=int, default=32, help='Minimum characters required for a result (default: 1000)')
    parser.add_argument('--output', type=str, default='./data/novelty_bench_search_results.json', help='Output file name (default: search_results.json)')
    args = parser.parse_args()

    all_results = process_queries_from_file(args.query_file, args.results, args.min_chars, DEFAULT_IGNORED_DOMAINS)
    with open(args.output, 'w', encoding='utf-8') as f:
        json_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'queries': {}
        }

        for query, results in all_results.items():
            json_data['queries'][query] = results
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Completed! Results for {len(all_results)} queries saved to {args.output} in JSON format")

if __name__ == "__main__":
    main()