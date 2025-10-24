import json
import time
import os
import sys
import requests
from multiprocessing import Pool
from random import uniform

# Use query '*' to fetch all public entries in Boston Public Library
BASE_URL = "https://www.digitalcommonwealth.org/search.json?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&per_page=100&q=*"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataCollector/1.0; +https://yourdomain.com/)"
}

def fetch_pages(args):
    start_page, end_page = args
    file_name = f"out_{start_page}_{end_page}.json"
    file_path = os.path.join("../data/raw", file_name)
    output = []
    
    print(f"ðŸ§µ Scraping pages {start_page} to {end_page}...")

    for page in range(start_page, end_page + 1):
        retries = 0
        while retries < 5:
            try:
                url = f"{BASE_URL}&page={page}"
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.raise_for_status()
                data = response.json()

                items = data.get("data", [])
                if not items:
                    print(f"Page {page} returned 0 items")
                else:
                    for item in items:
                        entry = item.get("attributes", {})
                        entry["id"] = item.get("id") 
                        output.append(entry)
                    print(f"Page {page}: {len(items)} items saved")

                # Respect API rate limits
                time.sleep(uniform(1.5, 3.0))
                break

            except requests.exceptions.RequestException as e:
                retries += 1
                backoff = 2 ** retries + uniform(0, 1)
                print(f"[Page {page}] Error: {e}. Retry {retries}/5 in {backoff:.2f}s")
                time.sleep(backoff)

    with open(file_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(output)} entries to {file_name}")
    return file_name

def divide_work(start_page, end_page, chunks):
    total = end_page - start_page + 1
    size = (total + chunks - 1) // chunks
    return [(i, min(i + size - 1, end_page)) for i in range(start_page, end_page + 1, size)]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python load_scraper_bpl.py START_PAGE END_PAGE")
        sys.exit(1)

    start_page = int(sys.argv[1])
    end_page = int(sys.argv[2])
    num_workers = 2

    work_chunks = divide_work(start_page, end_page, num_workers)
    print(f"Using {num_workers} workers to scrape pages {start_page}â€“{end_page}")

    start_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(fetch_pages, work_chunks)

    print(f"\nDone. Files saved: {results}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
