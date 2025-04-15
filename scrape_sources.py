import json
import time
import requests

# List of direct JSON metadata URLs
LINKS = [
    "https://www.digitalcommonwealth.org/search/commonwealth:2j62s565w.json",
    "https://www.digitalcommonwealth.org/search/commonwealth-oai:w0895g04t.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:5h73vx18p.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:fx71d9445.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:fx71d9712.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:fx71d959s.json",
    "https://www.digitalcommonwealth.org/search/commonwealth-oai:mc87qf31g.json",
    "https://www.digitalcommonwealth.org/search/commonwealth-oai:mc87qf39p.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:z603vg83r.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:70796j511.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:6108z041f.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:8g84ms67v.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:cr56qx57k.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:3n206v73b.json",
    "https://www.digitalcommonwealth.org/search/commonwealth:n8712j541.json"
]

def fetch_specific_items():
    start = time.time()
    output = []

    for link in LINKS:
        try:
            print(f"Fetching {link}")
            response = requests.get(link)
            response.raise_for_status()
            output.append(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {link}: {e}")

    # Save to JSON file
    file_name = "selected_items.json"
    with open(file_name, 'w') as f:
        json.dump(output, f, indent=2)

    end = time.time()
    print(f"✅ Fetched {len(output)} items in {end - start:.2f} seconds. Saved to {file_name}.")

if __name__ == "__main__":
    fetch_specific_items()
