import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import pytz  

# --- Question for this run ---
query = "What were some important historical events that happened in Boston in 1919?"

# --- URLs you manually curated ---
urls = [
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Molasses+Disaster%2C+Boston%2C+Mass.%2C+1919&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Riots&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Strikes&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=World+War%2C+1914-1918&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Parades+%26+processions&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Boston+Elevated+Railway+Company&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
]

# --- Teammate name for record---
author = "Nathan"

# --- Output path ---
json_path = "current_fall2025/evaluation/test_ground_truth.ndjson"
os.makedirs(os.path.dirname(json_path), exist_ok=True)

# --- Get current timestamp in Boston time ---
boston_tz = pytz.timezone("America/New_York")
timestamp = datetime.now(boston_tz).strftime("%Y-%m-%d %H:%M:%S %Z")

# --- Scrape IDs from each URL ---
results = []
for url in urls:
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    for item in soup.select(".document"):
        link_tag = item.select_one("a[href]")
        if not link_tag:
            continue
        doc_id = link_tag["href"].split("/")[-1]
        results.append(doc_id)

unique_ids = sorted(set(results))

# --- Save one NDJSON record ---
record = {
    "question": query,
    "ground_truth": unique_ids,
    "urls": urls,
    "author": author,
    "timestamp": timestamp
}

with open(json_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Appended record with {len(unique_ids)} ground_truth to {json_path}")