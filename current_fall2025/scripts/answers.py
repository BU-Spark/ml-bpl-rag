import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os

# --- Question for this run ---
query = "What were some important historical events that happened in Boston in 1919?"

# --- URLs to scrape ---
urls = [
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Molasses+Disaster%2C+Boston%2C+Mass.%2C+1919&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Riots&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Strikes&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=World+War%2C+1914-1918&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Parades+%26+processions&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
    "https://www.digitalcommonwealth.org/search?f%5Bphysical_location_ssim%5D%5B%5D=Boston+Public+Library&f%5Bsubject_facet_ssim%5D%5B%5D=Boston+Elevated+Railway+Company&f%5Bsubject_geographic_sim%5D%5B%5D=Boston&per_page=100&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D=1919&range%5Bdate_facet_yearly_itim%5D%5Bend%5D=1919",
]

# --- Output path ---
json_path = "current_fall2025/evaluation/test_answers.ndjson"
os.makedirs(os.path.dirname(json_path), exist_ok=True)

# --- Scrape results ---
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

# --- Append one JSON object per line ---
record = {"question": query, "answers": unique_ids}

with open(json_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Appended record with {len(unique_ids)} answers to {json_path}")
