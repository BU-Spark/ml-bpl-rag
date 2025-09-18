import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
from pinecone import Pinecone
from tqdm import tqdm

# Load env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Sample settings
NUM_BATCHES = 1000  # ~20,000 docs
TOP_K = 100
rows = []

for _ in tqdm(range(NUM_BATCHES), desc="Sampling metadata from Pinecone"):
    rand_vec = np.random.normal(0, 1, 768).tolist()
    try:
        response = index.query(vector=rand_vec, top_k=TOP_K, include_metadata=True)
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            row = {
                "Genre": ", ".join(meta.get("genre_basic_ssim", [])),
                "Year": ", ".join(meta.get("date_facet_yearly_itim", [])),
                "City": ", ".join(meta.get("subject_geographic_sim", [])),
                "Topic": ", ".join(meta.get("subject_topic_tsim", [])),
                "Collection": ", ".join(meta.get("collection_name_ssim", [])),
            }
            rows.append(row)
    except Exception as e:
        print(f"Query error: {e}")

# Save as CSV
df = pd.DataFrame(rows)
df.to_csv("pinecone_metadata_sample_100k.csv", index=False)
print("✅ Saved as pinecone_metadata_sample_100k.csv")
