from pinecone import Pinecone
import os
from dotenv import load_dotenv
from collections import defaultdict
import json

# Load environment variables (for Pinecone API Key + env)
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "gcp-starter"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index("audio-index")  

# 🔍 Query for all audio-related entries
# Assumes metadata has "field": "format" and "content" mentions "Audio"
query_vector = [0.0] * 384  # dummy vector (replace with real audio query embedding)
results = index.query(vector=query_vector, top_k=100, include_metadata=True)

# 🧠 Group metadata by source_id
grouped_metadata = defaultdict(lambda: defaultdict(list))

for match in results['matches']:
    metadata = match['metadata']
    source_id = metadata.get('source', 'unknown')
    field = metadata.get('field', 'unknown')
    content = match.get('metadata', {}).get('text', match.get('page_content', ''))

    # Skip if it’s not audio-related (adjust based on your actual metadata structure)
    if "audio" not in content.lower() and "audio" not in field.lower():
        continue

    grouped_metadata[source_id][field].append(content)

# 📝 Format the grouped entries
output_entries = []
for source_id, fields in grouped_metadata.items():
    combined_text = ""
    for field_name, contents in fields.items():
        combined_text += f"{field_name.capitalize()}:\n" + "\n".join(contents) + "\n\n"

    output_entries.append({
        "source_id": source_id,
        "content": combined_text.strip()
    })

# 💾 Save to JSON file for inspection / future use
with open("grouped_audio_results.json", "w", encoding="utf-8") as f:
    json.dump(output_entries, f, indent=2)

print(f"Grouped and saved {len(output_entries)} audio metadata entries.")
