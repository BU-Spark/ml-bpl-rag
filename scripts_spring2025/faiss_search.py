import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Config
DATA_FILE = '../out1_500.json'
INDEX_FILE = 'digital_commonwealth.index'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
FIELDS = ['abstract_tsi', 'title_info_primary_tsi', 'title_info_primary_subtitle_tsi', 'title_info_alternative_tsim']

def load_metadata():
    with open(DATA_FILE, 'r') as file:
        return json.load(file)

def extract_documents(data):
    documents = []
    for page in data:
        for item in page['data']:
            content = []
            for field in FIELDS:
                if field in item['attributes']:
                    content.append(str(item['attributes'][field]))
            documents.append(" ".join(content))
    return documents

def build_faiss_index(documents, model):
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings_np = np.array(embeddings).astype('float32')

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    # Save index for future use
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS index saved to {INDEX_FILE}")
    return index

def load_or_build_index(documents, model):
    if os.path.exists(INDEX_FILE):
        print(f"Loading existing FAISS index from {INDEX_FILE}")
        index = faiss.read_index(INDEX_FILE)
    else:
        print(f"Building new FAISS index...")
        index = build_faiss_index(documents, model)
    return index

def search_query(query, model, index, documents, top_k=5):
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "distance": distances[0][rank],
            "content": documents[idx]
        })
    return results


def main():
    # Load data & documents
    data = load_metadata()
    documents = extract_documents(data)

    # Load model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Load or build FAISS index
    index = load_or_build_index(documents, model)

    # Run query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting search")
            break

        results = search_query(query, model, index, documents)

        print(f"\nResults for: {query}")
        for result in results:
            print(f"Rank {result['rank']} (Distance={result['distance']}):")
            print(result['content'])
            print("-" * 40)

if __name__ == "__main__":
    main()
