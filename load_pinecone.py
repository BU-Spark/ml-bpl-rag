from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from uuid import uuid4
from dotenv import load_dotenv
import ijson
import os
import sys
import time
import hashlib

# Load environment variables
load_dotenv()

# CLI args: python3 load_pinecone.py path/to/data.json pinecone-index-name
PATH = sys.argv[1]
INDEX_NAME = sys.argv[2]

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("Initializing Embeddings and VectorStore...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Fields to embed
fields = ['abstract_tsi', 'title_info_primary_tsi', 'title_info_primary_subtitle_tsi', 'title_info_alternative_tsim']

# Deterministic ID generator
def make_doc_id(source, field, content):
    raw = f"{source}_{field}_{content}"
    return hashlib.md5(raw.encode()).hexdigest()

print("Starting streaming and embedding process...")
start = time.time()
doc_count = 0

with open(PATH, "r") as f:
    items = ijson.items(f, "data.item.data.item")  
    for item in items:
        try:
            item_id = item.get("id")
            attributes = item.get("attributes", {})
            documents = []
            uuids = []

            for field in attributes:
                if field in fields or "note" in field:
                    entry = str(attributes[field])
                    if not entry.strip():
                        continue
                    if len(entry) > 1000:
                        chunks = text_splitter.split_text(entry)
                        for chunk in chunks:
                            documents.append(Document(page_content=chunk, metadata={"source": item_id, "field": field}))
                            uuids.append(make_doc_id(item_id, field, chunk))
                    else:
                        documents.append(Document(page_content=entry, metadata={"source": item_id, "field": field}))
                        uuids.append(make_doc_id(item_id, field, entry))

            if documents:
                vector_store.add_documents(documents=documents, ids=uuids)
                doc_count += len(documents)

                if doc_count % 100 == 0:
                    print(f"{doc_count} documents embedded... latest ID: {item_id}")

        except Exception as e:
            print(f" Error processing item {item.get('id', 'unknown')}: {e}")

end = time.time()
print(f"\n Embedded {doc_count} total documents in {end - start:.2f} seconds.")
