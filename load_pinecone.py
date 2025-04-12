from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from uuid import uuid4
import hashlib
import os
import sys
import time
import json
import ijson
from dotenv import load_dotenv

load_dotenv()

# Args
args = sys.argv[1:]
if len(args) < 2:
    print("Usage: python3 script.py <path_to_json> <pinecone_index> [--static]")
    sys.exit(1)

PATH = args[0]
INDEX_NAME = args[1]
USE_IJSON = "--static" in args

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

fields = ['abstract_tsi', 'title_info_primary_tsi', 'title_info_primary_subtitle_tsi', 'title_info_alternative_tsim']

def make_doc_id(source, field, content):
    raw = f"{source}_{field}_{content}"
    return hashlib.md5(raw.encode()).hexdigest()

print("Starting embedding process...")
start = time.time()
doc_count = 0

def process_item(item):
    global doc_count
    item_id = item.get("id")
    attributes = item.get("attributes", {})
    documents = []
    uuids = []

    for field in attributes:
        if field in fields or "note" in field:
            entry = str(attributes[field])
            if len(entry) > 1000:
                chunks = text_splitter.split_text(entry)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata={"source": item_id, "field": field}))
                    uuids.append(make_doc_id(item_id, field, chunk))
            else:
                documents.append(Document(page_content=entry, metadata={"source": item_id, "field": field}))
                uuids.append(make_doc_id(item_id, field, entry))

    if documents:
        existing = index.fetch(ids=uuids)
        existing_ids = set(existing.get('vectors', {}).keys())

        new_documents = []
        new_ids = []
        for doc, uid in zip(documents, uuids):
            if uid not in existing_ids:
                new_documents.append(doc)
                new_ids.append(uid)

        if new_documents:
            vector_store.add_documents(documents=new_documents, ids=new_ids)
            doc_count += len(new_documents)
            if doc_count % 100 == 0:
                print(f"✅ {doc_count} documents embedded... latest ID: {item_id}")

# Load data
if USE_IJSON:
    with open(PATH, "r") as f:
        parser = ijson.items(f, "Data.item.data.item")
        for item in parser:
            try:
                process_item(item)
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
else:
    with open(PATH, "r") as f:
        meta = json.load(f)
        all_items = []
        for page in meta:
            all_items.extend(page.get("data", []))
        for item in all_items:
            try:
                process_item(item)
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")

end = time.time()
print(f"\n✅ Embedded {doc_count} total documents in {end - start:.2f} seconds.")
