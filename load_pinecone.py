# File: load_pinecone.py

import os
import sys
import json
import time
import hashlib
import ijson
import requests
from uuid import uuid4
from dotenv import load_dotenv
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from image_captioning import ImageCaptioner
from image_scraper import DigitalCommonwealthScraper
from audio_embedding import AudioEmbedder

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Args ---
args = sys.argv[1:]
if len(args) < 4:
    print("Usage: python load_pinecone.py <BEGIN_INDEX> <END_INDEX> <PATH_TO_JSON> <PINECONE_INDEX_NAME> [--static] [--no-images] [--no-audio]")
    sys.exit(1)

BEGIN_INDEX = int(args[0])
END_INDEX = int(args[1])
PATH = args[2]
INDEX_NAME = args[3]
USE_IJSON = "--static" in args
SKIP_IMAGES = "--no-images" in args
SKIP_AUDIO = "--no-audio" in args

# --- Setup ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
captioner = ImageCaptioner()
scraper = DigitalCommonwealthScraper()
audio_embedder = AudioEmbedder(index_name=INDEX_NAME, namespace="bpl_audio")
doc_count = 0

fields = ['abstract_tsi', 'title_info_primary_tsi', 'title_info_primary_subtitle_tsi', 'title_info_alternative_tsim']

def make_doc_id(source, field, content):
    raw = f"{source}_{field}_{content}"
    return hashlib.md5(raw.encode()).hexdigest()

def safe_join(val):
    return " ".join(val) if isinstance(val, list) else str(val or "")

def sanitize_metadata(metadata):
    def sanitize_value(value):
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, float):
            return int(value) if value.is_integer() else value
        elif isinstance(value, list):
            return [str(v) for v in value]
        elif isinstance(value, dict):
            return sanitize_metadata(value)
        return value
    return {k: sanitize_value(v) for k, v in metadata.items()}

def batch_insert_to_pinecone(documents, ids, batch_size=50):
    def insert_batch(batch_docs, batch_ids):
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    with ThreadPoolExecutor() as executor:
        for i in range(0, len(documents), batch_size):
            executor.submit(insert_batch, documents[i:i + batch_size], ids[i:i + batch_size])

def process_item(item):
    global doc_count
    item_id = item.get("id")
    attributes = item
    documents = []
    uuids = []

    summary_text = f"""
    Title: {attributes.get('title_info_primary_tsi')}
    Subtitle: {attributes.get('title_info_primary_subtitle_tsi')}
    Abstract: {attributes.get('abstract_tsi')}
    Notes: {safe_join(attributes.get('note_tsim', []))}
    Subjects: {safe_join(attributes.get('subject_topic_tsim', []))}
    People: {safe_join(attributes.get('subject_name_tsim', []))}
    Locations: {safe_join(attributes.get('subject_geographic_sim', []))}
    Date: {safe_join(attributes.get('date_tsim', []))}
    Type: {safe_join(attributes.get('type_of_resource_ssim', []))}
    Collection: {safe_join(attributes.get('collection_name_ssim', []))}
    """.strip()

    metadata = sanitize_metadata({**attributes, "source": item_id, "chunk_source_text": summary_text})
    documents.append(Document(page_content=summary_text, metadata=metadata))
    uuids.append(make_doc_id(item_id, "semantic_summary", summary_text))

    for field in attributes:
        if field in fields or "note" in field:
            entry = str(attributes[field])
            chunks = text_splitter.split_text(entry) if len(entry) > 1000 else [entry]
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata=sanitize_metadata({
                        **attributes,
                        "source": item_id,
                        "field": field,
                        "chunk_source_text": entry
                    })
                ))
                uuids.append(make_doc_id(item_id, field, chunk))

    if not SKIP_IMAGES:
        try:
            item_url = f"https://www.digitalcommonwealth.org/search/{item_id}"
            images = scraper.extract_images(item_url)
            if images:
                image_info = images[0]
                image_url = image_info['url']
                analysis = captioner.analyze_image(image_url)
                caption = analysis.get("description")
                tags = analysis.get("tags", [])
                if caption:
                    doc = Document(
                        page_content=caption,
                        metadata={
                            "source": item_id,
                            "field": "image_caption",
                            "tags": tags,
                            "image_url": image_url,
                            "source_page": item_url,
                            "alt": image_info.get("alt", "")
                        }
                    )
                    uid = make_doc_id(item_id, "image_caption", caption)
                    documents.append(doc)
                    uuids.append(uid)
        except Exception as e:
            print(f"Error processing image for {item_id}: {e}")

    if not SKIP_AUDIO:
        try:
            audio_url = attributes.get("audio_url")
            if audio_url:
                response = requests.get(audio_url)
                if response.status_code == 200:
                    audio_bytes = response.content
                    audio_embedder.add_audios([audio_bytes], metadatas=[attributes], ids=[f"{item_id}_audio"])
        except Exception as e:
            print(f"Error processing audio for {item_id}: {e}")

    if documents:
        existing = index.fetch(ids=uuids)
        existing_ids = set(existing.get('vectors', {}).keys())
        new_docs = [doc for i, doc in enumerate(documents) if uuids[i] not in existing_ids]
        new_ids = [uid for uid in uuids if uid not in existing_ids]

        if new_docs:
            batch_insert_to_pinecone(new_docs, new_ids)
            doc_count += len(new_docs)
            if doc_count % 100 == 0:
                print(f"{doc_count} documents embedded... latest ID: {item_id}")

# --- Load and process ---
start = time.time()
if USE_IJSON:
    with open(PATH, "r") as f:
        parser = ijson.items(f, "item")
        for item in parser:
            try:
                process_item(item)
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
else:
    with open(PATH, "r") as f:
        meta = json.load(f)
        items = meta.get("data", [])[BEGIN_INDEX:END_INDEX+1]
        for item in items:
            try:
                process_item(item)
            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")

end = time.time()
print(f"\nEmbedded {doc_count} total documents in {end - start:.2f} seconds.")
