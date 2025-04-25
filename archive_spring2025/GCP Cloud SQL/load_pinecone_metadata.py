import os
import ijson
import time
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from decimal import Decimal


# Unzip the uploaded env_file.zip to extract the .env for SCC
with zipfile.ZipFile("env_file.zip", "r") as zip_ref:
    zip_ref.extractall() 

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')  

# --- Init Pinecone ---
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

#FILE_PATH = "/projectnb/sparkgrp/ml-bpl-rag-data/extraneous/metadata/bpl_data.json"
FILE_PATH = "selected_items.json"
NUM_WORKERS = 6
BATCH_SIZE = 10 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def safe_join(val):
    return " ".join(val) if isinstance(val, list) else str(val or "")

def sanitize_metadata(metadata):
    def sanitize_value(value):
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, float):
            return int(value) if value.is_integer() else value
        elif isinstance(value, list):
            # Convert all elements in lists to strings
            return [str(v) for v in value]
        elif isinstance(value, dict):
            return sanitize_metadata(value)
        return value


    return {k: sanitize_value(v) for k, v in metadata.items()}



def process_item(item):
    try:
        id = item["id"]
        attributes = item["attributes"]
        documents = []

        # --- Semantic summary ---
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
        
        base_metadata = sanitize_metadata({
            **attributes,
            "source": id,
            "chunk_source_text": summary_text
        })


        documents.append(Document(page_content=summary_text, metadata=base_metadata))

        fields_to_embed = [
            'abstract_tsi', 'title_info_primary_tsi',
            'title_info_primary_subtitle_tsi', 'title_info_alternative_tsim'
        ]
        for field in fields_to_embed + [f for f in attributes if "note" in f]:
            if field in attributes:
                val = safe_join(attributes[field])
                if val.strip():
                    chunks = text_splitter.split_text(val) if len(val) > 1000 else [val]
                    for chunk in chunks:
                        documents.append(Document(page_content=chunk, metadata=sanitize_metadata({
                            **attributes,
                            "source": id,
                            "field": field,
                            "chunk_source_text": val
                        })))

        print(f"Processed item: {id} with {len(documents)} documents")
        return documents
    except Exception as e:
        print(f"Error on {item.get('id')}: {e}")
        return []


start = time.time()
print("Starting streaming + embedding...")

with open(FILE_PATH, 'rb') as f:
    items = ijson.items(f, "Data.item.data.item")
    total_embedded = 0
    total_processed = 0  # To count processed items

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for item in tqdm(items, desc="Processing items"):
            documents = process_item(item) 
            total_processed += 1

            if documents:
                uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
                print(f"Inserting batch of {len(documents)} documents to Pinecone...")
                try:
                    vector_store.add_documents(documents=documents, ids=uuids)
                    total_embedded += len(documents)
                    print(f"Total embedded so far: {total_embedded}")
                except Exception as e:
                    print(f"Error inserting documents: {e}")

            if total_processed % BATCH_SIZE == 0:
                print(f"Processed {total_processed} items, embedded {total_embedded} documents so far.")

end = time.time()
print(f"Done in {end - start:.2f} seconds.")
