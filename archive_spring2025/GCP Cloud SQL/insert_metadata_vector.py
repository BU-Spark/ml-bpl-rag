import os
import json
import uuid
import ijson
import psycopg2
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Database connection details
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bpl_metadata")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FILE_PATH = "bpl_data.json"

# Fields to embed
FIELDS_TO_EMBED = [
    'abstract_tsi',
    'title_info_primary_tsi',
    'title_info_primary_subtitle_tsi',
    'title_info_alternative_tsim'
]

# Initialize DB connection
conn = psycopg2.connect(
    host=DB_HOST,
    port=int(DB_PORT),
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    source_id TEXT,
    field TEXT,
    content TEXT,
    embedding VECTOR(384),
    metadata JSONB
);
""")
conn.commit()

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Stream and process JSON
with open(FILE_PATH, "r") as f:
    parser = ijson.items(f, "Data.item.data.item")
    for item in parser:
        item_id = item.get("id")
        attributes = item.get("attributes", {})
        metadata = json.dumps(attributes)

        for field in attributes:
            if (field in FIELDS_TO_EMBED) or ("note" in field):
                text = str(attributes[field])
                if not text.strip():
                    continue

                chunks = text_splitter.split_text(text) if len(text) > 1000 else [text]

                for chunk in chunks:
                    try:
                        vector = embeddings.embed_documents([chunk])[0]
                        doc_id = str(uuid.uuid4())

                        cur.execute(
                            """
                            INSERT INTO documents (id, source_id, field, content, embedding, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING;
                            """,
                            (doc_id, item_id, field, chunk, vector, metadata)
                        )
                    except Exception as e:
                        print(f"Error embedding or inserting for {item_id} field {field}: {e}")

conn.commit()
cur.close()
conn.close()
