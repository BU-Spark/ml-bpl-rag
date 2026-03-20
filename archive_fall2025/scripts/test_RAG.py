import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

from dotenv import load_dotenv
import psycopg2, os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from RAG import RAG

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    database=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode=os.getenv("PGSSLMODE", "prefer")
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = ChatOpenAI(model="gpt-4o-mini")

answer, docs = RAG(llm, conn, embeddings, query="Boston flood 1919")
print(answer)
