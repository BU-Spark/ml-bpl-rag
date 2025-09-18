import os
import psycopg2
import numpy as np
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load DB config from environment
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5435")
DB_NAME = os.getenv("DB_NAME", "bpl_metadata")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

class CloudSQLVectorStore:
    def __init__(self, embedding):
        self.embeddings = embedding

    def similarity_search_with_score(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        logger.debug(f"Embedding query: '{query}'")
        query_vector = self.embeddings.embed_query(query)
        logger.debug(f"Query vector (first 5 dims): {query_vector[:5]}... Length: {len(query_vector)}")

        logger.debug("Attempting to connect to PostgreSQL...")
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname="bpl_metadata",
                user="postgres",
                password=DB_PASSWORD
            )
            cur = conn.cursor()
            logger.debug("PostgreSQL connection established.")
            try:
                pg_vector = f"[{','.join(map(str, query_vector))}]"
                logger.debug(f"PostgreSQL vector format: {pg_vector[:60]}...")
            except Exception as e:
                logger.error(f"Error formatting pg_vector: {e}")


            logger.info("Executing similarity search...")
            cur.execute(
                f"""
                SELECT id, source_id, field, content, metadata,
                    1 - (embedding <#> %s::vector) AS score
                FROM documents
                ORDER BY embedding <#> %s::vector ASC
                LIMIT %s;
                """,
                (pg_vector, pg_vector, k)
            )

            rows = cur.fetchall()
            logger.info(f"Number of results: {len(rows)}")
            if not rows:
                logger.warning("No results found — your vector store might be empty or not indexed yet!")

            results = []
            for row in rows:
                _id, source_id, field, content, metadata, score = row
                logger.debug(f"Found: {source_id} | Score: {score}")

                parsed_metadata = metadata if metadata else {}
                parsed_metadata["source"] = source_id
                parsed_metadata["field"] = field

                doc = Document(page_content=content, metadata=parsed_metadata)
                #doc = Document(page_content=content, metadata={"source": source_id, "field": field})
                results.append((doc, float(score)))

            return results

        except Exception as e:
            logger.error(f"Error in similarity_search_with_score: {str(e)}")
            return []

        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

