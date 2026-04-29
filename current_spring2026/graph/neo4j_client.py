"""
graph/neo4j_client.py

Neo4j connection manager and graph schema setup.
Includes vector index on Entity embeddings for semantic entity matching.

Run once:
    python -m graph.neo4j_client
"""

from __future__ import annotations
from contextlib import contextmanager
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ── Driver singleton ──────────────────────────────────────────────────────────

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
    return _driver


@contextmanager
def get_session():
    driver = get_driver()
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# ── Schema setup ──────────────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    # Document node constraints
    "CREATE CONSTRAINT document_ark IF NOT EXISTS FOR (d:Document) REQUIRE d.ark_id IS UNIQUE",

    # Entity node constraints
    "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",

    # Indexes for fast lookup
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX document_year IF NOT EXISTS FOR (d:Document) ON (d.year)",
]

# Vector index — created separately since it requires different syntax
VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
FOR (e:Entity) ON e.embedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
    }
}
"""


def create_schema():
    print("Creating Neo4j schema...")
    with get_session() as session:
        for query in SCHEMA_QUERIES:
            try:
                session.run(query)
                print(f"  OK: {query[:60]}...")
            except Exception as e:
                print(f"  Skip: {e}")

        # Create vector index
        try:
            session.run(VECTOR_INDEX_QUERY)
            print("  OK: vector index on Entity.embedding (1024 dims, cosine)")
        except Exception as e:
            print(f"  Skip vector index: {e}")

    print("Schema ready.")


if __name__ == "__main__":
    create_schema()
    print("\nVerifying connection...")
    with get_session() as session:
        result = session.run("RETURN 1 AS ok")
        print("Neo4j connected OK:", result.single()["ok"])
    close_driver()