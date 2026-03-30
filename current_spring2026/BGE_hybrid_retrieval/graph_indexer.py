#!/usr/bin/env python3
"""
GraphRAG indexing pipeline for the BGE-M3 hybrid RAG system.

Wraps Microsoft's GraphRAG library to build a knowledge graph from
newspaper records, detect communities via Leiden clustering, and
generate community summaries.

Output structure
----------------
graphrag_index/{year}/
    input/                    -- plain-text files fed to GraphRAG
    output/                   -- GraphRAG parquet artefacts
        entities.parquet
        relationships.parquet
        communities.parquet
        community_reports.parquet
        text_units.parquet
    settings.yaml             -- GraphRAG config for this year

Usage:
    python -m BGE_hybrid_retrieval.graph_indexer --year 1900 \
        --data Boston_Traveler_The_Boston_Traveler/1900.json

    # Or index all years at once:
    python -m BGE_hybrid_retrieval.graph_indexer --all \
        --data-dir Boston_Traveler_The_Boston_Traveler
"""

import argparse
import asyncio
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Max characters per input document fed to GraphRAG (avoid token-limit blowup)
MAX_DOC_CHARS = 15_000
# Max records per year to index (GraphRAG entity extraction is LLM-heavy)
MAX_RECORDS_PER_YEAR = 50

DEFAULT_GRAPHRAG_DIR = Path(__file__).parent / "graphrag_index"


# ---------------------------------------------------------------------------
# settings.yaml template
# ---------------------------------------------------------------------------

SETTINGS_TEMPLATE = textwrap.dedent("""\
    # Auto-generated GraphRAG settings for year {year}
    # Docs: https://microsoft.github.io/graphrag/config/yaml/

    completion_models:
      default:
        model_provider: openai
        model: gpt-4o-mini
        api_key: ${{GRAPHRAG_API_KEY}}
        max_retries: 3
        tokens_per_minute: 80000
        requests_per_minute: 40

    embedding_models:
      default:
        model_provider: openai
        model: text-embedding-3-small
        api_key: ${{GRAPHRAG_API_KEY}}

    input:
      type: text
      storage:
        type: file
        base_dir: input
      file_pattern: ".*\\\\.txt$"
      encoding: utf-8

    chunking:
      type: tokens
      size: 1200
      overlap: 100

    extract_graph:
      completion_model_id: default
      entity_types:
        - person
        - organization
        - location
        - event
        - date
        - topic
      max_gleanings: 1

    community_reports:
      completion_model_id: default
      max_length: 1500
      max_input_length: 8000

    cluster_graph:
      max_cluster_size: 10
      seed: 42

    embed_graph:
      enabled: true
      embedding_model_id: default

    vector_store:
      type: lancedb
      db_uri: output/lancedb

    output:
      type: file
      base_dir: output

    local_search:
      text_unit_prop: 0.5
      community_prop: 0.1
      top_k_entities: 10
      top_k_relationships: 10
      max_context_tokens: 12000

    global_search:
      dynamic_search_threshold: 0.5
      dynamic_search_keep_parent: true
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_records(data_path: Path) -> List[dict]:
    """Load newspaper records from a year JSON file."""
    with open(data_path) as f:
        raw = json.load(f)

    # Handle both formats: list of records, or {records: [...]}
    if isinstance(raw, dict) and "records" in raw:
        records = raw["records"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unexpected JSON structure in {data_path}")

    logger.info(f"Loaded {len(records)} records from {data_path}")
    return records


def _records_to_text_files(records: List[dict], input_dir: Path, max_records: int = MAX_RECORDS_PER_YEAR) -> int:
    """
    Convert newspaper records to plain-text files for GraphRAG ingestion.
    Each record becomes one .txt file with metadata header + truncated body.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for rec in records[:max_records]:
        ark_id = rec.get("ark_id", rec.get("doc_id", f"doc_{count}"))
        newspaper = rec.get("newspaper", "Unknown")
        issue_date = rec.get("issue_date", "Unknown date")
        title = rec.get("title", "")
        topics = ", ".join(rec.get("topics", []))
        geography = ", ".join(rec.get("geography", []))
        clean_text = rec.get("clean_text", rec.get("full_text", ""))

        if not clean_text:
            continue

        # Truncate to avoid excessive LLM costs during entity extraction
        body = clean_text[:MAX_DOC_CHARS]

        header = (
            f"NEWSPAPER: {newspaper}\n"
            f"TITLE: {title}\n"
            f"DATE: {issue_date}\n"
            f"TOPICS: {topics}\n"
            f"GEOGRAPHY: {geography}\n"
            f"SOURCE ID: {ark_id}\n"
            f"---\n"
        )

        out_path = input_dir / f"{ark_id}.txt"
        out_path.write_text(header + body, encoding="utf-8")
        count += 1

    logger.info(f"Wrote {count} text files to {input_dir}")
    return count


def _write_settings(project_dir: Path, year: str) -> Path:
    """Write a settings.yaml for this year's GraphRAG project."""
    settings_path = project_dir / "settings.yaml"
    content = SETTINGS_TEMPLATE.format(year=year)
    settings_path.write_text(content, encoding="utf-8")
    logger.info(f"Settings written to {settings_path}")
    return settings_path


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

async def _run_graphrag_index(project_dir: Path) -> None:
    """Run GraphRAG indexing pipeline via the Python API."""
    from graphrag.config.load_config import load_config
    import graphrag.api as api

    config = load_config(project_dir)
    logger.info(f"Starting GraphRAG indexing for {project_dir}...")

    results = await api.build_index(config=config)

    for r in results:
        status = f"ERROR: {r.errors}" if r.errors else "OK"
        logger.info(f"  Workflow: {r.workflow} — {status}")

    logger.info(f"GraphRAG indexing complete for {project_dir}")


def index_year(year: str, data_path: Path, graphrag_dir: Path = DEFAULT_GRAPHRAG_DIR,
               max_records: int = MAX_RECORDS_PER_YEAR) -> Path:
    """
    Build GraphRAG index for a single year.

    Args:
        year:         Year label (e.g. "1900").
        data_path:    Path to the year JSON file.
        graphrag_dir: Root directory for GraphRAG indexes.
        max_records:  Max records to index per year.

    Returns:
        Path to the year's GraphRAG project directory.
    """
    project_dir = graphrag_dir / year
    project_dir.mkdir(parents=True, exist_ok=True)

    # Ensure GRAPHRAG_API_KEY is set (fall back to OPENAI_API_KEY)
    if not os.environ.get("GRAPHRAG_API_KEY"):
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            os.environ["GRAPHRAG_API_KEY"] = openai_key
            logger.info("Set GRAPHRAG_API_KEY from OPENAI_API_KEY")

    # Step 1: Convert records to text files
    records = _load_records(data_path)
    input_dir = project_dir / "input"
    n = _records_to_text_files(records, input_dir, max_records=max_records)
    if n == 0:
        raise ValueError(f"No valid records found in {data_path}")

    # Step 2: Write settings.yaml
    _write_settings(project_dir, year)

    # Step 3: Run GraphRAG indexing
    asyncio.run(_run_graphrag_index(project_dir))

    logger.info(f"GraphRAG index for year {year} saved to {project_dir}")
    return project_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build GraphRAG indexes for newspaper data.")
    parser.add_argument("--year", help="Single year to index (e.g. 1900)")
    parser.add_argument("--data", help="Path to year JSON file (required with --year)")
    parser.add_argument("--all", action="store_true", help="Index all years in --data-dir")
    parser.add_argument(
        "--data-dir",
        default="Boston_Traveler_The_Boston_Traveler",
        help="Directory containing year JSON files (used with --all)",
    )
    parser.add_argument(
        "--graphrag-dir",
        default=str(DEFAULT_GRAPHRAG_DIR),
        help="Root directory for GraphRAG indexes",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS_PER_YEAR,
        help=f"Max records per year to index (default {MAX_RECORDS_PER_YEAR})",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    graphrag_dir = Path(args.graphrag_dir)

    if args.all:
        data_dir = Path(args.data_dir)
        json_files = sorted(data_dir.glob("*.json"))
        json_files = [f for f in json_files if f.stem.isdigit()]  # only year files
        logger.info(f"Found {len(json_files)} year files in {data_dir}")

        for jf in json_files:
            year = jf.stem
            # Skip if already indexed
            output_dir = graphrag_dir / year / "output"
            if (output_dir / "community_reports.parquet").exists():
                logger.info(f"Year {year} already indexed, skipping.")
                continue
            try:
                index_year(year, jf, graphrag_dir, max_records=args.max_records)
            except Exception as e:
                logger.error(f"Failed to index year {year}: {e}")
                continue

    elif args.year and args.data:
        index_year(args.year, Path(args.data), graphrag_dir, max_records=args.max_records)

    else:
        parser.error("Provide either --year + --data, or --all + --data-dir")


if __name__ == "__main__":
    main()
