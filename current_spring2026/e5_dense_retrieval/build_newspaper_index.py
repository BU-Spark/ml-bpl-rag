#!/usr/bin/env python3
"""
Standalone script to ingest a newspaper JSON file and build FAISS indexes.

Usage:
    python build_newspaper_index.py \
        --json  Data/Boston_Evening_Transcript_Boston_Evening_Transcript/1940.json \
        --out   indexes/1940 \
        [--model intfloat/e5-base-v2] \
        [--chunk-words 400] \
        [--overlap-words 50] \
        [--batch-size 32]

The script writes five files to --out/:
    title_summary.faiss   — FAISS index for Title/Summary DB
    title_meta.pkl        — aligned metadata (list[dict])
    full_article.faiss    — FAISS index for Full Article DB
    chunk_meta.pkl        — aligned metadata (list[dict])
    doc_chunks_map.pkl    — doc_id → list[chunk_meta] lookup

These files are consumed by newspaper_app.py at runtime.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Make sure the package is importable when running from Documents/BPL/
sys.path.insert(0, str(Path(__file__).parent))
from newspaper_RAG.ingest import ingest_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build FAISS indexes from a newspaper JSON file."
    )
    p.add_argument(
        "--json",
        required=True,
        help="Path to the newspaper JSON file (e.g. Data/.../1940.json)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory for FAISS indexes and metadata pickles.",
    )
    p.add_argument(
        "--model",
        default="intfloat/e5-base-v2",
        help="HuggingFace model ID for E5 embeddings (default: intfloat/e5-base-v2)",
    )
    p.add_argument(
        "--chunk-words",
        type=int,
        default=400,
        help="Words per chunk in the Full Article index (default: 400)",
    )
    p.add_argument(
        "--overlap-words",
        type=int,
        default=50,
        help="Overlap words between consecutive chunks (default: 50)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading E5 model '{args.model}' on {device}...")
    model = SentenceTransformer(args.model, device=device)
    logger.info("Model loaded.")

    ingest_json(
        json_path=str(json_path),
        model=model,
        output_dir=args.out,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        batch_size=args.batch_size,
    )

    logger.info(f"Done. Indexes saved to: {args.out}")


if __name__ == "__main__":
    main()
