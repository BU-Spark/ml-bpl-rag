#!/usr/bin/env python3
"""
BGE-M3 embedding wrapper for the hybrid RAG system.

BGE-M3 simultaneously produces:
  - Dense embeddings  (1024-dim): used for approximate nearest-neighbor search via pgvector
  - Sparse embeddings (SPLADE-style lexical weights): used for sparse token-overlap search

References:
  - BAAI/bge-m3 on HuggingFace
  - FlagEmbedding library: https://github.com/FlagOpen/FlagEmbedding
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BGEM3Embedder:
    """
    Wrapper around BAAI/bge-m3 that returns both dense and sparse embeddings.

    Usage:
        embedder = BGEM3Embedder()

        # Single text
        dense, sparse = embedder.embed_query("Boston 1919 photographs")

        # Batch of passages (for indexing)
        dense_list, sparse_list = embedder.embed_passages(["text1", "text2"])
    """

    MODEL_ID = "BAAI/bge-m3"

    def __init__(
        self,
        use_fp16: bool = True,
        batch_size: int = 12,
        max_length: int = 8192,
        device: str | None = None,
    ) -> None:
        """
        Args:
            use_fp16:    Use half-precision for GPU inference (faster, less VRAM).
            batch_size:  Encoding batch size.
            max_length:  Max token length per document chunk.
            device:      'cuda', 'cpu', or None (auto-detect).
        """
        from FlagEmbedding import BGEM3FlagModel  # lazy import

        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading {self.MODEL_ID} on {device} (fp16={use_fp16})...")
        self.model = BGEM3FlagModel(
            self.MODEL_ID,
            use_fp16=use_fp16 and device == "cuda",
        )
        logger.info("BGE-M3 model loaded.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Embed a single query string.

        Returns:
            dense  – list of 1024 floats (L2-normalised cosine embedding)
            sparse – dict mapping str(token_id) → float weight
        """
        dense_list, sparse_list = self._encode([query])
        return dense_list[0], sparse_list[0]

    def embed_passages(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        Embed a batch of passage texts.

        Returns:
            dense_list  – list of N dense vectors (list[float], dim=1024)
            sparse_list – list of N sparse dicts {str(token_id): weight}
        """
        return self._encode(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs: np.ndarray = output["dense_vecs"]          # (N, 1024)
        lexical_weights: List[Dict] = output["lexical_weights"] # list of {token_id: weight}

        dense_list = [vec.tolist() for vec in dense_vecs]

        # Normalise token IDs to str keys (consistent with JSONB storage)
        sparse_list: List[Dict[str, float]] = [
            {str(k): float(v) for k, v in lw.items()}
            for lw in lexical_weights
        ]
        return dense_list, sparse_list

    @staticmethod
    def sparse_dot_product(
        sparse_a: Dict[str, float], sparse_b: Dict[str, float]
    ) -> float:
        """Compute the dot product between two sparse vectors (Python-side)."""
        if not sparse_a or not sparse_b:
            return 0.0
        # Iterate over the shorter dict for efficiency
        if len(sparse_a) > len(sparse_b):
            sparse_a, sparse_b = sparse_b, sparse_a
        return sum(v * sparse_b.get(k, 0.0) for k, v in sparse_a.items())
