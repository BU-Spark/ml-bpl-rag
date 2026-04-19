from __future__ import annotations
from typing import List
import numpy as np
from config import BGE_MODEL_NAME, BGE_DEVICE, BGE_BATCH_SIZE
from FlagEmbedding import BGEM3FlagModel


class BGEEmbedder:

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            print(f"Loading BGE-M3 model on {BGE_DEVICE}")
            self._model = BGEM3FlagModel(
                BGE_MODEL_NAME,
                use_fp16=True,
                device=BGE_DEVICE,
            )

    def encode_both(self, texts: List[str]) -> dict:
        self._load()
        texts = [t if t.strip() else " " for t in texts]
        output = self._model.encode(
            texts,
            batch_size=BGE_BATCH_SIZE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense":  output["dense_vecs"].astype(np.float32),
            "sparse": output["lexical_weights"],
        }

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.encode_both(texts)["dense"]

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        return self.encode_both(texts)["sparse"]

    def embed_one(self, text: str, is_query: bool = False) -> np.ndarray:
        if is_query:
            text = f"Represent this sentence for searching relevant passages: {text}"
        return self.encode_both([text])["dense"][0]

    def embed_one_sparse(self, text: str, is_query: bool = False) -> dict:
        return {}

    def encode_one_both(self, text: str, is_query: bool = False) -> dict:
        if is_query:
            text = f"Represent this sentence for searching relevant passages: {text}"
        output = self.encode_both([text])
        sparse_list = output["sparse"]
        return {
            "dense":  output["dense"][0],
            "sparse": sparse_list[0] if sparse_list else {},
        }

    @staticmethod
    def build_metadata_text(record: dict) -> str:
        parts = []

        if record.get("title"):
            parts.append(f"Title: {record['title']}")

        genres = record.get("genre") or []
        if genres:
            parts.append(f"Format: {', '.join(genres)}")

        topics = record.get("topics") or []
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")

        geography = record.get("geography") or []
        if geography:
            parts.append(f"Geography: {', '.join(geography)}")

        place = record.get("place") or []
        if place:
            parts.append(f"Place: {', '.join(place)}")

        year = record.get("year") or []
        if year:
            parts.append(f"Year: {', '.join(str(y) for y in year)}")

        collection = record.get("collection") or ""
        if collection and collection != record.get("title"):
            parts.append(f"Collection: {collection}")

        # HTML already stripped at parse time, just truncate
        abstract = record.get("abstract") or ""
        if abstract:
            parts.append(f"Description: {abstract}")

        return " | ".join(parts)


embedder = BGEEmbedder()