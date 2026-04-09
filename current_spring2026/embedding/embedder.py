from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config import BGE_MODEL_NAME, BGE_DEVICE, BGE_BATCH_SIZE


class BGEEmbedder:

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            print(f"Loading BGE-M3 model on {BGE_DEVICE}")
            self._model = SentenceTransformer(BGE_MODEL_NAME, device=BGE_DEVICE)

    def encode_both(self, texts: List[str]) -> dict:
        self._load()
        texts = [t if t.strip() else " " for t in texts]
        embeddings = self._model.encode(
            texts,
            batch_size=BGE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > BGE_BATCH_SIZE,
        )
        # sentence-transformers doesn't support sparse natively
        # return empty sparse dicts as placeholders
        return {
            "dense":  embeddings.astype(np.float32),
            "sparse": [{} for _ in texts],
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
        return {
            "dense":  output["dense"][0],
            "sparse": output["sparse"][0],
        }

    @staticmethod
    def build_metadata_text(record: dict) -> str:
        parts = []
        if record.get("title"):
            parts.append(f"Title: {record['title']}")
        if record.get("newspaper"):
            parts.append(f"Newspaper: {record['newspaper']}")
        if record.get("institution"):
            parts.append(f"Institution: {record['institution']}")
        if record.get("issue_date"):
            parts.append(f"Date: {record['issue_date']}")
        topics = record.get("topics") or []
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")
        geography = record.get("geography") or []
        if geography:
            parts.append(f"Geography: {', '.join(geography)}")
        place = record.get("place") or []
        if place:
            parts.append(f"Place: {', '.join(place)}")
        genres = record.get("genre") or []
        if genres:
            parts.append(f"Formats: {', '.join(genres)}")
        # Include abstract — most information-rich field for collection records
        abstract = record.get("abstract") or record.get("abstract_tsi") or ""
        if abstract:
            import re
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
            parts.append(f"Description: {abstract}")
        return " | ".join(parts)


embedder = BGEEmbedder()