from __future__ import annotations
from typing import List
from transformers import AutoTokenizer
from config import CHUNK_SIZE, CHUNK_OVERLAP

class Chunker:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc           = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        tokens = self.enc.encode(text, add_special_tokens=False)

        if len(tokens) == 0:
            return []

        chunks = []
        start  = 0
        step   = self.chunk_size - self.chunk_overlap

        while start < len(tokens):
            end        = min(start + self.chunk_size, len(tokens))
            chunk_ids  = tokens[start:end]
            chunk_text = self.enc.decode(chunk_ids, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end == len(tokens):
                break
            start += step

        return chunks

    def chunk_record(self, record: dict) -> List[dict]:
        raw_text = record.get("clean_text") or record.get("raw_text") or ""
        texts    = self.chunk(raw_text)
        return [
            {
                "ark_id":      record["ark_id"],
                "chunk_index": i,
                "chunk_text":  text,
            }
            for i, text in enumerate(texts)
        ]


# Module-level singleton
chunker = Chunker()