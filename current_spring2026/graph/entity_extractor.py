"""
graph/entity_extractor.py

Extracts named entities from document chunks using spaCy.
Entity types: PERSON, GPE (places), ORG, EVENT, DATE

Usage:
    from graph.entity_extractor import extractor
    entities = extractor.extract("Mayor Fitzgerald attended the rally in Boston.")
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import spacy


# Entity types we care about
RELEVANT_TYPES = {"PERSON", "GPE", "LOC", "ORG", "EVENT", "DATE", "FAC", "NORP"}

# Map spaCy types to our simplified types
TYPE_MAP = {
    "PERSON": "PERSON",
    "GPE":    "PLACE",     # Geo-political entity (cities, countries)
    "LOC":    "PLACE",     # Non-GPE locations
    "FAC":    "PLACE",     # Facilities (buildings, bridges)
    "ORG":    "ORG",
    "EVENT":  "EVENT",
    "DATE":   "DATE",
    "NORP":   "ORG",       # Nationalities, religious groups, political groups
}


@dataclass
class Entity:
    text:       str   # normalized entity text
    type:       str   # simplified type (PERSON, PLACE, ORG, EVENT, DATE)
    raw_type:   str   # original spaCy label
    count:      int = 1


class EntityExtractor:

    def __init__(self, model: str = "en_core_web_sm"):
        self._nlp = None
        self._model = model

    def _load(self):
        if self._nlp is None:
            print(f"Loading spaCy model: {self._model}")
            try:
                self._nlp = spacy.load(self._model)
            except OSError:
                print(f"Model {self._model} not found. Downloading...")
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", self._model],
                    check=True
                )
                self._nlp = spacy.load(self._model)

    def extract(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        Processes in 100K character windows to avoid memory issues
        while covering the full document.
        """
        self._load()

        if not text or not text.strip():
            return []

        # Process in overlapping windows to cover full document
        WINDOW_SIZE = 100_000
        OVERLAP     = 1_000    # overlap to avoid missing entities at boundaries

        entity_counts: dict[tuple, int] = {}
        entity_raw:    dict[tuple, str] = {}

        start = 0
        while start < len(text):
            end     = min(start + WINDOW_SIZE, len(text))
            window  = text[start:end]
            doc     = self._nlp(window)

            for ent in doc.ents:
                if ent.label_ not in RELEVANT_TYPES:
                    continue
                normalized = ent.text.strip().lower()
                if len(normalized) < 2 or len(normalized) > 100:
                    continue
                if ent.label_ != "DATE" and normalized.replace(" ", "").isdigit():
                    continue

                mapped_type = TYPE_MAP.get(ent.label_, ent.label_)
                key = (normalized, mapped_type)
                entity_counts[key] = entity_counts.get(key, 0) + 1
                entity_raw[key]    = ent.label_

            # Move window forward with overlap
            start += WINDOW_SIZE - OVERLAP
            if end == len(text):
                break

        return [
            Entity(
                text     = text_,
                type     = type_,
                raw_type = entity_raw[(text_, type_)],
                count    = count,
            )
            for (text_, type_), count in sorted(
                entity_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

    def extract_top(self, text: str, n: int = 20) -> List[Entity]:
        """Return only the top N most frequent entities."""
        return self.extract(text)[:n]


# Module-level singleton
extractor = EntityExtractor()
