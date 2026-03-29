#!/usr/bin/env python3
"""
Data models for the BGE-M3 hybrid RAG system.
"""

from typing import Optional, List, Dict
from enum import Enum
from pydantic import BaseModel


class QueryRewrite(BaseModel):
    improved_query: str
    expanded_query: Optional[str] = ""


class CatalogResponse(BaseModel):
    summary: str


class MaterialType(str, Enum):
    STILL_IMAGE = "Still image"
    CARTOGRAPHIC = "Cartographic"
    MANUSCRIPT = "Manuscript"
    MOVING_IMAGE = "Moving image"
    NOTATED_MUSIC = "Notated music"
    ARTIFACT = "Artifact"
    AUDIO = "Audio"


class SearchFilters(BaseModel):
    year_exact: Optional[int] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    material_types: Optional[List[MaterialType]] = None


class EmbeddingOutput(BaseModel):
    """Output from BGE-M3 encoding a single text."""
    dense: List[float]                   # 1024-dim dense vector
    sparse: Dict[str, float]             # {token_id_str: weight} sparse vector

    class Config:
        arbitrary_types_allowed = True
