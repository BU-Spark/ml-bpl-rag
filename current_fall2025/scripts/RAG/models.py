#!/usr/bin/env python3
"""
Data models for RAG system.
Contains Pydantic models for query rewriting, filters, and responses.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel


class QueryRewrite(BaseModel):
    """Model for query rewriting with expansion."""
    improved_query: str
    expanded_query: Optional[str] = ""


class CatalogResponse(BaseModel):
    """Simplified response for catalog search - no YES/NO validation needed."""
    summary: str


class MaterialType(str, Enum):
    """Enumeration of material types available in the BPL catalog."""
    STILL_IMAGE = "Still image"
    CARTOGRAPHIC = "Cartographic"
    MANUSCRIPT = "Manuscript"
    MOVING_IMAGE = "Moving image"
    NOTATED_MUSIC = "Notated music"
    ARTIFACT = "Artifact"
    AUDIO = "Audio"


class SearchFilters(BaseModel):
    """Search filters for temporal and material type filtering."""
    year_exact: Optional[int] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    material_types: Optional[List[MaterialType]] = None

