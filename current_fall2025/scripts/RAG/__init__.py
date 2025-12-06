#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) module for Boston Public Library catalog search.

This package provides a modular RAG system for searching and discovering materials
in the Boston Public Library's Digital Commonwealth collection.

Main exports:
- RAG: Main pipeline function
- Models: QueryRewrite, CatalogResponse, MaterialType, SearchFilters
"""

from .pipeline import RAG
from .models import QueryRewrite, CatalogResponse, MaterialType, SearchFilters
from .query_enhancement import rephrase_and_expand_query
from .filters import extract_filters_with_llm, build_sql_filter
from .retrieval import retrieve_from_pg
from .reranking import rerank
from .response import generate_catalog_summary, parse_json_response

__all__ = [
    # Main pipeline
    "RAG",
    
    # Models
    "QueryRewrite",
    "CatalogResponse",
    "MaterialType",
    "SearchFilters",
    
    # Components (for testing/debugging)
    "rephrase_and_expand_query",
    "extract_filters_with_llm",
    "build_sql_filter",
    "retrieve_from_pg",
    "rerank",
    "generate_catalog_summary",
    "parse_json_response",
]

