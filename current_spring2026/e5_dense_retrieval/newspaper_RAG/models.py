#!/usr/bin/env python3
"""
Data models for the historical newspaper RAG pipeline.
Based on: Retrieval Augmented Generation for Historical Newspapers (ACM 2024)
"""

from typing import Optional, List
from pydantic import BaseModel


class TitleSummaryEntry(BaseModel):
    """Metadata entry stored in the Title/Summary index."""
    doc_id: str
    record_id: str
    source_url: str
    newspaper: str
    issue_date: str
    date_iso: List[str] = []
    title: str
    topics: List[str] = []
    geography: List[str] = []
    page_count: int = 0


class ArticleResult(BaseModel):
    """
    A retrieved article returned by the two-stage retrieval.
    Aggregated from individual chunks; full_text holds the reconstructed article.
    """
    doc_id: str
    record_id: str
    source_url: str
    newspaper: str
    issue_date: str
    retrieval_score: float
    full_text: str           # complete article text (capped for LLM)
    rerank_text: str         # shorter excerpt used for Cohere reranking
    topics: List[str] = []
    geography: List[str] = []
    # Populated after reranking
    cohere_score: float = 0.0
    ner_score: float = 0.0
    combined_score: float = 0.0


class NewsRAGResponse(BaseModel):
    """LLM-generated answer from newspaper context."""
    answer: str
