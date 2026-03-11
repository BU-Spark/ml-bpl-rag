#!/usr/bin/env python3
"""
Hybrid reranking for the historical newspaper RAG pipeline.

Implements the reranking strategy described in:
  "Retrieval Augmented Generation for Historical Newspapers" (ACM 2024)

Two scores are combined:
  1. Cohere cross-encoder score  — semantic relevance via a Cohere rerank model.
  2. NER-based TF-IDF score      — named entities are extracted from query and
                                   document, embedded with TF-IDF, and cosine
                                   similarity is computed. This handles OCR noise
                                   by anchoring on recognisable proper nouns.

Final score = cohere_weight * norm(cohere_score)
            + ner_weight   * norm(ner_tfidf_score)

Both component scores are min-max normalised before combining.
"""

import logging
from typing import List, Optional

import numpy as np
import cohere
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ArticleResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NER helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ner_string(text: str, nlp) -> str:
    """
    Run spaCy NER on `text` and return a space-joined string of entity texts.
    Text is truncated to 10 000 chars for performance.
    """
    doc = nlp(text[:10_000])
    return " ".join(ent.text for ent in doc.ents)


def _ner_tfidf_scores(query: str, doc_texts: List[str], nlp) -> List[float]:
    """
    Compute NER-based TF-IDF cosine similarities between the query and each doc.

    Steps:
      1. Extract named entities from query and each document.
      2. Fit TF-IDF on [query_ners] + [doc_ners].
      3. Return cosine similarity of query row vs every doc row.

    Returns a list of floats (one per document), 0.0 on failure.
    """
    query_ner = _extract_ner_string(query, nlp)
    doc_ners  = [_extract_ner_string(t, nlp) for t in doc_texts]

    if not query_ner.strip():
        logger.debug("No named entities found in query; NER scores set to 0.")
        return [0.0] * len(doc_texts)

    corpus = [query_ner] + doc_ners
    try:
        vec = TfidfVectorizer(min_df=1, sublinear_tf=True)
        tfidf_matrix = vec.fit_transform(corpus)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return sims.tolist()
    except Exception as exc:
        logger.warning(f"TF-IDF computation failed: {exc}")
        return [0.0] * len(doc_texts)


# ─────────────────────────────────────────────────────────────────────────────
# Cohere reranking
# ─────────────────────────────────────────────────────────────────────────────

def _cohere_scores(
    query: str,
    doc_texts: List[str],
    cohere_client: cohere.ClientV2,
    model: str = "rerank-english-v3.0",
) -> List[float]:
    """
    Call the Cohere Rerank API and return relevance scores in original doc order.

    On API failure, returns all-zeros so the pipeline can still continue with
    NER TF-IDF alone.
    """
    if not doc_texts:
        return []
    try:
        response = cohere_client.rerank(
            query=query,
            documents=doc_texts,
            model=model,
            top_n=len(doc_texts),
            return_documents=False,
        )
        scores = [0.0] * len(doc_texts)
        for result in response.results:
            scores[result.index] = result.relevance_score
        return scores
    except Exception as exc:
        logger.error(f"Cohere rerank API error: {exc}")
        return [0.0] * len(doc_texts)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _minmax_normalize(values: List[float]) -> List[float]:
    """Min-max normalise to [0, 1]. Returns 0.5 for constant inputs."""
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [0.5] * len(values)
    return ((arr - mn) / (mx - mn)).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Public reranking function
# ─────────────────────────────────────────────────────────────────────────────

def rerank_hybrid(
    query: str,
    articles: List[ArticleResult],
    cohere_client: cohere.ClientV2,
    nlp,
    top_k: int = 5,
    cohere_weight: float = 0.7,
    ner_weight: float = 0.3,
) -> List[ArticleResult]:
    """
    Rerank retrieved articles using a weighted hybrid of Cohere and NER TF-IDF.

    Args:
        query:          User query string.
        articles:       ArticleResult objects from the retrieval stage.
        cohere_client:  Authenticated Cohere V2 client.
        nlp:            Loaded spaCy language model (e.g. en_core_web_sm).
        top_k:          Number of top articles to return.
        cohere_weight:  Weight for Cohere cross-encoder score (0–1).
        ner_weight:     Weight for NER TF-IDF score (0–1).

    Returns:
        Top-k ArticleResult objects with cohere_score, ner_score,
        and combined_score fields populated, sorted descending by combined_score.
    """
    if not articles:
        return []

    # Text used for scoring: first ~1500 chars of each article (rerank_text)
    doc_texts = [a.rerank_text for a in articles]

    # ── Compute component scores ──────────────────────────────────────────────
    logger.info(f"Cohere reranking {len(articles)} articles...")
    cohere_raw = _cohere_scores(query, doc_texts, cohere_client)

    logger.info("Computing NER TF-IDF scores...")
    ner_raw    = _ner_tfidf_scores(query, doc_texts, nlp)

    # ── Normalise ─────────────────────────────────────────────────────────────
    cohere_norm = _minmax_normalize(cohere_raw)
    ner_norm    = _minmax_normalize(ner_raw)

    # ── Weighted combination and sort ─────────────────────────────────────────
    ranked: List[ArticleResult] = []
    for i, article in enumerate(articles):
        combined = cohere_weight * cohere_norm[i] + ner_weight * ner_norm[i]
        article.cohere_score   = cohere_raw[i]
        article.ner_score      = ner_raw[i]
        article.combined_score = combined
        ranked.append(article)

    ranked.sort(key=lambda a: a.combined_score, reverse=True)
    logger.info(f"Reranking complete — returning top {min(top_k, len(ranked))} articles.")
    return ranked[:top_k]
