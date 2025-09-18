import os
import time
import re
import requests
import json
import logging

from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def retrieve(query: str, vectorstore: PineconeVectorStore, k: int = 1000) -> Tuple[List[Document], List[float]]:
    start = time.time()
    results = vectorstore.similarity_search_with_score(query, k=k)
    documents = []
    scores = []
    for res, score in results:
        if len(res.page_content) > 4000:
            res.page_content = res.page_content[:4000]
        documents.append(res)
        scores.append(score)
    logging.info(f"Finished Retrieval: {time.time() - start}")
    return documents, scores


def safe_get_json(url: str) -> Optional[Dict]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching from {url}: {str(e)}")
        return None


def extract_text_from_json(json_data: Dict) -> str:
    if not json_data:
        return ""
    text_parts = []
    text_fields = ["title_info_primary_tsi", "abstract_tsi", "subject_geographic_sim", "genre_basic_ssim", "genre_specific_ssim", "date_tsim"]
    for field in text_fields:
        if field in json_data['data']['attributes'] and json_data['data']['attributes'][field]:
            text_parts.append(str(json_data['data']['attributes'][field]))
    return " ".join(text_parts) if text_parts else "No content available"


def get_metadata(document_ids: List[str], pinecone_docs: List[Document]) -> Dict[str, str]:
    metadata_dict = {}
    known_fields = {"title_info_primary_tsi", "abstract_tsi", "subject_geographic_sim", "genre_basic_ssim", "genre_specific_ssim", "date_tsim"}

    pinecone_metadata_by_id = {doc.metadata.get("source"): doc.metadata for doc in pinecone_docs}

    for doc_id in document_ids:
        metadata = pinecone_metadata_by_id.get(doc_id, {})
        if all(field in metadata and metadata[field] for field in known_fields):
            text = " ".join(str(metadata[field]) for field in known_fields if metadata.get(field))
            metadata_dict[doc_id] = text
        else:
            url = f"https://www.digitalcommonwealth.org/search/{doc_id}.json"
            json_data = safe_get_json(url)
            metadata_dict[doc_id] = extract_text_from_json(json_data) if json_data else ""

    return metadata_dict


def rephrase_and_expand_query(query: str, llm: Any) -> str:
    prompt_template = PromptTemplate.from_template(
        """
        You are a professional librarian skilled at historical research.
        Rewrite and expand the query to match metadata tags. Include related terms (synonyms, historical names, places, events).
        
        <IMPROVED_QUERY>your improved query here</IMPROVED_QUERY>
        <EXPANDED_QUERY>your expanded query here</EXPANDED_QUERY>

        Original Query: {query}
        """
    )
    prompt = prompt_template.invoke({"query": query})
    response = llm.invoke(prompt)
    improved_match = re.search(r"<IMPROVED_QUERY>(.*?)</IMPROVED_QUERY>", response.content, re.DOTALL)
    expanded_match = re.search(r"<EXPANDED_QUERY>(.*?)</EXPANDED_QUERY>", response.content, re.DOTALL)
    improved_query = improved_match.group(1).strip() if improved_match else query
    expanded_query = expanded_match.group(1).strip() if expanded_match else ""
    return f"{improved_query} {expanded_query}".strip()


def extract_years_from_query(query: str) -> List[str]:
    return re.findall(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2}|22\d{2}|23\d{2})\b", query)


weights = {
    "title_info_primary_tsi": 1.5,
    "name_role_tsim": 1.4,
    "date_tsim": 1.3,
    "abstract_tsi": 1.0,
    "note_tsim": 0.8,
    "subject_geographic_sim": 0.5,
    "genre_basic_ssim": 0.5,
    "genre_specific_ssim": 0.5,
}


def rerank(documents: List[Document], query: str) -> List[Document]:
    if not documents:
        return []

    query_years = extract_years_from_query(query)
    grouped = defaultdict(list)
    for doc in documents:
        source_id = doc.metadata.get("source")
        if source_id:
            grouped[source_id].append(doc)

    full_docs = []
    for source_id, chunks in grouped.items():
        combined_text = " ".join(chunk.page_content for chunk in chunks if chunk.page_content)
        metadata = chunks[0].metadata if chunks else {}
        full_docs.append(Document(page_content=combined_text.strip(), metadata={**metadata, "source": source_id}))

    if not full_docs:
        return []

    bm25 = BM25Retriever.from_documents(full_docs, k=len(full_docs))
    bm25_ranked_docs = bm25.invoke(query)

    ranked_docs = []
    for doc in bm25_ranked_docs:
        bm25_score = 1.0
        metadata_multiplier = 1.0

        for field, weight in weights.items():
            if field in doc.metadata and doc.metadata[field]:
                metadata_multiplier += weight

        date_field = str(doc.metadata.get("date_tsim", ""))
        for year in query_years:
            if re.search(rf"\b{year}\b", date_field) or re.search(rf"{year[:-2]}\d{{2}}–{year[:-2]}\d{{2}}", date_field):
                metadata_multiplier += 50
                break

        final_score = bm25_score * metadata_multiplier
        ranked_docs.append((doc, final_score))

    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked_docs[:10]]


def parse_xml_and_check(xml_string: str) -> str:
    if not xml_string:
        return "No response generated."
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, xml_string, re.DOTALL)
    parsed_response = dict(matches)
    if parsed_response.get('VALID') == 'NO':
        return "Sorry, I was unable to find any documents for your query.\n\n Here are some documents I found that might be relevant."
    return parsed_response.get('RESPONSE', "No response found in the output")


def RAG(llm: Any, query: str, vectorstore: PineconeVectorStore, top: int = 10, k: int = 100) -> Tuple[str, List[Document]]:
    start = time.time()
    try:
        query = rephrase_and_expand_query(query, llm)
        logging.info(f"Rephrased Query for Retrieval: {query}")
        retrieved, _ = retrieve(query=query, vectorstore=vectorstore, k=k)

        if not retrieved:
            return "No documents found for your query.", []

        reranked = rerank(documents=retrieved, query=query)
        logging.info(f"RERANKED LENGTH: {len(reranked)}")
        if not reranked:
            return "Unable to process the retrieved documents.", []

        context = "\n\n".join(doc.page_content for doc in reranked[:top] if doc.page_content)
        if not context.strip():
            return "No relevant content found in the documents.", []

        answer_template = PromptTemplate.from_template(
            """Pretend you are a professional librarian. Please Summarize The Following Context as though you had retrieved it for a patron:
            Some of the retrieved results may include image descriptions, captions, or references to photos, rather than the images themselves. 
            Assume that content describing or captioning an image, or mentioning a place/person clearly, is valid and relevant — even if the actual image isn't embedded.
            Context:{context}
            Make sure to answer in the following format
            First, reason about the answer between <REASONING></REASONING> headers,
            based on the context determine if there is sufficient material for answering the exact question,
            return either <VALID>YES</VALID> or <VALID>NO</VALID>
            then return a response between <RESPONSE></RESPONSE> headers:
            <QUERY>{query}</QUERY>"""
        )

        ans_prompt = answer_template.invoke({"context": context, "query": query})
        response = llm.invoke(ans_prompt)

        logging.debug(f"RAW LLM RESPONSE:\n{response.content}")
        parsed = parse_xml_and_check(response.content)
        logging.debug(f"PARSED FINAL RESPONSE: {parsed}")
        logging.info(f"RAG Finished: {time.time()-start}\n---\n")
        return parsed, reranked

    except Exception as e:
        logging.error(f"Error in RAG function: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}", []
