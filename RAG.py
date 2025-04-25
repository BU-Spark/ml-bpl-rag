import getpass
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import requests
import psycopg2
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import json
import logging

def retrieve(query: str,vectorstore:PineconeVectorStore, k: int = 1000) -> Tuple[List[Document], List[float]]:    
    start = time.time()
    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
    )
    documents = []
    scores = []
    for res, score in results:
        # check to make sure response isnt too long for context window of 4o-mini
        if len(res.page_content) > 4000:
            res.page_content = res.page_content[:4000]
        documents.append(res)
        scores.append(score)
    logging.info(f"Finished Retrieval: {time.time() - start}")
    return documents, scores

def safe_get_json(url: str) -> Optional[Dict]:
    """Safely fetch and parse JSON from a URL."""
    print("Fetching JSON")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching from {url}: {str(e)}")
        return None

def extract_text_from_json(json_data: Dict) -> str:
    """Extract text content from JSON response."""
    if not json_data:
        return ""
    
    text_parts = []
    
    # Handle direct text fields
    text_fields = ["title_info_primary_tsi","abstract_tsi","subject_geographic_sim","genre_basic_ssim","genre_specific_ssim","date_tsim"]
    for field in text_fields:
        if field in json_data['data']['attributes'] and json_data['data']['attributes'][field]:
            # print(json_data[field])
            text_parts.append(str(json_data['data']['attributes'][field]))
    
    return " ".join(text_parts) if text_parts else "No content available"

def rephrase_and_expand_query(query: str, llm: Any) -> str:
    """Use LLM to rewrite and expand a query for better alignment with archive metadata."""
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
    """Extract 4-digit years from query for boosting."""
    return re.findall(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2}|22\d{2}|23\d{2})\b", query)

weights = {
    "title_info_primary_tsi": 1.5,  # Titles should be prioritized
    "name_role_tsim": 1.4,  # Author/role should be highly weighted
    "date_tsim": 1.3,  # Date should be considered
    "abstract_tsi": 1.0,  # Abstracts are important but less so
    "note_tsim": 0.8,  
    "subject_geographic_sim": 0.5,  
    "genre_basic_ssim": 0.5, 
    "genre_specific_ssim": 0.5,  
}

def get_metadata(document_ids: List[str]) -> Dict[str, Dict]:
    """ Fetch metadata from either PostgreSQL or the Commonwealth API, based on config """
    
    if USE_DB_FOR_METADATA:
        return get_metadata_from_db(document_ids)
    else:
        return get_metadata_from_api(document_ids)

def get_metadata_from_db(document_ids: List[str]) -> Dict[str, Dict]:
    """ Fetch metadata from PostgreSQL """
    conn = psycopg2.connect(
        host="127.0.0.1",
        port="5435",
        dbname="bpl_metadata",
        user="postgres",
        password="MNOF.MzLDjcgzAXu"  # Replace with real one or load with dotenv
    )
    cur = conn.cursor()

    sql_query = """
    SELECT id, title, abstract, subjects, institution, metadata_url, image_url 
    FROM metadata 
    WHERE id = ANY(%s);
    """
    cur.execute(sql_query, (document_ids,))
    results = cur.fetchall()
    cur.close()
    conn.close()

    # Convert results to a dictionary
    return {
        row[0]: {
            "title": row[1],
            "abstract": row[2],
            "subjects": row[3],
            "institution": row[4],
            "metadata_url": row[5],
            "image_url": row[6],
        }
        for row in results
    }

def get_metadata_from_api(document_ids: List[str]) -> Dict[str, Dict]:
    """ Fetch metadata from the Commonwealth API """
    metadata_dict = {}
    for doc_id in document_ids:
        url = f"https://www.digitalcommonwealth.org/search/{doc_id}.json"
        json_data = safe_get_json(url)
        if json_data:
            metadata_dict[doc_id] = extract_text_from_json(json_data)
    return metadata_dict

def rerank(documents: List[Document], query: str) -> List[Document]:
    """Rerank documents using BM25 and metadata, boost if year matches."""
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
        full_docs.append(Document(
            page_content=combined_text.strip(),
            metadata={**metadata, "source": source_id}
        ))

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

def parse_xml_and_query(query:str,xml_string:str) -> str:
    """parse xml and return rephrased query"""
    if not xml_string:
        return "No response generated."
    
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, xml_string, re.DOTALL)
    parsed_response = dict(matches)
    if parsed_response.get('VALID') == 'NO':
        return query
    return parsed_response.get('STATEMENT', query)


def parse_xml_and_check(xml_string: str) -> str:
    """Parse XML-style tags and handle validation."""
    if not xml_string:
        return "No response generated."
    
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, xml_string, re.DOTALL)
    parsed_response = dict(matches)
    
    if parsed_response.get('VALID') == 'NO':
        return "Sorry, I was unable to find any documents for your query.\n\n Here are some documents I found that might be relevant."
    
    return parsed_response.get('RESPONSE', "No response found in the output")

def RAG(llm: Any, query: str,vectorstore:PineconeVectorStore, top: int = 10, k: int = 100) -> Tuple[str, List[Document]]:
    """Main RAG function with improved error handling and validation."""
    start = time.time()
    try:

        # Query alignment is commented our, however I have decided to leave it in for potential future use.

      # 🔄 Rephrase and expand the user query for better Pinecone matching
        query = rephrase_and_expand_query(query, llm)
        logging.info(f"Rephrased Query for Retrieval: {query}")

        retrieved, _ = retrieve(query=query, vectorstore=vectorstore, k=k)

        if not retrieved:
            return "No documents found for your query.", []
        
        # Rerank documents
        reranked = rerank(documents=retrieved, query=query)
        logging.info(f"RERANKED LENGTH: {len(reranked)}")
        if not reranked:
            return "Unable to process the retrieved documents.", []
        
        # Prepare context from reranked documents
        context = "\n\n".join(doc.page_content for doc in reranked[:top] if doc.page_content)
        if not context.strip():
            return "No relevant content found in the documents.", []
        # change for the sake of another commit
        # Prepare prompt
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
            Here is an example
            <EXAMPLE>
            <QUERY>Are pineapples a good fuel for cars?</QUERY>
            <CONTEXT>Cars use gasoline for fuel. Some cars use electricity for fuel.Tesla stock has increased by 10 percent over the last quarter.</CONTEXT>
            <REASONING>Based on the context pineapples have not been explored as a fuel for cars. The context discusses gasoline, electricity, and tesla stock, therefore it is not relevant to the query about pineapples for fuel</REASONING>
            <VALID>NO</VALID>
            <RESPONSE>Pineapples are not a good fuel for cars, however with further research they might be</RESPONSE> 
            </EXAMPLE>
            Now it's your turn 
            <QUERY>
            {query}
            </QUERY>"""
        )
        
        # Generate response
        ans_prompt = answer_template.invoke({"context": context, "query": query})
        response = llm.invoke(ans_prompt)
        
        # Parse and return response
        logging.debug(f"RAW LLM RESPONSE:\n{response.content}")
        parsed = parse_xml_and_check(response.content)
        logging.debug(f"PARSED FINAL RESPONSE: {parsed}")
        #logging.info(f"RESPONSE: {parsed}\nRETRIEVED: {reranked}")
        logging.info(f"RAG Finished: {time.time()-start}\n---\n")
        return parsed, reranked
        
    except Exception as e:
        logging.error(f"Error in RAG function: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}", []