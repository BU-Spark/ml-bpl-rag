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
from RAG_cloudsql_vector import CloudSQLVectorStore
import json
import logging

USE_DB_FOR_METADATA = os.getenv("USE_DB_FOR_METADATA", "True").lower() == "true"
# Database connection details
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")  # Default to local proxy
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = "bpl_metadata_new"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")

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
    
    # Use LLM to rewrite and expand a query for better alignment with archive metadata.
    prompt_template = PromptTemplate.from_template(
        """
        You are a professional librarian skilled at historical research.
        Your task is to improve and expand the following search query to better match metadata in a historical archive.

        - First, rewrite the query to improve clarity and fit how librarians would search.
        - Second, expand the query by adding related terms (synonyms, related concepts, historical terminology, etc.).

        Return your output strictly in this format (no extra explanation):
        <IMPROVED_QUERY>your improved query here</IMPROVED_QUERY>
        <EXPANDED_QUERY>your expanded query here</EXPANDED_QUERY>

        Original Query: {query}
        """
    )

    prompt = prompt_template.invoke({"query": query})
    response = llm.invoke(prompt)

    # Extract just the improved and expanded queries
    improved_match = re.search(r"<IMPROVED_QUERY>(.*?)</IMPROVED_QUERY>", response.content, re.DOTALL)
    expanded_match = re.search(r"<EXPANDED_QUERY>(.*?)</EXPANDED_QUERY>", response.content, re.DOTALL)

    improved_query = improved_match.group(1).strip() if improved_match else query
    expanded_query = expanded_match.group(1).strip() if expanded_match else ""

    final_query = f"{improved_query} {expanded_query}".strip()

    logging.info(f"Original Query: {query}")
    logging.info(f"Improved Query: {improved_query}")
    logging.info(f"Expanded Query: {expanded_query}")
    logging.info(f"Final Query for Retrieval: {final_query}")

    return final_query



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



"""
def rerank(documents: List[Document], query: str) -> List[Document]:
    \"\"\"Ingest more metadata. Rerank documents using BM25\"\"\"
    start = time.time()
    if not documents:
        return []
    
    full_docs = []
    seen_sources = set()  
    meta_start = time.time()
    for doc in documents:
        source = doc.metadata.get('source')
        if not source or source in seen_sources:
            continue  # Skip duplicate sources
        seen_sources.add(source)

        url = f"https://www.digitalcommonwealth.org/search/{source}"
        json_data = safe_get_json(f"{url}.json")
        
        if json_data:
            text_content = extract_text_from_json(json_data)
            if text_content:  # Only add documents with actual content
                full_docs.append(Document(page_content=text_content, metadata={"source": source, "field": doc.metadata.get("field", ""), "URL": url}))

    logging.info(f"Took {time.time()-meta_start} seconds to retrieve all metadata")
    if not full_docs:
        return []

    # Create BM25 retriever with the processed documents
    bm25 = BM25Retriever.from_documents(full_docs, k=min(10, len(full_docs)))
    bm25_ranked_docs = bm25.invoke(query)

    ranked_docs = []
    for doc in bm25_ranked_docs:
        bm25_score = 1.0 

        # Compute metadata multiplier
        metadata_multiplier = 1.0 
        for field, weight in weights.items():
            if field in doc.metadata and doc.metadata[field]:
                metadata_multiplier += weight  

        # Compute final score: BM25 weight * Metadata multiplier
        final_score = bm25_score * metadata_multiplier
        ranked_docs.append((doc, final_score))

    # Sort by final score 
    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    logging.info(f"Finished reranking: {time.time()-start}")
    return [doc for doc, _ in ranked_docs]
"""

'''
def rerank(documents: List[Document], query: str) -> List[Document]:
    """Retrieve metadata from the database and rerank using BM25"""
    start = time.time()
    if not documents:
        return []

    document_ids = [doc.metadata.get('source') for doc in documents if doc.metadata.get('source')]
    
    # Fetch metadata from PostgreSQL
    metadata_dict = get_metadata_from_db(document_ids)

    full_docs = []
    for doc in documents:
        doc_id = doc.metadata.get('source')
        metadata = metadata_dict.get(doc_id, {})

        if metadata:
            text_content = " ".join([
                metadata.get("title", ""),
                metadata.get("abstract", ""),
                " ".join(metadata.get("subjects", [])),
                metadata.get("institution", "")
            ]).strip()


            if text_content:
                full_docs.append(Document(page_content=text_content, metadata={
                    "source": doc_id, 
                    "URL": metadata.get("metadata_url", ""), 
                    "image_url": metadata.get("image_url", "")
                }))

    logging.info(f"Took {time.time()-start} seconds to retrieve all metadata from PostgreSQL")

    if not full_docs:
        return []

    # Rerank using BM25
    bm25 = BM25Retriever.from_documents(full_docs, k=min(10, len(full_docs)))
    bm25_ranked_docs = bm25.invoke(query)

    ranked_docs = []
    for doc in bm25_ranked_docs:
        bm25_score = 1.0 

        # Compute metadata multiplier
        metadata_multiplier = 1.0 
        for field, weight in weights.items():
            if field in doc.metadata and doc.metadata[field]:
                metadata_multiplier += weight  

        # Compute final score: BM25 weight * Metadata multiplier
        final_score = bm25_score * metadata_multiplier
        ranked_docs.append((doc, final_score))

    # Sort by final score 
    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    logging.info(f"Finished reranking: {time.time()-start}")
    return [doc for doc, _ in ranked_docs]
'''

def rerank(documents: List[Document], query: str) -> List[Document]:
    """Rerank using BM25 and enhance scores using document metadata."""
    start = time.time()

    if not documents:
        return []

    # Group document chunks by source_id
    grouped = defaultdict(list)
    for doc in documents:
        source_id = doc.metadata.get("source")
        if source_id:
            grouped[source_id].append(doc)

    full_docs = []
    for source_id, chunks in grouped.items():
        combined_text = " ".join([chunk.page_content for chunk in chunks if chunk.page_content])
        representative_metadata = chunks[0].metadata or {}

        #logging.debug(f"Metadata for doc {source_id}: {representative_metadata}")

        if combined_text.strip():
            full_docs.append(Document(
                page_content=combined_text.strip(),
                metadata={
                    "source": source_id,
                    "URL": representative_metadata.get("metadata_url", ""),
                    "image_url": representative_metadata.get("image_url", ""),
                    **representative_metadata  # preserve all original fields
                }
            ))

    logging.info(f"Built {len(full_docs)} documents for reranking in {time.time() - start:.2f} seconds.")

    if not full_docs:
        return []

    # BM25 reranking
    bm25 = BM25Retriever.from_documents(full_docs, k=min(10, len(full_docs)))
    bm25_ranked_docs = bm25.invoke(query)

    # Score enhancement using metadata weights
    ranked_docs = []
    for doc in bm25_ranked_docs:
        bm25_score = 1.0  # BM25 returns sorted, so base score is 1
        metadata_multiplier = 1.0
        for field, weight in weights.items():
            if field in doc.metadata and doc.metadata[field]:
                metadata_multiplier += weight
        final_score = bm25_score * metadata_multiplier
        ranked_docs.append((doc, final_score))

    # Sort by enhanced score
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    logging.info(f"Finished reranking in {time.time() - start:.2f} seconds")

    return [doc for doc, _ in ranked_docs]



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

        # Retrieve initial documents using rephrased query -- not working as intended currently, maybe would be better for data with more words.
        # query_template = PromptTemplate.from_template(
        #     """
        #     Your job is to think about a query and then generate a statement that only includes information from the query that would answer the query.
        #     You will be provided with a query in <QUERY></QUERY> tags. 
        #     Then you will think about what kind of information the query is looking for between <REASONING></REASONING> tags.
        #     Then, based on the reasoning, you will generate a sample response to the query that only includes information from the query between <STATEMENT></STATEMENT> tags.
        #     Afterwards, you will determine and reason about whether or not the statement you generated only includes information from the original query and would answer the query between <DETERMINATION></DETERMINATION> tags.
        #     Finally, you will return a YES, or NO response between <VALID></VALID> tags based on whether or not you determined the statment to be valid.
        #     Let me provide you with an exmaple:

        #     <QUERY>I would really like to learn more about Bermudan geography<QUERY>

        #     <REASONING>This query is interested in geograph as it relates to Bermuda. Some things they might be interested in are Bermudan climate, towns, cities, and geography</REASONING>

        #     <STATEMENT>Bermuda's Climate is [blank]. Some of Bermuda's cities and towns are [blank]. Other points of interested about Bermuda's geography are [blank].</STATEMENT>

        #     <DETERMINATION>The query originally only mentions bermuda and geography. The answers do not provide any false information, instead replacing meaningful responses with a placeholder [blank]. If it had hallucinated, it would not be valid. Because the statements do not hallucinate anything, this is a valid statement.</DETERMINATION>
            
        #     <VALID>YES</VALID>

        #     Now it's your turn! Remember not to hallucinate:

        #     <QUERY>{query}</QUERY>
        #     """
        # )
        # query_prompt = query_template.invoke({"query":query})
        # query_response = llm.invoke(query_prompt)
        # new_query = parse_xml_and_query(query=query,xml_string=query_response.content)
        
        #logging.info(f"\n---\nQUERY: {query}")

        #new query rephrasing
        #query = rephrase_and_expand_query(query, llm)
        #logging.info(f"\n---\nRephrased QUERY: {query}")

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