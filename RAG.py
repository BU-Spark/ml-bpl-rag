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
from typing import Dict, Any, Optional, List, Tuple
import json
import logging

def retrieve(index_name: str, query: str, embeddings, k: int = 1000) -> Tuple[List[Document], List[float]]:        
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )
    documents = []
    scores = []
    for res, score in results:
        documents.append(res)
        scores.append(score)
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
    text_fields = ["title_info_primary_tsi","abstract_tsi","subject_geographic_sim","genre_specific_ssim"]
    for field in text_fields:
        if field in json_data['data']['attributes'] and json_data['data']['attributes'][field]:
            # print(json_data[field])
            text_parts.append(str(json_data['data']['attributes'][field]))
    
    return " ".join(text_parts) if text_parts else "No content available"

def rerank(documents: List[Document], query: str) -> List[Document]:
    """Rerank documents using BM25, with proper error handling."""
    if not documents:
        return []
    
    full_docs = []
    for doc in documents:
        if not doc.metadata.get('source'):
            continue
            
        url = f"https://www.digitalcommonwealth.org/search/{doc.metadata['source']}"
        json_data = safe_get_json(f"{url}.json")
        
        if json_data:
            text_content = extract_text_from_json(json_data)
            if text_content:  # Only add documents with actual content
                full_docs.append(Document(page_content=text_content, metadata={"source":doc.metadata['source'],"field":doc.metadata['field'],"URL":url}))
    
    # If no valid documents were processed, return empty list
    if not full_docs:
        return []
    
    # Create BM25 retriever with the processed documents
    reranker = BM25Retriever.from_documents(full_docs, k=min(10, len(full_docs)))
    reranked_docs = reranker.invoke(query)
    return reranked_docs

def parse_xml_and_check(xml_string: str) -> str:
    """Parse XML-style tags and handle validation."""
    if not xml_string:
        return "No response generated."
    
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, xml_string, re.DOTALL)
    parsed_response = dict(matches)
    
    if parsed_response.get('VALID') == 'NO':
        return "Sorry, I was unable to find any documents relevant to your query."
    
    return parsed_response.get('RESPONSE', "No response found in the output")

def RAG(llm: Any, query: str, index_name: str, embeddings: Any, top: int = 10, k: int = 100) -> Tuple[str, List[Document]]:
    """Main RAG function with improved error handling and validation."""
    try:
        # Retrieve initial documents
        retrieved, _ = retrieve(index_name=index_name, query=query, embeddings=embeddings, k=k)
        if not retrieved:
            return "No documents found for your query.", []
        
        # Rerank documents
        reranked = rerank(documents=retrieved, query=query)
        if not reranked:
            return "Unable to process the retrieved documents.", []
        
        # Prepare context from reranked documents
        context = "\n\n".join(doc.page_content for doc in reranked[:top] if doc.page_content)
        if not context.strip():
            return "No relevant content found in the documents.", []
        
        # Prepare prompt
        prompt_template = PromptTemplate.from_template(
            """Pretend you are a professional librarian. Please Summarize The Following Context as though you had retrieved it for a patron:
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
            <RESPONSE>Pineapples are not a good fuel for cars, however with further researach they migth be</RESPONSE> 
            </EXAMPLE>
            Now it's your turn 
            <QUERY>
            {query}
            </QUERY>"""
        )
        
        # Generate response
        prompt = prompt_template.invoke({"context": context, "query": query})
        print(prompt)
        response = llm.invoke(prompt)
        
        # Parse and return response
        parsed = parse_xml_and_check(response.content)
        return parsed, reranked
        
    except Exception as e:
        logging.error(f"Error in RAG function: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}", []