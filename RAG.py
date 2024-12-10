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
    start = time.time()    
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
    print(f"Finished Retrieval: {time.time() - start}")
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
    """Ingest more metadata. Rerank documents using BM25"""
    start = time.time()
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
    print(f"Finished reranking: {time.time()-start}")
    return reranked_docs

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
        return "Sorry, I was unable to find any documents relevant to your query."
    
    return parsed_response.get('RESPONSE', "No response found in the output")

def RAG(llm: Any, query: str, index_name: str, embeddings: Any, top: int = 10, k: int = 100) -> Tuple[str, List[Document]]:
    """Main RAG function with improved error handling and validation."""
    start = time.time()
    try:
        # Retrieve initial documents using rephrased query
        query_template = PromptTemplate.from_template(
            """
            Your job is to think about a query and then generate a statement that only includes information from the query that would answer the query.
            You will be provided with a query in <QUERY></QUERY> tags. 
            Then you will think about what kind of information the query is looking for between <REASONING></REASONING> tags.
            Then, based on the reasoning, you will generate a sample response to the query that only includes information from the query between <STATEMENT></STATEMENT> tags.
            Afterwards, you will determine and reason about whether or not the statement you generated only includes information from the original query and would answer the query between <DETERMINATION></DETERMINATION> tags.
            Finally, you will return a YES, or NO response between <VALID></VALID> tags based on whether or not you determined the statment to be valid.
            Let me provide you with an exmaple:

            <QUERY>I would really like to learn more about Bermudan geography<QUERY>

            <REASONING>This query is interested in geograph as it relates to Bermuda. Some things they might be interested in are Bermudan climate, towns, cities, and geography</REASONING>

            <STATEMENT>Bermuda's Climate is [blank]. Some of Bermuda's cities and towns are [blank]. Other points of interested about Bermuda's geography are [blank].</STATEMENT>

            <DETERMINATION>The query originally only mentions bermuda and geography. The answers do not provide any false information, instead replacing meaningful responses with a placeholder [blank]. If it had hallucinated, it would not be valid. Because the statements do not hallucinate anything, this is a valid statement.</DETERMINATION>
            
            <VALID>YES</VALID>

            Now it's your turn! Remember not to hallucinate:

            <QUERY>{query}</QUERY>
            """

        )
        query_prompt = query_template.invoke({"query":query})
        query_response = llm.invoke(query_prompt)
        new_query = parse_xml_and_query(query=query,xml_string=query_response.content)
        print(f"New_Query: {new_query}")

        retrieved, _ = retrieve(index_name=index_name, query=new_query, embeddings=embeddings, k=k)
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
        # change for the sake of another commit
        # Prepare prompt
        answer_template = PromptTemplate.from_template(
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
        ans_prompt = answer_template.invoke({"context": context, "query": query})
        response = llm.invoke(ans_prompt)
        
        # Parse and return response
        parsed = parse_xml_and_check(response.content)
        print(f"RAG Finished: {time.time()-start}")
        return parsed, reranked
        
    except Exception as e:
        logging.error(f"Error in RAG function: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}", []