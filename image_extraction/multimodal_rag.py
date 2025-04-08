import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import re
import requests
from PIL import Image
import io
import argparse
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Local imports
from load_pinecone_images import MultimodalPineconeManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MultimodalRAG:
    """
    Retrieval-Augmented Generation system with multimodal capabilities.
    Can retrieve and use both text and image content for generating responses.
    """
    
    def __init__(self, pinecone_index: str, namespace: str = "bpl"):
        """
        Initialize the multimodal RAG system
        
        :param pinecone_index: Name of the Pinecone index
        :param namespace: Namespace within Pinecone
        """
        self.pinecone_index = pinecone_index
        self.namespace = namespace
        
        # Initialize Pinecone manager
        self.pinecone_manager = MultimodalPineconeManager(
            index_name=pinecone_index,
            namespace=namespace
        )
        
        # Initialize LLM if API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.3
            )
            logger.info("Initialized LLM with OpenAI")
        else:
            self.llm = None
            logger.warning("No OpenAI API key found, LLM functionality will be limited")
        
        logger.info(f"Initialized MultimodalRAG with index: {pinecone_index}")
    
    def retrieve_with_text(self, query: str, k: int = 10) -> Tuple[List[Document], List[Dict]]:
        """
        Retrieve content using a text query
        
        :param query: Text query
        :param k: Number of results to retrieve
        :return: Tuple of (text_documents, image_results)
        """
        logger.info(f"Retrieving with text query: {query}")
        start_time = time.time()
        
        # Query for text documents
        text_results = self.pinecone_manager.query_text(
            query_text=query,
            top_k=k
        )
        
        # Query for images using the same text query
        image_results = self.pinecone_manager.index.query(
            vector=self.pinecone_manager.embed_text(query),
            top_k=k,
            namespace=f"{self.namespace}_image",
            include_metadata=True
        )
        
        # Extract documents from text results
        documents = []
        for match in text_results.get('matches', []):
            documents.append(match['document'])
        
        # Format image results
        formatted_images = []
        for match in image_results.get('matches', []):
            formatted_images.append({
                'id': match.get('id'),
                'score': match.get('score'),
                'metadata': match.get('metadata', {}),
                'url': match.get('metadata', {}).get('url')
            })
        
        logger.info(f"Retrieved {len(documents)} text documents and {len(formatted_images)} images in {time.time() - start_time:.2f}s")
        return documents, formatted_images
    
    def retrieve_with_image(self, 
                          image_source: Union[str, bytes, Image.Image],
                          k: int = 10) -> Tuple[List[Document], List[Dict]]:
        """
        Retrieve content using an image query
        
        :param image_source: Image to query with
        :param k: Number of results to retrieve
        :return: Tuple of (text_documents, image_results)
        """
        logger.info(f"Retrieving with image query")
        start_time = time.time()
        
        # Query using the image
        results = self.pinecone_manager.query_image(
            image_source=image_source,
            top_k=k,
            search_images=True,
            search_texts=True
        )
        
        if 'error' in results:
            logger.error(f"Error in image retrieval: {results['error']}")
            return [], []
        
        # Extract documents and image results
        documents = []
        images = []
        
        for match in results.get('matches', []):
            if match.get('type') == 'text' and 'document' in match:
                documents.append(match['document'])
            elif match.get('type') == 'image':
                images.append({
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'metadata': match.get('metadata', {}),
                    'url': match.get('metadata', {}).get('url')
                })
        
        logger.info(f"Retrieved {len(documents)} text documents and {len(images)} images in {time.time() - start_time:.2f}s")
        return documents, images
    
    def rerank(self, documents: List[Document], query: str, k: int = 10) -> List[Document]:
        """
        Rerank documents using BM25
        
        :param documents: List of documents to rerank
        :param query: Query text
        :param k: Number of results to return
        :return: Reranked documents
        """
        if not documents:
            return []
        
        # Create BM25 retriever and rerank
        reranker = BM25Retriever.from_documents(documents, k=min(k, len(documents)))
        reranked_docs = reranker.invoke(query)
        
        return reranked_docs
    
    def generate_response(self, 
                        query: str, 
                        documents: List[Document], 
                        images: List[Dict] = None) -> str:
        """
        Generate a response using the retrieved content
        
        :param query: User query
        :param documents: Retrieved documents
        :param images: Retrieved images
        :return: Generated response
        """
        if not self.llm:
            return "LLM not initialized. Please provide an OpenAI API key."
        
        if not documents and not images:
            return "I couldn't find any relevant information to answer your query."
        
        # Prepare text context
        text_context = ""
        if documents:
            text_context = "\n\n".join([doc.page_content for doc in documents[:5]])
        
        # Prepare image context
        image_context = ""
        if images:
            image_descriptions = []
            for img in images[:3]:
                metadata = img.get('metadata', {})
                desc = f"Image: {metadata.get('alt', 'No description available')}"
                if 'metadata' in metadata and metadata['metadata']:
                    if isinstance(metadata['metadata'], dict):
                        title = metadata['metadata'].get('title')
                        if title:
                            desc += f"\nTitle: {title}"
                        description = metadata['metadata'].get('description')
                        if description:
                            desc += f"\nDescription: {description}"
                image_descriptions.append(desc)
            
            if image_descriptions:
                image_context = "Related Images:\n" + "\n\n".join(image_descriptions)
        
        # Combine contexts
        combined_context = text_context
        if image_context:
            combined_context += "\n\n" + image_context
        
        # Generate response using template
        template = PromptTemplate.from_template(
            """You are a helpful librarian assistant for the Boston Public Library.
            
            A user has asked the following question:
            {query}
            
            Based on the information provided below, please provide a comprehensive, informative answer.
            If you don't have enough information to answer the question, acknowledge this and suggest what might be relevant.
            
            INFORMATION:
            {context}
            
            Your answer should be well-structured, accurate, and helpful. If there are images mentioned,
            you can reference them in your answer. Use clear language that is accessible to a general audience.
            """
        )
        
        prompt = template.invoke({
            "query": query,
            "context": combined_context
        })
        
        # Generate response
        response = self.llm.invoke(prompt.text)
        return response.content
    
    def process_query(self, 
                     query: str, 
                     image_source: Optional[Union[str, bytes, Image.Image]] = None,
                     k: int = 10) -> Dict:
        """
        Process a query and generate a response
        
        :param query: Text query
        :param image_source: Optional image query
        :param k: Number of results to retrieve
        :return: Dictionary with query results and response
        """
        start_time = time.time()
        
        # Retrieve content based on query type
        if image_source:
            # Multimodal query (both text and image)
            documents, images = self.retrieve_with_image(image_source, k=k)
            
            # If we have an image query but also text, supplement with text results
            if query:
                text_docs, more_images = self.retrieve_with_text(query, k=k)
                documents.extend(text_docs)
                images.extend(more_images)
                
                # Remove duplicates
                unique_docs = {}
                for doc in documents:
                    if doc.page_content not in unique_docs:
                        unique_docs[doc.page_content] = doc
                documents = list(unique_docs.values())
                
                unique_images = {}
                for img in images:
                    if img['id'] not in unique_images:
                        unique_images[img['id']] = img
                images = list(unique_images.values())
        else:
            # Text-only query
            documents, images = self.retrieve_with_text(query, k=k)
        
        # Rerank documents if we have any
        if documents:
            documents = self.rerank(documents, query, k=min(5, len(documents)))
        
        # Generate response
        response = self.generate_response(query, documents, images)
        
        # Build result object
        result = {
            "query": query,
            "processing_time": time.time() - start_time,
            "documents_retrieved": len(documents),
            "images_retrieved": len(images),
            "response": response
        }
        
        return result


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


# Example usage as a script
def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG for Boston Public Library")
    
    # Query options
    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--image", type=str, help="Path or URL to image for query")
    
    # Configuration
    parser.add_argument("--pinecone-index", type=str, required=True, help="Name of Pinecone index")
    parser.add_argument("--namespace", type=str, default="bpl", help="Namespace within Pinecone")
    parser.add_argument("--k", type=int, default=10, help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    if not args.query and not args.image:
        parser.error("At least one of --query or --image must be provided")
    
    # Initialize RAG system
    rag = MultimodalRAG(
        pinecone_index=args.pinecone_index,
        namespace=args.namespace
    )
    
    # Load image if provided
    image_source = None
    if args.image:
        if args.image.startswith(('http://', 'https://')):
            # URL provided
            try:
                response = requests.get(args.image)
                response.raise_for_status()
                image_source = Image.open(io.BytesIO(response.content))
                print(f"Loaded image from URL: {args.image}")
            except Exception as e:
                print(f"Error loading image from URL: {e}")
        else:
            # Local file path
            try:
                image_source = Image.open(args.image)
                print(f"Loaded image from file: {args.image}")
            except Exception as e:
                print(f"Error loading image from file: {e}")
    
    # Process query
    result = rag.process_query(
        query=args.query or "",
        image_source=image_source,
        k=args.k
    )
    
    # Print results
    print("\n=== Query Results ===")
    print(f"Query: {result['query']}")
    if args.image:
        print(f"Image: {args.image}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    print(f"Documents retrieved: {result['documents_retrieved']}")
    print(f"Images retrieved: {result['images_retrieved']}")
    print("\n=== Response ===")
    print(result['response'])


if __name__ == "__main__":
    main()