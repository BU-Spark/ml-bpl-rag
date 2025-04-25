import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import numpy as np

# Import the Pinecone manager
from archive_spring2025.pinecone_manager import MultimodalPineconeManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OpenAIQueryInterface:
    """
    Natural language query interface using OpenAI API for historical image search
    """
    
    def __init__(self, pinecone_index: str, namespace: str = "default"):
        """
        Initialize the OpenAI query interface
        
        :param pinecone_index: Name of the Pinecone index to search
        :param namespace: Namespace in Pinecone
        """
        self.pinecone_index = pinecone_index
        self.namespace = namespace
        
        # Get OpenAI API key from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize Pinecone manager
        logger.info(f"Initializing Pinecone manager with index: {pinecone_index}")
        self.pinecone_manager = MultimodalPineconeManager(
            index_name=pinecone_index,
            namespace=namespace
        )
        
        # OpenAI API settings
        self.openai_model = "text-embedding-ada-002"  # Embedding model
        self.openai_completion_model = "gpt-4o-mini"  # For enhancing natural language queries
        self.openai_api_url = "https://api.openai.com/v1/embeddings"
        self.openai_completion_url = "https://api.openai.com/v1/chat/completions"
    
    def get_openai_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI's API
        
        :param text: Text to embed
        :return: Embedding vector
        """
        # IMPORTANT: We'll skip using OpenAI embedding and use the same embedding model
        # as our Pinecone index to avoid dimension mismatch
        logger.info("Using CLIP text embedding instead of OpenAI to match index dimensions")
        return self.pinecone_manager.embed_text(text)
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance the natural language query using OpenAI's GPT model
        
        :param query: Original user query
        :return: Enhanced query optimized for image search
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        # System prompt for enhancing image search queries
        system_prompt = """You are an AI assistant specialized in converting natural language queries into 
        optimized image search queries for historical Boston images. Your task is to enhance the query to include visual details, 
        historical context, and descriptive elements that would help find relevant historical images.
        Include specific Boston landmarks, architectural features, historical periods, and visual elements.
        Keep your response brief and to the point - just return the enhanced query text with no 
        explanations or other text."""
        
        # Example for few-shot learning
        examples = [
            {
                "role": "user",
                "content": "Boston after World War 2"
            },
            {
                "role": "assistant",
                "content": "photographs of crowds celebrating in Boston streets after World War II victory 1945 with American flags parades"
            },
            {
                "role": "user", 
                "content": "old bridges"
            },
            {
                "role": "assistant",
                "content": "historic photographs of stone or wooden bridges spanning rivers in Boston 19th century with horse carriages or pedestrians crossing"
            }
        ]
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(examples)
        messages.append({"role": "user", "content": query})
        
        data = {
            "model": self.openai_completion_model,
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more focused responses
            "max_tokens": 100    # Keep it concise
        }
        
        try:
            response = requests.post(self.openai_completion_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            enhanced_query = result["choices"][0]["message"]["content"].strip()
            
            logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query with OpenAI: {e}")
            # Fall back to original query
            return query
    
    def search_images(self, query: str, top_k: int = 5, enhance: bool = True) -> Dict:
        """
        Search for images matching the natural language query using field-based approach
        
        :param query: Natural language query
        :param top_k: Number of results to return
        :param enhance: Whether to enhance the query using OpenAI
        :return: Search results with images and descriptions
        """
        # Enhance query if requested
        search_query = self.enhance_query(query) if enhance else query
        
        # Search using the Pinecone manager's field-based search
        results = self.pinecone_manager.search_fields(search_query, top_k=top_k)
        
        # Format results for display
        formatted_results = []
        
        for match in results.get('matches', []):
            source_id = match.get('source_id')
            fields = match.get('fields', {})
            
            # Construct result with available fields
            result = {
                'type': 'image',
                'score': match.get('score', 0),
                'source_id': source_id,
                'url': match.get('url', '')
            }
            
            # Add caption if available
            if 'caption' in fields:
                result['caption'] = fields['caption'].get('content', 'No caption available')
            
            # Add description if available
            if 'description' in fields:
                result['description'] = fields['description'].get('content', '')
            
            # Add tags if available
            if 'tags' in fields:
                tags_content = fields['tags'].get('content', [])
                if isinstance(tags_content, list):
                    result['tags'] = tags_content
                else:
                    # Handle case where tags might be stored as string
                    result['tags'] = str(tags_content).split()
            
            # Add other metadata fields
            for field_type, field_data in fields.items():
                if field_type not in ['caption', 'description', 'tags', 'image']:
                    result[field_type] = field_data.get('content')
            
            formatted_results.append(result)
        
        return {
            'query': query,
            'enhanced_query': search_query if enhance else None,
            'results': formatted_results
        }
    
    def openai_generate_response(self, query: str, enhanced_query: str, results: List[Dict]) -> str:
        """
        Generate a natural language response to the query based on search results
        
        :param query: Original query
        :param enhanced_query: Enhanced query (if available)
        :param results: Formatted search results
        :return: Natural language response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        # Create a summary of the search results for the prompt
        result_summaries = []
        for i, result in enumerate(results):
            if result.get('type') == 'image':
                summary = f"Image {i+1}: {result.get('caption', 'No caption')}"
                
                # Add tags if available
                if 'tags' in result and result['tags']:
                    tags_str = ', '.join(result['tags'][:10]) if isinstance(result['tags'], list) else result['tags']
                    summary += f" - Tags: {tags_str}"
                
                # Add URL
                if 'url' in result and result['url']:
                    summary += f" - URL: {result['url']}"
                
                # Add source ID
                if 'source_id' in result:
                    summary += f" - Source ID: {result['source_id']}"
                
                result_summaries.append(summary)
        
        # System prompt for natural language response
        system_prompt = """You are an AI assistant specialized in historical image search for Boston Public Library's historical image collection. 
        Your task is to respond to user queries about historical images with helpful, informative responses based on the search results provided.
        Be conversational but concise, highlighting the most relevant aspects of the images found. Focus on historical 
        context, visual details, and interesting aspects of the images that relate to the user's query."""
        
        # Construct the full prompt
        user_prompt = f"""User query: {query}

Search results:
{chr(10).join([f"- {summary}" for summary in result_summaries])}

Please provide a helpful response to the user's query based on these search results. 
Mention the number of relevant images found and describe the most relevant ones in relation to their query.
Keep your response conversational but concise (3-4 sentences total)."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        data = {
            "model": self.openai_completion_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 250
        }
        
        try:
            response = requests.post(self.openai_completion_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            # Fall back to simple response
            response_text = f"I found {len(results)} historical images matching your query '{query}'."
            if results:
                response_text += f" The top result is: {results[0].get('caption', 'No caption available')}"
        
        return response_text
    
    def generate_response(self, query: str, top_k: int = 5, enhance: bool = True) -> Dict:
        """
        Generate a complete natural language response to the query
        
        :param query: Natural language query
        :param top_k: Number of results to return
        :param enhance: Whether to enhance the query
        :return: Structured response with results and generated text
        """
        # First get the search results using the field-based approach
        search_results = self.search_images(query, top_k, enhance)
        
        # If no results, return early
        if not search_results['results']:
            return {
                'query': query,
                'response': f"I couldn't find any historical images matching '{query}'. Try a different search term or refine your query.",
                'results': []
            }
        
        # Generate a natural language response using OpenAI
        response_text = self.openai_generate_response(
            query=query,
            enhanced_query=search_results.get('enhanced_query', query),
            results=search_results['results']
        )
        
        # Construct the final response
        return {
            'query': query,
            'enhanced_query': search_results.get('enhanced_query'),
            'response': response_text,
            'results': search_results['results']
        }
    
    def find_image_by_source_id(self, source_id: str) -> Optional[Dict]:
        """
        Find image information by its source ID in the field-based structure
        
        :param source_id: Source ID to find
        :return: Image information or None if not found
        """
        try:
            # Query for vectors with matching source_id
            filter_dict = {"source_id": {"$eq": source_id}}
            
            result = self.pinecone_manager.index.query(
                vector=[0.0] * 512,  # Dummy vector, we're just using the filter
                top_k=100,  # Get all fields for this source ID
                namespace=f"{self.namespace}_fields",
                include_metadata=True,
                filter=filter_dict
            )
            
            if not result.get('matches'):
                return None
            
            # Group by field type
            fields = {}
            image_url = ""
            
            for match in result.get('matches', []):
                metadata = match.get('metadata', {})
                field_type = metadata.get('field_type')
                
                if field_type:
                    if field_type == 'image':
                        image_url = metadata.get('url', '')
                    
                    content = metadata.get('content')
                    if content:
                        fields[field_type] = content
            
            return {
                'source_id': source_id,
                'url': image_url,
                'fields': fields
            }
            
        except Exception as e:
            logger.error(f"Error finding image by source ID: {e}")
            return None

# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Check if API keys are set
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
    else:
        print("OpenAI API key found.")
    
    if not pinecone_api_key:
        print("ERROR: PINECONE_API_KEY environment variable not set")
    else:
        print("Pinecone API key found.")
    
    if openai_api_key and pinecone_api_key:
        # Test the OpenAI query interface
        query_interface = OpenAIQueryInterface(
            pinecone_index="historical-images",
            namespace="bpl"
        )
        
        # Test query
        test_query = "old photographs of Boston Common with people"
        print(f"\nSearching for: '{test_query}'")
        
        # Test field-based search
        search_results = query_interface.search_images(test_query, top_k=3)
        
        print(f"\nFound {len(search_results['results'])} results:")
        for i, result in enumerate(search_results['results']):
            print(f"\nResult {i+1}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Source ID: {result.get('source_id', 'N/A')}")
            print(f"  Caption: {result.get('caption', 'No caption')}")
            print(f"  URL: {result.get('url', 'No URL')}")
            
            if 'tags' in result and result['tags']:
                tags_str = ', '.join(result['tags'][:5]) if isinstance(result['tags'], list) else result['tags']
                print(f"  Tags: {tags_str}")
        
        # Test natural language response generation
        print("\n\nGenerating natural language response:")
        response = query_interface.generate_response(test_query, top_k=3)
        
        print(f"\nQuery: {test_query}")
        if response.get('enhanced_query'):
            print(f"Enhanced query: {response['enhanced_query']}")
        
        print(f"\nResponse: {response['response']}")