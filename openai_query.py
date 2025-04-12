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
from pinecone_manager import MultimodalPineconeManager

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
        Search for images matching the natural language query
        
        :param query: Natural language query
        :param top_k: Number of results to return
        :param enhance: Whether to enhance the query using OpenAI
        :return: Search results with images and descriptions
        """
        # Enhance query if requested
        search_query = self.enhance_query(query) if enhance else query
        
        # Search using the Pinecone manager directly - this avoids the dimension mismatch
        results = self.pinecone_manager.search(search_query, top_k=top_k)
        
        # Format results for display
        formatted_results = []
        
        for match in results.get('matches', []):
            if match['type'] == 'image':
                # Image result
                metadata = match.get('metadata', {})
                result = {
                    'type': 'image',
                    'score': match.get('score'),
                    'caption': metadata.get('caption', 'No caption available'),
                    'url': metadata.get('url', ''),
                    'source_page': metadata.get('source_page', ''),
                    'tags': metadata.get('tags', [])
                }
                formatted_results.append(result)
            elif match['type'] == 'text':
                # Text result
                # Get the linked image ID
                image_id = match.get('metadata', {}).get('image_id')
                if image_id:
                    # Try to find the image info
                    image_info = self.find_image_by_id(image_id)
                    if image_info:
                        result = {
                            'type': 'image',
                            'score': match.get('score'),
                            'content': match.get('content', ''),
                            'caption': image_info.get('caption', 'No caption available'),
                            'url': image_info.get('url', ''),
                            'source_page': image_info.get('source_page', ''),
                            'tags': image_info.get('tags', [])
                        }
                        formatted_results.append(result)
                    else:
                        # Add just the text result if image not found
                        result = {
                            'type': 'text',
                            'score': match.get('score'),
                            'content': match.get('content', 'No content available'),
                            'image_id': image_id
                        }
                        formatted_results.append(result)
        
        return {
            'query': query,
            'enhanced_query': search_query if enhance else None,
            'results': formatted_results
        }
    
    def find_image_by_id(self, image_id: str) -> Optional[Dict]:
        """
        Find image information by its ID
        
        :param image_id: Image ID to find
        :return: Image information or None if not found
        """
        try:
            # Query for the specific image ID
            result = self.pinecone_manager.index.fetch(
                ids=[image_id],
                namespace=f"{self.namespace}_image"
            )
            
            if image_id in result.get('vectors', {}):
                return result['vectors'][image_id].get('metadata', {})
            
            return None
        except Exception as e:
            logger.error(f"Error finding image by ID: {e}")
            return None
    
    def generate_response(self, query: str, top_k: int = 5, enhance: bool = True) -> Dict:
        """
        Generate a complete natural language response to the query
        
        :param query: Natural language query
        :param top_k: Number of results to return
        :param enhance: Whether to enhance the query
        :return: Structured response with results and generated text
        """
        # First get the search results
        search_results = self.search_images(query, top_k, enhance)
        
        # If no results, return early
        if not search_results['results']:
            return {
                'query': query,
                'response': f"I couldn't find any historical images matching '{query}'. Try a different search term or refine your query.",
                'results': []
            }
        
        # Use OpenAI to generate a natural language response
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        # Create a summary of the search results for the prompt
        result_summaries = []
        for i, result in enumerate(search_results['results']):
            if result['type'] == 'image':
                summary = f"Image {i+1}: {result.get('caption', 'No caption')}"
                if 'content' in result and result['content']:
                    summary += f" - {result['content'][:200]}..."
                if 'tags' in result and result['tags']:
                    summary += f" - Tags: {', '.join(result['tags'][:10])}"
                if 'url' in result and result['url']:
                    summary += f" - URL: {result['url']}"
                result_summaries.append(summary)
            else:
                summary = f"Text {i+1}: {result.get('content', 'No content')[:200]}..."
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
            response_text = f"I found {len(search_results['results'])} historical images matching your query '{query}'."
            if search_results['results']:
                response_text += f" The top result is: {search_results['results'][0].get('caption', 'No caption available')}"
        
        # Construct the final response
        return {
            'query': query,
            'enhanced_query': search_results.get('enhanced_query'),
            'response': response_text,
            'results': search_results['results']
        }


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
        
        response = query_interface.generate_response(test_query, top_k=3)
        
        print(f"\nResponse: {response['response']}")
        print("\nTop results:")
        
        for i, result in enumerate(response['results']):
            print(f"\nResult {i+1}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Type: {result['type']}")
            
            if result['type'] == 'image':
                print(f"  Caption: {result.get('caption', 'No caption')}")
                print(f"  URL: {result.get('url', 'No URL')}")
            else:
                print(f"  Content: {result.get('content', 'No content')[:100]}...")