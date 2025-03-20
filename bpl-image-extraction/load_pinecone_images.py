import getpass
import os
import time
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import re
from langchain_core.documents import Document
import requests
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import logging
import numpy as np
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MultimodalPineconeManager:
    """
    Handles both text and image vectors in Pinecone.
    Supports hybrid search across modalities.
    """
    
    def __init__(self, index_name: str, namespace: str = "default"):
        """
        Initialize connection to Pinecone for multimodal vectors
        
        :param index_name: Name of the Pinecone index
        :param namespace: Base namespace for vectors
        """
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        
        # Initialize text embeddings using HuggingFace
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize image embeddings using CLIP
        self.image_model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.image_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.image_model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        
        # Create text vector store with LangChain
        self.text_vector_store = PineconeVectorStore(
            index=self.index, 
            embedding=self.text_embeddings,
            namespace=f"{namespace}_text"
        )
        
        logger.info(f"Initialized MultimodalPineconeManager with index: {index_name}")
    
    def _flatten_metadata(self, metadata: Dict) -> Dict:
        """
        Flatten nested metadata structures to be compatible with Pinecone
        
        :param metadata: Original metadata dictionary
        :return: Flattened metadata dictionary
        """
        flattened = {}
        
        for key, value in metadata.items():
            # Handle nested dictionaries
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    # Pinecone only accepts strings, numbers, booleans, or lists of strings as values
                    if isinstance(subvalue, (str, int, float, bool)):
                        flattened[f"{key}_{subkey}"] = subvalue
                    elif isinstance(subvalue, list) and all(isinstance(item, str) for item in subvalue):
                        flattened[f"{key}_{subkey}"] = subvalue
                    else:
                        # Convert complex values to strings
                        flattened[f"{key}_{subkey}"] = str(subvalue)
            # Handle direct values
            elif isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                flattened[key] = value
            else:
                # Convert complex values to strings
                flattened[key] = str(value)
        
        return flattened
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate text embedding using HuggingFace embeddings
        
        :param text: Text to embed
        :return: Embedding vector
        """
        return self.text_embeddings.embed_query(text)
    
    def embed_image(self, image_source: Union[str, bytes, Image.Image]) -> Optional[List[float]]:
        """
        Generate image embedding using CLIP
        
        :param image_source: Image source (path, URL, bytes, or PIL Image)
        :return: Embedding vector or None if embedding fails
        """
        try:
            # Load the image if it's not already a PIL Image
            if not isinstance(image_source, Image.Image):
                if isinstance(image_source, str):
                    # Check if it's a URL
                    if image_source.startswith(('http://', 'https://')):
                        response = requests.get(image_source)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                    else:
                        # Local file path
                        image = Image.open(image_source)
                elif isinstance(image_source, bytes):
                    image = Image.open(io.BytesIO(image_source))
                else:
                    raise ValueError("Unsupported image source type")
            else:
                image = image_source
            
            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Normalize the embedding
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Add text documents to Pinecone
        
        :param texts: List of text content
        :param metadatas: List of metadata dictionaries
        :param ids: List of document IDs
        """
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas or [{}] * len(texts))]
        self.text_vector_store.add_documents(documents=documents, ids=ids)
        logger.info(f"Added {len(texts)} text documents to Pinecone")
    
    def add_images(self, image_sources: List[Union[str, bytes, Image.Image]], 
                  metadatas: Optional[List[Dict]] = None, 
                  ids: Optional[List[str]] = None,
                  batch_size: int = 100):
        """
        Add images to Pinecone
        
        :param image_sources: List of image sources
        :param metadatas: List of metadata dictionaries
        :param ids: List of document IDs
        :param batch_size: Number of vectors to upload in each batch
        """
        if metadatas is None:
            metadatas = [{}] * len(image_sources)
        
        if ids is None:
            ids = [f"img_{i}" for i in range(len(image_sources))]
        
        # Process images in batches
        for i in range(0, len(image_sources), batch_size):
            batch_sources = image_sources[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            batch_vectors = []
            
            # Generate embeddings for each image in batch
            for j, (img_src, meta, id_) in enumerate(zip(batch_sources, batch_metadatas, batch_ids)):
                embedding = self.embed_image(img_src)
                
                if embedding:
                    # Flatten the metadata to be compatible with Pinecone
                    flattened_meta = self._flatten_metadata(meta)
                    
                    # Add vector_type to metadata
                    flattened_meta['vector_type'] = 'image'
                    
                    # Ensure URL is preserved as a string
                    if 'url' in meta:
                        flattened_meta['url'] = str(meta['url'])
                    
                    # Ensure source_page is preserved as a string
                    if 'source_page' in meta:
                        flattened_meta['source_page'] = str(meta['source_page'])
                    
                    # Ensure alt text is preserved as a string
                    if 'alt' in meta:
                        flattened_meta['alt'] = str(meta['alt'])
                    
                    vector_entry = {
                        'id': id_,
                        'values': embedding,
                        'metadata': flattened_meta
                    }
                    batch_vectors.append(vector_entry)
                else:
                    logger.warning(f"Failed to generate embedding for image {j+i}")
            
            # Upsert batch to Pinecone
            if batch_vectors:
                self.index.upsert(vectors=batch_vectors, namespace=f"{self.namespace}_image")
                logger.info(f"Added batch of {len(batch_vectors)} image vectors to Pinecone")
            
            # Respect Pinecone rate limits
            time.sleep(1)
        
        logger.info(f"Finished adding images to Pinecone")
    
    def process_image_from_metadata(self, metadata_list: List[Dict]) -> int:
        """
        Process images from metadata and add them to Pinecone
        
        :param metadata_list: List of image metadata dictionaries with URLs
        :return: Number of images successfully added
        """
        image_sources = []
        metadatas = []
        ids = []
        
        # Extract image URLs and metadata
        for i, metadata in enumerate(metadata_list):
            if 'url' not in metadata:
                continue
            
            image_sources.append(metadata['url'])
            metadatas.append(metadata)
            ids.append(f"img_{i}")
        
        # Add images to Pinecone
        if image_sources:
            self.add_images(
                image_sources=image_sources,
                metadatas=metadatas,
                ids=ids
            )
        
        return len(image_sources)
    
    def query_text(self, query_text: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        """
        Query the text vectors using text
        
        :param query_text: Text query
        :param top_k: Number of results to return
        :param filter_dict: Filter to apply to query
        :return: Query results
        """
        results = self.text_vector_store.similarity_search_with_score(
            query=query_text,
            k=top_k,
            filter=filter_dict
        )
        
        # Format results
        matches = []
        for doc, score in results:
            matches.append({
                'document': doc,
                'score': score,
                'type': 'text'
            })
        
        return {'matches': matches}
    
    def query_image(self, 
                   image_source: Union[str, bytes, Image.Image], 
                   top_k: int = 5, 
                   filter_dict: Optional[Dict] = None,
                   search_images: bool = True,
                   search_texts: bool = False) -> Dict:
        """
        Query using an image
        
        :param image_source: Image to query with
        :param top_k: Number of results to return
        :param filter_dict: Filter to apply to query
        :param search_images: Whether to search image vectors
        :param search_texts: Whether to search text vectors
        :return: Query results
        """
        # Generate embedding for query image
        embedding = self.embed_image(image_source)
        
        if embedding is None:
            return {'error': 'Failed to generate embedding for query image'}
        
        results = {'matches': []}
        
        # Query image namespace
        if search_images:
            image_results = self.index.query(
                vector=embedding,
                top_k=top_k,
                namespace=f"{self.namespace}_image",
                include_metadata=True,
                filter=filter_dict
            )
            
            for match in image_results.get('matches', []):
                results['matches'].append({
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'metadata': match.get('metadata', {}),
                    'type': 'image'
                })
        
        # Optionally query text namespace using the image embedding
        if search_texts:
            text_results = self.index.query(
                vector=embedding,
                top_k=top_k,
                namespace=f"{self.namespace}_text",
                include_metadata=True,
                filter=filter_dict
            )
            
            for match in text_results.get('matches', []):
                metadata = match.get('metadata', {})
                content = metadata.get('page_content', 'No content available')
                
                results['matches'].append({
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'document': Document(
                        page_content=content,
                        metadata={k: v for k, v in metadata.items() if k != 'page_content'}
                    ),
                    'type': 'text'
                })
        
        # Sort all matches by score
        results['matches'] = sorted(results['matches'], key=lambda x: x['score'], reverse=True)[:top_k]
        
        return results
    
    def hybrid_query(self, 
                    query_text: str,
                    image_source: Optional[Union[str, bytes, Image.Image]] = None,
                    top_k: int = 5,
                    text_weight: float = 0.7,
                    filter_dict: Optional[Dict] = None) -> Dict:
        """
        Perform a hybrid query with both text and image
        
        :param query_text: Text query
        :param image_source: Optional image query
        :param top_k: Number of results to return
        :param text_weight: Weight for text results (0-1)
        :param filter_dict: Filter to apply to query
        :return: Combined query results
        """
        results = {'matches': []}
        
        # Get text query results
        text_results = self.query_text(
            query_text=query_text,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        # Add text results with text_weight
        for match in text_results.get('matches', []):
            match['score'] *= text_weight
            results['matches'].append(match)
        
        # If image provided, get image query results
        if image_source:
            image_results = self.query_image(
                image_source=image_source,
                top_k=top_k,
                filter_dict=filter_dict,
                search_images=True,
                search_texts=False
            )
            
            # Add image results with (1-text_weight)
            for match in image_results.get('matches', []):
                match['score'] *= (1 - text_weight)
                results['matches'].append(match)
        
        # Sort combined results
        results['matches'] = sorted(results['matches'], key=lambda x: x['score'], reverse=True)[:top_k]
        
        return results