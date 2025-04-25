import os
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MultimodalPineconeManager:
    """
    Handles storing and retrieving multimodal vectors (text and image) in Pinecone
    """
    
    def __init__(self, index_name: str, namespace: str = "default"):
        """
        Initialize connection to Pinecone
        
        :param index_name: Name of the Pinecone index
        :param namespace: Base namespace for vectors
        """
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize Pinecone
        logger.info(f"Connecting to Pinecone with index: {index_name}")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Try to connect to the index, create if it doesn't exist
        try:
            # Check if index exists
            available_indexes = self.pc.list_indexes()
            
            if not any(idx['name'] == index_name for idx in available_indexes):
                logger.info(f"Creating new index: {index_name}")
                # Create the index with 512 dimensions for CLIP
                # Define the server spec for the index
                spec = {
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
                self.pc.create_index(
                    name=index_name,
                    dimension=512,  # CLIP embedding size
                    metric="cosine",
                    spec=spec
                )
                # Wait for index to be ready
                time.sleep(5)
            
            # Connect to the index
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
        
        # Initialize CLIP model for embeddings
        logger.info("Loading CLIP model for embeddings...")
        self.model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)
        self.clip_model = CLIPModel.from_pretrained(self.model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        logger.info(f"CLIP model loaded on: {self.device}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using CLIP's text encoder
        
        :param text: Text to embed
        :return: Embedding vector
        """
        # Process text with CLIP
        inputs = self.clip_processor(
            text=text,
            images=None,  # No image needed for text embedding
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's maximum context length
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**{k: v for k, v in inputs.items() if k != 'pixel_values'})
        
        # Convert to numpy and normalize
        embedding = text_features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def embed_image(self, image_source: Union[str, bytes, Image.Image]) -> Optional[List[float]]:
        """
        Generate embedding for image using CLIP's image encoder
        
        :param image_source: Path to image, URL, bytes, or PIL Image
        :return: Embedding vector or None if embedding fails
        """
        try:
            # Load the image if not already a PIL Image
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
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None
    
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
                    # Pinecone only accepts strings, numbers, booleans, or lists of strings
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
    
    def add_text_descriptions(self, descriptions: List[str], metadatas: List[Dict], ids: Optional[List[str]] = None):
        """
        Add text descriptions to Pinecone
        
        :param descriptions: List of text descriptions
        :param metadatas: List of metadata dictionaries
        :param ids: List of document IDs (generated if not provided)
        """
        if ids is None:
            ids = [f"txt_{i}" for i in range(len(descriptions))]
        
        vectors = []
        
        # Process each description
        for i, (description, metadata, id_) in enumerate(zip(descriptions, metadatas, ids)):
            # Generate embedding
            embedding = self.embed_text(description)
            
            # Flatten metadata and add the description as page_content
            flattened_meta = self._flatten_metadata(metadata)
            flattened_meta["page_content"] = description
            flattened_meta["vector_type"] = "text"
            
            # Create vector entry
            vector = {
                "id": id_,
                "values": embedding,
                "metadata": flattened_meta
            }
            
            vectors.append(vector)
        
        # Upsert vectors to Pinecone
        self.index.upsert(vectors=vectors, namespace=f"{self.namespace}_text")
        logger.info(f"Added {len(vectors)} text descriptions to Pinecone")
    
    def add_images(self, image_sources: List[Union[str, bytes, Image.Image]], 
                 metadatas: List[Dict],
                 ids: Optional[List[str]] = None,
                 batch_size: int = 100):
        """
        Add images to Pinecone
        
        :param image_sources: List of image sources
        :param metadatas: List of metadata dictionaries
        :param ids: List of document IDs (generated if not provided)
        :param batch_size: Number of vectors to upload in each batch
        """
        if ids is None:
            ids = [f"img_{i}" for i in range(len(image_sources))]
        
        # Process images in batches
        for i in range(0, len(image_sources), batch_size):
            batch_sources = image_sources[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            batch_vectors = []
            
            # Generate embeddings for each image in batch
            for j, (img_src, metadata, id_) in enumerate(zip(batch_sources, batch_metadatas, batch_ids)):
                embedding = self.embed_image(img_src)
                
                if embedding:
                    # Flatten metadata
                    flattened_meta = self._flatten_metadata(metadata)
                    flattened_meta["vector_type"] = "image"
                    
                    # Create vector entry
                    vector = {
                        "id": id_,
                        "values": embedding,
                        "metadata": flattened_meta
                    }
                    
                    batch_vectors.append(vector)
                else:
                    logger.warning(f"Failed to generate embedding for image {i+j}")
            
            # Upsert batch to Pinecone
            if batch_vectors:
                self.index.upsert(vectors=batch_vectors, namespace=f"{self.namespace}_image")
                logger.info(f"Added batch of {len(batch_vectors)} image vectors to Pinecone")
            
            # Respect rate limits
            time.sleep(1)
    
    def process_image_with_captions(self, image_source: Union[str, bytes, Image.Image], 
                                  caption: str, 
                                  detailed_description: Optional[str] = None,
                                  tags: Optional[List[str]] = None,
                                  metadata: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Process an image with its captions, storing both in Pinecone
        
        :param image_source: Image source
        :param caption: Short caption of the image
        :param detailed_description: Detailed description of the image
        :param tags: List of tags/keywords
        :param metadata: Additional metadata (source URL, etc.)
        :return: Image ID and description ID
        """
        if metadata is None:
            metadata = {}
        
        # Generate a unique ID for this image
        import uuid
        image_id = f"img_{str(uuid.uuid4())[:8]}"
        desc_id = f"desc_{str(uuid.uuid4())[:8]}"
        
        # Create metadata for image
        image_meta = metadata.copy()
        image_meta["caption"] = caption
        if tags:
            image_meta["tags"] = tags
        
        # Store image embedding
        self.add_images(
            image_sources=[image_source],
            metadatas=[image_meta],
            ids=[image_id]
        )
        
        # Create description metadata
        desc_meta = metadata.copy()
        desc_meta["image_id"] = image_id
        if tags:
            desc_meta["tags"] = tags
        
        # Store description embedding(s)
        descriptions = []
        desc_metadatas = []
        
        # Add caption as a text entry
        descriptions.append(caption)
        caption_meta = desc_meta.copy()
        caption_meta["description_type"] = "caption"
        desc_metadatas.append(caption_meta)
        
        # Add detailed description if available
        if detailed_description:
            descriptions.append(detailed_description)
            detail_meta = desc_meta.copy()
            detail_meta["description_type"] = "detailed"
            desc_metadatas.append(detail_meta)
        
        # Store all descriptions
        desc_ids = [desc_id]
        if detailed_description:
            desc_ids.append(f"{desc_id}_detail")
        
        self.add_text_descriptions(
            descriptions=descriptions,
            metadatas=desc_metadatas,
            ids=desc_ids
        )
        
        return image_id, desc_id
    
    def process_image_with_fields(self, image_source: Union[str, bytes, Image.Image],
                                caption: str,
                                detailed_description: Optional[str] = None,
                                tags: Optional[List[str]] = None,
                                metadata: Optional[Dict] = None) -> str:
        """
        Process an image using a field-based approach where each aspect is stored separately
        but references a common source ID
        
        :param image_source: Image source
        :param caption: Short caption of the image
        :param detailed_description: Detailed description of the image
        :param tags: List of tags/keywords
        :param metadata: Additional metadata (source URL, etc.)
        :return: Source ID that all fields reference
        """
        if metadata is None:
            metadata = {}
            
        # Generate a unique source ID for this image
        import uuid
        source_id = f"src_{str(uuid.uuid4())[:8]}"
        
        # Create base metadata with source ID reference
        base_meta = {"source_id": source_id}
        
        # Add source metadata to base metadata
        for key, value in metadata.items():
            # Don't overwrite source_id if it's in metadata
            if key != "source_id":
                base_meta[key] = value
        
        # Create a list to collect all vectors to upsert
        vectors = []
        
        # Add image embedding
        image_embedding = self.embed_image(image_source)
        if image_embedding:
            vectors.append({
                "id": f"{source_id}_image",
                "values": image_embedding,
                "metadata": {
                    **base_meta,
                    "field_type": "image",
                    "url": metadata.get("url", ""),
                    "caption": caption  # Include caption with image for convenience
                }
            })
        
        # Add caption as separate field
        if caption:
            caption_embedding = self.embed_text(caption)
            vectors.append({
                "id": f"{source_id}_caption",
                "values": caption_embedding,
                "metadata": {
                    **base_meta,
                    "field_type": "caption",
                    "content": caption
                }
            })
        
        # Add detailed description as separate field
        if detailed_description:
            desc_embedding = self.embed_text(detailed_description)
            vectors.append({
                "id": f"{source_id}_description",
                "values": desc_embedding,
                "metadata": {
                    **base_meta,
                    "field_type": "description",
                    "content": detailed_description
                }
            })
        
        # Add tags as separate field
        if tags and len(tags) > 0:
            tags_text = " ".join(tags)
            tags_embedding = self.embed_text(tags_text)
            vectors.append({
                "id": f"{source_id}_tags",
                "values": tags_embedding,
                "metadata": {
                    **base_meta, 
                    "field_type": "tags",
                    "content": tags
                }
            })
        
        # Process any additional metadata fields from BPL that need their own entries
        for key, value in metadata.items():
            # Skip already processed fields or empty values
            if key in ['url', 'source_id'] or not value:
                continue
                
            # Create embedding for this metadata field
            if isinstance(value, str):
                field_embedding = self.embed_text(value)
                vectors.append({
                    "id": f"{source_id}_{key}",
                    "values": field_embedding,
                    "metadata": {
                        **base_meta,
                        "field_type": key,
                        "content": value
                    }
                })
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                # Handle list of strings (like multiple subjects)
                field_text = " ".join(value)
                field_embedding = self.embed_text(field_text)
                vectors.append({
                    "id": f"{source_id}_{key}",
                    "values": field_embedding,
                    "metadata": {
                        **base_meta,
                        "field_type": key,
                        "content": value
                    }
                })
        
        # Upsert all vectors to Pinecone
        if vectors:
            self.index.upsert(vectors=vectors, namespace=f"{self.namespace}_fields")
            logger.info(f"Added {len(vectors)} field vectors for source_id: {source_id}")
        
        return source_id
    
    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None, mode: str = "all") -> Dict:
        """
        Search for vectors using text query
        
        :param query: Text query
        :param top_k: Number of results to return
        :param filter_dict: Filter to apply to query
        :param mode: Search mode (all, text, image)
        :return: Search results
        """
        # Generate embedding for query
        query_embedding = self.embed_text(query)
        
        results = {"matches": []}
        
        # Search text namespace
        if mode in ["all", "text"]:
            text_results = self.index.query(
                vector=query_embedding,
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
                    'content': content,
                    'metadata': {k: v for k, v in metadata.items() if k != 'page_content'},
                    'type': 'text'
                })
        
        # Search image namespace
        if mode in ["all", "image"]:
            image_results = self.index.query(
                vector=query_embedding,
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
        
        # Sort results by score
        results['matches'] = sorted(results['matches'], key=lambda x: x['score'], reverse=True)[:top_k]
        
        return results
    
    def search_fields(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        """
        Search for images using the field-based approach
        
        :param query: Text query
        :param top_k: Number of results to return
        :param filter_dict: Filter to apply to query
        :return: Search results grouped by source_id
        """
        # Generate embedding for query
        query_embedding = self.embed_text(query)
        
        # Search in the fields namespace
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Get more results since we'll group them
            namespace=f"{self.namespace}_fields",
            include_metadata=True,
            filter=filter_dict
        )
        
        # Group results by source_id
        grouped_results = {}
        
        for match in search_results.get('matches', []):
            metadata = match.get('metadata', {})
            source_id = metadata.get('source_id')
            
            if not source_id:
                continue
                
            # Initialize source entry if not exists
            if source_id not in grouped_results:
                grouped_results[source_id] = {
                    'source_id': source_id,
                    'score': match.get('score', 0),  # Use the highest score for now
                    'fields': {},
                    'url': metadata.get('url', '')
                }
            elif match.get('score', 0) > grouped_results[source_id]['score']:
                # Update score if this match has a higher one
                grouped_results[source_id]['score'] = match.get('score', 0)
            
            # Add field data to grouped result
            field_type = metadata.get('field_type')
            if field_type:
                grouped_results[source_id]['fields'][field_type] = {
                    'content': metadata.get('content'),
                    'score': match.get('score', 0)
                }
                
                # Update URL if not already set
                if field_type == 'image' and 'url' in metadata and not grouped_results[source_id]['url']:
                    grouped_results[source_id]['url'] = metadata.get('url', '')
        
        # Convert to list and sort by score
        results_list = list(grouped_results.values())
        results_list.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to top_k after grouping
        results_list = results_list[:top_k]
        
        return {'matches': results_list}

# Example usage
if __name__ == "__main__":
    # Test the Pinecone manager
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    pinecone_manager = MultimodalPineconeManager(
        index_name="historical-images",
        namespace="test"
    )
    
    # Test text embedding
    text = "Historic photograph of Boston Common with people celebrating"
    text_embedding = pinecone_manager.embed_text(text)
    print(f"Text embedding length: {len(text_embedding)}")
    
    # Test search
    if os.path.exists("sample_images"):
        from archive_spring2025.image_captioning import ImageCaptioner
        
        # Create image captioner
        captioner = ImageCaptioner()
        
        # Process a sample image
        import glob
        sample_images = glob.glob("sample_images/*")
        
        if sample_images:
            image_path = sample_images[0]
            print(f"Processing sample image: {image_path}")
            
            # Generate caption
            caption = captioner.generate_caption(image_path)
            description = captioner.generate_caption(image_path, detail_level="detailed")
            tags = captioner.generate_tags(image_path)
            
            # Process image with fields
            source_id = pinecone_manager.process_image_with_fields(
                image_source=image_path,
                caption=caption,
                detailed_description=description,
                tags=tags,
                metadata={"source": "sample"}
            )
            
            print(f"Stored image with fields, source ID: {source_id}")
            
            # Test search
            search_query = "historic photograph"
            results = pinecone_manager.search_fields(search_query, top_k=3)
            
            print(f"\nSearch results for '{search_query}':")
            for i, match in enumerate(results.get('matches', [])):
                print(f"Result {i+1}:")
                print(f"  Score: {match['score']:.4f}")
                print(f"  Source ID: {match['source_id']}")
                print(f"  Fields: {', '.join(match['fields'].keys())}")
                
                if 'caption' in match['fields']:
                    print(f"  Caption: {match['fields']['caption']['content'][:100]}...")
                print()