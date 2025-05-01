#!/usr/bin/env python3

import os
import json
import time
import logging
import uuid
import hashlib
import threading
import concurrent.futures
import ijson
import requests
from typing import List, Dict, Set, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from image_captioning import ImageCaptioner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BPLDescriptionProcessor:
    """
    Process descriptions for BPL images and store their embeddings in Pinecone
    """
    
    def __init__(self, 
                 json_file: str, 
                 index_name: str = "bpl-images",
                 namespace: str = "default",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 caption_model: str = "blip",
                 max_workers: int = 8,
                 batch_size: int = 100,
                 checkpoint_file: str = "description_processor_checkpoint.json",
                 detail_level: str = "detailed",
                 request_timeout: int = 30):
        """
        Initialize the description processor
        
        :param json_file: Path to the large JSON file
        :param index_name: Pinecone index name
        :param namespace: Pinecone namespace
        :param embedding_model: Model to use for text embeddings (must be 768-dim)
        :param caption_model: Model to use for image captioning
        :param max_workers: Maximum number of worker threads
        :param batch_size: Number of items to process in a batch
        :param checkpoint_file: File to store progress
        :param detail_level: Detail level for image descriptions
        :param request_timeout: Timeout in seconds for image requests
        """
        self.json_file = json_file
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model_name = embedding_model
        self.caption_model = caption_model
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.detail_level = detail_level
        self.request_timeout = request_timeout
        
        # Initialize components
        self._init_components()
        
        # Progress tracking
        self.processed_ids = set()
        self.processed_urls = set()  # Track processed URLs to avoid duplicates
        self.error_urls = set()      # Track URLs that caused errors
        self.processed_count = 0
        self.total_count = 0
        self.lock = threading.Lock()
        
        # Load checkpoint if exists
        self.load_checkpoint()
    
    def _init_components(self):
        """Initialize the required components"""
        logger.info("Initializing components...")
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists, create it if not
        try:
            indexes = self.pc.list_indexes()
            index_exists = any(idx['name'] == self.index_name for idx in indexes)
            
            if not index_exists:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # For sentence-transformers/all-mpnet-base-v2
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                logger.info(f"Waiting for index to be ready...")
                time.sleep(30)  # Give time for the index to initialize
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
        
        # Initialize sentence transformer (768-dim embedding model)
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize image captioner
        logger.info(f"Loading image captioner model: {self.caption_model}")
        self.captioner = ImageCaptioner(model_type=self.caption_model)
    
    def load_checkpoint(self):
        """Load progress from checkpoint file if it exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.processed_ids = set(checkpoint.get('processed_ids', []))
                    self.processed_urls = set(checkpoint.get('processed_urls', []))
                    self.error_urls = set(checkpoint.get('error_urls', []))
                    self.processed_count = checkpoint.get('processed_count', 0)
                    self.total_count = checkpoint.get('total_count', 0)
                logger.info(f"Resumed from checkpoint: {len(self.processed_ids)} items, {self.processed_count} images processed, {len(self.error_urls)} error URLs")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            with self.lock:  # Ensure thread safety
                checkpoint = {
                    'processed_ids': list(self.processed_ids),
                    'processed_urls': list(self.processed_urls),
                    'error_urls': list(self.error_urls),
                    'processed_count': self.processed_count,
                    'total_count': self.total_count,
                    'timestamp': time.time()
                }
                
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                
                # Also save a backup in case the main file gets corrupted
                with open(f"{self.checkpoint_file}.bak", 'w') as f:
                    json.dump(checkpoint, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def extract_image_url(self, item: Dict) -> Optional[str]:
        """
        Extract the image URL from a BPL/American Archive item
        
        :param item: Item dictionary from data
        :return: Image URL if found, None otherwise
        """
        try:
            # Check if there's a direct URL in the item
            if 'attributes' in item:
                attributes = item['attributes']
                item_id = item.get('id', '')
                
                # Skip URLs that are known to cause errors
                if item_id in self.error_urls:
                    return None
                
                # American Archive specific - direct thumbnail URL
                if 'identifier_uri_preview_ss' in attributes:
                    url = attributes['identifier_uri_preview_ss']
                    if isinstance(url, str) and url.startswith(('http://', 'https://')):
                        # Skip known problematic domains
                        if any(domain in url for domain in ['credo.library.umass.edu', 'repository.library.northeastern.edu']):
                            self.error_urls.add(item_id)
                            return None
                        return url
                    elif isinstance(url, list) and url and isinstance(url[0], str) and url[0].startswith(('http://', 'https://')):
                        # Skip known problematic domains
                        if any(domain in url[0] for domain in ['credo.library.umass.edu', 'repository.library.northeastern.edu']):
                            self.error_urls.add(item_id)
                            return None
                        return url[0]
                
                # Try to construct URL from identifier and oai_header_id pattern
                if 'identifier_uri_ss' in attributes and 'oai_header_id_ssi' in attributes:
                    identifier = attributes['identifier_uri_ss']
                    oai_header = attributes['oai_header_id_ssi']
                    if isinstance(identifier, str) and 'americanarchive.org' in identifier:
                        if isinstance(oai_header, str) and oai_header.startswith('cpb-aacip'):
                            # Based on the seen pattern in the JSON
                            return f"https://s3.amazonaws.com/americanarchive.org/thumbnail/{oai_header}.jpg"
                
                # Fallback: Try other common image URL fields
                for field in ['thumbnail_url_ss', 'preview_ssim', 'image_url', 'thumbnail']:
                    if field in attributes:
                        url_value = attributes[field]
                        if isinstance(url_value, list):
                            for url in url_value:
                                if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
                                    # Skip known problematic domains
                                    if any(domain in url for domain in ['credo.library.umass.edu', 'repository.library.northeastern.edu']):
                                        continue
                                    return url
                        elif isinstance(url_value, str) and url_value.startswith(('http://', 'https://')):
                            # Skip known problematic domains
                            if any(domain in url_value for domain in ['credo.library.umass.edu', 'repository.library.northeastern.edu']):
                                continue
                            return url_value
                
                # Digital Commonwealth specific URLs (for items with commonwealth prefix)
                if isinstance(item_id, str) and item_id.startswith('commonwealth-oai:'):
                    # Try a more reliable pattern first
                    try:
                        # Extract the ID segment
                        clean_id = item_id.replace('commonwealth-oai:', '')
                        return f"https://www.digitalcommonwealth.org/search/{item_id}/thumbnail"
                    except Exception:
                        pass
            
            return None
        
        except Exception as e:
            logger.error(f"Error extracting image URL for item {item.get('id', 'unknown')}: {e}")
            return None
    
    def safe_join(self, val) -> str:
        """Join list values safely or convert to string"""
        if not val:
            return ""
        return " ".join(val) if isinstance(val, list) else str(val)
    
    def create_text_summary(self, item: Dict) -> str:
        """
        Create a text summary from item attributes to enrich the image description
        
        :param item: Item dictionary from BPL data
        :return: Text summary
        """
        if 'attributes' not in item:
            return ""
        
        attributes = item['attributes']
        
        summary_text = f"""
        Title: {attributes.get('title_info_primary_tsi', '')}
        Subtitle: {attributes.get('title_info_primary_subtitle_tsi', '')}
        Abstract: {attributes.get('abstract_tsi', '')}
        Notes: {self.safe_join(attributes.get('note_tsim', []))}
        Subjects: {self.safe_join(attributes.get('subject_topic_tsim', []))}
        People: {self.safe_join(attributes.get('subject_name_tsim', []))}
        Locations: {self.safe_join(attributes.get('subject_geographic_sim', []))}
        Date: {self.safe_join(attributes.get('date_tsim', []))}
        Type: {self.safe_join(attributes.get('type_of_resource_ssim', []))}
        Collection: {self.safe_join(attributes.get('collection_name_ssim', []))}
        """.strip()
        
        return summary_text
    
    def embed_description(self, description: str) -> List[float]:
        """
        Generate embedding for a description
        
        :param description: Text description
        :return: Embedding vector
        """
        embedding = self.embedding_model.encode(description)
        return embedding.tolist()
    
    def generate_caption_with_timeout(self, image_url: str, detail_level: str, max_length: int) -> Optional[str]:
        """
        Generate a caption with a timeout to handle unresponsive image servers
        
        :param image_url: URL of the image
        :param detail_level: Detail level for caption
        :param max_length: Maximum length of caption
        :return: Caption text or None if timeout or error
        """
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = self.captioner.generate_caption(image_url, detail_level=detail_level, max_length=max_length)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(self.request_timeout)
        
        if thread.is_alive():
            logger.warning(f"Caption generation timed out for URL: {image_url}")
            return None  # Timeout occurred
        
        if exception[0]:
            logger.error(f"Error during caption generation: {exception[0]}")
            return None
            
        return result[0]
    
    def process_item(self, item: Dict) -> bool:
        """
        Process a single item that has an image
        
        :param item: Dictionary containing item data
        :return: True if processed successfully, False otherwise
        """
        try:
            # Extract item ID
            if 'id' not in item:
                logger.warning("Item missing ID field")
                return False
            
            item_id = item['id']
            
            # Skip if already processed
            if item_id in self.processed_ids:
                return False
            
            # Extract image URL - skip items without images
            image_url = self.extract_image_url(item)
            if not image_url:
                return False  # Skip items without images
            
            # Skip if URL has already been processed (avoid duplicates)
            if image_url in self.processed_urls:
                with self.lock:
                    self.processed_ids.add(item_id)
                return False
            
            # Create text summary from metadata
            text_summary = self.create_text_summary(item)
            
            # Generate image description with timeout
            try:
                # Generate detailed description using the VLM with timeout
                detailed_desc = self.generate_caption_with_timeout(
                    image_url, 
                    detail_level=self.detail_level, 
                    max_length=200
                )
                
                if not detailed_desc:
                    logger.warning(f"No description generated for {item_id}")
                    
                    # Add to error URLs to avoid retrying in the future
                    with self.lock:
                        self.error_urls.add(item_id)
                        
                    # Create a metadata-only description
                    if text_summary:
                        detailed_desc = f"Image description not available. Metadata summary: {text_summary}"
                    else:
                        # If no metadata either, skip this item
                        return False
                
                # Create combined description
                description_parts = []
                description_parts.append(detailed_desc)
                
                if text_summary and "Metadata summary" not in detailed_desc:
                    description_parts.append(f"Metadata: {text_summary}")
                
                combined_description = "\n\n".join(description_parts)
            
            except Exception as e:
                logger.error(f"Error generating description for {item_id}: {e}")
                
                # Add to error URLs
                with self.lock:
                    self.error_urls.add(item_id)
                    
                # Fall back to metadata-only description if available
                if text_summary:
                    combined_description = f"Image description not available. Metadata summary: {text_summary}"
                else:
                    # No metadata either, so skip this item
                    return False
            
            # Generate embedding
            embedding = self.embed_description(combined_description)
            
            # Create deterministic vector ID based on image URL to prevent duplicates
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            vector_id = f"bpl_{url_hash}"
            
            # Create vector entry with metadata
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "source_id": item_id,
                    "content": combined_description,
                    "image_url": image_url,
                    "field_type": "image_description"
                }
            }
            
            # Upsert to Pinecone
            self.index.upsert(vectors=[vector], namespace=self.namespace)
            
            # Update progress with lock to ensure thread safety
            with self.lock:
                self.processed_ids.add(item_id)
                self.processed_urls.add(image_url)
                self.processed_count += 1
                
                # Save checkpoint periodically
                if self.processed_count % 50 == 0:
                    self.save_checkpoint()
                    logger.info(f"Progress: {self.processed_count}/{self.total_count if self.total_count else 'unknown'} images processed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            return False
    
    def process_batch(self, batch: List[Dict]) -> int:
        """
        Process a batch of items
        
        :param batch: List of items to process
        :return: Number of items processed successfully
        """
        processed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all items for processing
            futures = [executor.submit(self.process_item, item) for item in batch]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Error in processing thread: {e}")
        
        return processed_count
    
    def process_json_file(self):
        """Process the BPL data JSON file using ijson for streaming, focusing on items with images"""
        start_time = time.time()
        logger.info(f"Starting processing of {self.json_file}")
        
        # Initialize progress tracking
        batch = []
        batch_count = 0
        total_processed = 0
        total_items = 0
        items_with_images = 0
        
        # Sample tracking for debugging
        sample_items = []
        sample_count = 0
        
        try:
            # Process BPL data using ijson for streaming
            with open(self.json_file, 'rb') as f:
                # Stream items using the path
                items = ijson.items(f, "Data.item.data.item")
                
                # Process items in batches
                for item in tqdm(items, desc="Processing items"):
                    total_items += 1
                    
                    # Save a few sample items for debugging URL extraction
                    if sample_count < 5:
                        sample_items.append(item)
                        sample_count += 1
                    
                    # Check if item has an image before adding to batch
                    image_url = self.extract_image_url(item)
                    if image_url:
                        items_with_images += 1
                        batch.append(item)
                        batch_count += 1
                        
                        # Process batch when it reaches the batch size
                        if len(batch) >= self.batch_size:
                            processed = self.process_batch(batch)
                            total_processed += processed
                            batch = []
                            
                            logger.info(f"Processed batch: {processed}/{self.batch_size} items successfully embedded")
                            logger.info(f"Progress: Found {items_with_images}/{total_items} items with images")
                
                # Process any remaining items
                if batch:
                    processed = self.process_batch(batch)
                    total_processed += processed
            
            # Save sample items for debugging
            with open('sample_bpl_items.json', 'w') as f:
                json.dump(sample_items, f, indent=2)
            logger.info("Saved 5 sample items to sample_bpl_items.json for debugging")
        
        except Exception as e:
            logger.error(f"Error processing JSON file: {e}")
            # Save checkpoint to allow resuming
            self.save_checkpoint()
        
        # Final checkpoint
        with self.lock:
            self.total_count = items_with_images
            self.save_checkpoint()
        
        end_time = time.time()
        logger.info(f"Processing complete: {total_processed}/{items_with_images} images processed in {end_time - start_time:.2f} seconds")
        logger.info(f"Found {items_with_images} items with images out of {total_items} total items")
        logger.info(f"Encountered errors with {len(self.error_urls)} items")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process BPL images and store descriptions in Pinecone")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("--index", default="bpl-images", help="Pinecone index name")
    parser.add_argument("--namespace", default="default", help="Pinecone namespace")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2", help="Sentence Transformer model for embeddings")
    parser.add_argument("--caption-model", default="blip", choices=["blip", "git", "llava"], help="Model to use for image captioning")
    parser.add_argument("--detail-level", default="detailed", choices=["basic", "detailed", "analysis"], help="Detail level for image descriptions")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of worker threads")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of items to process in a batch")
    parser.add_argument("--checkpoint-file", default="description_processor_checkpoint.json", help="File to store progress")
    parser.add_argument("--request-timeout", type=int, default=30, help="Timeout in seconds for image requests")
    
    args = parser.parse_args()
    
    processor = BPLDescriptionProcessor(
        json_file=args.json_file,
        index_name=args.index,
        namespace=args.namespace,
        embedding_model=args.embedding_model,
        caption_model=args.caption_model,
        detail_level=args.detail_level,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        checkpoint_file=args.checkpoint_file,
        request_timeout=args.request_timeout
    )
    
    processor.process_json_file()