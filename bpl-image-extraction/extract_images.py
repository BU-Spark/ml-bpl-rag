import os
import argparse
import logging
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from image_scraper import DigitalCommonwealthScraper
from load_pinecone_images import MultimodalPineconeManager
import time
from langchain_core.documents import Document
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ImageExtractor:
    """Handles extraction of images from Digital Commonwealth and uploads to Pinecone"""
    
    def __init__(self, 
                 pinecone_index: str,
                 output_dir: str = "extracted_images",
                 save_locally: bool = False):
        """
        Initialize the image extractor
        
        :param pinecone_index: Name of the Pinecone index
        :param output_dir: Directory to save extracted images and metadata
        :param save_locally: Whether to save images locally
        """
        self.pinecone_index = pinecone_index
        self.output_dir = output_dir
        self.save_locally = save_locally
        
        # Create output directory if saving locally
        if save_locally:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        # Initialize components
        self.scraper = DigitalCommonwealthScraper()
        self.pinecone_manager = MultimodalPineconeManager(
            index_name=pinecone_index,
            namespace="bpl"
        )
        
        logger.info(f"Initialized ImageExtractor with index: {pinecone_index}")
    
    def extract_from_url(self, url: str) -> Dict:
        """
        Extract images from a single URL
        
        :param url: URL to extract images from
        :return: Statistics about the extraction
        """
        logger.info(f"Extracting images from URL: {url}")
        start_time = time.time()
        
        # Extract images and metadata using the scraper
        images = self.scraper.extract_images(url)
        
        if not images:
            logger.warning(f"No images found at URL: {url}")
            return {
                "url": url,
                "images_found": 0,
                "images_processed": 0,
                "processing_time": time.time() - start_time
            }
        
        logger.info(f"Found {len(images)} images at URL: {url}")
        
        # Save images locally if requested
        if self.save_locally:
            download_paths = self.scraper.download_images(
                images, 
                output_dir=os.path.join(self.output_dir, "images")
            )
            
            # Save metadata to JSON
            metadata_path = os.path.join(
                self.output_dir, 
                "metadata", 
                f"{url.split('/')[-1]}_metadata.json"
            )
            with open(metadata_path, 'w') as f:
                json.dump(images, f, indent=2)
        
        # Process images and add to Pinecone
        processed_count = self.pinecone_manager.process_image_from_metadata(images)
        
        # Create statistics
        stats = {
            "url": url,
            "images_found": len(images),
            "images_processed": processed_count,
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Completed extraction from {url}: {stats}")
        return stats
    
    def extract_from_ids(self, item_ids: List[str]) -> List[Dict]:
        """
        Extract images from a list of Digital Commonwealth item IDs
        
        :param item_ids: List of item IDs
        :return: List of statistics for each URL
        """
        results = []
        
        for item_id in item_ids:
            # Convert ID to URL
            url = f"https://www.digitalcommonwealth.org/search/{item_id}"
            result = self.extract_from_url(url)
            results.append(result)
        
        return results
    
    def extract_from_metadata_json(self, json_path: str) -> List[Dict]:
        """
        Extract images from Digital Commonwealth metadata JSON
        
        :param json_path: Path to metadata JSON file
        :return: List of statistics for each processed item
        """
        logger.info(f"Extracting images from metadata JSON: {json_path}")
        
        # Load metadata JSON
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata JSON: {e}")
            return []
        
        # Extract item IDs from metadata
        item_ids = []
        
        if isinstance(metadata, list) and metadata and "data" in metadata[0]:
            # Handle format with pages containing data array
            for page in metadata:
                content = page.get('data', [])
                for item in content:
                    if "id" in item:
                        item_ids.append(item["id"])
        elif isinstance(metadata, dict) and "data" in metadata:
            # Handle format with single data array
            content = metadata.get('data', [])
            for item in content:
                if "id" in item:
                    item_ids.append(item["id"])
        elif isinstance(metadata, list):
            # Handle flat list of items
            for item in metadata:
                if "id" in item:
                    item_ids.append(item["id"])
        
        logger.info(f"Found {len(item_ids)} item IDs in metadata")
        
        # Extract images from each item ID
        return self.extract_from_ids(item_ids)
    
    def extract_images_and_store_texts(self, json_path: str, text_fields: List[str]) -> Dict:
        """
        Extract both images and text content from metadata
        
        :param json_path: Path to metadata JSON file
        :param text_fields: List of text fields to extract from metadata
        :return: Statistics about the extraction
        """
        logger.info(f"Extracting images and texts from metadata: {json_path}")
        
        # Load metadata JSON
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata JSON: {e}")
            return {"error": str(e)}
        
        # Extract and process content
        image_stats = []
        text_count = 0
        
        # Process metadata and extract content
        if isinstance(metadata, list) and metadata and "data" in metadata[0]:
            # Process each page of results
            for page in metadata:
                content = page.get('data', [])
                stats = self._process_content_items(content, text_fields)
                image_stats.extend(stats["image_stats"])
                text_count += stats["text_count"]
        elif isinstance(metadata, dict) and "data" in metadata:
            # Process single page of results
            content = metadata.get('data', [])
            stats = self._process_content_items(content, text_fields)
            image_stats.extend(stats["image_stats"])
            text_count += stats["text_count"]
        
        return {
            "items_processed": len(image_stats),
            "images_processed": sum(stat["images_processed"] for stat in image_stats),
            "texts_processed": text_count,
            "image_stats": image_stats
        }
    
    def _process_content_items(self, content: List[Dict], text_fields: List[str]) -> Dict:
        """
        Process a list of content items to extract images and texts
        
        :param content: List of content items
        :param text_fields: List of text fields to extract
        :return: Statistics about processing
        """
        image_stats = []
        text_count = 0
        
        for item in content:
            item_id = item.get("id")
            if not item_id:
                continue
            
            # Extract images
            url = f"https://www.digitalcommonwealth.org/search/{item_id}"
            image_stat = self.extract_from_url(url)
            image_stats.append(image_stat)
            
            # Extract text fields
            if "attributes" in item:
                attributes = item["attributes"]
                documents = []
                
                for field in text_fields:
                    if field in attributes and attributes[field]:
                        text_content = str(attributes[field])
                        if text_content:
                            documents.append(Document(
                                page_content=text_content,
                                metadata={
                                    "source": item_id,
                                    "field": field,
                                    "URL": url
                                }
                            ))
                
                # Add texts to Pinecone if any found
                if documents:
                    uuids = [str(uuid4()) for _ in range(len(documents))]
                    self.pinecone_manager.text_vector_store.add_documents(
                        documents=documents,
                        ids=uuids
                    )
                    text_count += len(documents)
        
        return {
            "image_stats": image_stats,
            "text_count": text_count
        }


def main():
    parser = argparse.ArgumentParser(description="Extract images from Digital Commonwealth and store in Pinecone")
    
    # Input sources - mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--url", type=str, help="URL to extract images from")
    input_group.add_argument("--ids", nargs="+", help="List of Digital Commonwealth item IDs")
    input_group.add_argument("--metadata-json", type=str, help="Path to metadata JSON file")
    
    # Configuration
    parser.add_argument("--pinecone-index", type=str, required=True, help="Name of Pinecone index")
    parser.add_argument("--save-locally", action="store_true", help="Save images locally")
    parser.add_argument("--output-dir", type=str, default="extracted_images", help="Directory to save extracted content")
    parser.add_argument("--text-fields", nargs="+", default=["abstract_tsi", "title_info_primary_tsi", "title_info_primary_subtitle_tsi", "title_info_alternative_tsim"], 
                      help="Text fields to extract when using --metadata-json")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ImageExtractor(
        pinecone_index=args.pinecone_index,
        output_dir=args.output_dir,
        save_locally=args.save_locally
    )
    
    # Process based on input type
    if args.url:
        result = extractor.extract_from_url(args.url)
        print(f"Extracted {result['images_processed']} images from {args.url}")
        
    elif args.ids:
        results = extractor.extract_from_ids(args.ids)
        total_processed = sum(r["images_processed"] for r in results)
        print(f"Extracted {total_processed} images from {len(args.ids)} items")
        
    elif args.metadata_json:
        results = extractor.extract_images_and_store_texts(
            json_path=args.metadata_json,
            text_fields=args.text_fields
        )
        print(f"Processed {results['items_processed']} items")
        print(f"Extracted {results['images_processed']} images")
        print(f"Processed {results['texts_processed']} text fields")


if __name__ == "__main__":
    main()