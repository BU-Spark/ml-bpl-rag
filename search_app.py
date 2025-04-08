#!/usr/bin/env python3

import os
import argparse
import json
import logging
import random
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import application components
from image_scraper import DigitalCommonwealthScraper
from image_captioning import ImageCaptioner
from pinecone_manager import MultimodalPineconeManager
from openai_query import OpenAIQueryInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HistoricalImageSearchApp:
    """
    Main application for the Boston Public Library historical image search system
    """
    
    def __init__(self, 
                 pinecone_index: str = "historical-images",
                 namespace: str = "bpl",
                 caption_model: str = "blip",
                 output_dir: str = "output"):
        """
        Initialize the application
        
        :param pinecone_index: Name of the Pinecone index to use
        :param namespace: Namespace in Pinecone
        :param caption_model: Model to use for image captioning
        :param output_dir: Directory for downloaded files
        """
        self.pinecone_index = pinecone_index
        self.namespace = namespace
        self.caption_model = caption_model
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        # Check for required API keys
        self._check_environment()
        
        # Initialize components
        logger.info("Initializing components...")
        self.scraper = DigitalCommonwealthScraper()
        self.captioner = ImageCaptioner(model_type=caption_model)
        self.pinecone_manager = MultimodalPineconeManager(
            index_name=pinecone_index,
            namespace=namespace
        )
        
        # Initialize query interface if OpenAI API key is available
        if os.getenv("OPENAI_API_KEY"):
            self.query_interface = OpenAIQueryInterface(
                pinecone_index=pinecone_index,
                namespace=namespace
            )
        else:
            self.query_interface = None
            logger.warning("OpenAI API key not found, natural language query interface disabled")
        
        logger.info("Application initialized successfully")
    
    def _check_environment(self):
        """Check if all required API keys are set"""
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set - some features will be limited")
    
    def scrape_and_process(self, 
                          queries: List[str], 
                          limit_items: int = 3, 
                          max_images_per_item: int = 2, 
                          save_images: bool = True) -> Dict:
        """
        Scrape images from Digital Commonwealth, generate captions, and store in Pinecone
        
        :param queries: List of search queries to use for diversity
        :param limit_items: Maximum number of items to process per query
        :param max_images_per_item: Maximum number of images to process per item
        :param save_images: Whether to save images locally
        :return: Processing results
        """
        all_processed_items = []
        total_images = 0
        
        for query in queries:
            logger.info(f"Searching Digital Commonwealth for: '{query}'")
            item_ids = self.scraper.search_query(query, limit=limit_items)
            
            if not item_ids:
                logger.warning(f"No items found for query: '{query}'")
                continue
            
            logger.info(f"Found {len(item_ids)} items for query '{query}', processing...")
            
            processed_items = []
            
            # Process each item
            for i, item_id in enumerate(item_ids):
                logger.info(f"Processing item {i+1}/{len(item_ids)}: {item_id}")
                
                # Get the item URL
                item_url = f"https://www.digitalcommonwealth.org/search/{item_id}"
                
                # Extract images
                images = self.scraper.extract_images(item_url)
                logger.info(f"Found {len(images)} images for item: {item_id}")
                
                # Limit images per item and select random ones for diversity
                if len(images) > max_images_per_item:
                    logger.info(f"Limiting to {max_images_per_item} random images for diversity")
                    images = random.sample(images, max_images_per_item)
                
                # Process each image
                processed_images = []
                
                for j, image in enumerate(images):
                    try:
                        image_url = image.get('url')
                        if not image_url:
                            continue
                        
                        # Save image locally if requested
                        local_path = None
                        if save_images:
                            response = self.scraper.download_images(
                                [image], 
                                output_dir=os.path.join(self.output_dir, "images")
                            )
                            if response:
                                local_path = response[0]
                        
                        # Use the local path if available, otherwise use URL
                        image_source = local_path if local_path else image_url
                        
                        # Generate image analysis
                        logger.info(f"Generating analysis for image {j+1}/{len(images)}")
                        analysis = self.captioner.analyze_image(image_source)
                        
                        if analysis:
                            caption = analysis.get('caption', '')
                            description = analysis.get('description', '')
                            tags = analysis.get('tags', [])
                            
                            # Add analysis to image metadata
                            image['caption'] = caption
                            image['detailed_description'] = description
                            image['tags'] = tags
                            
                            # Store in Pinecone
                            logger.info("Adding image and descriptions to Pinecone")
                            image_id, desc_id = self.pinecone_manager.process_image_with_captions(
                                image_source=image_source,
                                caption=caption,
                                detailed_description=description,
                                tags=tags,
                                metadata={
                                    'url': image_url,
                                    'source_page': item_url,
                                    'alt': image.get('alt', ''),
                                    'item_id': item_id,
                                    'source': 'Digital Commonwealth',
                                    'query': query  # Add the search query used to find this image
                                }
                            )
                            
                            processed_images.append({
                                'image_id': image_id,
                                'description_id': desc_id,
                                'url': image_url,
                                'local_path': local_path,
                                'caption': caption,
                                'tags': tags
                            })
                            
                            # Save detailed metadata
                            metadata_file = os.path.join(
                                self.output_dir, 
                                "metadata", 
                                f"{item_id}_{j}.json"
                            )
                            with open(metadata_file, 'w') as f:
                                json.dump({
                                    'item_id': item_id,
                                    'image_url': image_url,
                                    'local_path': local_path,
                                    'analysis': analysis,
                                    'pinecone_ids': {
                                        'image_id': image_id,
                                        'description_id': desc_id
                                    }
                                }, f, indent=2)
                            
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                
                # Add to processed items
                processed_items.append({
                    'item_id': item_id,
                    'item_url': item_url,
                    'query': query,
                    'images_found': len(images),
                    'images_processed': len(processed_images),
                    'processed_images': processed_images
                })
                
                total_images += len(processed_images)
            
            all_processed_items.extend(processed_items)
        
        # Generate result summary
        result = {
            "success": True,
            "queries": queries,
            "items_found": len(all_processed_items),
            "total_images_processed": total_images,
            "processed_items": all_processed_items
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "process_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Processing complete: {total_images} images from {len(all_processed_items)} items")
        return result
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        """
        Search for images using the Pinecone manager
        
        :param query: Search query
        :param top_k: Number of results to return
        :return: Search results
        """
        if not query:
            return {
                "success": False,
                "message": "Query cannot be empty",
                "results": []
            }
        
        logger.info(f"Searching for: '{query}'")
        
        # Create embedding from query
        query_embedding = self.pinecone_manager.embed_text(query)
        
        # Search in both namespaces
        results = self.pinecone_manager.search(query, top_k=top_k)
        
        # Format results for display
        formatted_results = []
        
        for match in results.get('matches', []):
            result = {}
            
            if match['type'] == 'image':
                # Image result
                metadata = match['metadata']
                result = {
                    'type': 'image',
                    'score': match['score'],
                    'caption': metadata.get('caption', 'No caption available'),
                    'url': metadata.get('url', ''),
                    'source_page': metadata.get('source_page', ''),
                    'tags': metadata.get('tags', [])
                }
            else:
                # Text result
                result = {
                    'type': 'text',
                    'score': match['score'],
                    'content': match['content'],
                    'image_id': match['metadata'].get('image_id', '')
                }
            
            formatted_results.append(result)
        
        return {
            "success": True,
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
    
    def natural_language_search(self, query: str, top_k: int = 5, enhance: bool = True) -> Dict:
        """
        Search for images using natural language via OpenAI interface
        
        :param query: Natural language search query
        :param top_k: Number of results to return
        :param enhance: Whether to enhance the query using OpenAI
        :return: Search results with natural language response
        """
        if not self.query_interface:
            return {
                "success": False,
                "message": "Natural language search requires OpenAI API key",
                "results": []
            }
        
        logger.info(f"Performing natural language search: '{query}'")
        
        # Use the OpenAI query interface
        response = self.query_interface.generate_response(
            query=query,
            top_k=top_k,
            enhance=enhance
        )
        
        return {
            "success": True,
            "query": query,
            "enhanced_query": response.get('enhanced_query'),
            "response": response.get('response'),
            "results_count": len(response.get('results', [])),
            "results": response.get('results', [])
        }

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Boston Public Library Historical Image Search")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Scrape and process images")
    process_parser.add_argument("--queries", type=str, nargs="+", default=["boston common historical"], 
                              help="List of search queries for diversity")
    process_parser.add_argument("--limit-items", type=int, default=2, 
                              help="Maximum number of items to process per query")
    process_parser.add_argument("--max-images", type=int, default=2, 
                              help="Maximum number of images to process per item")
    process_parser.add_argument("--index", type=str, default="historical-images", help="Pinecone index name")
    process_parser.add_argument("--namespace", type=str, default="bpl", help="Pinecone namespace")
    process_parser.add_argument("--save-images", action="store_true", help="Save images locally")
    process_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    process_parser.add_argument("--model", type=str, default="blip", choices=["blip", "git"], help="Image captioning model")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for images")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument("--index", type=str, default="historical-images", help="Pinecone index name")
    search_parser.add_argument("--namespace", type=str, default="bpl", help="Pinecone namespace")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    # Natural language search command
    nl_search_parser = subparsers.add_parser("nl-search", help="Natural language search")
    nl_search_parser.add_argument("--query", type=str, required=True, help="Natural language query")
    nl_search_parser.add_argument("--index", type=str, default="historical-images", help="Pinecone index name")
    nl_search_parser.add_argument("--namespace", type=str, default="bpl", help="Pinecone namespace") 
    nl_search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    nl_search_parser.add_argument("--no-enhance", action="store_true", help="Disable query enhancement")
    
    args = parser.parse_args()
    
    try:
        # Initialize the application
        app = HistoricalImageSearchApp(
            pinecone_index=args.index if hasattr(args, 'index') else "historical-images",
            namespace=args.namespace if hasattr(args, 'namespace') else "bpl",
            caption_model=args.model if hasattr(args, 'model') else "blip",
            output_dir=args.output_dir if hasattr(args, 'output_dir') else "output"
        )
        
        # Process command
        if args.command == "process":
            result = app.scrape_and_process(
                queries=args.queries,
                limit_items=args.limit_items,
                max_images_per_item=args.max_images,
                save_images=args.save_images
            )
            
            print(json.dumps(result, indent=2))
        
        # Search command
        elif args.command == "search":
            result = app.search(
                query=args.query,
                top_k=args.top_k
            )
            
            print(json.dumps(result, indent=2))
        
        # Natural language search command
        elif args.command == "nl-search":
            result = app.natural_language_search(
                query=args.query,
                top_k=args.top_k,
                enhance=not args.no_enhance
            )
            
            # Print natural language response
            print(f"\nQuery: {args.query}")
            if result.get('enhanced_query'):
                print(f"Enhanced query: {result['enhanced_query']}")
            print(f"\nResponse: {result['response']}")
            
            # Print results
            print(f"\nFound {result['results_count']} results:")
            for i, res in enumerate(result['results']):
                print(f"\n{i+1}. {res.get('caption', res.get('content', ''))}")
                if 'url' in res:
                    print(f"   URL: {res['url']}")
                if 'tags' in res and res['tags']:
                    print(f"   Tags: {', '.join(res['tags'][:10])}")
                print(f"   Score: {res['score']}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()