import requests
from bs4 import BeautifulSoup
import os
import json
import re
from typing import List, Dict
import logging
from urllib.parse import urljoin, urlparse

class DigitalCommonwealthScraper:
    def __init__(self, base_url: str = "https://www.digitalcommonwealth.org"):
        """
        Initialize the scraper with base URL and logging
        
        :param base_url: Base URL for Digital Commonwealth
        """
        self.base_url = base_url
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_page(self, url: str) -> requests.Response:
        """
        Fetch webpage content with error handling
        
        :param url: URL to fetch
        :return: Response object
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_json_metadata(self, url: str) -> Dict:
        """
        Extract JSON metadata from the page
        
        :param url: URL of the page
        :return: Dictionary of metadata
        """
        json_url = f"{url}.json"
        response = self.fetch_page(json_url)
        
        if response:
            try:
                return response.json()
            except json.JSONDecodeError:
                self.logger.error(f"Could not parse JSON from {json_url}")
                return {}
        return {}
    
    def extract_images(self, url: str) -> List[Dict]:
        """
        Extract images from the page
        
        :param url: URL of the page to scrape
        :return: List of image dictionaries
        """
        # Fetch page content
        response = self.fetch_page(url)
        if not response:
            return []
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract JSON metadata
        metadata = self.extract_json_metadata(url)
        
        # List to store images
        images = []
        
        # Strategy 1: Look for image viewers or specific image containers
        image_containers = [
            soup.find('div', class_='viewer-container'),
            soup.find('div', class_='image-viewer'),
            soup.find('div', id='image-container')
        ]
        
        # Strategy 2: Find all image tags
        img_tags = soup.find_all('img')
        
        # Combine image sources
        for img in img_tags:
            # Get image source
            src = img.get('src')
            if not src:
                continue
            
            # Resolve relative URLs
            full_src = urljoin(url, src)
            
            # Extract alt text or use filename
            alt = img.get('alt', os.path.basename(urlparse(full_src).path))
            
            # Create image dictionary
            image_info = {
                'url': full_src,
                'alt': alt,
                'source_page': url
            }
            
            # Try to add metadata if available
            if metadata:
                try:
                    # Extract relevant metadata from JSON if possible
                    image_info['metadata'] = {
                        'title': metadata.get('data', {}).get('attributes', {}).get('title_info_primary_tsi'),
                        'description': metadata.get('data', {}).get('attributes', {}).get('abstract_tsi'),
                        'subject': metadata.get('data', {}).get('attributes', {}).get('subject_geographic_sim')
                    }
                except Exception as e:
                    self.logger.warning(f"Error extracting metadata: {e}")
            
            images.append(image_info)
        
        return images
    
    def download_images(self, images: List[Dict], output_dir: str = 'downloaded_images') -> List[str]:
        """
        Download images to local directory
        
        :param images: List of image dictionaries
        :param output_dir: Directory to save images
        :return: List of downloaded file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        downloaded_files = []
        
        for i, image in enumerate(images):
            try:
                response = requests.get(image['url'], headers=self.headers)
                response.raise_for_status()
                
                # Generate filename
                ext = os.path.splitext(urlparse(image['url']).path)[1] or '.jpg'
                filename = os.path.join(output_dir, f'image_{i}{ext}')
                
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(filename)
                self.logger.info(f"Downloaded: {filename}")
                
            except Exception as e:
                self.logger.error(f"Error downloading {image['url']}: {e}")
        
        return downloaded_files