import os
import torch
import logging
from typing import Union, Optional, Dict, List
from PIL import Image
import io
import requests
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up NLTK data path
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk_data_env = os.getenv("NLTK_DATA", nltk_data_dir)
os.environ["NLTK_DATA"] = nltk_data_env

# Load environment variables
load_dotenv()

class ImageCaptioner:
    """
    Generate detailed descriptions of images using vision language models
    """
    
    def __init__(self, model_type: str = "blip"):
        """
        Initialize the captioner with the specified model type
        
        :param model_type: Type of model to use. Options: "blip", "git", "llava"
        """
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing ImageCaptioner with model type: {model_type} on {self.device}")
        
        if model_type == "blip":
            # BLIP model - good balance of quality and speed
            logger.info("Loading BLIP model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(self.device)
            
        elif model_type == "git":
            # GIT model - good for detailed descriptions
            logger.info("Loading GIT model...")
            self.processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/git-large-coco"
            ).to(self.device)
            
        elif model_type == "llava":
            # LLaVA model - more complex captioning
            logger.info("Loading LLaVA model...")
            try:
                from llava.model import LlavaLlamaForCausalLM
                from llava.conversation import conv_templates
                from llava.utils import disable_torch_init
                from llava.mm_utils import process_images, tokenizer_image_token
                
                self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
                self.model = LlavaLlamaForCausalLM.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf", 
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.conv_mode = "llava_v1"
                self.tokenizer = self.processor.tokenizer
            except ImportError:
                logger.error("LLaVA dependencies not installed. Falling back to BLIP.")
                self.model_type = "blip"
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large"
                ).to(self.device)
        else:
            # Default to BLIP if model_type not recognized
            logger.warning(f"Unknown model type: {model_type}. Defaulting to BLIP.")
            self.model_type = "blip"
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(self.device)
    
    def load_image(self, image_source: Union[str, bytes, Image.Image]) -> Optional[Image.Image]:
        """
        Load image from various sources
        
        :param image_source: Path, URL, bytes, or PIL Image
        :return: PIL Image or None if loading fails
        """
        try:
            if isinstance(image_source, Image.Image):
                return image_source
            
            if isinstance(image_source, str):
                # Check if it's a URL
                if image_source.startswith(('http://', 'https://')):
                    response = requests.get(image_source)
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content))
                else:
                    # Local file path
                    return Image.open(image_source)
            
            if isinstance(image_source, bytes):
                return Image.open(io.BytesIO(image_source))
            
            raise ValueError("Unsupported image source type")
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def generate_caption(self, image_source: Union[str, bytes, Image.Image], 
                         detail_level: str = "detailed",
                         max_length: int = 150) -> Optional[str]:
        """
        Generate a caption for the image
        
        :param image_source: Path, URL, bytes, or PIL Image
        :param detail_level: Level of detail for the caption (basic, detailed, analysis)
        :param max_length: Maximum caption length in tokens
        :return: Caption text or None if generation fails
        """
        image = self.load_image(image_source)
        
        if image is None:
            return None
        
        try:
            if self.model_type == "blip":
                # Use BLIP model for captioning
                prompt = ""
                if detail_level == "detailed":
                    prompt = "a detailed description of"
                elif detail_level == "analysis":
                    prompt = "a comprehensive visual analysis of"
                
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    min_length=10,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1.0
                )
                
                caption = self.processor.decode(output[0], skip_special_tokens=True)
                
            elif self.model_type == "git":
                # Use GIT model for captioning
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                generated_ids = self.model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=max_length,
                    num_beams=5,
                    min_length=10
                )
                
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            elif self.model_type == "llava":
                # Use LLaVA model for captioning
                from llava.conversation import conv_templates
                from llava.utils import disable_torch_init
                from llava.mm_utils import process_images, tokenizer_image_token
                
                disable_torch_init()
                conv = conv_templates[self.conv_mode].copy()
                
                if detail_level == "basic":
                    prompt = "Provide a short caption for this image."
                elif detail_level == "detailed":
                    prompt = "Provide a detailed description of this image."
                else:  # analysis
                    prompt = "Analyze this image in detail. Describe what you see, including subjects, setting, colors, composition, and any notable elements."
                
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                image_tensor = process_images([image], self.processor)
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX=self.processor.tokenizer.encode("<image>")[1])
                
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids=input_ids.unsqueeze(0).to(self.device),
                        images=image_tensor.to(self.device),
                        max_new_tokens=max_length,
                        temperature=0.7,
                        do_sample=True
                    )
                
                outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[0]:]).strip()
                caption = outputs.replace("<s>", "").replace("</s>", "")
            
            # Clean up caption
            caption = caption.replace("\n", " ").strip()
            if caption.lower().startswith(("a photo of", "this is a picture of", "this image shows")):
                # Remove common prefixes for cleaner caption
                words = caption.split()
                if len(words) > 4:  # Only trim if caption is long enough
                    caption = " ".join(words[3:])
            
            logger.info(f"Generated caption: {caption[:50]}...")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return None
    
    def generate_tags(self, image_source: Union[str, bytes, Image.Image], 
                     max_tags: int = 10) -> Optional[List[str]]:
        """
        Generate a list of tags/keywords for the image
        
        :param image_source: Path, URL, bytes, or PIL Image
        :param max_tags: Maximum number of tags to generate
        :return: List of tags or None if generation fails
        """
        # First get a caption
        caption = self.generate_caption(image_source, detail_level="detailed")
        
        if not caption:
            return None
        
        # Extract keywords from caption
        try:
            # Simple keyword extraction based on nouns and adjectives
            # This is a simplified approach - for production use, consider using NLP libraries
            import re
            import nltk
            from nltk.corpus import stopwords
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
            
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
            
            # Tokenize and tag parts of speech
            words = nltk.word_tokenize(caption.lower())
            tagged_words = nltk.pos_tag(words)
            
            # Keep nouns and adjectives, filter out stopwords
            stop_words = set(stopwords.words('english'))
            tags = []
            
            for word, tag in tagged_words:
                # Keep only nouns (NN*) and adjectives (JJ*)
                is_noun = tag.startswith('NN')
                is_adjective = tag.startswith('JJ')
                
                if (is_noun or is_adjective) and word not in stop_words and len(word) > 2:
                    # Clean the word of any punctuation
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    if clean_word and clean_word not in tags:
                        tags.append(clean_word)
            
            # Limit to max_tags
            return tags[:max_tags]
            
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            # If NLTK processing fails, fall back to simple word splitting
            words = caption.lower().split()
            tags = list(set([w for w in words if len(w) > 3 and w not in ['and', 'the', 'with', 'that', 'this']]))
            return tags[:max_tags]
    
    def analyze_image(self, image_source: Union[str, bytes, Image.Image]) -> Dict:
        """
        Complete image analysis including caption, tags, and other features
        
        :param image_source: Path, URL, bytes, or PIL Image
        :return: Dictionary with analysis results
        """
        image = self.load_image(image_source)
        
        if image is None:
            return {"error": "Failed to load image"}
        
        analysis = {}
        
        # Generate basic caption
        analysis["caption"] = self.generate_caption(image, detail_level="basic", max_length=50)
        
        # Generate detailed description
        analysis["description"] = self.generate_caption(image, detail_level="detailed", max_length=200)
        
        # Generate tags
        analysis["tags"] = self.generate_tags(image, max_tags=15)
        
        # Add basic image metadata
        try:
            analysis["image_metadata"] = {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode
            }
        except:
            analysis["image_metadata"] = {}
        
        return analysis

# Example usage
if __name__ == "__main__":
    # Test the captioner
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_captioning.py <image_path_or_url>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    captioner = ImageCaptioner(model_type="blip")
    print("Generating caption...")
    caption = captioner.generate_caption(image_path, detail_level="detailed")
    print(f"Caption: {caption}")
    
    print("\nGenerating tags...")
    tags = captioner.generate_tags(image_path)
    print(f"Tags: {tags}")
    
    print("\nFull analysis...")
    analysis = captioner.analyze_image(image_path)
    for key, value in analysis.items():
        print(f"{key}: {value}")