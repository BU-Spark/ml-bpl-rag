import os
import io
import sys
import torch
import logging
import requests
import re
from PIL import Image
from typing import Union, Optional, Dict, List
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk import word_tokenize, pos_tag, download, data
from nltk.corpus import stopwords

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def ensure_nltk_dependencies():
    for resource in ["punkt", "stopwords", "averaged_perceptron_tagger"]:
        try:
            data.find(resource)
        except LookupError:
            download(resource, quiet=True)


class ImageCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BLIP model on {self.device}")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        ensure_nltk_dependencies()

    def load_image(self, source: Union[str, bytes, Image.Image]) -> Optional[Image.Image]:
        try:
            if isinstance(source, Image.Image):
                return source
            if isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    response = requests.get(source)
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content))
                return Image.open(source)
            if isinstance(source, bytes):
                return Image.open(io.BytesIO(source))
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
        return None

    def generate_caption(self, source: Union[str, bytes, Image.Image],
                         detail_level: str = "detailed",
                         max_length: int = 150) -> Optional[str]:
        image = self.load_image(source)
        if image is None:
            return None

        try:
            prompt = {
                "basic": "",
                "detailed": "a detailed description of",
                "analysis": "a comprehensive visual analysis of"
            }.get(detail_level, "")

            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_length=max_length, num_beams=5,
                                         min_length=10, top_p=0.9, repetition_penalty=1.5,
                                         length_penalty=1.0, temperature=1.0)
            caption = self.processor.decode(output[0], skip_special_tokens=True).strip()
            return self.clean_caption(caption)
        except Exception as e:
            logger.error(f"Captioning error: {e}")
            return None

    def clean_caption(self, text: str) -> str:
        text = text.replace("\n", " ").strip()
        prefixes = ["a photo of", "this is a picture of", "this image shows"]
        for p in prefixes:
            if text.lower().startswith(p):
                return " ".join(text.split()[3:])
        return text

    def generate_tags(self, source: Union[str, bytes, Image.Image],
                      max_tags: int = 10) -> Optional[List[str]]:
        caption = self.generate_caption(source)
        if not caption:
            return None

        try:
            words = word_tokenize(caption.lower())
            tagged = pos_tag(words)
            stop_words = set(stopwords.words('english'))

            tags = []
            for word, tag in tagged:
                if (tag.startswith('NN') or tag.startswith('JJ')) and word not in stop_words:
                    clean = re.sub(r'\W+', '', word)
                    if clean and clean not in tags:
                        tags.append(clean)
            return tags[:max_tags]
        except Exception as e:
            logger.error(f"Tagging error: {e}")
            return caption.lower().split()[:max_tags]

    def analyze_image(self, source: Union[str, bytes, Image.Image]) -> Dict:
        image = self.load_image(source)
        if image is None:
            return {"error": "Failed to load image"}

        return {
            "caption": self.generate_caption(image, "basic", max_length=50),
            "description": self.generate_caption(image, "detailed", max_length=200),
            "tags": self.generate_tags(image, max_tags=15),
            "image_metadata": {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode
            }
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_captioning.py <image_path_or_url>")
        sys.exit(1)

    path = sys.argv[1]
    captioner = ImageCaptioner()

    print("Caption:", captioner.generate_caption(path))
    print("Tags:", captioner.generate_tags(path))
    print("Full analysis:")
    print(captioner.analyze_image(path))
