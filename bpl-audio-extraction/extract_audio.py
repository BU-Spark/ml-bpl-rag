# File: extract_audio.py
import os
import argparse
import json
import logging
from typing import List, Dict, Optional
import time
from dotenv import load_dotenv

from load_pinecone_audio import AudioCapablePineconeManager  # from the snippet above

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AudioExtractor:
    """
    Similar to ImageExtractor, but for audio.
    You might parse metadata, find direct audio links, and store them in Pinecone.
    """

    def __init__(self, pinecone_index: str, namespace: str = "bpl_audio"):
        self.pinecone_index = pinecone_index
        self.namespace = namespace
        self.pinecone_manager = AudioCapablePineconeManager(index_name=pinecone_index, 
                                                            namespace=namespace)
        logger.info(f"AudioExtractor initialized with index={pinecone_index}, namespace={namespace}")

    def extract_from_metadata_json(self, json_path: str) -> None:
        """
        Suppose the JSON has a structure like:
        {
          "Data": [
            { "id": "some-audio-id", "attributes": {..., "audio_url":"..."} },
            ...
          ]
        }
        or it’s just a dictionary keyed by ID. 
        We'll look for a field "audio_url" or something similar.
        """
        logger.info(f"Loading metadata from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        # You might need to adapt for your data's structure
        if isinstance(data, dict) and "Data" in data:
            records = data["Data"]
        elif isinstance(data, list):
            records = data
        else:
            # fallback if different structure
            records = []

        audio_sources = []
        metadatas = []
        for record in records:
            # record might look like { "id": "...", "attributes":{...} }
            # find a field that has an audio URL
            audio_url = None
            if "attributes" in record and "audio_url" in record["attributes"]:
                audio_url = record["attributes"]["audio_url"]
            if audio_url:
                # We'll store it for embedding
                audio_sources.append(audio_url)
                # maybe you store the entire record as metadata
                meta = {"id": record.get("id", "unknown")}
                meta.update(record.get("attributes", {}))
                metadatas.append(meta)

        if audio_sources:
            logger.info(f"Found {len(audio_sources)} audio references in {json_path}")
            self.pinecone_manager.add_audios(audio_sources, metadatas=metadatas)
        else:
            logger.warning(f"No audio URLs found in the metadata JSON at {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and embed audio data for BPL.")
    parser.add_argument("--metadata-json", required=True, 
                        help="Path to a JSON file that references audio items.")
    parser.add_argument("--pinecone-index", required=True, 
                        help="Name of the Pinecone index.")
    parser.add_argument("--namespace", default="bpl_audio",
                        help="Namespace to store audio vectors in Pinecone.")
    args = parser.parse_args()

    extractor = AudioExtractor(pinecone_index=args.pinecone_index, namespace=args.namespace)
    extractor.extract_from_metadata_json(args.metadata_json)
    print("Done extracting audio. Check Pinecone for stored embeddings!")

if __name__ == "__main__":
    main()
