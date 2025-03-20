# File: load_pinecone_audio.py
import os
import numpy as np
import torch
import logging
import time
from typing import Dict, Optional, List, Union
import requests
from PIL import Image
import io

# from your existing code:
from load_pinecone_images import MultimodalPineconeManager  # or rename if needed

# Additional imports for audio
from transformers import AutoProcessor, AutoModel  # Example for CLAP or other audio model

logger = logging.getLogger(__name__)

class AudioCapablePineconeManager(MultimodalPineconeManager):
    """
    Extends the existing MultimodalPineconeManager to handle audio embeddings.
    You still have text + image methods, plus new audio code.
    """

    def __init__(self, index_name: str, namespace: str = "bpl_audio"):
        super().__init__(index_name=index_name, namespace=namespace)

        # For demonstration, let's suppose we use a "clap" huggingface model:
        # E.g. laion/clap-htsat-fused
        # (Check https://huggingface.co/models?q=clap for available models)
        # You must `pip install git+https://github.com/lucidrains/laion_clap.git`
        # or whichever library is required

        self.audio_model_name = "laion/clap-htsat-fused"
        try:
            self.audio_processor = AutoProcessor.from_pretrained(self.audio_model_name)
            self.audio_model = AutoModel.from_pretrained(self.audio_model_name)
        except Exception as e:
            logger.error(f"Could not load audio model {self.audio_model_name}: {e}")
            self.audio_processor = None
            self.audio_model = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.audio_model is not None:
            self.audio_model.to(self.device)
            logger.info(f"Loaded audio model '{self.audio_model_name}' on {self.device}")

    def embed_audio(self, audio_source: Union[str, bytes]) -> Optional[List[float]]:
        """
        Convert audio to an embedding vector, using a CLAP-like model or any 
        other audio embedding approach.

        :param audio_source: Path to local audio file, or raw bytes
        :return: A 768- or 512-dim embedding (depends on your chosen model),
                 or None if there's an error
        """
        if self.audio_model is None or self.audio_processor is None:
            logger.error("Audio model not initialized; cannot embed audio.")
            return None

        try:
            # Load the audio
            if isinstance(audio_source, str):
                # If it's a local path, read the file as bytes
                with open(audio_source, "rb") as f:
                    audio_data = f.read()
            else:
                # If already raw bytes
                audio_data = audio_source

            # We'll do a basic approach: pass the raw wave data to the processor
            inputs = self.audio_processor(audio_data, sampling_rate=16000, return_tensors="pt")
            # Move to device
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)

            # forward pass
            with torch.no_grad():
                audio_features = self.audio_model(**inputs)
                # Example: some models produce `audio_embeds` or `last_hidden_state`
                # Check your model's documentation for the correct output key
                # For demonstration, let's assume there's a 'pooler_output' or something similar
                # If not, you might do something like:
                if hasattr(audio_features, "pooler_output"):
                    embedding = audio_features.pooler_output[0].cpu().numpy()
                else:
                    # Fallback: take the mean of last_hidden_state
                    embedding = audio_features.last_hidden_state.mean(dim=1)[0].cpu().numpy()

            # Normalize
            embedding_norm = embedding / np.linalg.norm(embedding)
            return embedding_norm.tolist()

        except Exception as e:
            logger.error(f"Error embedding audio: {e}")
            return None

    def add_audios(
        self, 
        audio_sources: List[Union[str, bytes]], 
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ):
        """
        Similar to add_images, but for audio. 
        :param audio_sources: Local file paths or raw bytes
        :param metadatas: Optional metadata dicts
        :param ids: Unique IDs for each
        :param batch_size: how many vectors to upsert per batch
        """
        if metadatas is None:
            metadatas = [{} for _ in audio_sources]
        if ids is None:
            ids = [f"audio_{i}" for i in range(len(audio_sources))]

        for i in range(0, len(audio_sources), batch_size):
            batch_sources = audio_sources[i : i+batch_size]
            batch_metas = metadatas[i : i+batch_size]
            batch_ids = ids[i : i+batch_size]

            vectors_to_upsert = []
            for src, meta, doc_id in zip(batch_sources, batch_metas, batch_ids):
                emb = self.embed_audio(src)
                if emb is None:
                    continue  # skip if we can't embed

                # flatten metadata
                flattened = self._flatten_metadata(meta)
                flattened["vector_type"] = "audio"

                # store
                vectors_to_upsert.append({
                    "id": doc_id,
                    "values": emb,
                    "metadata": flattened
                })

            if vectors_to_upsert:
                self.index.upsert(
                    vectors=vectors_to_upsert,
                    namespace=f"{self.namespace}_audio"
                )
                logger.info(f"Upserted {len(vectors_to_upsert)} audio embeddings")

            time.sleep(1)

    def query_audio(self,
                    audio_source: Union[str, bytes],
                    top_k: int = 5,
                    filter_dict: Optional[Dict] = None) -> Dict:
        """
        Query Pinecone with an audio clip to find similar audio docs.

        :param audio_source: Path or bytes
        :param top_k: Number of results
        :param filter_dict: Additional Pinecone filter
        :return: Dict with 'matches'
        """
        emb = self.embed_audio(audio_source)
        if emb is None:
            return {"error": "Failed to embed audio"}

        results = self.index.query(
            vector=emb,
            top_k=top_k,
            namespace=f"{self.namespace}_audio",
            include_metadata=True,
            filter=filter_dict
        )
        return results or {}
