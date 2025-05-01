# File: audio_embedding.py

import os
import time
import torch
import logging
import numpy as np
import requests
from typing import Dict, Optional, List, Union
from transformers import AutoProcessor, AutoModel

from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioEmbedder:
    """
    Manages audio embedding and upserting to Pinecone using CLAP-style models.
    """

    def __init__(self, index_name: str, namespace: str = "bpl_audio"):
        self.namespace = namespace
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_model_name = "laion/clap-htsat-fused"

        try:
            self.audio_processor = AutoProcessor.from_pretrained(self.audio_model_name)
            self.audio_model = AutoModel.from_pretrained(self.audio_model_name).to(self.device)
            logger.info(f"✅ Loaded audio model '{self.audio_model_name}' on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load audio model '{self.audio_model_name}': {e}")
            self.audio_processor = None
            self.audio_model = None

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Missing PINECONE_API_KEY in environment.")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)

    def _flatten_metadata(self, metadata: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        flat = {}
        for k, v in metadata.items():
            key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat.update(self._flatten_metadata(v, key, sep))
            else:
                flat[key] = v
        return flat

    def embed_audio(self, audio_source: Union[str, bytes]) -> Optional[List[float]]:
        if not self.audio_model or not self.audio_processor:
            logger.error("⚠️ Audio model is not initialized.")
            return None

        try:
            if isinstance(audio_source, str):
                with open(audio_source, "rb") as f:
                    audio_data = f.read()
            else:
                audio_data = audio_source

            inputs = self.audio_processor(audio_data, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.audio_model(**inputs)
                if hasattr(output, "pooler_output"):
                    embedding = output.pooler_output[0].cpu().numpy()
                else:
                    embedding = output.last_hidden_state.mean(dim=1)[0].cpu().numpy()

            return (embedding / np.linalg.norm(embedding)).tolist()

        except Exception as e:
            logger.error(f"❌ Error embedding audio: {e}")
            return None

    def add_audios(
        self,
        audio_sources: List[Union[str, bytes]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ):
        metadatas = metadatas or [{} for _ in audio_sources]
        ids = ids or [f"audio_{i}" for i in range(len(audio_sources))]

        for i in range(0, len(audio_sources), batch_size):
            batch_sources = audio_sources[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            vectors = []
            for src, meta, doc_id in zip(batch_sources, batch_metas, batch_ids):
                embedding = self.embed_audio(src)
                if embedding:
                    vectors.append({
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            **self._flatten_metadata(meta),
                            "vector_type": "audio"
                        }
                    })

            if vectors:
                self.index.upsert(vectors=vectors, namespace=self.namespace)
                logger.info(f"🎧 Upserted {len(vectors)} audio embeddings to namespace '{self.namespace}'")

            time.sleep(1)

    def query_audio(
        self,
        audio_source: Union[str, bytes],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        embedding = self.embed_audio(audio_source)
        if embedding is None:
            return {"error": "Failed to generate audio embedding."}

        return self.index.query(
            vector=embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter=filter_dict
        )
