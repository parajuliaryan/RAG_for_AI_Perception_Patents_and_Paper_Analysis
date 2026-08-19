import os
import requests
from typing import List
from src.embeddings.base_embedder import BaseEmbedder

class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "nomic-embed-text"):
        """
        Make sure you have pulled the model in your host terminal first!
        Command: ollama pull nomic-embed-text
        """
        self.model_name = model_name
        # Grabs the host URL from docker-compose, defaults to localhost if running outside docker
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.api_url = f"{self.base_url}/api/embeddings"

    def _get_embedding(self, text: str) -> List[float]:
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json().get("embedding", [])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # In a production app, you might want to batch these or run them asynchronously
        # For our pipeline, we will embed them one by one.
        print(f"Embedding {len(texts)} chunks using Ollama ({self.model_name})...")
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)