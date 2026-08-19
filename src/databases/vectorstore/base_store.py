from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):

    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Takes chunked documents, embeds them using the injected embedder, 
        and saves them to the persistent index.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds a search query and returns the top_k closest matching chunks.
        """
        pass