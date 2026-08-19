from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        #Converts a list of text chunks into a list of vector embeddings.
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        #Converts a single search query into a vector embedding.
        pass