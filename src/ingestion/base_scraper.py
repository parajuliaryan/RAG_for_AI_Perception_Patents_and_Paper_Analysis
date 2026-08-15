from abc import ABC, abstractmethod
from typing import List
from src.schemas.document import DocumentSchema

class BaseScraper(ABC):
    
    @abstractmethod
    def fetch(self, query: str, max_results: int) -> List[DocumentSchema]:
        # Takes a search query and returns a list of standardized Document schemas.
        pass