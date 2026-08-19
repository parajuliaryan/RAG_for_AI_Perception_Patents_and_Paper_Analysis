from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, standardized_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass