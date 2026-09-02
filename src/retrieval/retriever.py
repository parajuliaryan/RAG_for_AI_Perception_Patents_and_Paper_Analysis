"""
Retriever — query-time semantic search abstraction.

Wraps ChromaStore to create a clean separation of concerns:
  - ChromaStore  → owns data persistence and vector storage
  - Retriever    → owns query-time logic (filtering, top-k, future: re-ranking)

Future extension points in this class:
  - Hybrid search (BM25 + dense vectors)
  - MMR (Maximal Marginal Relevance) for result diversity
  - Metadata pre-filtering (e.g., source == "arxiv")
  - Cross-encoder re-ranking
"""

from typing import List, Dict, Any

from src.databases.vectorstore.chroma_store import ChromaStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Provides a clean retrieval interface over a ChromaStore instance.

    Args:
        vector_store: An initialized ChromaStore instance (injected by RAGService).
    """

    def __init__(self, vector_store: ChromaStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3, where_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most semantically similar chunks for a given query.

        Args:
            query:  The user's natural language research question.
            top_k:  Number of chunks to retrieve from the vector store.
            where_filter: Optional metadata filter dict for ChromaDB.

        Returns:
            A list of chunk dicts, each containing:
              - id                  (str)   Chunk identifier
              - text                (str)   Raw chunk text
              - metadata            (dict)  Source metadata (title, date, url, etc.)
              - similarity_distance (float) Cosine distance (lower = more similar)
        """
        logger.info(f"Retrieving top-{top_k} chunks for query: '{query[:80]}...'")
        results = self.vector_store.search(query=query, top_k=top_k, where_filter=where_filter)
        logger.debug(f"Retrieved {len(results)} chunk(s) from vector store.")
        return results

