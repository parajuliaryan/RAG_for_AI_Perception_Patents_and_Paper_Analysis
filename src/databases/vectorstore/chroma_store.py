import os
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from src.databases.vectorstore.base_store import BaseVectorStore
from src.embeddings.base_embedder import BaseEmbedder

class ChromaStore(BaseVectorStore):
    def __init__(
        self, 
        embedder: BaseEmbedder, 
        collection_name: str = "ai_perception_docs",
        persist_dir: str = "data/vector_store"
    ):
        self.embedder = embedder
        self.persist_dir = persist_dir
        
        # Ensure destination directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create collection with cosine distance
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all metadata fields are compatible with Chroma (no None or nested types)."""
        clean_meta = {}
        for k, v in metadata.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        return clean_meta

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            print("No chunks provided to store.")
            return

        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_documents(texts)

        print(f"Upserting {len(ids)} records into Chroma collection '{self.collection.name}'...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print("Storage complete.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
                formatted_results.append({
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": meta,
                    "similarity_distance": dist
                })

        return formatted_results