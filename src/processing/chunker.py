from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.processing.base_chunker import BaseChunker

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, standardized_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_results = []
        for doc in standardized_docs:
            text_splits = self.splitter.split_text(doc["text"])
            
            for i, split in enumerate(text_splits):
                chunk_record = {
                    "id": f"{doc['id']}_chunk_{i}",
                    "parent_id": doc["id"],
                    "text": split,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i
                    }
                }
                chunked_results.append(chunk_record)
                
        return chunked_results