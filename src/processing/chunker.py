from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.processing.base_chunker import BaseChunker
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StructuralChunker(BaseChunker):
    """
    Hybrid chunker: 
    1. Splits structurally based on Markdown headers.
    2. Uses RecursiveCharacterTextSplitter for oversized sections.
    3. Injects document metadata into every final chunk payload.
    """
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, standardized_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_results = []
        for doc in standardized_docs:
            text = doc["text"]
            metadata = doc["metadata"]
            
            title = metadata.get("title", "Unknown Title")
            pub_date = metadata.get("published_date", "Unknown Date")
            
            # Pass 1: Structural Split
            md_splits = self.markdown_splitter.split_text(text)
            
            if not md_splits:
                # Fallback class if no markdown splits were found
                class DummyDoc:
                    def __init__(self, c, m):
                        self.page_content = c
                        self.metadata = m
                md_splits = [DummyDoc(text, {})]
                
            global_chunk_idx = 0
            
            for md_doc in md_splits:
                section_text = md_doc.page_content
                section_meta = md_doc.metadata
                
                section_name = section_meta.get("Header 1", section_meta.get("Header 2", "General Context"))
                
                # Pass 2: Recursive Sub-splitting
                sub_splits = self.recursive_splitter.split_text(section_text)
                
                for sub_text in sub_splits:
                    # Pass 3: Metadata Injection
                    injected_text = (
                        f"Document Title: {title}\n"
                        f"Publication Date: {pub_date}\n"
                        f"Section: {section_name}\n"
                        f"---\n"
                        f"{sub_text}"
                    )
                    
                    chunk_record = {
                        "id": f"{doc['id']}_chunk_{global_chunk_idx}",
                        "parent_id": doc["id"],
                        "text": injected_text,
                        "metadata": {
                            **metadata,
                            "section": section_name,
                            "chunk_index": global_chunk_idx
                        }
                    }
                    chunked_results.append(chunk_record)
                    global_chunk_idx += 1
                    
        return chunked_results