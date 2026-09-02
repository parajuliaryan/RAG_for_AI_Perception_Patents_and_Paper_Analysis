from typing import List, Dict, Any
from src.schemas.document import DocumentSchema
from src.processing.pdf_parser import PDFParser
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentStandardizer:
    def __init__(self, use_full_text: bool = True):
        self.use_full_text = use_full_text
        self.pdf_parser = PDFParser()

    def process(self, documents: List[DocumentSchema]) -> List[Dict[str, Any]]:
        standardized_docs = []
        for doc in documents:
            
            content_body = f"Abstract:\n{doc.abstract}"
            
            # Try to fetch full text if enabled
            if self.use_full_text:
                full_text = None
                if getattr(doc, "local_path", None):
                    full_text = self.pdf_parser.extract_text_from_file(doc.local_path)
                elif doc.pdf_url:
                    full_text = self.pdf_parser.extract_text_from_url(doc.pdf_url, doc_id=doc.id)
                    
                if full_text:
                    content_body = f"Full Text:\n{full_text}"
                    logger.info(f"Successfully incorporated full text for {doc.id}")
                else:
                    logger.debug(f"Falling back to abstract for {doc.id}")

            combined_text = (
                f"Title: {doc.title}\n"
                f"Authors: {', '.join(doc.authors)}\n"
                f"Published Date: {doc.published_date}\n\n"
                f"{content_body}"
            )
            
            # Save the final text representation to disk for transparency
            import src.config as cfg
            safe_id = doc.id.replace('/', '_')
            text_path = cfg.EXTRACTED_TEXT_DIR / f"{safe_id}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(combined_text)
            
            standardized_docs.append({
                "id": doc.id,
                "text": combined_text,
                "metadata": {
                    "source": doc.source,
                    "title": doc.title,
                    "published_date": doc.published_date,
                    "pdf_url": doc.pdf_url
                }
            })
        return standardized_docs