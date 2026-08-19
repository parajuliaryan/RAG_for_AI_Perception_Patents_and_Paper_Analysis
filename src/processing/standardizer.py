from typing import List, Dict, Any
from src.schemas.document import DocumentSchema

class DocumentStandardizer:
    def process(self, documents: List[DocumentSchema]) -> List[Dict[str, Any]]:
        standardized_docs = []
        for doc in documents:
            combined_text = (
                f"Title: {doc.title}\n"
                f"Authors: {', '.join(doc.authors)}\n"
                f"Published Date: {doc.published_date}\n\n"
                f"Abstract:\n{doc.abstract}"
            )
            
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