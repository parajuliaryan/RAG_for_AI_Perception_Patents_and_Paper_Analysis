from pydantic import BaseModel
from typing import List, Optional

class DocumentSchema(BaseModel):
    source: str          # e.g., "arxiv" or "patent"
    id: str              # arXiv ID or Patent Number
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    pdf_url: Optional[str] = None