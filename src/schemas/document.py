from pydantic import BaseModel
from typing import List, Optional

class DocumentSchema(BaseModel):
    source: str          # e.g., "arxiv", "epo_ops", or "patent"
    id: str              # arXiv ID, Patent Number, or filename stem
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    pdf_url: Optional[str] = None
    local_path: Optional[str] = None
    full_text: Optional[str] = None
    publication_number: Optional[str] = None
    affiliated_institutions: Optional[List[str]] = None
    patent_classifications: Optional[List[str]] = None
    priority_or_grant_date: Optional[str] = None
    independent_claims: Optional[List[str]] = None